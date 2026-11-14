// PNG adapter: the bridge between the legacy dependency-free PNG codec and the
// ImageTensor contract.
//
// Old PNG contract (src/core/png.js):
//   decode -> { width, height, data: Uint16Array RGBA, bitDepth: 8|16, max,
//               hasAlpha, info }
//             Every input is rescaled to nominal 16-bit (an 8-bit file is
//             expanded with /255*65535 scaling); output is always 4 channels.
//   encode <- { width, height, channels: 1..4, bitDepth: 8|16, data, text? }
//
// New contract exposed here: ImageTensor with explicit channelOrder,
// valueRange and transfer. Every conversion the legacy format cannot carry
// without loss is either refused or performed only on demand with a reported
// loss. Nothing is ever silently flattened to 8-bit.

import { decodePNG, encodePNG } from '../../../src/core/png.js';
import { KIND_MAX, isFloatKind } from './format.js';
import { validateTensor, quantizeFloat, convertInt } from './tensor.js';

/**
 * Decode PNG bytes into an RGBA ImageTensor.
 * The legacy codec always emits nominal 16-bit samples, so the tensor kind is
 * u16 even when the source file was 8-bit; the source bit depth is preserved
 * in metadata together with the codec's own info string.
 */
export async function decodePNGTensor(bytes) {
  const img = await decodePNG(bytes);
  const tensor = {
    width: img.width,
    height: img.height,
    channels: 4,
    kind: 'u16',
    data: img.data,
    channelOrder: ['R', 'G', 'B', 'A'],
    valueRange: { min: 0, max: img.max ?? 65535 },
    transfer: 'srgb',
    metadata: {
      format: 'png',
      sourceBitDepth: img.sourceBitDepth ?? img.bitDepth,
      sourceMax: img.max,
      hasAlpha: img.hasAlpha,
      codecInfo: img.info,
    },
  };
  validateTensor(tensor);
  return tensor;
}

function pngEncodeOptions(tensor, { allowLoss = false, bitDepth } = {}) {
  validateTensor(tensor);
  if (tensor.channels !== 4 || tensor.channelOrder.join(',') !== 'R,G,B,A') {
    throw new Error(
      `PNG encode: only 4-channel RGBA is representable, got ${tensor.channels} channel(s) ` +
      `[${tensor.channelOrder.join(', ')}]. Extra/non-RGBA channels must use the FFI container.`,
    );
  }
  let work = tensor;
  const losses = [];
  if (isFloatKind(tensor.kind)) {
    if (!allowLoss) {
      throw new Error(
        'PNG encode: float samples are not representable in PNG without quantisation; ' +
        'pass { allowLoss: true } or use encodeFFI for exact output',
      );
    }
    const q = quantizeFloat(tensor, bitDepth === 8 ? 'u8' : 'u16', { clamp: true });
    work = q.tensor;
    losses.push(q.loss);
  }
  const targetDepth = bitDepth ?? (work.kind === 'u8' ? 8 : 16);
  if (targetDepth !== 8 && targetDepth !== 16) {
    throw new Error(`PNG encode: bitDepth must be 8 or 16, got ${targetDepth}`);
  }
  const targetKind = targetDepth === 8 ? 'u8' : 'u16';
  if (work.kind !== targetKind) {
    if (targetDepth === 8 && !allowLoss) {
      throw new Error(
        `PNG encode: writing ${work.kind} into an 8-bit PNG loses depth; ` +
        'pass { allowLoss: true, bitDepth: 8 } to acknowledge',
      );
    }
    const c = convertInt(work, targetKind);
    work = c.tensor;
    if (c.loss) losses.push(c.loss);
  }
  // PNG cannot store an arbitrary valueRange either: integer samples must be
  // the standard 0..max domain.
  const expectedMax = KIND_MAX[targetKind];
  const workRange = work.valueRange ?? { min: 0, max: expectedMax };
  if (workRange.min !== 0 || workRange.max !== expectedMax) {
    throw new Error(
      `PNG encode: valueRange [${workRange.min}, ${workRange.max}] is not the standard ` +
      `0..${expectedMax} domain; rescale explicitly or use FFI`,
    );
  }
  return { work, targetDepth, losses };
}

/**
 * Encode an RGBA ImageTensor to PNG bytes.
 * @param {object} tensor
 * @param {{allowLoss?:boolean, bitDepth?:8|16, text?:Array}} [opts]
 * @returns {Promise<{bytes:Uint8Array, losses:Array}>}
 */
export async function encodePNGTensor(tensor, opts = {}) {
  const { work, targetDepth, losses } = pngEncodeOptions(tensor, opts);
  const text = buildText(tensor);
  const bytes = await encodePNG({
    width: work.width,
    height: work.height,
    channels: 4,
    bitDepth: targetDepth,
    data: work.data,
    text,
  });
  return { bytes, losses, bitDepth: targetDepth };
}

function buildText(tensor) {
  const text = [{ keyword: 'Software', value: 'focus-fusion imageio' }];
  const md = tensor.metadata || {};
  if (typeof md.title === 'string' && md.title.length <= 60) text.push({ keyword: 'Title', value: md.title });
  return text;
}
