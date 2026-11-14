// Adapters between the ImageTensor contract and the focus-fusion engine's
// in-memory pixel contract.
//
// The engine (src/core/fuse.js) consumes slices as
//   { name, width, height, rgba: Uint16Array RGBA, sRGB, nominal 16-bit }
// and produces a linear-light Float32 planar RGB result plus an 8/16-bit sRGB
// render. This module makes every step between the two worlds explicit:
//   * a 16-bit sRGB RGBA tensor reaches the engine byte-for-byte, no loss;
//   * extra channels are refused unless the caller opts into dropping them;
//   * float or high-bit-depth inputs are quantised only on demand and every
//     such loss is returned, never applied silently.

import { isFloatKind } from './format.js';
import { validateTensor, convertInt, quantizeFloat, rangeViolations } from './tensor.js';

function srgbLinearToEncoded16(c) {
  if (!(c > 0)) return 0;
  if (c >= 1) return 65535;
  const v = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  return Math.round(v * 65535);
}

// Re-encode linear-light u16 samples (domain 0..65535 interpreted as 0..1
// linear) onto the sRGB u16 scale.
function linearU16ToSrgbU16(rgba) {
  const out = new Uint16Array(rgba.length);
  for (let i = 0; i < rgba.length; i++) {
    out[i] = (i & 3) === 3 ? rgba[i] : srgbLinearToEncoded16(rgba[i] / 65535);
  }
  return out;
}

function selectRGBA(tensor, allowLoss) {
  const labels = tensor.channelOrder;
  const pick = (label) => labels.indexOf(label);
  const idx = [pick('R'), pick('G'), pick('B'), pick('A')];
  const used = new Set(idx.filter((i) => i >= 0));
  const dropped = labels.filter((_, i) => !used.has(i));
  if (dropped.length && !allowLoss) {
    throw new Error(
      `fusion input: ${dropped.length} extra channel(s) [${dropped.join(', ')}] cannot enter the RGBA engine; ` +
      'pass { allowLoss: true } to drop them, or keep them via the FFI container',
    );
  }
  const n = tensor.width * tensor.height;
  const out = new Uint16Array(n * 4);
  const src = tensor.data;
  const srcCh = tensor.channels;
  for (let p = 0; p < n; p++) {
    for (let c = 0; c < 4; c++) {
      const si = idx[c];
      if (si < 0) { out[p * 4 + c] = c === 3 ? 65535 : 0; continue; }
      out[p * 4 + c] = src[p * srcCh + si];
    }
  }
  return { data: out, dropped };
}

/**
 * Convert an ImageTensor into an engine slice (RGBA sRGB, nominal 16-bit).
 * @returns {{rgba:Uint16Array, width:number, height:number, losses:Array, info:object}}
 */
export function tensorToSlice(tensor, { allowLoss = false, name } = {}) {
  validateTensor(tensor);
  const losses = [];
  let work = tensor;

  if (isFloatKind(work.kind)) {
    const v = rangeViolations(work);
    if (v.nan > 0) throw new Error('fusion input: float tensor has non-finite samples');
    const clamp = allowLoss && v.outside > 0;
    if (v.outside > 0 && !clamp) {
      throw new Error(
        `fusion input: ${v.outside} float samples are outside ${JSON.stringify(work.valueRange)}; ` +
        'pass { allowLoss: true } to clamp before quantisation',
      );
    }
    const q = quantizeFloat(work, 'u16', { clamp });
    work = q.tensor;
    losses.push(q.loss);
  } else if (work.kind !== 'u16') {
    const c = convertInt(work, 'u16');
    work = c.tensor;
    if (c.loss) losses.push(c.loss);
  }

  let rgba;
  if (work.channels === 4 && work.channelOrder.join(',') === 'R,G,B,A') {
    rgba = new Uint16Array(work.data);
  } else {
    const sel = selectRGBA(work, allowLoss);
    rgba = sel.data;
    if (sel.dropped.length) {
      losses.push({ kind: 'dropped-channels', channels: sel.dropped });
    }
  }

  if (work.transfer === 'linear') {
    rgba = linearU16ToSrgbU16(rgba);
    losses.push({ kind: 'transfer', from: 'linear', to: 'srgb', encoding: 'u16' });
  } else if (work.transfer === 'unknown') {
    losses.push({ kind: 'transfer-assumption', assumed: 'srgb', note: 'source transfer was untagged; engine assumes sRGB' });
  }

  return {
    name: name ?? tensor.metadata?.name ?? 'slice',
    width: work.width,
    height: work.height,
    rgba,
    losses,
    info: {
      sourceKind: tensor.kind,
      sourceChannels: tensor.channels,
      sourceChannelOrder: [...tensor.channelOrder],
      sourceTransfer: tensor.transfer,
    },
  };
}

/**
 * Wrap the engine's linear-light planar RGB fusion result as an f32 RGBA
 * ImageTensor. The actual numeric range is measured rather than assumed, so
 * HDR content above 1.0 is preserved and declared instead of clipped.
 */
export function fusedLinearToTensor(result, { name = 'fused', metadata = {} } = {}) {
  const n = result.w * result.h;
  const data = new Float32Array(n * 4);
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const r = result.fused[i * 3];
    const g = result.fused[i * 3 + 1];
    const b = result.fused[i * 3 + 2];
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = result.covered[i] ? 1 : 0;
    if (result.covered[i]) {
      if (r < min) min = r;
      if (g < min) min = g;
      if (b < min) min = b;
      if (r > max) max = r;
      if (g > max) max = g;
      if (b > max) max = b;
    }
  }
  if (!Number.isFinite(min)) { min = 0; max = 1; }
  const tensor = {
    width: result.w,
    height: result.h,
    channels: 4,
    kind: 'f32',
    data,
    channelOrder: ['R', 'G', 'B', 'A'],
    valueRange: { min, max },
    transfer: 'linear',
    metadata: {
      name,
      coverage: result.coverage?.validFraction ?? null,
      ...metadata,
    },
  };
  validateTensor(tensor);
  return tensor;
}
