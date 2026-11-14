// Fusion worker: runs the shared core pipeline off the UI thread.
//
// Message in:
//   { id, inputs: [{name, width, height, rgba: Uint16Array (transferable)}],
//     options, roi: (box[]|null)[], excludedSlices: number[] }
// Message back:
//   { id, ok: true, preview: {scale,width,height,rgba,weights,relClass,reliability},
//     stats, edgeWidth, ghosting, warnings, registration, refIndex, skip,
//     weightsFull: Float32Array[] (transferable), w, h, gains, coverage }
//   or { id, ok: false, error: string }
//
// Exports go through the @focus-fusion/imageio writer:
//   exportFormat 'png' + exportBitDepth 8|16 -> PNG (8-bit requires
//     allowLoss and the loss is surfaced back, never applied silently);
//   exportFormat 'ffi' -> lossless f32 linear FFI v1 container.

import { fuseStack, computePreview, renderFused } from '../core/fuse.js';
import { writeImage, fusedLinearToTensor } from '../../packages/imageio/src/index.js';

self.onmessage = async (e) => {
  const { id, inputs, options, roi, excludedSlices, exportBitDepth, exportFormat = 'png' } = e.data;
  try {
    const result = fuseStack(inputs, options || {}, {
      roi: roi || null,
      excludedSlices: new Set(excludedSlices || []),
      onProgress: (p) => self.postMessage({ id, progress: p }),
    });
    const preview = computePreview(result);
    const message = {
      id,
      ok: true,
      w: result.w,
      h: result.h,
      preview,
      weightsFull: result.weights,
      stats: result.stats,
      edgeWidth: result.edgeWidth,
      ghosting: result.ghosting,
      warnings: result.warnings,
      registration: result.registration,
      refIndex: result.refIndex,
      skip: result.skip,
      gains: result.gains,
      coverage: result.coverage,
      relClassFull: result.relClass,
    };
    const transfer = [
      preview.rgba.buffer,
      ...preview.weights.map((p) => p.buffer),
      preview.relClass.buffer,
      preview.reliability.buffer,
      ...result.weights.map((p) => p.buffer),
    ];
    if (exportFormat === 'ffi') {
      const tensor = fusedLinearToTensor(result, {
        metadata: { name: 'fused', referenceIndex: result.refIndex, writer: '@focus-fusion/imageio' },
      });
      const out = await writeImage(tensor, 'ffi');
      message.exportInfo = {
        format: 'ffi', extension: out.extension, valueRange: tensor.valueRange, losses: [],
      };
      message.fusedBytes = out.bytes;
      transfer.push(out.bytes.buffer);
    } else if (exportBitDepth === 8 || exportBitDepth === 16) {
      const fused = renderFused(result, { bitDepth: exportBitDepth });
      const tensor = {
        width: fused.width, height: fused.height, channels: 4, kind: 'u16', data: fused.data,
        channelOrder: ['R', 'G', 'B', 'A'], valueRange: { min: 0, max: 65535 },
        transfer: 'srgb', metadata: { name: 'fused' },
      };
      const out = await writeImage(tensor, 'png', {
        bitDepth: exportBitDepth,
        allowLoss: true, // GUI explicitly offers the 8-bit option; report the loss
      });
      message.exportInfo = {
        format: 'png', extension: out.extension, bitDepth: out.bitDepth, losses: out.losses,
      };
      message.fusedBytes = out.bytes;
      transfer.push(out.bytes.buffer);
    }
    self.postMessage(message, transfer);
  } catch (err) {
    self.postMessage({ id, ok: false, error: err && err.message ? err.message : String(err) });
  }
};
