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

import { fuseStack, computePreview, renderFused } from '../core/fuse.js';
import { encodePNG } from '../core/png.js';

self.onmessage = async (e) => {
  const { id, inputs, options, roi, excludedSlices, exportBitDepth } = e.data;
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
    if (exportBitDepth === 8 || exportBitDepth === 16) {
      const fused = renderFused(result, { bitDepth: exportBitDepth });
      const png = await encodePNG({
        width: fused.width, height: fused.height, channels: 4, bitDepth: fused.bitDepth,
        data: fused.data, text: [{ keyword: 'Software', value: 'focus-fusion 1.0' }],
      });
      message.fusedPng = png;
      transfer.push(png.buffer);
    }
    self.postMessage(message, transfer);
  } catch (err) {
    self.postMessage({ id, ok: false, error: err && err.message ? err.message : String(err) });
  }
};

