// Unified imageio entry point. Format dispatch is by byte signature, never by
// file extension, so a misnamed file cannot silently take the wrong path.

import { signatureOf, FFI_EXTENSION, FFI_MEDIA_TYPE } from './format.js';
import { decodeFFI, encodeFFI } from './ffi.js';
import { decodePNGTensor, encodePNGTensor } from './png-adapter.js';
import { validateTensor } from './tensor.js';
import { tensorToSlice, fusedLinearToTensor } from './fusion-adapter.js';

export { signatureOf, FFI_EXTENSION, FFI_MEDIA_TYPE } from './format.js';
export { validateTensor, withDefaults, rangeViolations, quantizeFloat, convertInt } from './tensor.js';
export { decodeFFI, encodeFFI } from './ffi.js';
export { decodePNGTensor, encodePNGTensor } from './png-adapter.js';
export { tensorToSlice, fusedLinearToTensor } from './fusion-adapter.js';
export {
  fromLegacyManifest, fromFocusGroup, parseProject, projectIdFingerprint, PROJECT_SCHEMA,
} from './project.js';

/**
 * Read any supported byte container into an ImageTensor.
 * Unknown signatures are refused with a clear message instead of a guess.
 */
export async function readImage(bytes) {
  const kind = signatureOf(bytes);
  if (kind === 'png') return decodePNGTensor(bytes);
  if (kind === 'ffi') return decodeFFI(bytes);
  throw new Error(
    'imageio: unsupported image signature; supported formats are PNG (8/16-bit) and FFI v1 ' +
    '(u8/u16/u32/f32, arbitrary channels)',
  );
}

/**
 * Write an ImageTensor.
 *   format 'png' -> legacy-compatible PNG; lossy conversions require
 *                   { allowLoss: true } and are returned in `losses`.
 *   format 'ffi' -> exact FFI v1 container, always lossless.
 */
export async function writeImage(tensor, format = 'ffi', opts = {}) {
  validateTensor(tensor);
  if (format === 'ffi') {
    const bytes = await encodeFFI(tensor);
    return { bytes, format: 'ffi', extension: FFI_EXTENSION, mediaType: FFI_MEDIA_TYPE, losses: [] };
  }
  if (format === 'png') {
    const r = await encodePNGTensor(tensor, opts);
    return { bytes: r.bytes, format: 'png', extension: '.png', mediaType: 'image/png', losses: r.losses, bitDepth: r.bitDepth };
  }
  throw new Error(`imageio: unknown output format "${format}" (expected 'png' or 'ffi')`);
}
