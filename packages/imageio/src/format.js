// Shared constants and low-level byte helpers for the imageio component.
//
// The component owns two byte formats:
//   * PNG, through the legacy dependency-free codec (see ./png-adapter.js);
//   * FFI v1 ("Focus-Fusion Image"), the component's own container that can
//     carry arbitrary channel counts, high integer bit depths and float
//     samples without any lossy representation.
//
// Everything here uses standard Web/Node APIs only, so the same modules run
// unchanged in the browser, the Web Worker and Node.

export const FFI_MAGIC = new Uint8Array([0x46, 0x46, 0x49, 0x4d, 0x47, 0x30, 0x30, 0x31]); // "FFIMG001"
export const FFI_VERSION = 1;
export const FFI_MEDIA_TYPE = 'application/x-focus-fusion-image';
export const FFI_EXTENSION = '.ffi';

export const PNG_SIGNATURE = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

// Integer sample value domains supported by the FFI container.
export const INT_KINDS = new Set(['u8', 'u16', 'u32']);
export const FLOAT_KINDS = new Set(['f32']);

export const KIND_BYTES = { u8: 1, u16: 2, u32: 4, f32: 4 };
export const KIND_MAX = { u8: 255, u16: 65535, u32: 4294967295 };

export function isFloatKind(kind) {
  return FLOAT_KINDS.has(kind);
}

export function signatureOf(bytes) {
  if (!bytes || bytes.length < 8) return 'unknown';
  let png = true;
  for (let i = 0; i < 8; i++) png = png && bytes[i] === PNG_SIGNATURE[i];
  if (png) return 'png';
  let ffi = true;
  for (let i = 0; i < 8; i++) ffi = ffi && bytes[i] === FFI_MAGIC[i];
  if (ffi) return 'ffi';
  return 'unknown';
}

export function writeU32LE(out, offset, value) {
  out[offset] = value & 255;
  out[offset + 1] = (value >>> 8) & 255;
  out[offset + 2] = (value >>> 16) & 255;
  out[offset + 3] = (value >>> 24) & 255;
}

export function readU32LE(src, offset) {
  return (src[offset] | (src[offset + 1] << 8) | (src[offset + 2] << 16) | (src[offset + 3] << 24)) >>> 0;
}

// CRC-32 (IEEE), used only as a corruption detector for FFI payloads.
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(buf, start = 0, end = buf.length) {
  let c = 0xffffffff;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

async function streamThrough(bytes, transform) {
  const stream = new Blob([bytes]).stream().pipeThrough(transform);
  const ab = await new Response(stream).arrayBuffer();
  return new Uint8Array(ab);
}

export function zlibDeflate(bytes) {
  return streamThrough(bytes, new CompressionStream('deflate'));
}

export function zlibInflate(bytes) {
  return streamThrough(bytes, new DecompressionStream('deflate'));
}
