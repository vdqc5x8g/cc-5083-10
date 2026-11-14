// FFI v1 codec: the imageio component's native, lossless container.
//
// Byte layout (all integers little-endian):
//   8  magic        "FFIMG001"
//   4  version      1
//   4  headerLen    byte length of the UTF-8 JSON header
//   …  header       JSON: { width, height, channels, kind, channelOrder,
//                           valueRange, transfer, metadata }
//   4  payloadLen   byte length of the compressed payload
//   …  payload      zlib-deflate of the interleaved sample bytes
//   4  payloadCrc   CRC-32 of the *uncompressed* sample bytes
//
// Integer samples are stored at their native width in their declared domain;
// f32 samples are IEEE-754 little-endian and are never clamped, so HDR and
// out-of-0..1 float content survives exactly.

import {
  FFI_MAGIC, FFI_VERSION, KIND_BYTES, isFloatKind,
  writeU32LE, readU32LE, crc32, zlibDeflate, zlibInflate,
} from './format.js';
import { validateTensor, withDefaults } from './tensor.js';

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function samplesToBytes(tensor) {
  const bps = KIND_BYTES[tensor.kind];
  const bytes = new Uint8Array(tensor.data.length * bps);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < tensor.data.length; i++) {
    const v = tensor.data[i];
    const off = i * bps;
    if (tensor.kind === 'u8') view.setUint8(off, v);
    else if (tensor.kind === 'u16') view.setUint16(off, v, true);
    else if (tensor.kind === 'u32') view.setUint32(off, v, true);
    else view.setFloat32(off, v, true);
  }
  return bytes;
}

function bytesToSamples(bytes, kind, count) {
  const bps = KIND_BYTES[kind];
  if (bytes.length !== count * bps) {
    throw new Error(`FFI: payload size ${bytes.length} does not match ${count} x ${kind} samples`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (kind === 'u8') return Uint8Array.from(bytes);
  const Ctor = kind === 'u16' ? Uint16Array : kind === 'u32' ? Uint32Array : Float32Array;
  const out = new Ctor(count);
  for (let i = 0; i < count; i++) {
    const off = i * bps;
    out[i] = kind === 'u16' ? view.getUint16(off, true)
      : kind === 'u32' ? view.getUint32(off, true)
        : view.getFloat32(off, true);
  }
  return out;
}

/**
 * Encode an ImageTensor to FFI v1 bytes.
 * @returns {Promise<Uint8Array>}
 */
export async function encodeFFI(input) {
  const tensor = withDefaults(input);
  const header = {
    format: 'ffi',
    version: FFI_VERSION,
    width: tensor.width,
    height: tensor.height,
    channels: tensor.channels,
    kind: tensor.kind,
    channelOrder: tensor.channelOrder,
    valueRange: tensor.valueRange,
    transfer: tensor.transfer,
    metadata: tensor.metadata,
  };
  const headerBytes = encoder.encode(JSON.stringify(header));
  const sampleBytes = samplesToBytes(tensor);
  const payload = await zlibDeflate(sampleBytes);
  const crc = crc32(sampleBytes);
  const out = new Uint8Array(8 + 4 + 4 + headerBytes.length + 4 + payload.length + 4);
  let off = 0;
  out.set(FFI_MAGIC, off); off += 8;
  writeU32LE(out, off, FFI_VERSION); off += 4;
  writeU32LE(out, off, headerBytes.length); off += 4;
  out.set(headerBytes, off); off += headerBytes.length;
  writeU32LE(out, off, payload.length); off += 4;
  out.set(payload, off); off += payload.length;
  writeU32LE(out, off, crc);
  return out;
}

/**
 * Decode FFI v1 bytes back into an ImageTensor. Structural problems, bad CRCs
 * and truncated containers throw instead of returning partial data.
 */
export async function decodeFFI(bytes) {
  if (!(bytes instanceof Uint8Array)) throw new Error('FFI: input must be Uint8Array');
  for (let i = 0; i < 8; i++) {
    if (bytes[i] !== FFI_MAGIC[i]) throw new Error('FFI: bad magic');
  }
  if (bytes.length < 16) throw new Error('FFI: truncated header');
  const version = readU32LE(bytes, 8);
  if (version !== FFI_VERSION) throw new Error(`FFI: unsupported version ${version}`);
  const headerLen = readU32LE(bytes, 12);
  let off = 16;
  if (off + headerLen + 8 > bytes.length) throw new Error('FFI: header length exceeds container');
  let header;
  try {
    header = JSON.parse(decoder.decode(bytes.subarray(off, off + headerLen)));
  } catch (err) {
    throw new Error(`FFI: header is not valid JSON (${err.message})`);
  }
  off += headerLen;
  const payloadLen = readU32LE(bytes, off); off += 4;
  if (off + payloadLen + 4 > bytes.length) throw new Error('FFI: payload length exceeds container');
  const payload = bytes.subarray(off, off + payloadLen); off += payloadLen;
  const storedCrc = readU32LE(bytes, off);
  let sampleBytes;
  try {
    sampleBytes = await zlibInflate(payload);
  } catch (err) {
    throw new Error(`FFI: payload cannot be decompressed (${err.message})`);
  }
  const actualCrc = crc32(sampleBytes);
  if (actualCrc !== storedCrc) throw new Error('FFI: payload CRC mismatch (data corrupted)');
  const { width, height, channels, kind, channelOrder, valueRange, transfer, metadata } = header;
  if (!Number.isInteger(width) || !Number.isInteger(height) || !Number.isInteger(channels) ||
      width <= 0 || height <= 0 || channels <= 0) {
    throw new Error('FFI: header dimensions are invalid');
  }
  const data = bytesToSamples(sampleBytes, kind, width * height * channels);
  const tensor = {
    width, height, channels, kind, data,
    channelOrder,
    valueRange: valueRange ?? undefined,
    transfer: transfer ?? 'unknown',
    metadata: metadata ?? {},
  };
  validateTensor(tensor);
  if (isFloatKind(kind)) {
    for (let i = 0; i < data.length; i++) {
      if (!Number.isFinite(data[i])) {
        throw new Error('FFI: float payload contains non-finite samples');
      }
    }
  }
  return tensor;
}
