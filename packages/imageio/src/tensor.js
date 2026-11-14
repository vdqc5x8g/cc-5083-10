// ImageTensor: the single in-memory image contract used by every imageio path.
//
// New contract (explicit, no hidden normalisation):
//   width, height    positive integers
//   channels         positive integer; arbitrary channel counts are allowed
//   kind             'u8' | 'u16' | 'u32' | 'f32'
//   data             interleaved typed array, length width*height*channels
//                      u8/u16/u32 -> Uint8Array/Uint16Array/Uint32Array
//                      f32        -> Float32Array
//   channelOrder     explicit label per channel, e.g. ['R','G','B','A'];
//                      never inferred from the channel count
//   valueRange       { min, max } the numerical domain the samples live in
//   transfer         'srgb' | 'linear' | 'unknown'
//   metadata         plain JSON-serialisable object (may be empty)
//
// Old contract (the legacy PNG codec in src/core/png.js):
//   * always 4 channels, RGBA, one interleaved Uint16Array;
//   * every input is rescaled to nominal 16-bit, max 65535, even 8-bit files;
//   * out-of-range/extra channels do not exist (PNG is capped at RGBA);
//   * sRGB transfer is assumed by the fusion pipeline, but not recorded.
// The PNG adapter documents and isolates every conversion between the two.

import { INT_KINDS, FLOAT_KINDS, KIND_BYTES, KIND_MAX, isFloatKind } from './format.js';

export const SRGB = 'srgb';
export const LINEAR = 'linear';
export const UNKNOWN_TRANSFER = 'unknown';

export function defaultRangeFor(kind) {
  if (isFloatKind(kind)) return { min: 0, max: 1 };
  return { min: 0, max: KIND_MAX[kind] };
}

function isPlainObject(v) {
  if (v === null || typeof v !== 'object') return false;
  const proto = Object.getPrototypeOf(v);
  return proto === Object.prototype || proto === null;
}

/**
 * Validate an ImageTensor. Throws with a precise message; this module never
 * silently repairs or clamps caller data.
 */
export function validateTensor(t) {
  if (!t || typeof t !== 'object') throw new Error('ImageTensor: not an object');
  const { width, height, channels, kind, data, channelOrder } = t;
  if (!Number.isInteger(width) || width <= 0) throw new Error('ImageTensor: width must be a positive integer');
  if (!Number.isInteger(height) || height <= 0) throw new Error('ImageTensor: height must be a positive integer');
  if (!Number.isInteger(channels) || channels <= 0) throw new Error('ImageTensor: channels must be a positive integer');
  if (!INT_KINDS.has(kind) && !FLOAT_KINDS.has(kind)) throw new Error(`ImageTensor: unsupported kind "${kind}"`);
  const expected = width * height * channels;
  if (!ArrayBuffer.isView(data)) throw new Error('ImageTensor: data must be a typed array view');
  if (data.length !== expected) {
    throw new Error(`ImageTensor: data length ${data.length} != width*height*channels ${expected}`);
  }
  const ctor = { u8: Uint8Array, u16: Uint16Array, u32: Uint32Array, f32: Float32Array }[kind];
  if (!(data instanceof ctor)) {
    throw new Error(`ImageTensor: kind ${kind} requires ${ctor.name}, got ${data.constructor.name}`);
  }
  if (!Array.isArray(channelOrder) || channelOrder.length !== channels) {
    throw new Error('ImageTensor: channelOrder must be an array with one label per channel');
  }
  if (channelOrder.some((c) => typeof c !== 'string' || !c.length)) {
    throw new Error('ImageTensor: channelOrder entries must be non-empty strings');
  }
  if (new Set(channelOrder).size !== channelOrder.length) {
    throw new Error('ImageTensor: channelOrder labels must be unique');
  }
  const range = t.valueRange ?? defaultRangeFor(kind);
  if (!range || typeof range.min !== 'number' || typeof range.max !== 'number' || range.max <= range.min) {
    throw new Error('ImageTensor: valueRange needs numeric min < max');
  }
  const transfer = t.transfer ?? UNKNOWN_TRANSFER;
  if (![SRGB, LINEAR, UNKNOWN_TRANSFER].includes(transfer)) {
    throw new Error(`ImageTensor: unknown transfer "${transfer}"`);
  }
  if (t.metadata !== undefined && !isPlainObject(t.metadata)) {
    throw new Error('ImageTensor: metadata must be a plain JSON-serialisable object');
  }
  return true;
}

export function withDefaults(t) {
  validateTensor(t);
  return {
    width: t.width,
    height: t.height,
    channels: t.channels,
    kind: t.kind,
    data: t.data,
    channelOrder: [...t.channelOrder],
    valueRange: t.valueRange ?? defaultRangeFor(t.kind),
    transfer: t.transfer ?? UNKNOWN_TRANSFER,
    metadata: t.metadata ? JSON.parse(JSON.stringify(t.metadata)) : {},
  };
}

/**
 * Integer-range check. Returns the count and list of out-of-range samples so
 * callers can reject explicitly instead of wrapping or clamping.
 */
export function rangeViolations(t) {
  validateTensor(t);
  const range = t.valueRange ?? defaultRangeFor(t.kind);
  if (isFloatKind(t.kind)) {
    let nan = 0;
    let outside = 0;
    for (let i = 0; i < t.data.length; i++) {
      const v = t.data[i];
      if (!Number.isFinite(v)) { nan++; continue; }
      if (v < range.min || v > range.max) outside++;
    }
    return { nan, outside, total: t.data.length };
  }
  let outside = 0;
  for (let i = 0; i < t.data.length; i++) {
    const v = t.data[i];
    if (v < range.min || v > range.max) outside++;
  }
  return { nan: 0, outside, total: t.data.length };
}

// ---------------------------------------------------------------- conversions

/**
 * Quantise an f32 tensor to an integer kind. NaN/Inf and values outside the
 * destination range are refused unless `clamp` is set; clamping and rounding
 * are reported as losses rather than applied silently.
 */
export function quantizeFloat(tensor, kind, { clamp = false } = {}) {
  validateTensor(tensor);
  if (!isFloatKind(tensor.kind)) throw new Error('quantizeFloat: source must be f32');
  const max = KIND_MAX[kind];
  const range = tensor.valueRange ?? defaultRangeFor(tensor.kind);
  const { min: lo, max: hi } = range;
  const v = rangeViolations(tensor);
  if (v.nan > 0) throw new Error(`quantizeFloat: ${v.nan} non-finite samples cannot be represented as ${kind}`);
  if (!clamp && v.outside > 0) {
    throw new Error(
      `quantizeFloat: ${v.outside} samples fall outside [${lo}, ${hi}] and cannot map onto 0..${max}; ` +
      'pass { clamp: true } to acknowledge the clipping loss',
    );
  }
  const span = hi - lo;
  const Ctor = kind === 'u8' ? Uint8Array : kind === 'u16' ? Uint16Array : Uint32Array;
  const out = new Ctor(tensor.width * tensor.height * tensor.channels);
  let clippedLow = 0;
  let clippedHigh = 0;
  for (let i = 0; i < tensor.data.length; i++) {
    let x = tensor.data[i];
    if (x < lo) { x = lo; clippedLow++; }
    else if (x > hi) { x = hi; clippedHigh++; }
    const q = Math.round(((x - lo) / span) * max);
    out[i] = kind === 'u32' ? q : Math.min(max, q);
  }
  const result = {
    width: tensor.width,
    height: tensor.height,
    channels: tensor.channels,
    kind,
    data: out,
    channelOrder: [...tensor.channelOrder],
    valueRange: { min: 0, max },
    transfer: tensor.transfer,
    metadata: { ...tensor.metadata },
  };
  validateTensor(result);
  return {
    tensor: result,
    loss: {
      kind: 'quantize',
      from: 'f32',
      to: kind,
      clippedLow,
      clippedHigh,
      levels: max + 1,
      note: 'float samples rounded to integer levels; sub-level detail is lost',
    },
  };
}

/**
 * Convert between integer kinds. Down-conversion is a real, reported loss;
 * up-conversion is exact (bit-shift scaling).
 */
export function convertInt(tensor, kind) {
  validateTensor(tensor);
  if (isFloatKind(tensor.kind) || isFloatKind(kind)) throw new Error('convertInt: both kinds must be integer');
  if (tensor.kind === kind) return { tensor: withDefaults(tensor), loss: null };
  const srcMax = KIND_MAX[tensor.kind];
  const dstMax = KIND_MAX[kind];
  const Ctor = kind === 'u8' ? Uint8Array : kind === 'u16' ? Uint16Array : Uint32Array;
  const out = new Ctor(tensor.data.length);
  let rounded = 0;
  if (dstMax < srcMax) {
    for (let i = 0; i < tensor.data.length; i++) {
      const scaled = (tensor.data[i] * dstMax) / srcMax;
      const q = Math.round(scaled);
      if (q !== scaled) rounded++;
      out[i] = q;
    }
  } else {
    for (let i = 0; i < tensor.data.length; i++) {
      out[i] = Math.round((tensor.data[i] * dstMax) / srcMax);
    }
  }
  const result = {
    width: tensor.width,
    height: tensor.height,
    channels: tensor.channels,
    kind,
    data: out,
    channelOrder: [...tensor.channelOrder],
    valueRange: { min: 0, max: dstMax },
    transfer: tensor.transfer,
    metadata: { ...tensor.metadata },
  };
  validateTensor(result);
  return {
    tensor: result,
    loss: dstMax < srcMax
      ? { kind: 'bit-depth', from: tensor.kind, to: kind, roundedSamples: rounded, levels: dstMax + 1 }
      : null,
  };
}

export function bytesPerSample(kind) {
  return KIND_BYTES[kind];
}
