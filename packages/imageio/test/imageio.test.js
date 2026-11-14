import test from 'node:test';
import assert from 'node:assert/strict';
import {
  readImage, writeImage, decodeFFI, encodeFFI, signatureOf, FFI_EXTENSION,
  validateTensor, rangeViolations, quantizeFloat, convertInt,
  decodePNGTensor, encodePNGTensor,
} from '../src/index.js';
import { encodePNG } from '../../../src/core/png.js';

function makeTensor(over = {}) {
  const w = over.width ?? 6;
  const h = over.height ?? 4;
  const channels = over.channels ?? 4;
  const kind = over.kind ?? 'u16';
  const Ctor = kind === 'u8' ? Uint8Array : kind === 'u16' ? Uint16Array : kind === 'u32' ? Uint32Array : Float32Array;
  const data = new Ctor(w * h * channels);
  const mod = kind === 'u8' ? 256 : kind === 'u16' ? 65536 : 100000;
  for (let i = 0; i < data.length; i++) {
    data[i] = kind === 'f32' ? ((i * 0.137) % 1.7) - 0.3 : (i * 7919 + 13) % mod;
  }
  return {
    width: w, height: h, channels, kind, data,
    channelOrder: over.channelOrder ?? ['R', 'G', 'B', 'A'].slice(0, channels),
    valueRange: over.valueRange ?? (
      kind === 'u8' ? { min: 0, max: 255 } : kind === 'u16' ? { min: 0, max: 65535 }
        : kind === 'u32' ? { min: 0, max: 99999 } : { min: 0, max: 1 }
    ),
    transfer: over.transfer ?? 'srgb',
    metadata: over.metadata ?? { probe: 'x' },
  };
}

test('FFI u16 RGBA round-trips sample-exact with metadata and range', async () => {
  const t = makeTensor({ metadata: { title: 'probe', n: 7 } });
  const bytes = await encodeFFI(t);
  assert.equal(signatureOf(bytes), 'ffi');
  const back = await decodeFFI(bytes);
  assert.equal(back.width, 6);
  assert.equal(back.height, 4);
  assert.deepEqual(back.channelOrder, ['R', 'G', 'B', 'A']);
  assert.deepEqual(back.metadata, { title: 'probe', n: 7 });
  for (let i = 0; i < t.data.length; i++) assert.equal(back.data[i], t.data[i], `sample ${i}`);
});

test('FFI carries u8, u16, u32 and f32 at their native ranges', async () => {
  for (const kind of ['u8', 'u16', 'u32', 'f32']) {
    const t = makeTensor({ kind });
    const back = await decodeFFI(await encodeFFI(t));
    assert.equal(back.kind, kind);
    if (kind === 'f32') {
      // Float samples, including values outside 0..1, must survive exactly.
      const sawOutside = back.data.some((v) => v > 1 || v < 0);
      assert.ok(sawOutside, 'f32 tensor must keep out-of-0..1 samples');
    }
    for (let i = 0; i < t.data.length; i++) assert.equal(back.data[i], t.data[i], `${kind}[${i}]`);
  }
});

test('FFI preserves arbitrary extra channels and their order', async () => {
  const t = makeTensor({ channels: 6, channelOrder: ['R', 'G', 'B', 'A', 'Depth', 'Confidence'], kind: 'f32' });
  const back = await decodeFFI(await encodeFFI(t));
  assert.equal(back.channels, 6);
  assert.deepEqual(back.channelOrder, ['R', 'G', 'B', 'A', 'Depth', 'Confidence']);
  const p = 5 * 6 + 4; // Depth sample of pixel 5
  assert.equal(back.data[p], t.data[p]);
});

test('FFI detects a corrupted payload via CRC', async () => {
  const bytes = await encodeFFI(makeTensor());
  const bad = Uint8Array.from(bytes);
  bad[bad.length - 40] ^= 0xff;
  await assert.rejects(() => decodeFFI(bad), /CRC|corrupt|decompress/i);
});

test('FFI rejects truncated containers and bad magic', async () => {
  const bytes = await encodeFFI(makeTensor());
  await assert.rejects(() => decodeFFI(bytes.subarray(0, 10)), /truncated|exceeds/i);
  const wrong = Uint8Array.from(bytes);
  wrong[0] = 0;
  await assert.rejects(() => decodeFFI(wrong), /magic/i);
});

test('tensor validation refuses malformed contracts', () => {
  assert.throws(() => validateTensor(null), /not an object/);
  const t = makeTensor();
  assert.throws(() => validateTensor({ ...t, width: 0 }), /width/);
  assert.throws(() => validateTensor({ ...t, data: t.data.subarray(0, 4) }), /length/);
  assert.throws(() => validateTensor({ ...t, channelOrder: ['R', 'G', 'B'] }), /channelOrder/);
  assert.throws(() => validateTensor({ ...t, channelOrder: ['R', 'R', 'G', 'B'] }), /unique/);
  assert.throws(() => validateTensor({
    ...t, kind: 'u32', valueRange: { min: 0, max: 99999 }, data: new Uint16Array(t.data.length),
  }), /requires/);
});

test('rangeViolations flags out-of-domain integer and non-finite float samples', () => {
  const u16 = makeTensor({ kind: 'u16', valueRange: { min: 0, max: 60000 } });
  u16.data.fill(30000);
  u16.data[0] = 62000;
  assert.equal(rangeViolations(u16).outside, 1);
  const f32 = makeTensor({ kind: 'f32', valueRange: { min: 0, max: 1 } });
  f32.data[0] = NaN;
  f32.data[1] = 1.5;
  const v = rangeViolations(f32);
  assert.equal(v.nan, 1);
  assert.ok(v.outside >= 1);
});

test('float quantisation refuses unrepresentable content unless clamping is acknowledged', () => {
  const t = makeTensor({ kind: 'f32', valueRange: { min: 0, max: 1 } });
  t.data[0] = 2;
  assert.throws(() => quantizeFloat(t, 'u16'), /outside/);
  const { tensor, loss } = quantizeFloat(t, 'u16', { clamp: true });
  assert.equal(tensor.kind, 'u16');
  assert.ok(loss.clippedHigh >= 1);
  t.data[1] = Infinity;
  assert.throws(() => quantizeFloat(t, 'u16', { clamp: true }), /non-finite/);
});

test('integer down-conversion reports rounding loss; up-conversion is exact', () => {
  const t = makeTensor({ kind: 'u16' });
  const down = convertInt(t, 'u8');
  assert.equal(down.tensor.kind, 'u8');
  assert.ok(down.loss.roundedSamples > 0);
  const back = convertInt(down.tensor, 'u16');
  assert.equal(back.loss, null);
  assert.equal(back.tensor.data[3], Math.round((t.data[3] * 255) / 65535) * 257);
});

test('PNG adapter decodes legacy 8/16-bit files into explicit RGBA sRGB tensors', async () => {
  const data16 = new Uint16Array(2 * 2 * 4);
  for (let i = 0; i < data16.length; i++) data16[i] = 1000 + i * 111;
  const png16 = await encodePNG({ width: 2, height: 2, channels: 4, bitDepth: 16, data: data16 });
  const t16 = await decodePNGTensor(png16);
  assert.equal(t16.kind, 'u16');
  assert.equal(t16.transfer, 'srgb');
  assert.deepEqual(t16.channelOrder, ['R', 'G', 'B', 'A']);
  assert.equal(t16.metadata.sourceBitDepth, 16);
  for (let i = 0; i < data16.length; i++) assert.equal(t16.data[i], data16[i]);

  const data8 = new Uint8Array(2 * 2 * 4).fill(200);
  const png8 = await encodePNG({ width: 2, height: 2, channels: 4, bitDepth: 8, data: data8 });
  const t8 = await decodePNGTensor(png8);
  assert.equal(t8.metadata.sourceBitDepth, 8);
  assert.equal(t8.data[0], Math.round((200 * 65535) / 255));
});

test('PNG adapter refuses extra channels and silent 8-bit flattening', async () => {
  const five = makeTensor({ channels: 5, channelOrder: ['R', 'G', 'B', 'A', 'Depth'] });
  await assert.rejects(() => encodePNGTensor(five), /only 4-channel RGBA/);
  const rgba = makeTensor();
  await assert.rejects(() => encodePNGTensor(rgba, { bitDepth: 8 }), /loses depth/);
  const ok = await encodePNGTensor(rgba, { bitDepth: 8, allowLoss: true });
  assert.equal(ok.bitDepth, 8);
  assert.ok(ok.losses.some((l) => l.kind === 'bit-depth'));
});

test('PNG float output is refused without explicit loss consent', async () => {
  const f = makeTensor({ kind: 'f32', transfer: 'linear', valueRange: { min: 0, max: 1 } });
  await assert.rejects(() => encodePNGTensor(f), /float samples are not representable/);
  const ok = await encodePNGTensor(f, { allowLoss: true });
  assert.equal(ok.format ?? 'png', 'png');
  assert.ok(ok.losses.some((l) => l.kind === 'quantize'));
});

test('writeImage dispatches by signature and never trusts extensions', async () => {
  const t = makeTensor();
  const png = await writeImage(t, 'png', { allowLoss: true, bitDepth: 16 });
  assert.equal(png.extension, '.png');
  assert.equal(signatureOf(png.bytes), 'png');
  const ffi = await writeImage(t, 'ffi');
  assert.equal(ffi.extension, FFI_EXTENSION);
  assert.equal(signatureOf(ffi.bytes), 'ffi');
  const reread = await readImage(ffi.bytes);
  assert.equal(reread.kind, 'u16');
  await assert.rejects(() => readImage(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8])), /unsupported image signature/);
});
