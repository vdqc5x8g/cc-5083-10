import test from 'node:test';
import assert from 'node:assert/strict';
import { tensorToSlice, fusedLinearToTensor, readImage, writeImage } from '../src/index.js';

function u16RGBA(w, h, fill) {
  const data = new Uint16Array(w * h * 4);
  for (let i = 0; i < data.length; i++) data[i] = fill(i);
  return {
    width: w, height: h, channels: 4, kind: 'u16', data,
    channelOrder: ['R', 'G', 'B', 'A'], valueRange: { min: 0, max: 65535 },
    transfer: 'srgb', metadata: { name: 'probe' },
  };
}

test('16-bit sRGB RGBA tensor reaches the engine byte-for-byte', () => {
  const t = u16RGBA(3, 2, (i) => (i * 4019) % 65536);
  const slice = tensorToSlice(t);
  assert.equal(slice.width, 3);
  assert.equal(slice.height, 2);
  assert.deepEqual(Array.from(slice.rgba), Array.from(t.data));
  assert.equal(slice.losses.length, 0);
});

test('u8 input is up-scaled to nominal 16-bit without reported loss', () => {
  const data = new Uint8Array(2 * 2 * 4).fill(128);
  const t = {
    width: 2, height: 2, channels: 4, kind: 'u8', data,
    channelOrder: ['R', 'G', 'B', 'A'], valueRange: { min: 0, max: 255 }, transfer: 'srgb', metadata: {},
  };
  const slice = tensorToSlice(t);
  assert.equal(slice.rgba[0], Math.round((128 * 65535) / 255));
  assert.equal(slice.losses.length, 0);
});

test('extra channels are refused by default and only dropped on explicit consent', () => {
  const data = new Uint16Array(2 * 2 * 5);
  data.fill(1000);
  const t = {
    width: 2, height: 2, channels: 5, kind: 'u16', data,
    channelOrder: ['R', 'G', 'B', 'A', 'Depth'],
    valueRange: { min: 0, max: 65535 }, transfer: 'srgb', metadata: {},
  };
  assert.throws(() => tensorToSlice(t), /extra channel/);
  const slice = tensorToSlice(t, { allowLoss: true });
  assert.equal(slice.rgba.length, 2 * 2 * 4);
  assert.deepEqual(slice.losses.find((l) => l.kind === 'dropped-channels').channels, ['Depth']);
  // The depth channel content stays available through FFI regardless.
});

test('float inputs across ranges are quantised only with consent and clipping is reported', () => {
  const data = new Float32Array(2 * 2 * 4);
  for (let i = 0; i < data.length; i++) data[i] = (i % 4) === 3 ? 1 : (i * 0.17) % 1.4;
  assert.ok(data.some((v) => v > 1), 'fixture must include out-of-range samples');
  const t = {
    width: 2, height: 2, channels: 4, kind: 'f32', data,
    channelOrder: ['R', 'G', 'B', 'A'], valueRange: { min: 0, max: 1 }, transfer: 'srgb', metadata: {},
  };
  assert.throws(() => tensorToSlice(t), /outside/);
  const slice = tensorToSlice(t, { allowLoss: true });
  const qLoss = slice.losses.find((l) => l.kind === 'quantize');
  assert.ok(qLoss && qLoss.clippedHigh > 0);
  assert.ok(slice.rgba.every((v) => v <= 65535));
});

test('linear-light float input is converted to sRGB with a declared transfer loss', () => {
  const data = new Float32Array(1 * 1 * 4);
  data.set([0.5, 0.5, 0.5, 1]);
  const t = {
    width: 1, height: 1, channels: 4, kind: 'f32', data,
    channelOrder: ['R', 'G', 'B', 'A'], valueRange: { min: 0, max: 1 }, transfer: 'linear', metadata: {},
  };
  const slice = tensorToSlice(t, { allowLoss: true });
  assert.ok(slice.losses.some((l) => l.kind === 'transfer' && l.to === 'srgb'));
  // sRGB(0.5 linear) is well above the linear encoding.
  assert.ok(slice.rgba[0] > 0.5 * 65535);
});

test('fused linear result becomes an f32 tensor whose real range is declared, then round-trips via FFI', async () => {
  const w = 3;
  const h = 2;
  const fused = new Float32Array(w * h * 3);
  const covered = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    fused[i * 3] = (i % 3) * 0.4;        // reaches 0.8
    fused[i * 3 + 1] = 0.1;
    fused[i * 3 + 2] = 1.3;              // HDR above 1.0 must survive
    covered[i] = 1;
  }
  const result = { w, h, fused, covered, coverage: { validFraction: 1 } };
  const tensor = fusedLinearToTensor(result, { metadata: { job: 'e2e' } });
  assert.equal(tensor.kind, 'f32');
  assert.equal(tensor.transfer, 'linear');
  assert.ok(Math.abs(tensor.valueRange.max - 1.3) < 1e-6);
  assert.equal(tensor.metadata.job, 'e2e');
  const back = await readImage((await writeImage(tensor, 'ffi')).bytes);
  assert.ok(Math.abs(back.valueRange.max - 1.3) < 1e-6);
  for (let i = 0; i < tensor.data.length; i++) {
    assert.ok(Math.abs(back.data[i] - tensor.data[i]) < 1e-6, `sample ${i}`);
  }
});

test('HDR float content cannot be silently flattened into PNG at 8-bit', async () => {
  const result = {
    w: 1, h: 1,
    fused: new Float32Array([1.3, 0.2, 0.2]),
    covered: new Uint8Array([1]),
    coverage: { validFraction: 1 },
  };
  const tensor = fusedLinearToTensor(result);
  tensor.valueRange = { min: 0, max: 1 }; // standard photometric domain; 1.3 is now out-of-domain
  await assert.rejects(() => writeImage(tensor, 'png'), /float samples are not representable/);
  // The declared domain is 0..1 while samples reach 1.3, so acknowledged PNG
  // export must record the clipping loss instead of hiding it.
  const png = await writeImage(tensor, 'png', { allowLoss: true, bitDepth: 8 });
  const quant = png.losses.find((l) => l.kind === 'quantize');
  assert.ok(quant && quant.clippedHigh > 0, JSON.stringify(png.losses));
});
