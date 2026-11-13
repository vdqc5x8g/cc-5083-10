import test from 'node:test';
import assert from 'node:assert/strict';
import { decodePNG } from '../src/core/png.js';
import { makeChartStack } from './lib/fixtures.mjs';

// The browser image-restoration window runs every fusion in this worker. Node
// has no DOM, so the worker's message contract is exercised here against a
// stub `self`: this is the same code path the GUI uses, minus the canvas.
async function loadWorker() {
  const messages = [];
  globalThis.self = {
    postMessage(msg, transfer) { messages.push({ msg, transfer }); },
  };
  // Cache-bust so repeated imports in one process still re-register onmessage.
  await import(`../src/web/fusion-worker.js?t=${Math.random()}`);
  return { messages, post: (data) => globalThis.self.onmessage({ data }) };
}

function inputsOf(slices) {
  return slices.map((s) => ({ name: s.name, width: s.width, height: s.height, rgba: new Uint16Array(s.rgba) }));
}

test('worker returns fuse result + preview + transferables for the GUI', async () => {
  const { messages, post } = await loadWorker();
  const { slices } = makeChartStack(160, 120, 3, { noise: 0.001 });
  await post({ id: 1, inputs: inputsOf(slices), options: {}, roi: null, excludedSlices: [] });
  const done = messages.find((m) => m.msg.ok);
  assert.ok(done, 'worker must answer ok');
  const m = done.msg;
  assert.equal(m.id, 1);
  assert.equal(m.w, 160);
  assert.equal(m.h, 120);
  assert.equal(m.weightsFull.length, 3);
  assert.equal(m.relClassFull.length, 160 * 120);
  assert.equal(m.preview.width, 160);
  assert.equal(m.preview.height, 120);
  assert.equal(m.preview.rgba.length, 160 * 120 * 4);
  // Every buffer named in the transfer list is real and detached after send;
  // check the list is consistent with what the message references.
  assert.ok(done.transfer.includes(m.preview.rgba.buffer));
  assert.ok(done.transfer.includes(m.preview.relClass.buffer));
  assert.ok(done.transfer.includes(m.weightsFull[0].buffer));
  assert.equal(m.stats.reliable > 0.5, true);
  assert.ok(Array.isArray(m.warnings));
  assert.equal(m.registration.length, 3);
  assert.equal(m.registration.filter((r) => r == null).length, 1, 'exactly one reference entry (null)');
});

test('worker forwards progress messages before the result', async () => {
  const { messages, post } = await loadWorker();
  const { slices } = makeChartStack(120, 90, 3, { noise: 0.001 });
  await post({ id: 2, inputs: inputsOf(slices), options: {}, roi: null, excludedSlices: [] });
  const progress = messages.filter((m) => m.msg.progress != null);
  assert.ok(progress.length > 0, 'progress must be reported');
  const firstProgress = messages.findIndex((m) => m.msg.progress != null);
  const resultAt = messages.findIndex((m) => m.msg.ok);
  assert.ok(firstProgress < resultAt);
});

test('worker export path returns a real 16-bit PNG', async () => {
  const { messages, post } = await loadWorker();
  const { slices } = makeChartStack(120, 90, 3, { noise: 0.001 });
  await post({
    id: 3, inputs: inputsOf(slices), options: {}, roi: null, excludedSlices: [], exportBitDepth: 16,
  });
  const m = messages.find((x) => x.msg.ok).msg;
  assert.ok(m.fusedPng && m.fusedPng.length > 0);
  const png = await decodePNG(new Uint8Array(m.fusedPng));
  assert.equal(png.width, 120);
  assert.equal(png.height, 90);
  assert.equal(png.bitDepth, 16);
  assert.equal(png.max, 65535);
  // Distinct low bytes => genuinely more than 8 bits of output.
  const low = new Set();
  for (let i = 0; i < 120 * 90; i++) low.add(png.data[i * 4] & 255);
  assert.ok(low.size > 32, `distinct low bytes ${low.size}`);
});

test('worker surfaces a size mismatch as an error instead of throwing', async () => {
  const { messages, post } = await loadWorker();
  const { slices } = makeChartStack(120, 90, 3, { noise: 0.001 });
  const inputs = inputsOf(slices);
  inputs[1].width = 121;
  await post({ id: 4, inputs, options: {}, roi: null, excludedSlices: [] });
  const err = messages.find((m) => m.msg.ok === false);
  assert.ok(err, 'worker must answer with ok:false');
  assert.equal(err.msg.id, 4);
  assert.match(err.msg.error, /size mismatch/);
});

test('worker applies a per-slice ROI (valid region) restriction', async () => {
  const { messages, post } = await loadWorker();
  const { slices } = makeChartStack(120, 90, 3, { noise: 0.001 });
  const inputs = inputsOf(slices);
  const roi = [[{ x: 0, y: 0, w: 0.5, h: 1 }], null, null];
  await post({ id: 5, inputs, options: { registration: false }, roi, excludedSlices: [] });
  const m = messages.find((x) => x.msg.ok).msg;
  // The preview weights for slice 0 must be zero on the right half.
  const pv = m.preview;
  let rightWeight = 0;
  for (let y = 0; y < pv.height; y++) {
    for (let x = Math.ceil(pv.width / 2); x < pv.width; x++) {
      rightWeight += pv.weights[0][y * pv.width + x];
    }
  }
  assert.equal(rightWeight, 0, 'slice 0 must carry no weight outside its ROI');
});
