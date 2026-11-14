import test from 'node:test';
import assert from 'node:assert/strict';
import { fuseStack } from '../src/core/fuse.js';
import { makeChartStack } from './lib/fixtures.mjs';

function asInputs(slices) {
  return slices.map((s) => ({ name: s.name, width: s.width, height: s.height, rgba: s.rgba }));
}

test('a slice restricted to a near-empty ROI triggers a weight-distribution anomaly', () => {
  const { slices } = makeChartStack(160, 120, 3, { noise: 0.001 });
  // A 6% x 6% box is ~0.36% of the frame even before border erosion, well
  // under the 0.5% weight-collapse coverage gate.
  const roi = [null, [{ x: 0, y: 0, w: 0.06, h: 0.06 }], null];
  const res = fuseStack(asInputs(slices), { registration: false }, { roi });
  const collapse = res.warnings.filter((w) => w.code === 'weight-collapse');
  assert.ok(collapse.length >= 1, `expected a weight-collapse warning, got: ${JSON.stringify(res.warnings.map((w) => w.code))}`);
  assert.equal(collapse[0].slice, 1);
  assert.match(collapse[0].message, /权重分布/);
  // The depth-of-field fusion verdict is still produced for every pixel.
  assert.equal(res.relClass.length, 160 * 120);
  // The restricted slice's weight is zero outside its admitted region.
  let leak = 0;
  for (let y = 12; y < 120; y++) for (let x = 12; x < 160; x++) leak += res.weights[1][y * 160 + x];
  assert.equal(leak, 0);
});

test('a normal stack does not raise weight-collapse warnings', () => {
  const { slices } = makeChartStack(200, 150, 4);
  const res = fuseStack(asInputs(slices), {});
  assert.equal(res.warnings.some((w) => w.code === 'weight-collapse'), false);
});
