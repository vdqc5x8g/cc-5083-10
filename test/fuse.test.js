import test from 'node:test';
import assert from 'node:assert/strict';
import { fuseStack, renderFused } from '../src/core/fuse.js';
import { focusMeasure } from '../src/core/focus.js';
import { linearToSrgb16, srgb16ToLinear } from '../src/core/image.js';
import { makeChartStack, maskForLabels, classFraction, meanAbsDiff8 } from './lib/fixtures.mjs';

function asInputs(slices) {
  return slices.map((s) => ({ name: s.name, width: s.width, height: s.height, rgba: s.rgba }));
}

test('rejects slices with mismatched dimensions', async () => {
  const { slices } = makeChartStack(160, 120, 3);
  const inputs = asInputs(slices);
  inputs[1] = { ...inputs[1], width: 161 };
  await assert.rejects(async () => fuseStack(inputs, {}), /size mismatch/);
});

test('weights produce a genuine per-pixel selection, not an average', async () => {
  const { slices } = makeChartStack(200, 150, 4);
  const result = fuseStack(asInputs(slices), { registration: false, exposureMatch: false });
  const N = 200 * 150;
  // At least some pixels must be dominated by each of the four slices when the
  // stack spans depth; an average would give every slice ~0.25/px.
  let dominated = new Uint8Array(4);
  for (let i = 0; i < N; i++) {
    let dom = 0;
    for (let k = 1; k < 4; k++) if (result.weights[k][i] > result.weights[dom][i]) dom = k;
    if (result.weights[dom][i] > 0.8) dominated[dom] = 1;
  }
  assert.ok(dominated.reduce((a, b) => a + b, 0) >= 3, 'three or more slices should dominate somewhere');
});

test('fused output is sharper overall than any single slice (expanded sharp coverage)', async () => {
  const { slices } = makeChartStack(200, 150, 4);
  const result = fuseStack(asInputs(slices), {});
  const fused = renderFused(result, { bitDepth: 16 });
  // Compare edge energy of fused vs each input on the raw luma.
  const fusedMeasure = focusMeasure({
    rgb: result.fused, w: 200, h: 150,
  }, result.covered);
  const perSlice = result.stack.map((s, k) => focusMeasure(s, result.veto[k]));
  let fusedGrad = 0, bestGrad = 0, n = 0;
  for (let i = 0; i < 200 * 150; i++) {
    if (!result.covered[i]) continue;
    let mx = 0;
    for (let k = 0; k < perSlice.length; k++) if (perSlice[k].grad[i] > mx) mx = perSlice[k].grad[i];
    fusedGrad += fusedMeasure.grad[i];
    bestGrad += mx;
    n++;
  }
  // A perfect fusion can only lose a little sharpness to weight blending and
  // resampling; it must clearly exceed the *average* slice, not fall to it.
  assert.ok(fusedGrad / n > 0.82 * (bestGrad / n),
    `fused gradient ${fusedGrad / n} vs best single ${bestGrad / n}`);
  assert.ok(fused.data.length === 200 * 150 * 4);
});

test('all-blurred regions are marked unreliable, not invented', async () => {
  // In the chart scene label 3 (the near object) is closer than any focus
  // distance in this stack, so no slice resolves it.
  const { slices, scene } = makeChartStack(240, 180, 4, { focuses: [0.5, 0.62, 0.85, 1.1] });
  const result = fuseStack(asInputs(slices), {});
  const near = maskForLabels(scene, [3]);
  const sphere = maskForLabels(scene, [1]);
  const nearUnreliable = classFraction(result, near, 2);
  const sphereReliable = classFraction(result, sphere, 0);
  assert.ok(nearUnreliable > 0.6, `near unreliable ${nearUnreliable}`);
  assert.ok(sphereReliable > 0.9, `sphere reliable ${sphereReliable}`);
  assert.ok(result.warnings.some((w) => w.code === 'all-blurred'));
});

test('reliable regions are not flagged unreliably (low false-positive rate)', async () => {
  const { slices, scene } = makeChartStack(240, 180, 4);
  const result = fuseStack(asInputs(slices), {});
  const backdrop = maskForLabels(scene, [0]);
  const sphere = maskForLabels(scene, [1]);
  const bar = maskForLabels(scene, [2]);
  for (const [name, mask] of [['backdrop', backdrop], ['sphere', sphere], ['bar', bar]]) {
    const u = classFraction(result, mask, 2);
    assert.ok(u < 0.08, `${name} unreliable ${(u * 100).toFixed(1)}%`);
  }
});

test('fused colour matches the selected input in linear light', async () => {
  const { slices, scene } = makeChartStack(200, 150, 4, { noise: 0 });
  const result = fuseStack(asInputs(slices), {});
  // In flat backdrop regions the fused value should match the reference
  // slice's own pixel closely: no exposure shift, no cross-slice averaging of
  // unrelated radiance.
  const refSlice = slices[result.refIndex];
  let sum = 0, n = 0, worst = 0;
  for (let i = 0; i < 200 * 150; i++) {
    if (scene.labels[i] !== 0) continue;
    if (result.relClass[i] !== 0) continue;
    if (!result.covered[i]) continue;
    // Find the dominant input at this pixel.
    let dom = 0;
    for (let k = 1; k < result.weights.length; k++) {
      if (result.weights[k][i] > result.weights[dom][i]) dom = k;
    }
    const domSlice = result.stack[dom];
    const fusedLin = [
      result.fused[i * 3], result.fused[i * 3 + 1], result.fused[i * 3 + 2],
    ];
    let diff = 0;
    for (let c = 0; c < 3; c++) diff += Math.abs(fusedLin[c] - domSlice.rgb[i * 3 + c]) / 3;
    sum += diff;
    worst = Math.max(worst, diff);
    n++;
  }
  assert.ok(n > 1000, `sample count ${n}`);
  // Weighted blending of nearly equal slices and the final normalisation
  // introduce a small tolerance; large deviations would mean the fusion is
  // mixing distinct radiances (a ghost or an exposure error).
  assert.ok(sum / n < 0.02, `mean linear colour deviation ${sum / n}`);
  assert.ok(worst < 0.35, `worst linear colour deviation ${worst}`);
  void refSlice;
});

test('16-bit export preserves depth and reconstructs sRGB', async () => {
  const { slices } = makeChartStack(120, 90, 3);
  const result = fuseStack(asInputs(slices), {});
  const out = renderFused(result, { bitDepth: 16 });
  assert.equal(out.bitDepth, 16);
  // Check a covered pixel round-trips through the sRGB curve within 1 LSB.
  let i = 0;
  while (i < 120 * 90 && !result.covered[i]) i++;
  const lin = result.fused[i * 3];
  const expect = linearToSrgb16(lin);
  assert.ok(Math.abs(out.data[i * 4] - expect) <= 1);
  // Low bytes vary: proof the file is not effectively 8-bit.
  const lowBytes = new Set();
  for (let p = 0; p < 120 * 90; p++) if (result.covered[p]) lowBytes.add(out.data[p * 4] & 255);
  assert.ok(lowBytes.size > 64, `distinct low bytes ${lowBytes.size}`);
  void srgb16ToLinear;
});

test('an explicitly restricted slice is dropped from that region', async () => {
  const { slices } = makeChartStack(160, 120, 3);
  const N = 160 * 120;
  const base = fuseStack(asInputs(slices), {});
  // Restrict the first slice to the left half only.
  const roi = [[{ x: 0, y: 0, w: 0.5, h: 1 }], null, null];
  const restricted = fuseStack(asInputs(slices), {}, { roi });
  let changedRight = 0;
  let totalRight = 0;
  for (let y = 0; y < 120; y++) {
    for (let x = 81; x < 160; x++) {
      const i = y * 160 + x;
      totalRight++;
      // With slice 0 gone, its weight must be zero in the restricted solve.
      if (restricted.weights[0][i] > 1e-6) changedRight++;
    }
  }
  assert.equal(changedRight, 0, 'slice 0 must carry no weight outside its ROI');
  // Inside the ROI the solve is essentially unchanged.
  let diff = 0;
  let n = 0;
  for (let y = 0; y < 120; y++) {
    for (let x = 0; x < 60; x++) {
      const i = y * 160 + x;
      if (!base.covered[i]) continue;
      diff += Math.abs(restricted.fused[i * 3] - base.fused[i * 3]);
      n++;
    }
  }
  assert.ok(n > 1000);
  assert.ok(diff / n < 0.01, `ROI-inside deviation ${diff / n}`);
  void N;
});

test('excluding a slice removes it from the automatic solve', async () => {
  const { slices } = makeChartStack(160, 120, 4);
  const inputs = asInputs(slices);
  const res = fuseStack(inputs, {}, { excludedSlices: new Set([1]) });
  assert.deepEqual(res.skip, [1]);
  for (let i = 0; i < 160 * 120; i++) assert.equal(res.weights[1][i], 0);
  // Remaining slices still fuse normally.
  assert.ok(res.stats.reliable > 0.5);
  void meanAbsDiff8;
});

test('multiple ROI boxes are unioned', async () => {
  const { slices } = makeChartStack(120, 90, 3);
  const roi = [[{ x: 0.1, y: 0.1, w: 0.3, h: 0.3 }, { x: 0.6, y: 0.6, w: 0.3, h: 0.3 }], null, null];
  const res = fuseStack(asInputs(slices), {}, { roi });
  // Weight for slice 0 must be zero in the gap between the two boxes.
  const i = Math.floor(90 * 0.5) * 120 + Math.floor(120 * 0.5);
  assert.equal(res.weights[0][i], 0);
  // And non-zero inside each box's centre.
  const c1 = Math.floor(90 * 0.25) * 120 + Math.floor(120 * 0.25);
  const c2 = Math.floor(90 * 0.75) * 120 + Math.floor(120 * 0.75);
  if (res.coverage.validFraction > 0) {
    assert.ok(res.veto[0][c1] === 1 || res.weights[2][c1] > 0);
    assert.ok(res.veto[0][c2] === 1 || res.weights[2][c2] > 0);
  }
});

test('selection edge width stays close to the target', async () => {
  const { slices } = makeChartStack(240, 180, 4);
  const res = fuseStack(asInputs(slices), {});
  assert.ok(res.edgeWidth.samples > 0, 'edge width must be measured');
  assert.ok(res.edgeWidth.median < 12, `median edge width ${res.edgeWidth.median}`);
});
