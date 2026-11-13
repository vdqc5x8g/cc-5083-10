import test from 'node:test';
import assert from 'node:assert/strict';
import { fuseStack, renderFused } from '../src/core/fuse.js';
import { focusMeasure } from '../src/core/focus.js';
import { buildStillLife, renderStillLifeStack, stillLifeFocuses } from '../src/core/stilllife.js';
import { maskForLabels, classFraction, meanAbsDiff8 } from './lib/fixtures.mjs';

const W = 300;
const H = 225;

function asInputs(slices) {
  return slices.map((s) => ({ name: s.name, width: s.width, height: s.height, rgba: s.rgba }));
}

test('still-life stack has real depth structure and an object-focused slice for every object', () => {
  const scene = buildStillLife(W, H);
  // Objects at distinct depths: the closest (sprig, label 5) and the backdrop
  // (label 0) must be separated, or the sequence would not test occlusion.
  const depths = new Map();
  for (let i = 0; i < W * H; i++) {
    const l = scene.labels[i];
    if (!depths.has(l) || scene.depth[i] < depths.get(l)) depths.set(l, scene.depth[i]);
  }
  assert.ok(depths.has(5) && depths.has(0));
  assert.ok(depths.get(5) < depths.get(0) - 0.3, 'foreground must be much closer than backdrop');
  // Curved objects span a depth range; the focus sweep must land within
  // ~0.15 dioptre of each object's representative (median) surface depth so no
  // object is left permanently out of focus across its whole footprint.
  const medianDepth = new Map();
  for (let i = 0; i < W * H; i++) {
    const l = scene.labels[i];
    const arr = medianDepth.get(l) || [];
    arr.push(scene.depth[i]);
    medianDepth.set(l, arr);
  }
  const focuses = stillLifeFocuses(8);
  for (const [label, arr] of medianDepth) {
    if (label === 0) continue;
    arr.sort((a, b) => a - b);
    const d = arr[Math.floor(arr.length / 2)];
    const minGap = Math.min(...focuses.map((f) => Math.abs(1 / f - 1 / (d || 1e-6))));
    assert.ok(minGap < 0.15, `label ${label} median depth ${d.toFixed(3)} not covered by ${focuses}`);
  }
});

test('still-life fusion expands sharp coverage and picks the physically focused slice', () => {
  const { scene, slices } = renderStillLifeStack(W, H, 6, { noise: 0.002 });
  const result = fuseStack(asInputs(slices), {});
  const measures = result.stack.map((s, k) => focusMeasure(s, result.veto[k]));
  // Winner agreement: where the fused weight is decisive, it should pick the
  // slice with the highest real focus measure at that pixel.
  let agree = 0;
  let total = 0;
  for (let i = 0; i < W * H; i++) {
    if (!result.covered[i]) continue;
    let bestK = -1;
    let bestE = -1;
    for (let k = 0; k < measures.length; k++) {
      if (result.skip.includes(k)) continue;
      if (measures[k].energy[i] > bestE) { bestE = measures[k].energy[i]; bestK = k; }
    }
    let domK = -1;
    let domW = -1;
    for (let k = 0; k < result.weights.length; k++) {
      if (result.weights[k][i] > domW) { domW = result.weights[k][i]; domK = k; }
    }
    if (bestK < 0 || domW < 0.5) continue;
    total++;
    if (domK === bestK) agree++;
  }
  assert.ok(total > 20000);
  assert.ok(agree / total > 0.92, `winner agreement ${(100 * agree / total).toFixed(1)}%`);

  // Expanded coverage: the fused edge energy should meet or nearly meet the
  // best single slice's, averaged over the frame.
  const fusedMeasure = focusMeasure({ rgb: result.fused, w: W, h: H }, result.covered);
  let f = 0;
  let b = 0;
  for (let i = 0; i < W * H; i++) {
    if (!result.covered[i]) continue;
    let mx = 0;
    for (const m of measures) if (m.grad[i] > mx) mx = m.grad[i];
    f += fusedMeasure.grad[i];
    b += mx;
  }
  assert.ok(f > 0.72 * b, `fused edge energy ${f} vs best single ${b}`);
  void scene;
});

test('still-life keeps the sharp foreground and backdrop both in focus', () => {
  const { scene, slices } = renderStillLifeStack(W, H, 6, { noise: 0.002 });
  const result = fuseStack(asInputs(slices), {});
  // With a focus slice on the backdrop, the objects and the cloth are in
  // focus. Structured objects must be reliable; the cloth is mostly flat
  // low-contrast paper, so its "all inputs blurred" test legitimately has
  // little to work with, and a modest false-positive fraction there is the
  // physically honest outcome rather than a threshold failure.
  for (const label of [1, 2, 3, 4, 5, 6]) {
    const mask = maskForLabels(scene, [label]);
    let n = 0;
    for (let i = 0; i < mask.length; i++) if (mask[i]) n++;
    if (n < 300) continue;
    const unreliable = classFraction(result, mask, 2);
    assert.ok(unreliable < 0.2, `label ${label} unreliable ${(unreliable * 100).toFixed(1)}%`);
  }
  const cloth = classFraction(result, maskForLabels(scene, [0]), 2);
  assert.ok(cloth < 0.35, `flat cloth unreliable ${(cloth * 100).toFixed(1)}%`);
});

test('a stack that never focuses the backdrop marks it unreliable', () => {
  const { scene, slices } = renderStillLifeStack(W, H, 6, { noise: 0.002, includeBackdrop: false });
  const result = fuseStack(asInputs(slices), {});
  const backdrop = maskForLabels(scene, [0]);
  const sprig = maskForLabels(scene, [5]);
  const backdropReliable = classFraction(result, backdrop, 0);
  const sprigReliable = classFraction(result, sprig, 0);
  assert.ok(backdropReliable < 0.4, `backdrop reliable ${backdropReliable}`);
  assert.ok(sprigReliable > 0.8, `sprig reliable ${sprigReliable}`);
});

test('still-life fused colour tracks the dominant sharp slice, no ghost averaging', () => {
  const { slices } = renderStillLifeStack(W, H, 6, { noise: 0, drift: 0 });
  const result = fuseStack(asInputs(slices), {});
  const fused = renderFused(result, { bitDepth: 16 });
  // Compare against the dominant registered slice. Use only flat, reliable
  // pixels to avoid edge misregistration dominating the metric.
  let sum = 0;
  let n = 0;
  for (let y = 5; y < H - 5; y++) {
    for (let x = 5; x < W - 5; x++) {
      const i = y * W + x;
      if (result.relClass[i] !== 0 || !result.covered[i]) continue;
      const fusedMeasureCover = result.weights;
      let dom = 0;
      for (let k = 1; k < fusedMeasureCover.length; k++) {
        if (fusedMeasureCover[k][i] > fusedMeasureCover[dom][i]) dom = k;
      }
      for (let c = 0; c < 3; c++) {
        sum += Math.abs(result.fused[i * 3 + c] - result.stack[dom].rgb[i * 3 + c]) / 3;
        n++;
      }
    }
  }
  assert.ok(n > 5000);
  assert.ok(sum / n < 0.025, `mean linear deviation from dominant slice ${sum / n}`);
  void meanAbsDiff8; void fused;
});

test('edge width on the still-life scene is bounded (occlusions stay crisp)', () => {
  const { slices } = renderStillLifeStack(W, H, 6, { noise: 0.002 });
  const result = fuseStack(asInputs(slices), {});
  assert.ok(result.edgeWidth.samples > 0);
  assert.ok(result.edgeWidth.median < 28, `median edge width ${result.edgeWidth.median}`);
  assert.ok(result.ghosting.index < 0.2, `ghosting index ${result.ghosting.index}`);
});
