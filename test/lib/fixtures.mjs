// Shared helpers for the test suite. Kept free of side effects so importing it
// from any test never renders a stack by accident.

import { buildScene, renderSlice, renderSharp } from '../../src/core/synth.js';
import { renderStillLifeStack } from '../../src/core/stilllife.js';

const CHART_FOCUSES = [0.5, 0.62, 0.85, 1.1];

function sampleFocuses(count) {
  if (count <= 1) return [CHART_FOCUSES[0]];
  const out = [];
  for (let k = 0; k < count; k++) {
    const pos = (k * (CHART_FOCUSES.length - 1)) / (count - 1);
    const i0 = Math.floor(pos);
    const i1 = Math.min(CHART_FOCUSES.length - 1, i0 + 1);
    const t = pos - i0;
    const inv = 1 / CHART_FOCUSES[i0] + (1 / CHART_FOCUSES[i1] - 1 / CHART_FOCUSES[i0]) * t;
    out.push(Number((1 / inv).toFixed(4)));
  }
  return out;
}

export function makeChartStack(w = 240, h = 180, count = 4, opts = {}) {
  const scene = buildScene(w, h);
  const focuses = opts.focuses || sampleFocuses(count);
  const slices = focuses.map((focus, k) => {
    const rendered = renderSlice(scene, {
      focus,
      aperture: opts.aperture ?? 0.9,
      breathing: opts.breathing ?? 0.004,
      shift: opts.shift ? opts.shift(k) : { x: (k - (focuses.length - 1) / 2) * 1.2, y: k % 2 ? 1 : -1 },
      noise: opts.noise ?? 0.002,
    });
    return { name: `chart-${k}`, focus, ...rendered };
  });
  return { scene, slices, sharp: renderSharp(scene) };
}

export function makeStillStack(w = 240, h = 180, count = 6, opts = {}) {
  return renderStillLifeStack(w, h, count, opts);
}

/** Fraction of pixels (over a truth mask) whose relClass is a given value. */
export function classFraction(result, mask, cls) {
  let n = 0;
  let hit = 0;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    n++;
    if (result.relClass[i] === cls) hit++;
  }
  return n ? hit / n : 0;
}

export function maskForLabels(scene, labels) {
  const set = new Set(labels);
  const mask = new Uint8Array(scene.labels.length);
  for (let i = 0; i < mask.length; i++) if (set.has(scene.labels[i])) mask[i] = 1;
  return mask;
}

/** Pearson correlation between two equal-length Float32Arrays. */
export function correlation(a, b) {
  const n = a.length;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let va = 0, vb = 0, cov = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma;
    const db = b[i] - mb;
    va += da * da;
    vb += db * db;
    cov += da * db;
  }
  return cov / Math.sqrt(Math.max(1e-20, va * vb));
}

/** Mean absolute error, in 8-bit units, between a Uint16 RGBA and a Uint16 RGBA. */
export function meanAbsDiff8(a, b, mask) {
  let sum = 0;
  let n = 0;
  for (let i = 0; i < a.length / 4; i++) {
    if (mask && !mask[i]) continue;
    for (let c = 0; c < 3; c++) {
      sum += Math.abs(a[i * 4 + c] - b[i * 4 + c]) / 257;
      n++;
    }
  }
  return n ? sum / n : 0;
}
