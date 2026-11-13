// Registration between a reference slice and each focus slice.
//
// Model: similarity transform (translation + uniform scale + rotation) mapping
// reference absolute pixels to moving absolute pixels:
//   [x']   [a -b] [x]   [tx]
//   [y'] = [b  a] [y] + [ty]
// A full homography is intentionally NOT supported: with focus breathing the
// magnification changes with focus distance, and a homography would absorb
// that as a projective term while mis-registering the breathing; a similarity
// is the physically meaningful model here (one scale per slice).
//
// Staged optimisation:
//   1. translation-only, coarse-to-fine pyramid (large drift, robust),
//   2. full-resolution angle search with translation compensation,
//   3. full-resolution scale search,
//   4. joint sub-pixel angle/scale refinement and quarter-pixel polish.

import { blurPlane, gradientMagnitude } from './filters.js';
import { lumaOf } from './image.js';

function buildPyramid(rgb, w, h, levels) {
  const out = [];
  const cur = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) cur[i] = lumaOf(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
  out.push({ data: cur, w, h, f: 1 });
  for (let l = 1; l < levels; l++) {
    const prev = out[l - 1];
    const pw = Math.max(8, Math.floor(prev.w / 2));
    const ph = Math.max(8, Math.floor(prev.h / 2));
    const blurred = blurPlane(prev.data, prev.w, prev.h, 0.9);
    const next = new Float32Array(pw * ph);
    const fx = pw / prev.w;
    const fy = ph / prev.h;
    for (let y = 0; y < ph; y++) {
      const sy = Math.min(prev.h - 2, Math.max(0, Math.floor(y / fy)));
      for (let x = 0; x < pw; x++) {
        const sx = Math.min(prev.w - 2, Math.max(0, Math.floor(x / fx)));
        next[y * pw + x] = 0.25 * (
          blurred[sy * prev.w + sx] + blurred[sy * prev.w + sx + 1] +
          blurred[(sy + 1) * prev.w + sx] + blurred[(sy + 1) * prev.w + sx + 1]
        );
      }
    }
    out.push({ data: next, w: pw, h: ph, f: prev.f * fx });
  }
  return out;
}

function levelSamples(ref, budget) {
  const grad = gradientMagnitude(ref.data, ref.w, ref.h);
  let maxG = 1e-9;
  for (let i = 0; i < grad.length; i++) if (grad[i] > maxG) maxG = grad[i];
  const cand = [];
  for (let y = 3; y < ref.h - 3; y += 2) {
    for (let x = 3; x < ref.w - 3; x += 2) {
      const i = y * ref.w + x;
      if (ref.data[i] <= 0.001) continue;
      cand.push([x, y, i, grad[i] / maxG]);
    }
  }
  cand.sort((p, q) => q[3] - p[3]);
  const keep = Math.min(budget, cand.length);
  const out = [];
  const stride = Math.max(1, Math.floor(cand.length / Math.max(1, keep)));
  for (let i = 0; i < cand.length && out.length < keep; i += stride) out.push(cand[i]);
  return out;
}

function sampleBilinear(plane, w, h, x, y) {
  if (x < 0 || y < 0 || x > w - 1 || y > h - 1) return NaN;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = x0 + 1 < w ? x0 + 1 : x0;
  const y1 = y0 + 1 < h ? y0 + 1 : y0;
  const fx = x - x0;
  const fy = y - y0;
  return (plane[y0 * w + x0] * (1 - fx) + plane[y0 * w + x1] * fx) * (1 - fy) +
    (plane[y1 * w + x0] * (1 - fx) + plane[y1 * w + x1] * fx) * fy;
}

/**
 * Weighted NCC for transform P, evaluated on samples of the reference level.
 * `samples` are full-resolution [x, y, index, gradient] entries.
 */
export function evaluateNcc(ref, mov, samples, P, { minSamples = 24 } = {}) {
  const fref = ref.f;
  const fmov = mov.f;
  const mx = new Float64Array(samples.length);
  const my = new Float64Array(samples.length);
  const wt = new Float64Array(samples.length);
  let count = 0, sumW = 0, sA = 0, sB = 0;
  for (let k = 0; k < samples.length; k++) {
    const X = samples[k][0] / fref;
    const Y = samples[k][1] / fref;
    const u = fmov * (P.a * X - P.b * Y + P.tx);
    const v = fmov * (P.b * X + P.a * Y + P.ty);
    const mv = sampleBilinear(mov.data, mov.w, mov.h, u, v);
    if (!Number.isFinite(mv)) continue;
    const w = 0.15 + samples[k][3];
    mx[count] = mv;
    my[count] = ref.data[samples[k][2]];
    wt[count] = w;
    sA += mv * w;
    sB += my[count] * w;
    sumW += w;
    count++;
  }
  if (count < minSamples || sumW <= 0) return { ncc: -2, count };
  const mA = sA / sumW;
  const mB = sB / sumW;
  let vA = 0, vB = 0, cov = 0;
  for (let k = 0; k < count; k++) {
    const da = mx[k] - mA;
    const db = my[k] - mB;
    vA += da * da * wt[k];
    vB += db * db * wt[k];
    cov += da * db * wt[k];
  }
  if (vA < 1e-12 || vB < 1e-12) return { ncc: -2, count };
  return { ncc: cov / Math.sqrt(vA * vB), count };
}

function composeAbout(P, ang, sc, cx, cy) {
  const ci = sc * Math.cos(ang);
  const si = sc * Math.sin(ang);
  const dx = cx - (ci * cx - si * cy);
  const dy = cy - (si * cx + ci * cy);
  return {
    a: P.a * ci - P.b * si,
    b: P.b * ci + P.a * si,
    tx: P.a * dx - P.b * dy + P.tx,
    ty: P.b * dx + P.a * dy + P.ty,
  };
}

function searchTranslation(ref, mov, samples, P, radiusFull, stepFull) {
  let best = { score: -2, params: P, count: 0 };
  const n = Math.round(radiusFull / stepFull);
  for (let j = -n; j <= n; j++) {
    for (let i = -n; i <= n; i++) {
      const cand = { a: P.a, b: P.b, tx: P.tx + i * stepFull, ty: P.ty + j * stepFull };
      const e = evaluateNcc(ref, mov, samples, cand);
      if (e.ncc > best.score) best = { score: e.ncc, params: cand, count: e.count };
    }
  }
  return best;
}

function searchFamily(ref, mov, samples, P, angles, scales, transStepFull, transRadiusFull) {
  let best = { score: -2, params: P, count: 0 };
  const cx = (ref.w - 1) / 2;
  const cy = (ref.h - 1) / 2;
  for (const sc of scales) {
    for (const ang of angles) {
      const base = composeAbout(P, ang, sc, cx, cy);
      const local = searchTranslation(ref, mov, samples, base,
        transRadiusFull / ref.f, transStepFull);
      if (local.score > best.score) best = local;
    }
  }
  return best;
}

/**
 * Register `mov` to `ref` (planar linear RGB, same size expected).
 * @returns {{params, rotationDeg, scale, xShift, yShift, ncc, levels, perLevel}}
 */
export function registerSimilarity(ref, mov, options = {}) {
  const maxShiftFrac = options.maxShiftFrac ?? 0.03;
  const angleRangeDeg = options.angleRangeDeg ?? 1.5;
  const scaleRange = options.scaleRange ?? 0.015;
  const requestedLevels = options.levels ?? 3;
  const levels = Math.max(1, Math.min(requestedLevels,
    Math.floor(Math.log2(Math.min(ref.w, ref.h) / 60)) + 1));
  const pyrRef = buildPyramid(ref.rgb, ref.w, ref.h, levels);
  const pyrMov = buildPyramid(mov.rgb, mov.w, mov.h, levels);
  const nLevels = Math.min(pyrRef.length, pyrMov.length);
  let P = { a: 1, b: 0, tx: 0, ty: 0 };
  const perLevel = [];
  // Stage 1: translation only, coarse to fine.
  for (let l = nLevels - 1; l >= 0; l--) {
    const r = pyrRef[l];
    const m = pyrMov[l];
    const samples = levelSamples(r, l === 0 ? 2400 : 900);
    const radius = Math.ceil(maxShiftFrac * ref.w * (l === nLevels - 1 ? 2 : 1.3));
    const stepLevel = 1; // level pixels
    const res = searchTranslation(r, m, samples, P, radius / r.f, stepLevel / r.f);
    P = res.params;
    perLevel.push({ level: l, ncc: res.score });
  }
  const r0 = pyrRef[0];
  const m0 = pyrMov[0];
  const samples0 = levelSamples(r0, 3000);
  // Stage 2: angle guess.
  const angleRange = angleRangeDeg * Math.PI / 180;
  const anglesCoarse = [];
  const nA = Math.max(2, Math.round((2 * angleRange) / (0.25 * Math.PI / 180)));
  for (let i = 0; i <= nA; i++) anglesCoarse.push(-angleRange + (2 * angleRange * i) / nA);
  let res = searchFamily(r0, m0, samples0, P, anglesCoarse, [1], 1, 2.5);
  P = res.params;
  // Stage 3: scale guess. Re-centre the angle part so scale acts on the image
  // about its centre while the translation stays.
  const scales = [];
  const nS = Math.max(2, Math.round((2 * scaleRange) / 0.004));
  for (let i = 0; i <= nS; i++) scales.push(1 - scaleRange + (2 * scaleRange * i) / nS);
  res = searchFamily(r0, m0, samples0, P, [0], scales, 1, 2.5);
  P = res.params;
  // Stage 4: joint fine refinement.
  const anglesFine = [];
  const aStep = 0.04 * Math.PI / 180;
  const aR = 0.2 * Math.PI / 180;
  for (let a = -aR; a <= aR + 1e-9; a += aStep) anglesFine.push(a);
  const scalesFine = [];
  for (let s = 1 - 0.004; s <= 1.004 + 1e-9; s += 0.0008) scalesFine.push(s);
  res = searchFamily(r0, m0, samples0, P, anglesFine, scalesFine, 0.5, 1.5);
  P = res.params;
  res = searchTranslation(r0, m0, samples0, P, 0.75, 0.25);
  P = res.params;
  const ncc = evaluateNcc(r0, m0, samples0, P).ncc;
  return {
    params: P,
    rotationDeg: Math.atan2(P.b, P.a) * 180 / Math.PI,
    scale: Math.hypot(P.a, P.b),
    xShift: P.tx,
    yShift: P.ty,
    ncc,
    levels: nLevels,
    perLevel,
  };
}

export function invertSimilarity(p) {
  const det = p.a * p.a + p.b * p.b;
  if (!(det > 1e-12)) throw new Error('degenerate similarity transform');
  const a = p.a / det;
  const b = -p.b / det;
  return { a, b, tx: -(a * p.tx - b * p.ty), ty: -(b * p.tx + a * p.ty) };
}

/**
 * Resample the moving image at reference pixel coordinates using
 * x' = a x - b y + tx, y' = b x + a y + ty (params maps ref -> mov).
 */
export function sampleRegistered(mov, params, refW, refH) {
  const rgb = new Float32Array(refW * refH * 3);
  const valid = new Uint8Array(refW * refH);
  for (let y = 0; y < refH; y++) {
    for (let x = 0; x < refW; x++) {
      const sx = params.a * x - params.b * y + params.tx;
      const sy = params.b * x + params.a * y + params.ty;
      if (sx < 0 || sy < 0 || sx > mov.w - 1 || sy > mov.h - 1) continue;
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const x1 = Math.min(mov.w - 1, x0 + 1);
      const y1 = Math.min(mov.h - 1, y0 + 1);
      const fx = sx - x0;
      const fy = sy - y0;
      const w00 = (1 - fx) * (1 - fy);
      const w10 = fx * (1 - fy);
      const w01 = (1 - fx) * fy;
      const w11 = fx * fy;
      const p00 = (y0 * mov.w + x0) * 3;
      const p10 = (y0 * mov.w + x1) * 3;
      const p01 = (y1 * mov.w + x0) * 3;
      const p11 = (y1 * mov.w + x1) * 3;
      const d = (y * refW + x) * 3;
      rgb[d] = mov.rgb[p00] * w00 + mov.rgb[p10] * w10 + mov.rgb[p01] * w01 + mov.rgb[p11] * w11;
      rgb[d + 1] = mov.rgb[p00 + 1] * w00 + mov.rgb[p10 + 1] * w10 + mov.rgb[p01 + 1] * w01 + mov.rgb[p11 + 1] * w11;
      rgb[d + 2] = mov.rgb[p00 + 2] * w00 + mov.rgb[p10 + 2] * w10 + mov.rgb[p01 + 2] * w01 + mov.rgb[p11 + 2] * w11;
      valid[y * refW + x] = 255;
    }
  }
  return { rgb, valid, w: refW, h: refH };
}

/**
 * Resample a single-channel plane with the same transform used by
 * sampleRegistered. Uncovered samples are 0 and flagged in `valid`.
 */
export function sampleRegisteredPlane(plane, params, refW, refH, srcW, srcH) {
  const data = new Float32Array(refW * refH);
  const valid = new Uint8Array(refW * refH);
  for (let y = 0; y < refH; y++) {
    for (let x = 0; x < refW; x++) {
      const sx = params.a * x - params.b * y + params.tx;
      const sy = params.b * x + params.a * y + params.ty;
      if (sx < 0 || sy < 0 || sx > srcW - 1 || sy > srcH - 1) continue;
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const x1 = Math.min(srcW - 1, x0 + 1);
      const y1 = Math.min(srcH - 1, y0 + 1);
      const fx = sx - x0;
      const fy = sy - y0;
      const d = y * refW + x;
      data[d] = (plane[y0 * srcW + x0] * (1 - fx) + plane[y0 * srcW + x1] * fx) * (1 - fy) +
        (plane[y1 * srcW + x0] * (1 - fx) + plane[y1 * srcW + x1] * fx) * fy;
      valid[d] = 255;
    }
  }
  return { data, valid };
}
