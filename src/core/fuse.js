// Focal stacking pipeline.
//
// Policy summary (everything below is actual computation; no pre-drawn masks):
//  * size check + registration first; slices that fail the plausibility gate
//    are excluded from the automatic solve and reported;
//  * linear-light everything; per-slice exposure gain only corrects the
//    sharpness measure, not the fused radiance;
//  * focus energy is noise-floor corrected so flat regions blend smoothly;
//  * weights are normalised, then refined by a guided filter whose guide is
//    the fused luma, so the ramp snaps to real occlusion edges;
//  * high-contrast guard: clipped pixels are down-weighted, and a post-fusion
//    synthesis check compares edge energy of the chosen slice with the fused
//    result; a sharp input edge that does not reappear marks the pixel as
//    unreliable instead of inventing detail;
//  * "all inputs blurred" is reported when no slice carries meaningful edge
//    energy at a structured pixel.

import { toLinear, linearToSrgb16, lumaOf } from './image.js';
import { blurPlane, gradientMagnitude, boxMean } from './filters.js';
import { registerSimilarity, sampleRegistered, sampleRegisteredPlane } from './register.js';
import { focusMeasure, noiseFloor } from './focus.js';

export const DEFAULTS = {
  sigma: 1.6,
  smoothCoarse: 5,
  marginGamma: 6,
  angleRangeDeg: 1.5,
  scaleRange: 0.02,
  maxShiftFrac: 0.03,
  registration: true,
  exposureMatch: true,
  clipGuard: true,
  relLow: 0.5,
  relHigh: 0.9,
  sharpEnough: 0.008,
  focusMargin: 1.8,
  edgeWidthTarget: 4,
  previewMaxDim: 1100,
};

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

function clipPlaneOf(rgba, w, h) {
  const n = w * h;
  const out = new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    let c = 0;
    if (rgba[p] >= 65534) c++;
    if (rgba[p + 1] >= 65534) c++;
    if (rgba[p + 2] >= 65534) c++;
    if (rgba[p] <= 1) c++;
    if (rgba[p + 1] <= 1) c++;
    if (rgba[p + 2] <= 1) c++;
    out[i] = c / 6;
  }
  return out;
}

/**
 * Prepare the linear-light, registered stack.
 * @param {Array<{name,width,height,rgba}>} inputs
 */
export function prepareStack(inputs, options = {}) {
  const opts = { ...DEFAULTS, ...options };
  const n = inputs.length;
  if (n < 2) throw new Error('focus fusion needs at least two slices');
  const w = inputs[0].width;
  const h = inputs[0].height;
  for (const im of inputs) {
    if (im.width !== w || im.height !== h) {
      throw new Error(`size mismatch: "${im.name}" is ${im.width}x${im.height}, expected ${w}x${h}`);
    }
    if (!im.rgba || im.rgba.length < w * h * 4) throw new Error(`"${im.name}" has no pixel data`);
  }
  const linear = inputs.map((im) => toLinear(im.rgba, w, h));
  let ref = 0;
  for (let i = 1; i < n; i++) {
    const score = linear[i].median * Math.max(0.2, 1 - linear[i].clipped / (w * h) * 50);
    const best = linear[ref].median * Math.max(0.2, 1 - linear[ref].clipped / (w * h) * 50);
    if (score > best) ref = i;
  }
  const registration = [];
  const stack = [];
  const warnings = [];
  const border = 2;
  for (let k = 0; k < n; k++) {
    const clipSource = clipPlaneOf(inputs[k].rgba, w, h);
    if (k === ref) {
      registration.push(null);
      stack.push({
        rgb: linear[k].rgb, valid: new Uint8Array(linear[k].alpha),
        w, h, clip: clipSource, sourceIndex: k,
      });
      continue;
    }
    if (!opts.registration) {
      registration.push(null);
      stack.push({
        rgb: linear[k].rgb, valid: new Uint8Array(linear[k].alpha),
        w, h, clip: clipSource, sourceIndex: k, unregistered: true,
      });
      continue;
    }
    const r = registerSimilarity(
      { rgb: linear[ref].rgb, w, h },
      { rgb: linear[k].rgb, w, h },
      {
        maxShiftFrac: opts.maxShiftFrac,
        angleRangeDeg: opts.angleRangeDeg,
        scaleRange: opts.scaleRange,
      },
    );
    const shiftPx = Math.hypot(r.xShift, r.yShift);
    const plausible = r.ncc >= 0.3 &&
      shiftPx <= opts.maxShiftFrac * Math.max(w, h) * 1.6 &&
      Math.abs(r.scale - 1) <= opts.scaleRange * 2.5 &&
      Math.abs(r.rotationDeg) <= opts.angleRangeDeg * 1.5;
    registration.push({
      params: r.params, rotationDeg: r.rotationDeg, scale: r.scale,
      shiftPx: { x: r.xShift, y: r.yShift }, ncc: r.ncc, ok: plausible,
    });
    if (!plausible) {
      warnings.push({
        code: 'registration',
        slice: k, name: inputs[k].name,
        message: `配准置信度不足 (NCC ${r.ncc.toFixed(2)}, 位移 ${shiftPx.toFixed(1)}px, 缩放 ${r.scale.toFixed(3)}); 该片自动权重已排除`,
      });
    }
    // r.params maps reference pixels to the moving image's matching content;
    // the sampler evaluates the moving image at those reference locations.
    const s = sampleRegistered({ rgb: linear[k].rgb, w, h }, r.params, w, h);
    const sc = sampleRegisteredPlane(clipSource, r.params, w, h, w, h);
    const valid = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) valid[i] = (s.valid[i] && linear[k].alpha[i]) ? 1 : 0;
    erodeValidBorder(valid, w, h, border);
    stack.push({
      rgb: s.rgb, valid, w, h,
      clip: sc.data, sourceIndex: k,
      experimental: !plausible,
    });
  }
  return { stack, registration, refIndex: ref, w, h, warnings, options: opts, inputs };
}

/**
 * Drop pixels within `r` of the valid-region border. Resampling leaves
 * partially supported samples and interpolation ringing there; those must not
 * feed the weight solve or the reliability verdict.
 */
function erodeValidBorder(valid, w, h, r) {
  if (r <= 0) return;
  const src = Uint8Array.from(valid);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!src[i]) continue;
      let ok = 1;
      for (let dy = -r; dy <= r && ok; dy++) {
        const yy = y + dy;
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h || !src[yy * w + xx]) { ok = 0; break; }
        }
      }
      if (!ok) valid[i] = 0;
    }
  }
}

/**
 * Relative blur estimate between two slices, amplitude-independent.
 *
 * For a spatially smooth image patch blurred by Gaussian sigma, the gradient
 * variance after additional blur s is  E(s) = A/(sigma^2+s^2)  (for the 2D
 * Laplacian/Gaussian model). Measuring E at two extra blur levels and solving
 * the ratio isolates sigma without knowing the amplitude A. Used only to
 * evaluate *relative* focus and to declare "all inputs blurred"; absolute
 * values are not reported.
 */
/**
 * Weight refinement: box smoothing to suppress speckle, then a guided filter
 * whose guide is the fused luma so the transition snaps to the occluder edge.
 */
function refineWeights(weight, guide, valid, w, h, opts) {
  const coarseRadius = Math.max(1, Math.round(opts.smoothCoarse * Math.min(1, Math.max(w, h) / 1400)));
  const sm = boxMean(weight, w, h, coarseRadius);
  const guid = new Float32Array(weight.length);
  const inp = new Float32Array(weight.length);
  for (let i = 0; i < weight.length; i++) {
    guid[i] = valid[i] ? guide[i] : 0;
    inp[i] = valid[i] ? sm[i] : 0;
  }
  const radius = Math.max(2, coarseRadius);
  const eps = 0.006;
  const meanI = boxMean(guid, w, h, radius);
  const meanP = boxMean(inp, w, h, radius);
  const ip = new Float32Array(weight.length);
  const ii = new Float32Array(weight.length);
  for (let i = 0; i < weight.length; i++) {
    ip[i] = guid[i] * inp[i];
    ii[i] = guid[i] * guid[i];
  }
  const meanIP = boxMean(ip, w, h, radius);
  const meanII = boxMean(ii, w, h, radius);
  const out = new Float32Array(weight.length);
  for (let i = 0; i < weight.length; i++) {
    const varI = Math.max(0, meanII[i] - meanI[i] * meanI[i]);
    const cov = meanIP[i] - meanI[i] * meanP[i];
    const a = cov / (varI + eps);
    const b = meanP[i] - a * meanI[i];
    out[i] = valid[i] ? clamp(a * guid[i] + b, 0, 1) : 0;
  }
  return out;
}

/**
 * Fuse a stack.
 * @param inputs array of {name,width,height,rgba:Uint16Array}
 * @param options
 * @param hooks {roi?: (box|null)[], excludedSlices?: Set<number>, onProgress?: fn}
 */
export function fuseStack(inputs, options = {}, hooks = {}) {
  const opts = { ...DEFAULTS, ...options };
  const prepared = prepareStack(inputs, opts);
  const { stack, registration, refIndex, w, h } = prepared;
  const n = stack.length;
  const N = w * h;
  const roi = hooks.roi || null;
  const excluded = hooks.excludedSlices instanceof Set ? hooks.excludedSlices : new Set(hooks.excludedSlices || []);
  const warnings = [...prepared.warnings];
  const skip = [];
  for (let k = 0; k < n; k++) {
    if (excluded.has(k) || (k !== refIndex && stack[k].experimental)) skip.push(k);
  }
  if (skip.length >= n) throw new Error('all slices excluded from the automatic solve');
  const skipMask = new Uint8Array(n);
  for (const k of skip) skipMask[k] = 1;
  // Per-slice effective validity = source alpha ∩ registration support ∩ ROI.
  // A slice's ROI may be one box or a list of boxes (normalised 0..1); the
  // pixel is admitted if it falls inside *any* box.
  const veto = [];
  for (let k = 0; k < n; k++) {
    const v = new Uint8Array(N);
    const valid = stack[k].valid;
    const b = roi ? roi[k] : null;
    const boxes = !b ? null : Array.isArray(b) ? (b.length ? b : null) : [b];
    for (let y = 0; y < h; y++) {
      const ny = y / Math.max(1, h - 1);
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!valid[i]) continue;
        if (boxes) {
          const nx = x / Math.max(1, w - 1);
          let inside = false;
          for (const box of boxes) {
            if (nx >= box.x && ny >= box.y && nx <= box.x + box.w && ny <= box.y + box.h) {
              inside = true;
              break;
            }
          }
          if (!inside) continue;
        }
        v[i] = 1;
      }
    }
    veto.push(v);
  }
  // Focus measures.
  const measures = stack.map((s, k) => focusMeasure(s, veto[k], opts.sigma));
  if (hooks.onProgress) hooks.onProgress(0.25);
  // Exposure compensation of the sharpness measure only.
  const gains = new Array(n).fill(1);
  if (opts.exposureMatch) {
    for (let k = 0; k < n; k++) {
      if (k === refIndex || skipMask[k]) continue;
      let refSum = 0, movSum = 0, cnt = 0;
      for (let i = 0; i < N; i += 13) {
        if (!veto[k][i] || !veto[refIndex][i]) continue;
        refSum += measures[refIndex].luma[i];
        movSum += measures[k].luma[i];
        cnt++;
      }
      if (cnt > 8 && movSum > 1e-5) gains[k] = clamp(refSum / movSum, 0.4, 2.5);
    }
  }
  // Clipped-and-high-contrast guard: per-slice effective energy is reduced
  // where the source itself is clipped (energy there is not a focus cue).
  const energies = [];
  for (let k = 0; k < n; k++) {
    const e = new Float32Array(N);
    const g2 = gains[k] * gains[k];
    for (let i = 0; i < N; i++) {
      if (!veto[k][i]) continue;
      const c = opts.clipGuard ? stack[k].clip[i] : 0;
      const penalty = 0.4 + 0.6 * (1 - Math.min(1, c * 3));
      e[i] = measures[k].energy[i] * g2 * penalty;
    }
    energies.push(e);
  }
  const refMeasure = measures[refIndex];
  // Per-pixel focus discrimination features:
  //   A = best slice energy / local luma   (absolute "is it sharp at all")
  //   B = best slice energy / weakest slice energy  ("is any slice really
  //       focused"). B near 1 means every slice is equally (un)sharp at this
  //       location, which is the "all inputs blurred" signature: if one slice
  //       were genuinely focused its energy would stand clear of the blurred
  //       copies. A alone cannot tell a low-contrast sharp slab from a
  //       low-contrast blurred one, so both are required.
  const bestEnergy = new Float32Array(N);
  const weakestEnergy = new Float32Array(N);
  const sharpnessA = new Float32Array(N);
  const sharpnessB = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let m1 = 0, lo = Infinity, cnt = 0;
    for (let k = 0; k < n; k++) {
      if (skipMask[k] || !veto[k][i]) continue;
      cnt++;
      const e = energies[k][i];
      if (e > m1) m1 = e;
      if (e < lo) lo = e;
    }
    bestEnergy[i] = m1;
    weakestEnergy[i] = cnt ? lo : 0;
    const lc = Math.max(1e-5, refMeasure.localLuma[i]);
    sharpnessA[i] = m1 / lc;
    sharpnessB[i] = cnt >= 2 ? m1 / Math.max(1e-9, lo) : 1;
  }
  const texture = sharpnessA;
  const floor = noiseFloor(
    (() => {
      let count = 0;
      for (let k = 0; k < n; k++) if (!skipMask[k]) for (let i = 0; i < N; i++) if (veto[k][i]) count++;
      const comb = new Float32Array(count);
      let p = 0;
      for (let k = 0; k < n; k++) {
        if (skipMask[k]) continue;
        for (let i = 0; i < N; i++) if (veto[k][i]) comb[p++] = energies[k][i];
      }
      return comb;
    })(),
    N,
  );
  // Raw weights, built from the *relative* focus evidence: a slice earns
  // weight in proportion to how far above the local cross-slice competition it
  // is. This is the key against blurred bleed: a blurred copy of a strong
  // foreground edge still has energy, but it is clearly below the sharp copy,
  // so its weight collapses instead of averaging a phantom edge into the fused
  // image. In flat areas all slices sit at the floor and blend evenly.
  const weights = [];
  for (let k = 0; k < n; k++) weights.push(new Float32Array(N));
  const rawSum = new Float32Array(N);
  const margin = new Float32Array(N); // local best/second ratio, for reporting
  for (let i = 0; i < N; i++) {
    const denom = Math.max(1e-9, bestEnergy[i]);
    let sum = 0;
    for (let k = 0; k < n; k++) {
      if (skipMask[k] || !veto[k][i]) continue;
      const e = energies[k][i];
      // Relative evidence 0..1: how close this slice is to the best one.
      const rel = clamp(e / denom, 0, 1);
      // Emphasise the separation: a 2x weaker slice gets (0.5)^gamma.
      const v = Math.pow(rel, opts.marginGamma);
      weights[k][i] = v;
      sum += v;
    }
    rawSum[i] = sum;
    margin[i] = sharpnessB[i];
  }
  for (let i = 0; i < N; i++) {
    if (rawSum[i] > 0) for (let k = 0; k < n; k++) weights[k][i] /= rawSum[i];
  }
  if (hooks.onProgress) hooks.onProgress(0.4);
  // Optional user-supplied per-pixel veto (extra masks).
  if (typeof hooks.extraVeto === 'function') {
    const extra = hooks.extraVeto({ w, h, stack, veto });
    if (extra) {
      for (let k = 0; k < n; k++) if (extra[k]) for (let i = 0; i < N; i++) {
        if (!extra[k][i]) { veto[k][i] = 0; weights[k][i] = 0; }
      }
    }
  }
  // Provisional fused luma for the guide.
  const prov = new Float32Array(N * 3);
  const provSum = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let sm = 0;
    for (let k = 0; k < n; k++) sm += weights[k][i];
    provSum[i] = sm;
  }
  for (let i = 0; i < N; i++) {
    if (provSum[i] <= 1e-9) continue;
    for (let k = 0; k < n; k++) {
      const wk = weights[k][i] / provSum[i];
      if (wk <= 0) continue;
      prov[i * 3] += stack[k].rgb[i * 3] * wk;
      prov[i * 3 + 1] += stack[k].rgb[i * 3 + 1] * wk;
      prov[i * 3 + 2] += stack[k].rgb[i * 3 + 2] * wk;
    }
  }
  const guide = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    guide[i] = Math.max(0.0005, lumaOf(prov[i * 3], prov[i * 3 + 1], prov[i * 3 + 2]));
  }
  // Edge-preserving refinement.
  for (let pass = 0; pass < 2; pass++) {
    for (let k = 0; k < n; k++) {
      if (skipMask[k]) continue;
      const merged = refineWeights(weights[k], guide, veto[k], w, h, opts);
      for (let i = 0; i < N; i++) weights[k][i] = merged[i];
    }
  }
  if (hooks.onProgress) hooks.onProgress(0.55);
  // Final normalisation and fusion.
  const sumW = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let k = 0; k < n; k++) s += weights[k][i];
    sumW[i] = s;
  }
  const fused = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    if (sumW[i] <= 1e-9) continue;
    for (let k = 0; k < n; k++) {
      const wk = weights[k][i] / sumW[i];
      if (wk === 0) continue;
      fused[i * 3] += stack[k].rgb[i * 3] * wk;
      fused[i * 3 + 1] += stack[k].rgb[i * 3 + 1] * wk;
      fused[i * 3 + 2] += stack[k].rgb[i * 3 + 2] * wk;
    }
  }
  const normWeights = weights.map((plane) => {
    const out = new Float32Array(N);
    for (let i = 0; i < N; i++) out[i] = sumW[i] > 1e-9 ? plane[i] / sumW[i] : 0;
    return out;
  });
  const covered = new Uint8Array(N);
  for (let i = 0; i < N; i++) covered[i] = sumW[i] > 1e-9 ? 1 : 0;
  if (hooks.onProgress) hooks.onProgress(0.7);
  // Post-fusion analysis (synthesis check, reliability, edge-width, ghosting).
  const analysis = analyse({
    stack, energies, normWeights, fused, veto, skipMask, refIndex, opts, w, h, covered, floor,
    texture, sharpnessA, sharpnessB, bestEnergy, weakestEnergy,
  });
  if (hooks.onProgress) hooks.onProgress(0.85);
  for (const p of analysis.problems) warnings.push(p);
  return {
    w, h,
    fused, weights: normWeights, covered,
    reliability: analysis.reliability, relClass: analysis.relClass,
    edgeWidth: analysis.edgeWidth, ghosting: analysis.ghosting,
    sharpnessFloor: floor, gains,
    registration, refIndex, skip,
    coverage: {
      validFraction: covered.reduce((a, v) => a + v, 0) / N,
    },
    warnings,
    stats: analysis.stats,
    options: opts,
    stack, veto,
  };
}


function gxAt(plane, w, h, i) {
  const x = i % w;
  const xm = x > 0 ? i - 1 : i;
  const xp = x < w - 1 ? i + 1 : i;
  return (plane[xp] - plane[xm]) * 0.5;
}

function gyAt(plane, w, h, i) {
  const ym = i - w >= 0 ? i - w : i;
  const yp = i + w < w * h ? i + w : i;
  return (plane[yp] - plane[ym]) * 0.5;
}

function analyse(ctx) {
  const { stack, energies, normWeights, fused, veto, skipMask, refIndex, opts, w, h, covered, floor,
    texture, sharpnessA, sharpnessB, bestEnergy, weakestEnergy } = ctx;
  const N = w * h;
  const n = stack.length;
  // Best input edge energy actually selected.
  const selectedEnergy = new Float32Array(N);
  const allMax = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let sel = 0, mx = 0;
    for (let k = 0; k < n; k++) {
      if (skipMask[k]) continue;
      const e = energies[k][i];
      if (e > mx) mx = e;
      if (normWeights[k][i] > 0.05 && e > sel) sel = e;
    }
    selectedEnergy[i] = sel;
    allMax[i] = mx;
  }
  const fusedImg = { rgb: fused, w, h };
  const fusedMeasure = focusMeasure(fusedImg, covered, opts.sigma);
  const rel = new Float32Array(N);
  const relClass = new Uint8Array(N);
  // "All inputs blurred": no slice carries energy above the noise floor while
  // the local luma does vary (i.e. there is structure that should have detail).
  const structure = gradientMagnitude(blurPlane(fusedMeasure.luma, w, h, 3), w, h);
  const structureMax = (() => {
    let mx = 1e-9;
    for (let i = 0; i < N; i++) if (structure[i] > mx) mx = structure[i];
    return mx;
  })();
  const clipAtPixel = new Float32Array(N);
  // How many slices are actually usable at each pixel under the effective
  // region (source alpha, registration support and the user ROI). A single
  // usable slice cannot be cross-checked; that is a coverage limitation, not a
  // confirmed sharp region, so it is reported separately.
  const usableCount = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    let cnt = 0;
    for (let k = 0; k < n; k++) {
      if (skipMask[k]) continue;
      if (!veto[k][i]) continue;
      if (stack[k].clip[i] > 0.05 && clipAtPixel[i] < stack[k].clip[i]) clipAtPixel[i] = stack[k].clip[i];
      if (normWeights[k][i] > 0.02) cnt++;
    }
    usableCount[i] = 0;
    for (let k = 0; k < n; k++) {
      if (skipMask[k] || !veto[k][i]) continue;
      usableCount[i]++;
    }
  }
  for (let i = 0; i < N; i++) {
    if (!covered[i]) { rel[i] = 0; relClass[i] = 3; continue; }
    const want = Math.sqrt(Math.max(0, selectedEnergy[i]));
    const got = Math.sqrt(Math.max(0, fusedMeasure.energy[i]));
    // Synthesis check: the selected slice's edge energy must reappear in the
    // fused image. Where the selected slice carries essentially no edge
    // energy (flat cloth, wall, sky) the check has nothing to compare and
    // would test noise against noise; there it is reported as satisfied.
    // The threshold is on the *raw* energy, a few times the estimated floor,
    // not on the luma-normalised A, because flat low-contrast content legitimately
    // has very low A yet is perfectly sharp.
    const edgeful = want > 0 && selectedEnergy[i] > Math.max(4 * floor, 2e-3);
    let ratio = edgeful && want > 1e-5 ? got / want : 1;
    // Clipped high-contrast edge: the response is not a dependable focus cue,
    // so soften the judgment rather than claiming success or failure.
    if (clipAtPixel[i] > 0.04) ratio = ratio + (1 - ratio) * Math.min(1, clipAtPixel[i] * 4) * 0.7;
    // "Every input is blurred here": the best focus level is small AND the
    // spread across slices is small. Requiring both avoids calling a genuinely
    // sharp-but-low-contrast patch blurred, and avoids trusting a speck of
    // noise that happens to make one slice look sharper. `allMax` must also
    // stay near the floor for the pixel to count as structure-free.
    const noDistinction = sharpnessB[i] < opts.focusMargin;
    const noFocus = sharpnessA[i] < opts.sharpEnough;
    // Cross-slice distinction is undefined when only one slice covers the
    // pixel: B falls back to 1.0, so the test would fire on every single-source
    // pixel. That is a coverage limitation (reported as marginal below), not
    // evidence that every input is blurred.
    const noSharpSample = noFocus && noDistinction && usableCount[i] >= 2;
    // A pixel only supported by a single input cannot be cross-validated; mark
    // it as having no usable alternative rather than a confirmed result.
    const singleSource = usableCount[i] < 2;
    // Report the weakest of the independent checks: an unreliable answer from
    // any one of them must not be hidden by a good value from another.
    const edgeEcho = clamp(ratio, 0, 1);
    rel[i] = noSharpSample ? Math.min(edgeEcho, 0.25) : edgeEcho;
    if (singleSource) rel[i] = Math.min(rel[i], 0.45);
    if (noSharpSample) relClass[i] = 2;
    else if (singleSource) relClass[i] = 1;
    else if (rel[i] >= opts.relHigh) relClass[i] = 0;
    else if (rel[i] >= opts.relLow) relClass[i] = 1;
    else relClass[i] = 2;
  }
  // Reliability statistics.
  let r0 = 0, r1 = 0, r2 = 0, r3 = 0;
  for (let i = 0; i < N; i++) {
    if (relClass[i] === 0) r0++;
    else if (relClass[i] === 1) r1++;
    else if (relClass[i] === 2) r2++;
    else r3++;
  }
  const stats = {
    reliable: r0 / N, marginal: r1 / N, unreliable: r2 / N, noData: r3 / N,
    meanReliability: (() => {
      let s = 0, c = 0;
      for (let i = 0; i < N; i++) if (covered[i]) { s += rel[i]; c++; }
      return c ? s / c : 0;
    })(),
  };
  // Edge width of the selection map: for each pixel on a strong guide edge
  // (with a weight change), measure the 10-90% transition width of the
  // dominant weight plane along the direction of maximum guide gradient.
  const gw = gradientMagnitude(fusedMeasure.luma, w, h);
  let gMax = 1e-9;
  for (let i = 0; i < N; i++) if (gw[i] > gMax) gMax = gw[i];
  const widths = [];
  const ghostSamples = [];
  for (let y = 3; y < h - 3; y++) {
    for (let x = 3; x < w - 3; x++) {
      const i = y * w + x;
      if (!covered[i]) continue;
      if (gw[i] < gMax * 0.35) continue;
      // Find the dominant weight index and a neighbouring rival.
      let dom = -1, domV = -1, rival = -1, rivalV = -1;
      for (let k = 0; k < n; k++) {
        if (skipMask[k]) continue;
        const v = normWeights[k][i];
        if (v > domV) { rivalV = domV; rival = dom; domV = v; dom = k; }
        else if (v > rivalV) { rivalV = v; rival = k; }
      }
      if (dom < 0 || rival < 0) continue;
      if (domV < 0.55 || domV - rivalV < 0.25) continue;
      // Walk along the gradient normal and find the two positions where the
      // dominant weight crosses 50%; their separation is the edge width.
      const gx = gxAt(fusedMeasure.luma, w, h, i);
      const gy = gyAt(fusedMeasure.luma, w, h, i);
      const len = Math.hypot(gx, gy) || 1;
      const ux = gx / len;
      const uy = gy / len;
      let a = -1, b = -1;
      const maxWalk = 24;
      for (let s = -maxWalk; s <= maxWalk; s++) {
        const sx = Math.round(x + ux * s);
        const sy = Math.round(y + uy * s);
        if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
        const si = sy * w + sx;
        if (normWeights[dom][si] >= 0.5 && a < 0) a = s;
        if (a >= 0 && b < 0 && normWeights[dom][si] < 0.5 && s > 0) b = s;
      }
      if (a >= 0 && b >= 0 && b > a) widths.push(b - a);
      // Ghosting probe: is the rival (blurred) slice gaining weight inside a
      // region where the dominant slice has a much stronger edge?
      if (rivalV > 0.12 && energies[dom][i] > 4 * Math.max(floor, 1e-5) &&
          energies[rival][i] < 0.5 * energies[dom][i]) {
        ghostSamples.push(rivalV - energies[rival][i] / Math.max(1e-9, energies[dom][i]));
      }
    }
  }
  widths.sort((a, b) => a - b);
  const median = widths.length ? widths[Math.floor(widths.length / 2)] : 0;
  const p90 = widths.length ? widths[Math.floor(widths.length * 0.9)] : 0;
  let ghostSum = 0;
  for (const g of ghostSamples) ghostSum += g;
  const ghosting = {
    samples: ghostSamples.length,
    index: ghostSamples.length ? ghostSum / ghostSamples.length : 0,
  };
  const problems = [];
  if (stats.unreliable > 0.02) {
    problems.push({
      code: 'all-blurred',
      message: `约 ${(stats.unreliable * 100).toFixed(1)}% 的像素在所有输入中都找不到可确认的清晰边缘，已标记为不可靠（不补细节）`,
    });
  }
  if (ghosting.index > 0.18) {
    problems.push({
      code: 'ghosting',
      message: `遮挡边缘附近滞留权重偏高（指数 ${ghosting.index.toFixed(2)}），可能产生重影`,
    });
  }
  // Weight-distribution anomaly: an enabled slice whose effective support
  // (source alpha ∩ registration support ∩ ROI, after border erosion) covers
  // almost none of the frame can carry no meaningful weight budget. We gate on
  // the eligible-pixel fraction rather than the in-region win rate: inside a
  // tiny surviving ROI the slice still "wins" its own few pixels, which would
  // hide the fact that it is effectively absent from the fused image.
  const activeSlices = [];
  for (let k = 0; k < n; k++) if (!skipMask[k]) activeSlices.push(k);
  for (const k of activeSlices) {
    let eligible = 0;
    let weightSum = 0;
    let won = 0;
    for (let i = 0; i < N; i++) {
      if (!veto[k][i]) continue;
      eligible++;
      weightSum += normWeights[k][i];
      let isWinner = true;
      for (const j of activeSlices) {
        if (j !== k && normWeights[j][i] > normWeights[k][i]) { isWinner = false; break; }
      }
      if (isWinner) won++;
    }
    const coverageFraction = eligible / N;
    const meanWeight = eligible ? weightSum / eligible : 0;
    if (coverageFraction < 0.005) {
      problems.push({
        code: 'weight-collapse',
        slice: k,
        message: `权重分布异常：第 ${k + 1} 片有效像素仅占画面 ${(coverageFraction * 100).toFixed(2)}%（${eligible}/${N}），平均权重 ${meanWeight.toFixed(3)}，几乎未参与融合`,
      });
    }
  }
  return {
    reliability: rel, relClass, fusedMeasure,
    edgeWidth: { median, p90, samples: widths.length, target: opts.edgeWidthTarget },
    ghosting,
    stats,
    problems,
    clipAtPixel,
  };
}


/**
 * Render the fused result to RGBA at 8 or 16 bit.
 */
export function renderFused(result, { bitDepth = 16 } = {}) {
  const { w, h, fused, covered } = result;
  const out = new Uint16Array(w * h * 4);
  const maxOut = bitDepth === 16 ? 65535 : 255;
  for (let i = 0; i < w * h; i++) {
    if (!covered[i]) continue;
    const r = linearToSrgb16(fused[i * 3]);
    const g = linearToSrgb16(fused[i * 3 + 1]);
    const b = linearToSrgb16(fused[i * 3 + 2]);
    if (bitDepth === 16) {
      out[i * 4] = r;
      out[i * 4 + 1] = g;
      out[i * 4 + 2] = b;
      out[i * 4 + 3] = 65535;
    } else {
      out[i * 4] = Math.round(r / 257);
      out[i * 4 + 1] = Math.round(g / 257);
      out[i * 4 + 2] = Math.round(b / 257);
      out[i * 4 + 3] = 255;
    }
  }
  return { width: w, height: h, data: out, bitDepth, max: maxOut };
}

/**
 * Preview at a reduced size: fused RGBA8 plus downsampled weight planes.
 */
export function computePreview(result, maxDim = DEFAULTS.previewMaxDim) {
  const { w, h } = result;
  const scale = Math.min(1, maxDim / Math.max(w, h));
  const dw = Math.max(1, Math.round(w * scale));
  const dh = Math.max(1, Math.round(h * scale));
  const sx = w / dw;
  const sy = h / dh;
  const rgb = new Float32Array(dw * dh * 3);
  const cov = new Uint8Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const y0 = Math.floor(y * sy);
    const y1 = Math.min(h, Math.max(y0 + 1, Math.ceil((y + 1) * sy)));
    for (let x = 0; x < dw; x++) {
      const x0 = Math.floor(x * sx);
      const x1 = Math.min(w, Math.max(x0 + 1, Math.ceil((x + 1) * sx)));
      let r = 0, g = 0, b = 0, c = 0, total = 0;
      for (let yy = y0; yy < y1; yy++) {
        for (let xx = x0; xx < x1; xx++) {
          const si = yy * w + xx;
          total++;
          if (!result.covered[si]) continue;
          c++;
          r += result.fused[si * 3];
          g += result.fused[si * 3 + 1];
          b += result.fused[si * 3 + 2];
        }
      }
      const di = y * dw + x;
      if (c > 0) {
        rgb[di * 3] = r / c;
        rgb[di * 3 + 1] = g / c;
        rgb[di * 3 + 2] = b / c;
        cov[di] = c / total > 0.4 ? 1 : 0;
      }
    }
  }
  const rgba = new Uint8Array(dw * dh * 4);
  for (let i = 0; i < dw * dh; i++) {
    if (!cov[i]) continue;
    rgba[i * 4] = Math.round(linearToSrgb16(rgb[i * 3]) / 257);
    rgba[i * 4 + 1] = Math.round(linearToSrgb16(rgb[i * 3 + 1]) / 257);
    rgba[i * 4 + 2] = Math.round(linearToSrgb16(rgb[i * 3 + 2]) / 257);
    rgba[i * 4 + 3] = 255;
  }
  const weights = result.weights.map((plane) => boxDown(plane, w, h, dw, dh, result.covered));
  const relClass = boxDownClass(result.relClass, w, h, dw, dh, result.covered);
  const reliability = boxDown(result.reliability, w, h, dw, dh, result.covered);
  return { scale, width: dw, height: dh, rgba, weights, relClass, reliability };
}

function boxDown(plane, w, h, dw, dh, covered) {
  const out = new Float32Array(dw * dh);
  const sx = w / dw;
  const sy = h / dh;
  for (let y = 0; y < dh; y++) {
    const y0 = Math.floor(y * sy);
    const y1 = Math.min(h, Math.max(y0 + 1, Math.ceil((y + 1) * sy)));
    for (let x = 0; x < dw; x++) {
      const x0 = Math.floor(x * sx);
      const x1 = Math.min(w, Math.max(x0 + 1, Math.ceil((x + 1) * sx)));
      let s = 0, c = 0;
      for (let yy = y0; yy < y1; yy++) {
        for (let xx = x0; xx < x1; xx++) {
          const si = yy * w + xx;
          if (covered[si]) { s += plane[si]; c++; }
        }
      }
      out[y * dw + x] = c ? s / c : 0;
    }
  }
  return out;
}

function boxDownClass(cls, w, h, dw, dh, covered) {
  const out = new Uint8Array(dw * dh);
  const sx = w / dw;
  const sy = h / dh;
  for (let y = 0; y < dh; y++) {
    const y0 = Math.floor(y * sy);
    const y1 = Math.min(h, Math.max(y0 + 1, Math.ceil((y + 1) * sy)));
    for (let x = 0; x < dw; x++) {
      const x0 = Math.floor(x * sx);
      const x1 = Math.min(w, Math.max(x0 + 1, Math.ceil((x + 1) * sx)));
      // Worst class in the block wins: warning areas are not hidden by
      // downsampling.
      let worst = 0;
      for (let yy = y0; yy < y1; yy++) {
        for (let xx = x0; xx < x1; xx++) {
          const si = yy * w + xx;
          if (!covered[si]) continue;
          const c = cls[si];
          const rank = c === 2 ? 3 : c === 1 ? 2 : c === 3 ? 4 : 1;
          if (rank > worst) worst = rank;
        }
      }
      out[y * dw + x] = worst === 3 ? 2 : worst === 2 ? 1 : worst === 4 ? 3 : 0;
    }
  }
  return out;
}

export const WEIGHT_PALETTE = [
  [230, 57, 70], [69, 123, 157], [233, 196, 106], [42, 157, 143],
  [144, 190, 109], [131, 56, 236], [251, 133, 0], [120, 120, 120],
];

/**
 * Colourise the selection map. mode: 'winner' tints by dominant slice,
 * 'weight' shows the selected slice's own weight as a heat ramp.
 */
export function encodeWeightMap(weights, w, h, mode = 'winner', selected = 0) {
  const n = weights.length;
  const out = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    let total = 0, dom = -1, domV = -1;
    for (let k = 0; k < n; k++) {
      total += weights[k][i];
      if (weights[k][i] > domV) { domV = weights[k][i]; dom = k; }
    }
    if (total <= 1e-9) { out[i * 4 + 3] = 0; continue; }
    if (mode === 'winner') {
      const col = WEIGHT_PALETTE[dom % WEIGHT_PALETTE.length];
      const t = 0.3 + 0.7 * domV;
      out[i * 4] = Math.round(col[0] * t);
      out[i * 4 + 1] = Math.round(col[1] * t);
      out[i * 4 + 2] = Math.round(col[2] * t);
    } else {
      const v = weights[selected][i];
      out[i * 4] = Math.round(255 * clamp(v * 1.4, 0, 1));
      out[i * 4 + 1] = Math.round(255 * clamp(v * 1.4, 0, 1) * 0.75);
      out[i * 4 + 2] = Math.round(255 * clamp(1 - v * 1.6, 0, 1));
    }
    out[i * 4 + 3] = 255;
  }
  return out;
}

/**
 * Reliability overlay RGBA8 (amber = marginal, red = unreliable, grey = no data).
 */
export function encodeReliability(result) {
  const { w, h } = result;
  const cls = result.relClass;
  const out = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    if (cls[i] === 0) continue;
    const col = cls[i] === 1 ? [232, 168, 56] : cls[i] === 2 ? [206, 64, 64] : [120, 120, 120];
    out[i * 4] = col[0];
    out[i * 4 + 1] = col[1];
    out[i * 4 + 2] = col[2];
    out[i * 4 + 3] = cls[i] === 3 ? 100 : 185;
  }
  return out;
}

export { clamp };
