// Per-pixel focus measures.
//
// Sharpness is the Laplacian-of-Gaussian energy over a Gaussian window. The
// raw energy is deliberately not normalised: after subtracting the noise floor
// it acts as a focus cue, while the *relative* claim ("this slice is really
// sharper than that one") is made separately with the normalised blur estimate
// computed in fuse.js. Keeping the two apart is what lets the tool report
// "all inputs are blurred here" instead of inventing a winner.

import { blurPlane, gradientMagnitude, fillHoles } from './filters.js';
import { lumaOf } from './image.js';

/**
 * Build a zero-mean/unity-gain Laplacian-of-Gaussian response of a plane.
 * The LoG is close to the scale where defocus manifests as contrast loss,
 * which makes it a better focus measure than raw gradient magnitude on
 * textured, non-edge content.
 */
function logResponse(plane, w, h, sigma) {
  const sm = blurPlane(plane, w, h, sigma);
  const lap = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const ym = y > 0 ? y - 1 : 0;
    const yp = y < h - 1 ? y + 1 : h - 1;
    for (let x = 0; x < w; x++) {
      const xm = x > 0 ? x - 1 : 0;
      const xp = x < w - 1 ? x + 1 : w - 1;
      lap[y * w + x] = 4 * sm[y * w + x] - sm[ym * w + x] - sm[yp * w + x] -
        sm[y * w + xm] - sm[y * w + xp];
    }
  }
  return lap;
}

/**
 * Compute a focus energy plane for one registered slice.
 *
 * @param {{rgb:Float32Array,w:number,h:number}} img
 * @param {Uint8Array} valid registered-validity mask
 * @param {number} sigma LoG scale in pixels
 * @returns {{energy:Float32Array, grad:Float32Array, localLuma:Float32Array}}
 */
export function focusMeasure(img, valid, sigma = 1.6) {
  const { w, h, rgb } = img;
  const n = w * h;
  const luma = new Float32Array(n);
  for (let i = 0; i < n; i++) luma[i] = lumaOf(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
  // Invalid pixels are filled by smooth extension of the valid interior rather
  // than zeroed: a hard zero edge at the mask boundary would be measured as
  // high-contrast structure, which is exactly the failure mode the reliability
  // map is supposed to catch elsewhere.
  const masked = fillHoles(luma, valid, w, h);
  const lo = logResponse(masked, w, h, sigma);
  // Energy over a window slightly larger than the LoG support.
  const win = Math.max(1.2, sigma * 1.9);
  const energy = new Float32Array(n);
  const sq = new Float32Array(n);
  for (let i = 0; i < n; i++) sq[i] = lo[i] * lo[i];
  const bl = blurPlane(sq, w, h, win);
  for (let i = 0; i < n; i++) energy[i] = Math.sqrt(bl[i]);
  const grad = gradientMagnitude(masked, w, h);
  const localLuma = blurPlane(masked, w, h, Math.max(2, win * 1.5));
  return { energy, grad, localLuma, luma, lo };
}

/**
 * Noise floor estimate from the flattest regions: the 5th percentile of
 * focus energy is dominated by sensor noise rather than blur differences.
 */
export function noiseFloor(energies, n) {
  const hist = new Uint32Array(4096);
  let count = 0;
  for (let i = 0; i < n; i++) {
    const v = energies[i];
    if (!Number.isFinite(v)) continue;
    const bin = Math.min(4095, Math.max(0, Math.round(Math.sqrt(Math.max(0, v)) * 4095)));
    hist[bin]++;
    count++;
  }
  if (!count) return 0;
  let acc = 0;
  const target = count * 0.05;
  for (let i = 0; i < 4096; i++) {
    acc += hist[i];
    if (acc >= target) {
      const v = i / 4095;
      return v * v;
    }
  }
  return 0;
}
