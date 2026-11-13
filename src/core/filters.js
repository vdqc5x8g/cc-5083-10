// Separable blur, gradients, and single-channel morphology.
// Borders use replication so results match at the border regardless of radius.

function gaussianKernel(sigma) {
  if (sigma <= 0) return { k: new Float32Array([1]), r: 0 };
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    k[i + r] = v;
    sum += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  return { k, r };
}

/**
 * Gaussian blur on an arbitrary single-channel plane.
 * @param {Float32Array} src
 */
export function blurPlane(src, w, h, sigma) {
  if (!(sigma > 0)) return new Float32Array(src);
  const { k, r } = gaussianKernel(sigma);
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = -r; i <= r; i++) {
        let xx = x + i;
        if (xx < 0) xx = 0;
        else if (xx >= w) xx = w - 1;
        s += src[row + xx] * k[i + r];
      }
      tmp[row + x] = s;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = -r; i <= r; i++) {
        let yy = y + i;
        if (yy < 0) yy = 0;
        else if (yy >= h) yy = h - 1;
        s += tmp[yy * w + x] * k[i + r];
      }
      out[y * w + x] = s;
    }
  }
  return out;
}

function upsamplePlane(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh);
  const sx = sw / dw;
  const sy = sh / dh;
  for (let y = 0; y < dh; y++) {
    const fy = Math.min(sh - 1, Math.max(0, (y + 0.5) * sy - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(sh - 1, y0 + 1);
    const ty = fy - y0;
    for (let x = 0; x < dw; x++) {
      const fx = Math.min(sw - 1, Math.max(0, (x + 0.5) * sx - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(sw - 1, x0 + 1);
      const tx = fx - x0;
      out[y * dw + x] = (src[y0 * sw + x0] * (1 - tx) + src[y0 * sw + x1] * tx) * (1 - ty) +
        (src[y1 * sw + x0] * (1 - tx) + src[y1 * sw + x1] * tx) * ty;
    }
  }
  return out;
}

/**
 * Fill pixels outside `valid` by a smooth extension of the valid interior
 * (push-pull / pull-push pyramid). Zeroing them instead would insert a step
 * edge at the mask boundary that the focus operator would mistake for real
 * structure; this keeps the extension smooth so only genuine content is
 * measured.
 */
export function fillHoles(plane, valid, w, h) {
  const levels = [];
  let cur = {
    v: Float32Array.from(plane), wt: new Float32Array(w * h), w, h,
  };
  for (let i = 0; i < w * h; i++) cur.wt[i] = valid[i] ? 1 : 0;
  levels.push(cur);
  while (cur.w > 1 || cur.h > 1) {
    const nw = Math.max(1, Math.ceil(cur.w / 2));
    const nh = Math.max(1, Math.ceil(cur.h / 2));
    const P = new Float32Array(nw * nh);
    const W = new Float32Array(nw * nh);
    for (let y = 0; y < cur.h; y++) {
      const ty = y >> 1;
      for (let x = 0; x < cur.w; x++) {
        const i = y * cur.w + x;
        const wgt = cur.wt[i];
        if (wgt <= 0) continue;
        const j = ty * nw + (x >> 1);
        P[j] += cur.v[i] * wgt;
        W[j] += wgt;
      }
    }
    for (let j = 0; j < nw * nh; j++) if (W[j] > 0) P[j] /= W[j];
    cur = { v: P, wt: W, w: nw, h: nh };
    levels.push(cur);
  }
  // Coarsest level: any empty cell borrows the global valid mean.
  const top = levels[levels.length - 1];
  let sum = 0, cnt = 0;
  for (let i = 0; i < top.v.length; i++) if (top.wt[i] > 0) { sum += top.v[i] * top.wt[i]; cnt += top.wt[i]; }
  const mean = cnt > 0 ? sum / cnt : 0;
  for (let i = 0; i < top.v.length; i++) if (top.wt[i] <= 0) top.v[i] = mean;
  let up = top.v;
  let uw = top.w;
  let uh = top.h;
  for (let l = levels.length - 2; l >= 0; l--) {
    const lv = levels[l];
    const fine = upsamplePlane(up, uw, uh, lv.w, lv.h);
    const out = new Float32Array(lv.w * lv.h);
    for (let i = 0; i < out.length; i++) out[i] = lv.wt[i] > 0 ? lv.v[i] : fine[i];
    up = out;
    uw = lv.w;
    uh = lv.h;
  }
  return up;
}

/**
 * 3x3 Sobel gradient magnitude.
 */
export function gradientMagnitude(src, w, h) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const ym = y > 0 ? y - 1 : 0;
    const yp = y < h - 1 ? y + 1 : h - 1;
    for (let x = 0; x < w; x++) {
      const xm = x > 0 ? x - 1 : 0;
      const xp = x < w - 1 ? x + 1 : w - 1;
      const tl = src[ym * w + xm], tc = src[ym * w + x], tr = src[ym * w + xp];
      const ml = src[y * w + xm], mr = src[y * w + xp];
      const bl = src[yp * w + xm], bc = src[yp * w + x], br = src[yp * w + xp];
      const gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);
      const gy = (bl + 2 * bc + br) - (tl + 2 * tc + tr);
      out[y * w + x] = Math.hypot(gx, gy) * 0.25;
    }
  }
  return out;
}

export function growPlane(src, w, h, radius) {
  if (radius <= 0) return new Float32Array(src);
  let cur = src;
  const steps = Math.max(1, Math.round(radius));
  for (let r = 0; r < steps; r++) {
    const out = new Float32Array(src.length);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let best = cur[y * w + x];
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= w) continue;
            const v = cur[yy * w + xx];
            if (v > best) best = v;
          }
        }
        out[y * w + x] = best;
      }
    }
    cur = out;
  }
  return cur;
}

export function erodePlane(src, w, h, radius) {
  if (radius <= 0) return new Float32Array(src);
  let cur = src;
  const out = new Float32Array(src.length);
  for (let r = 0; r < Math.max(1, Math.round(radius)); r++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let best = cur[y * w + x];
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= w) continue;
            const v = cur[yy * w + xx];
            if (v < best) best = v;
          }
        }
        out[y * w + x] = best;
      }
    }
    cur = out;
  }
  return cur;
}

/**
 * Normalized box (moving-average) filter over a plane, replicating borders.
 * Used by the guided filter as the mean operator.
 */
export function boxMean(src, w, h, radius) {
  const r = Math.max(0, Math.round(radius));
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  const denom = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    let acc = 0;
    const row = y * w;
    for (let i = -r; i <= r; i++) acc += src[row + (i < 0 ? 0 : i >= w ? w - 1 : i)];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = acc / denom;
      const add = x + r + 1;
      const sub = x - r;
      acc += src[row + (add >= w ? w - 1 : add)] - src[row + (sub < 0 ? 0 : sub)];
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let i = -r; i <= r; i++) acc += tmp[(i < 0 ? 0 : i >= h ? h - 1 : i) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = acc / denom;
      const add = y + r + 1;
      const sub = y - r;
      acc += tmp[(add >= h ? h - 1 : add) * w + x] - tmp[(sub < 0 ? 0 : sub) * w + x];
    }
  }
  return out;
}

export function boxMean2(src, w, h, radius) {
  const r = Math.max(0, Math.round(radius));
  const m = boxMean(src, w, h, r);
  return boxMean(m, w, h, r);
}

/**
 * Guided filter (He) on a single-channel guide and input. Edge-preserving
 * smoothing whose "detail" is the difference from the base layer.
 */
export function guidedFilter(guide, input, w, h, radius, eps) {
  const meanI = boxMean(guide, w, h, radius);
  const meanP = boxMean(input, w, h, radius);
  const ii = new Float32Array(w * h);
  const ip = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    ii[i] = guide[i] * guide[i];
    ip[i] = guide[i] * input[i];
  }
  const meanII = boxMean(ii, w, h, radius);
  const meanIP = boxMean(ip, w, h, radius);
  const a = new Float32Array(w * h);
  const bb = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const varI = Math.max(0, meanII[i] - meanI[i] * meanI[i]);
    const cov = meanIP[i] - meanI[i] * meanP[i];
    const ai = cov / (varI + eps);
    a[i] = ai;
    bb[i] = meanP[i] - ai * meanI[i];
  }
  const meanA = boxMean(a, w, h, radius);
  const meanB = boxMean(bb, w, h, radius);
  const out = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = meanA[i] * guide[i] + meanB[i];
  return out;
}
