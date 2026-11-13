// Image math helpers. All color blending is done in linear light.

const SRGB_TO_LINEAR = (() => {
  const t = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) {
    const v = i / 65535;
    t[i] = v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  }
  return t;
})();

const LINEAR_TO_SRGB = new Float32Array(65536);
{
  for (let i = 0; i < 65536; i++) {
    const v = i / 65535;
    LINEAR_TO_SRGB[i] = v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
  }
}

export function srgb16ToLinear(v) {
  return SRGB_TO_LINEAR[v & 0xffff];
}

export function linearToSrgb16(v) {
  if (!(v > 0)) return 0;
  if (v >= 1) return 65535;
  return Math.round(LINEAR_TO_SRGB[Math.min(65535, Math.round(v * 65535))] * 65535);
}

export function linearToSrgbFloat(v) {
  if (!(v > 0)) return 0;
  if (v >= 1) return 1;
  return LINEAR_TO_SRGB[Math.min(65535, Math.round(v * 65535))];
}

export const LUMA_R = 0.2126;
export const LUMA_G = 0.7152;
export const LUMA_B = 0.0722;

export function lumaOf(r, g, b) {
  return LUMA_R * r + LUMA_G * g + LUMA_B * b;
}

/**
 * Convert a decoded RGBA (16-bit nominal) image to linear-light planar RGB.
 * @returns {{rgb:Float32Array,w:number,h:number,luma:Float32Array,alpha:Uint8Array,clipped:number,median:number}}
 */
export function toLinear(rgba, w, h, { gain = 1, alphaThreshold = 0.08 } = {}) {
  const n = w * h;
  const rgb = new Float32Array(n * 3);
  const luma = new Float32Array(n);
  const alpha = new Uint8Array(n);
  const hist = new Uint32Array(2048);
  let clipped = 0;
  let counted = 0;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const a = rgba[p + 3];
    const alphaNorm = a / 65535;
    if (alphaNorm < alphaThreshold) continue;
    alpha[i] = 1;
    const r = srgb16ToLinear(rgba[p]) * gain;
    const g = srgb16ToLinear(rgba[p + 1]) * gain;
    const b = srgb16ToLinear(rgba[p + 2]) * gain;
    rgb[i * 3] = r;
    rgb[i * 3 + 1] = g;
    rgb[i * 3 + 2] = b;
    const L = lumaOf(r, g, b);
    luma[i] = L;
    if (rgba[p] >= 65534 || rgba[p + 1] >= 65534 || rgba[p + 2] >= 65534) clipped++;
    if (rgba[p] <= 1 && rgba[p + 1] <= 1 && rgba[p + 2] <= 1) clipped++;
    const bin = Math.min(2047, Math.max(0, Math.round(Math.sqrt(Math.max(0, L)) * 2047)));
    hist[bin]++;
    counted++;
  }
  let median = 0;
  if (counted > 0) {
    const target = counted / 2;
    let acc = 0;
    for (let i = 0; i < 2048; i++) {
      acc += hist[i];
      if (acc >= target) {
        median = i / 2047;
        median *= median;
        break;
      }
    }
  }
  return { rgb, w, h, luma, alpha, clipped, median };
}

export function linearMedian(rgba, w, h, alphaThreshold = 0.08) {
  const hist = new Uint32Array(2048);
  let counted = 0;
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    if (rgba[p + 3] / 65535 < alphaThreshold) continue;
    const L = lumaOf(srgb16ToLinear(rgba[p]), srgb16ToLinear(rgba[p + 1]), srgb16ToLinear(rgba[p + 2]));
    hist[Math.min(2047, Math.round(Math.sqrt(Math.max(0, L)) * 2047))]++;
    counted++;
  }
  if (!counted) return 0;
  let acc = 0;
  for (let i = 0; i < 2048; i++) {
    acc += hist[i];
    if (acc >= counted / 2) {
      const v = i / 2047;
      return v * v;
    }
  }
  return 0;
}

/**
 * Area-average downscale of planar linear RGB (alpha weighted) and validity mask.
 */
export function resizeLinear(src, sw, sh, dw, dh, srcValid) {
  const rgb = new Float32Array(dw * dh * 3);
  const valid = new Uint8Array(dw * dh);
  const luma = new Float32Array(dw * dh);
  const sx = sw / dw;
  const sy = sh / dh;
  for (let y = 0; y < dh; y++) {
    const y0 = y * sy;
    const y1 = Math.min(sh, (y + 1) * sy);
    for (let x = 0; x < dw; x++) {
      const x0 = x * sx;
      const x1 = Math.min(sw, (x + 1) * sx);
      let wr = 0, wg = 0, wb = 0, wValid = 0, wsum = 0;
      for (let yy = Math.floor(y0); yy < Math.ceil(y1); yy++) {
        if (yy < 0 || yy >= sh) continue;
        const fy = Math.min(y1, yy + 1) - Math.max(y0, yy);
        if (fy <= 0) continue;
        for (let xx = Math.floor(x0); xx < Math.ceil(x1); xx++) {
          if (xx < 0 || xx >= sw) continue;
          const fx = Math.min(x1, xx + 1) - Math.max(x0, xx);
          if (fx <= 0) continue;
          const si = yy * sw + xx;
          const weight = fx * fy;
          const va = srcValid ? srcValid[si] / 255 : 1;
          const f = weight * va;
          wr += src[si * 3] * f;
          wg += src[si * 3 + 1] * f;
          wb += src[si * 3 + 2] * f;
          wValid += f;
          wsum += weight;
        }
      }
      const di = y * dw + x;
      if (wValid > 0 && wsum > 0) {
        const inv = 1 / wValid;
        rgb[di * 3] = wr * inv;
        rgb[di * 3 + 1] = wg * inv;
        rgb[di * 3 + 2] = wb * inv;
        valid[di] = wValid / wsum > 0.5 ? 255 : 0;
        luma[di] = lumaOf(rgb[di * 3], rgb[di * 3 + 1], rgb[di * 3 + 2]) * (valid[di] ? 1 : 0);
      }
    }
  }
  return { rgb, valid, luma, w: dw, h: dh };
}

/**
 * Downscale a single-channel float plane (area average over valid samples).
 */
export function resizePlane(src, sw, sh, dw, dh, valid) {
  const out = new Float32Array(dw * dh);
  const outValid = new Uint8Array(dw * dh);
  const sx = sw / dw;
  const sy = sh / dh;
  for (let y = 0; y < dh; y++) {
    const y0 = Math.floor(y * sy);
    const y1 = Math.min(sh, Math.ceil((y + 1) * sy));
    for (let x = 0; x < dw; x++) {
      const x0 = Math.floor(x * sx);
      const x1 = Math.min(sw, Math.ceil((x + 1) * sx));
      let sum = 0, count = 0;
      for (let yy = y0; yy < y1; yy++) {
        for (let xx = x0; xx < x1; xx++) {
          const si = yy * sw + xx;
          if (valid && !valid[si]) continue;
          sum += src[si];
          count++;
        }
      }
      const di = y * dw + x;
      if (count) {
        out[di] = sum / count;
        outValid[di] = 255;
      }
    }
  }
  return { data: out, valid: outValid };
}

export function grayFromRGBA8(rgba, w, h) {
  const out = new Float32Array(w * h);
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    const L = lumaOf(rgba[p] / 255, rgba[p + 1] / 255, rgba[p + 2] / 255);
    out[i] = Math.pow(L, 1 / 2.2);
  }
  return out;
}
