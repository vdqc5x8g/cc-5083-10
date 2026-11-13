// Synthetic focal stack with a controllable depth of field.
//
// The scene is built from explicit geometry (a "plane" texture, a textured
// sphere in front of it, and a foreground occluder bar) and a per-pixel depth.
// A focus slice at focus distance d applies a circle-of-confusion blur whose
// radius follows the thin-lens model around d, plus optional focus breathing
// (magnification changing with d), sensor noise, and a global 1/4-pixel shift.
//
// The ground truth used by the tests is NOT the fusion decision: it is the
// scene's own blur-free textures, so an independent pixel comparison and
// selection-map comparison are possible.

import { blurPlane } from './filters.js';

function hash2(x, y) {
  const v = Math.sin(x * 127.1 + y * 311.7) * 43758.5453123;
  return v - Math.floor(v);
}

function valueNoise(x, y) {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const fx = x - xi;
  const fy = y - yi;
  const sx = fx * fx * (3 - 2 * fx);
  const sy = fy * fy * (3 - 2 * fy);
  const a = hash2(xi, yi);
  const b = hash2(xi + 1, yi);
  const c = hash2(xi, yi + 1);
  const d = hash2(xi + 1, yi + 1);
  return a * (1 - sx) * (1 - sy) + b * sx * (1 - sy) + c * (1 - sx) * sy + d * sx * sy;
}

function fbm(x, y, octaves = 5) {
  let v = 0, amp = 0.5, freq = 1, norm = 0;
  for (let o = 0; o < octaves; o++) {
    v += valueNoise(x * freq, y * freq) * amp;
    norm += amp;
    amp *= 0.5;
    freq *= 2.03;
  }
  return v / norm;
}

/**
 * Build a scene: linear-light RGB texture, depth in "world" units (smaller is
 * closer), plus per-object masks and the sharp target image.
 * Coordinate system: u,v in [0,1] across the frame.
 */
export function buildScene(w, h, variant = 'standard') {
  const rgb = new Float32Array(w * h * 3);
  const depth = new Float32Array(w * h);
  // 0 backdrop, 1 sphere, 2 bar, 3 near object that is out of range for the
  // whole stack (the "every input is blurred here" test case).
  const labels = new Uint8Array(w * h);
  const aspect = w / h;
  const sphere = {
    cx: 0.60 * aspect,
    cy: 0.56,
    r: 0.21 * aspect,
    depth: 0.62,
  };
  const bar = { x0: 0.04 * aspect, x1: 0.30 * aspect, y0: 0.14, y1: 0.34, depth: 0.46 };
  const near = {
    cx: 0.16 * aspect, cy: 0.80, r: 0.15 * aspect, depth: 0.20,
  };
  for (let y = 0; y < h; y++) {
    const v = y / (h - 1);
    for (let x = 0; x < w; x++) {
      const u = (x / (w - 1)) * aspect;
      const i = y * w + x;
      let label = 0;
      let dep = 1.0;
      // Foreground bar (textured, high contrast) in front of everything.
      if (u >= bar.x0 && u <= bar.x1 && v >= bar.y0 && v <= bar.y1) {
        label = 2;
        dep = bar.depth;
      }
      // Sphere: front surface of a sphere at its centre.
      const dx = u - sphere.cx;
      const dy = v - sphere.cy;
      if (label === 0 && dx * dx + dy * dy < sphere.r * sphere.r) {
        label = 1;
        dep = sphere.depth;
      }
      const ndx = u - near.cx;
      const ndy = v - near.cy;
      if (label === 0 && ndx * ndx + ndy * ndy < near.r * near.r) {
        label = 3;
        dep = near.depth;
      }
      labels[i] = label;
      depth[i] = dep;
      // Textures. Backdrop: paper/fabric-like fbm with a printed grid; the
      // grid gives thin high-contrast lines that are hard to superpose.
      let r, g, b;
      if (label === 2) {
        // Painted wood: strong stripes plus grain.
        const stripe = Math.sin(u * 90) > 0.25 ? 0.75 : 0.12;
        const grain = fbm(u * 26, v * 26, 4);
        r = stripe * (0.8 + 0.2 * grain) * 0.95;
        g = stripe * (0.55 + 0.2 * grain) * (0.75 + 0.25 * grain);
        b = stripe * 0.25 * (0.7 + 0.3 * grain);
      } else if (label === 3) {
        // Near object: fine text-like print, the finest structure in the scene.
        const ang = Math.atan2(ndy, ndx);
        const rad = Math.sqrt(ndx * ndx + ndy * ndy) / near.r;
        const ink = fbm(u * 170, v * 170, 4) > 0.5 ? 0.1 : 0.85;
        const ring = Math.sin(rad * 34) > 0.4 ? 0.25 : 0;
        r = ink * (0.95 - ring) * (0.9 + 0.1 * Math.cos(ang));
        g = ink * (0.80 - ring * 0.8);
        b = ink * (0.62 - ring * 0.6);
      } else if (label === 1) {
        // Sphere: coloured map with a strong meridional pattern and glare.
        const ang = Math.atan2(dy, dx);
        const lat = Math.acos(Math.min(1, Math.sqrt(dx * dx + dy * dy) / sphere.r));
        const bands = Math.sin(lat * 12) * 0.5 + 0.5;
        const merid = Math.sin(ang * 7 + 0.7) * 0.5 + 0.5;
        const tex = 0.35 + 0.65 * (0.45 * bands + 0.55 * merid);
        const spec = Math.exp(-((dx + 0.03) ** 2 + (dy + 0.05) ** 2) / (0.0035));
        r = tex * 0.25 + spec * 0.9;
        g = tex * 0.55 + spec * 0.85;
        b = tex * 0.95 + spec * 0.7;
      } else {
        // Backdrop: off-white paper with a fine printed grid.
        const paper = 0.55 + 0.18 * fbm(u * 34, v * 34, 5);
        const gridX = Math.abs(((u * 30) % 1) + 1) % 1;
        const gridY = Math.abs(((v * 30) % 1) + 1) % 1;
        const line = (Math.min(gridX, 1 - gridX) < 0.02 || Math.min(gridY, 1 - gridY) < 0.02) ? 0.16 : 0;
        const text = fbm(u * 90 + 10, v * 26 + 4, 3);
        r = paper * 0.98 - line * 0.4 + 0.03 * text;
        g = paper * 0.94 - line * 0.4 + 0.02 * text;
        b = paper * 0.88 - line * 0.42 - 0.04 * text;
      }
      rgb[i * 3] = Math.max(0, r);
      rgb[i * 3 + 1] = Math.max(0, g);
      rgb[i * 3 + 2] = Math.max(0, b);
    }
  }
  return { rgb, depth, labels, w, h, sphere, bar, near, aspect };
}

/**
 * Blur the scene by a circle of confusion. Aperture controls the CoC slope;
 * `focus` is the focus distance in the same units as depth. Points at `focus`
 * are sharp; points at `far` are blurred by coc(far).
 */
export function blurScene(scene, { focus, aperture = 0.9, maxRadius = 12 }) {
  const { w, h, rgb, depth } = scene;
  const radii = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const coc = aperture * Math.abs(1 / depth[i] - 1 / focus);
    radii[i] = Math.min(maxRadius, coc * w * 0.075);
  }
  // Spatially varying blur: piecewise-constant groups would show banding, so
  // blend neighbouring layers by their radius weight. Group radii in 1/4 px
  // steps and blend the four surrounding layers.
  const out = new Float32Array(w * h * 3);
  const step = 0.5;
  // De-interleave the channels before blurring.
  const pr = new Float32Array(w * h);
  const pg = new Float32Array(w * h);
  const pb = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    pr[i] = rgb[i * 3];
    pg[i] = rgb[i * 3 + 1];
    pb[i] = rgb[i * 3 + 2];
  }
  const layers = [];
  for (let r = 0; r <= maxRadius; r += step) {
    const sigma = Math.max(0.35, r * 0.55);
    const bl = blurPlane(pr, w, h, sigma);
    const bg = blurPlane(pg, w, h, sigma);
    const bb = blurPlane(pb, w, h, sigma);
    layers.push({ r, bl, bg, bb });
  }
  for (let i = 0; i < w * h; i++) {
    const r = radii[i];
    const pos = r / step;
    const i0 = Math.min(layers.length - 1, Math.floor(pos));
    const i1 = Math.min(layers.length - 1, i0 + 1);
    const t = Math.min(1, Math.max(0, pos - i0));
    const L0 = layers[i0];
    const L1 = layers[i1];
    out[i * 3] = L0.bl[i] * (1 - t) + L1.bl[i] * t;
    out[i * 3 + 1] = L0.bg[i] * (1 - t) + L1.bg[i] * t;
    out[i * 3 + 2] = L0.bb[i] * (1 - t) + L1.bb[i] * t;
  }
  return { rgb: out, radii, w, h };
}

/**
 * Render one focus slice: blur, optional focus-breathing magnification,
 * optional global translation, optional noise. Returns RGBA Uint16 plus the
 * effective geometry used.
 */
export function renderSlice(scene, options) {
  const { w, h } = scene;
  const { focus, aperture = 0.9, breathing = 0.0, shift = { x: 0, y: 0 },
    noise = 0.0, exposure = 1.0, maxRadius = 12, rotationDeg = 0 } = options;
  const blurred = blurScene(scene, { focus, aperture, maxRadius });
  const rgba = new Uint16Array(w * h * 4);
  // Geometry: magnification about the image centre from focus breathing, plus
  // a rotation, then a translation. Forward map of a scene point s (relative
  // to the centre):
  //   p = m * R(rot) * (s - c) + c + shift + d(p)
  // Small lens distortion is omitted; the similarity model matches this.
  const m = 1 + breathing * (1 - focus);
  const th = rotationDeg * Math.PI / 180;
  const cos = Math.cos(th) * m;
  const sin = Math.sin(th) * m;
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  const shiftX = shift.x;
  const shiftY = shift.y;
  const det = cos * cos + sin * sin;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const d = (y * w + x) * 4;
      // Backward map: find the source pixel for destination (x,y).
      const rx = x - cx - shiftX;
      const ry = y - cy - shiftY;
      const sx = (cos * rx + sin * ry) / det + cx;
      const sy = (-sin * rx + cos * ry) / det + cy;
      if (sx < 0 || sy < 0 || sx > w - 1 || sy > h - 1) {
        rgba[d + 3] = 0;
        continue;
      }
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const x1 = Math.min(w - 1, x0 + 1);
      const y1 = Math.min(h - 1, y0 + 1);
      const fx = sx - x0;
      const fy = sy - y0;
      const sample = (px, py) => {
        const i = py * w + px;
        return [blurred.rgb[i * 3], blurred.rgb[i * 3 + 1], blurred.rgb[i * 3 + 2]];
      };
      const p00 = sample(x0, y0);
      const p10 = sample(x1, y0);
      const p01 = sample(x0, y1);
      const p11 = sample(x1, y1);
      const rr = (p00[0] * (1 - fx) + p10[0] * fx) * (1 - fy) + (p01[0] * (1 - fx) + p11[0] * fx) * fy;
      const gg = (p00[1] * (1 - fx) + p10[1] * fx) * (1 - fy) + (p01[1] * (1 - fx) + p11[1] * fx) * fy;
      const bb = (p00[2] * (1 - fx) + p10[2] * fx) * (1 - fy) + (p01[2] * (1 - fx) + p11[2] * fx) * fy;
      rgba[d] = toSrgb16(rr * exposure + (noise ? (hash2(x * 3.1, y * 5.7) - 0.5) * noise : 0));
      rgba[d + 1] = toSrgb16(gg * exposure + (noise ? (hash2(x * 7.3, y * 2.9) - 0.5) * noise : 0));
      rgba[d + 2] = toSrgb16(bb * exposure + (noise ? (hash2(x * 5.1, y * 9.3) - 0.5) * noise : 0));
      rgba[d + 3] = 65535;
    }
  }
  return {
    rgba, width: w, height: h,
    focus, aperture,
    magnification: m, rotationDeg, shift,
  };
}

const TO_SRGB = (() => {
  const t = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) {
    const v = i / 65535;
    t[i] = v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
  }
  return t;
})();

function toSrgb16(linear) {
  const v = Math.max(0, Math.min(1, linear));
  const idx = Math.min(65535, Math.round(v * 65535));
  return Math.round(TO_SRGB[idx] * 65535);
}

/**
 * Ground-truth "sharp scene" rendered in the reference geometry for tests.
 */
export function renderSharp(scene, { w, h } = scene) {
  const rgba = new Uint16Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    rgba[i * 4] = toSrgb16(scene.rgb[i * 3]);
    rgba[i * 4 + 1] = toSrgb16(scene.rgb[i * 3 + 1]);
    rgba[i * 4 + 2] = toSrgb16(scene.rgb[i * 3 + 2]);
    rgba[i * 4 + 3] = 65535;
  }
  return { rgba, width: w, height: h };
}
