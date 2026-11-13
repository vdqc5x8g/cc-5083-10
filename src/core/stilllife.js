// Self-made still-life scene for validation.
//
// This is a second scene, distinct from the controlled test chart in synth.js:
// a tabletop arrangement with real-looking materials (draped cloth, wood,
// glazed ceramic, fruit, a printed card) at a spread of depths. It is built
// from analytic surfaces so a per-pixel depth (and therefore a physically
// correct circle of confusion) is available, while the textures are procedural
// so nothing is downloaded. The stack is then rendered through the same
// thin-lens model as the synthetic chart (renderSlice in synth.js), including
// focus breathing, sub-pixel drift and sensor noise.
//
// labels: 0 backdrop cloth, 1 cup, 2 lemon, 3 book stack, 4 printed card,
//         5 herb sprig (closest), 6 tabletop.

import { renderSlice } from './synth.js';

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
  let v = 0;
  let amp = 0.5;
  let freq = 1;
  let norm = 0;
  for (let o = 0; o < octaves; o++) {
    v += valueNoise(x * freq, y * freq) * amp;
    norm += amp;
    amp *= 0.5;
    freq *= 2.03;
  }
  return v / norm;
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function smoothstep(a, b, x) {
  const t = clamp01((x - a) / (b - a));
  return t * t * (3 - 2 * t);
}

/**
 * Build the still-life scene. Coordinates u,v are [0,1] across the frame and
 * the plane aspect is honoured (u spans [0, aspect]).
 */
export function buildStillLife(w, h) {
  const aspect = w / h;
  const rgb = new Float32Array(w * h * 3);
  const depth = new Float32Array(w * h);
  const labels = new Uint8Array(w * h);

  // Objects, in painting order (front to back for occlusion). Each entry is an
  // analytic silhouette in the u,v plane; the closest surface wins.
  const card = { x0: 0.60 * aspect, x1: 0.94 * aspect, y0: 0.10, y1: 0.52, depth: 0.34, tilt: 0.03 };
  const bookTop = { x0: 0.30 * aspect, x1: 0.66 * aspect, y0: 0.42, y1: 0.56, depth: 0.60 };
  const bookBot = { x0: 0.27 * aspect, x1: 0.70 * aspect, y0: 0.56, y1: 0.72, depth: 0.66 };
  const cup = { cx: 0.13 * aspect, cy: 0.62, rx: 0.095 * aspect, ry: 0.16, depth: 0.70 };
  const lemon = { cx: 0.42 * aspect, cy: 0.78, r: 0.075 * aspect, depth: 0.48 };
  const sprig = { cx: 0.78 * aspect, cy: 0.83, r: 0.045 * aspect, depth: 0.26 };

  for (let y = 0; y < h; y++) {
    const v = y / (h - 1);
    for (let x = 0; x < w; x++) {
      const u = (x / (w - 1)) * aspect;
      const i = y * w + x;
      let dep = 1.0;
      let label = 0;
      let r = 0;
      let g = 0;
      let b = 0;

      // Draped cloth backdrop: folds modulate both texture and shading. Depth
      // stays on the backdrop plane; the folds are shading, not geometry.
      {
        const fold = Math.sin(u * 13.0 + 0.6) * 0.5 + Math.sin(u * 5.3 + 1.7) * 0.5;
        const weave = fbm(u * 150, v * 150, 4);
        const base = 0.40 + 0.10 * Math.sin(v * 9 + fold * 0.7) + 0.05 * fold;
        const tint = 0.55 + 0.25 * fold;
        r = (base + 0.03 * weave) * tint;
        g = (base + 0.02 * weave) * (tint * 0.92 + 0.05);
        b = base * (0.55 + 0.18 * fold) + 0.04 * weave;
      }

      // Tabletop in the lower third: wood grain, slightly closer than cloth.
      if (v > 0.68) {
        const t = smoothstep(0.68, 0.70, v);
        dep = dep * (1 - t) + 0.86 * t;
        const grain = fbm(u * 24 + 3, v * 90, 4);
        const ring = Math.sin(u * 40 + grain * 4) * 0.5 + 0.5;
        const wr = 0.32 + 0.20 * ring * (0.6 + 0.4 * grain);
        const wg = 0.19 + 0.12 * ring * (0.6 + 0.4 * grain);
        const wb = 0.10 + 0.06 * ring;
        r = r * (1 - t) + wr * t;
        g = g * (1 - t) + wg * t;
        b = b * (1 - t) + wb * t;
        if (t > 0.5) label = 6;
      }

      // Printed card: standing, thin, high-contrast type — the hardest
      // structure to superpose and the best occlusion-edge test.
      if (u > card.x0 && u < card.x1 && v > card.y0 && v < card.y1) {
        const edgeV = smoothstep(card.y0, card.y0 + 0.01, v) * smoothstep(card.y1, card.y1 - 0.01, v);
        const edgeU = smoothstep(card.x0, card.x0 + 0.008, u) * smoothstep(card.x1, card.x1 - 0.008, u);
        if (edgeU * edgeV > 0.5) {
          dep = card.depth;
          label = 4;
          const ax = (u - card.x0) / (card.x1 - card.x0);
          const ay = (v - card.y0) / (card.y1 - card.y0);
          const paper = 0.78 + 0.06 * fbm(u * 120, v * 120, 3);
          // Horizontal rules plus a block of "text" marks.
          const rule = Math.abs(ay - 0.22) < 0.012 || Math.abs(ay - 0.34) < 0.012;
          const mark = ay > 0.40 && ay < 0.92 && ax > 0.08 && ax < 0.92 && fbm(ax * 90 + 4, ay * 30, 3) > 0.52;
          const ink = rule || mark ? 0.12 : 0;
          const shade = 1 - 0.18 * Math.max(0, ax - 0.2);
          r = (paper * (1 - ink) + ink * 0.1) * shade;
          g = (paper * (1 - ink) + ink * 0.1) * shade * 0.97;
          b = (paper * (1 - ink) + ink * 0.1) * shade * 0.90;
        }
      }

      // Book stack: two slabs, cloth-bound with a page block.
      const inBook = (bk, dd) => u > bk.x0 && u < bk.x1 && v > bk.y0 && v < bk.y1;
      if (inBook(bookBot, bookBot.depth)) {
        dep = bookBot.depth;
        label = 3;
        const ax = (u - bookBot.x0) / (bookBot.x1 - bookBot.x0);
        const ay = (v - bookBot.y0) / (bookBot.y1 - bookBot.y0);
        const pages = ay > 0.72 ? 0.86 + 0.06 * fbm(ax * 60, ay * 80, 3) : 0;
        const cover = (0.16 + 0.05 * fbm(u * 60, v * 60, 3)) * (1 - 0.3 * ax);
        if (pages > 0) {
          r = pages * 0.95; g = pages * 0.90; b = pages * 0.78;
        } else {
          r = cover * 1.1; g = cover * 0.45; b = cover * 0.35;
        }
      }
      if (inBook(bookTop, bookTop.depth)) {
        dep = bookTop.depth;
        label = 3;
        const ax = (u - bookTop.x0) / (bookTop.x1 - bookTop.x0);
        const ay = (v - bookTop.y0) / (bookTop.y1 - bookTop.y0);
        const band = Math.abs(ay - 0.5) < 0.16 ? 0.85 : 0.30;
        r = band * (0.30 + 0.1 * fbm(ax * 40, ay * 40, 3));
        g = band * 0.55;
        b = band * 0.40;
      }

      // Ceramic cup (elliptical silhouette). Depth increases towards the rim
      // to model the cylindrical body; the rim ellipse gives the handle-free
      // silhouette. Glaze highlight.
      {
        const dx = (u - cup.cx) / cup.rx;
        const dy = (v - cup.cy) / cup.ry;
        const d2 = dx * dx + dy * dy;
        if (d2 < 1) {
          const rad = Math.sqrt(d2);
          dep = cup.depth + 0.045 * rad * rad;
          label = 1;
          const inside = v < cup.cy + cup.ry * 0.02 && y > 0.06 * h;
          const glaze = 0.72 + 0.12 * fbm(u * 70, v * 70, 3);
          const rim = Math.exp(-((rad - 0.52) ** 2) / 0.002) * 0.8;
          const spec = Math.exp(-((dx + 0.25) ** 2 + (dy + 0.30) ** 2) / 0.02);
          if (inside && rad < 0.62) {
            // Interior: dark tea surface with a bright rim.
            const tea = 0.10 + 0.03 * fbm(u * 90, v * 90, 2);
            r = tea * 0.9 + spec * 0.3; g = tea * 0.55 + spec * 0.3; b = tea * 0.25 + spec * 0.25;
            r = r + rim * 0.6; g = g + rim * 0.55; b = b + rim * 0.5;
          } else {
            r = glaze * 0.92 + spec * 0.9 + rim;
            g = glaze * 0.88 + spec * 0.85 + rim;
            b = glaze * 0.80 + spec * 0.7 + rim;
          }
        }
      }

      // Lemon: near-spherical, textured peel, strong shading. Closest after
      // the herb sprig.
      {
        const dx = (u - lemon.cx) / lemon.r;
        const dy = (v - lemon.cy) / lemon.r;
        const d2 = dx * dx + dy * dy;
        if (d2 < 1) {
          const nz = Math.sqrt(Math.max(0, 1 - d2));
          dep = lemon.depth - 0.10 * nz;
          label = 2;
          const peel = 0.85 + 0.15 * fbm(u * 220, v * 220, 3);
          const shade = 0.35 + 0.65 * Math.max(0, 0.35 + 0.65 * (-dx * 0.5 - dy * 0.5 + nz));
          r = (0.92 + 0.08 * peel) * shade;
          g = (0.80 + 0.06 * peel) * shade;
          b = 0.10 * shade;
          // Speckles.
          const spot = hash2(Math.floor(u * 200), Math.floor(v * 200));
          if (spot > 0.86) { r *= 0.8; g *= 0.72; }
        }
      }

      // Herb sprig: small, closest object, fine leaves — sets the front focus.
      {
        const dx = (u - sprig.cx) / sprig.r;
        const dy = (v - sprig.cy) / sprig.r;
        const d2 = dx * dx + dy * dy;
        if (d2 < 1.35) {
          const ang = Math.atan2(dy, dx);
          const rad = Math.sqrt(d2);
          const leaf = Math.sin(ang * 11 + rad * 5) > 0.1 ? 1 : 0;
          if (leaf && rad < 1.3) {
            dep = sprig.depth;
            label = 5;
            const leafTex = 0.55 + 0.25 * fbm(u * 260, v * 260, 3);
            const vein = Math.abs(Math.sin(ang * 11 + rad * 5)) < 0.06 ? 0.7 : 0;
            r = leafTex * 0.32 + vein * 0.3;
            g = leafTex * 0.75 + vein * 0.4;
            b = leafTex * 0.26;
          }
        }
      }

      rgb[i * 3] = Math.max(0, Math.min(1, r));
      rgb[i * 3 + 1] = Math.max(0, Math.min(1, g));
      rgb[i * 3 + 2] = Math.max(0, Math.min(1, b));
      depth[i] = dep;
      labels[i] = label;
    }
  }
  return { rgb, depth, labels, w, h, aspect };
}

/**
 * Focus sweep for the still-life arrangement. The stack is sampled at the
 * object depths the layout actually contains (near sprig, card, lemon, book
 * stack, cup, cloth) so every object has a slice focused on it. When
 * `includeBackdrop` is false the farthest slices are dropped, which leaves the
 * cloth out of focus everywhere — the demo's "unreliable far region" case.
 */
export function stillLifeFocuses(count = 8, { includeBackdrop = true } = {}) {
  const all = [0.26, 0.34, 0.42, 0.52, 0.62, 0.74, 0.88, 1.0];
  const usable = includeBackdrop ? all : all.filter((d) => d < 0.8);
  const out = [];
  for (let k = 0; k < count; k++) {
    const pos = count === 1 ? 0 : (k * (usable.length - 1)) / (count - 1);
    const i0 = Math.floor(pos);
    const i1 = Math.min(usable.length - 1, i0 + 1);
    const t = pos - i0;
    // Interpolate in dioptres so focus spacing matches lens behaviour.
    const inv = 1 / usable[i0] + (1 / usable[i1] - 1 / usable[i0]) * t;
    out.push(Number((1 / inv).toFixed(4)));
  }
  return out;
}

/**
 * Render the whole still-life stack, applying focus breathing (magnification
 * grows as the focus moves closer, i.e. as `focus` decreases) and sub-pixel
 * drift from the focus mechanism. Returns rendered slices plus the scene so
 * tests can check the result against known depth.
 */
export function renderStillLifeStack(w, h, count = 6, opts = {}) {
  const scene = buildStillLife(w, h);
  const {
    aperture = 0.55, breathing = 0.005, noise = 0.004, drift = 0.9,
    includeBackdrop = true,
  } = opts;
  const focuses = stillLifeFocuses(count, { includeBackdrop });
  const slices = focuses.map((focus, k) => {
    const phase = k / Math.max(1, count - 1);
    const rendered = renderSlice(scene, {
      focus,
      aperture,
      breathing,
      noise,
      shift: { x: (phase - 0.5) * 2 * drift, y: (phase - 0.3) * 1.1 },
    });
    return {
      name: `still-${String(k + 1).padStart(2, '0')}`,
      focus,
      ...rendered,
    };
  });
  return { scene, focuses, slices };
}
