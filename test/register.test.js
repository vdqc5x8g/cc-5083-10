import test from 'node:test';
import assert from 'node:assert/strict';
import { registerSimilarity, sampleRegistered } from '../src/core/register.js';
import { buildScene } from '../src/core/synth.js';
import { blurPlane } from '../src/core/filters.js';

const W = 240;
const H = 180;

function texturedReference() {
  const scene = buildScene(W, H);
  // Mild blur removes hard aliasing that would bias the NCC.
  const rgb = new Float32Array(W * H * 3);
  for (let c = 0; c < 3; c++) {
    const p = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) p[i] = scene.rgb[i * 3 + c];
    const bp = blurPlane(p, W, H, 1.2);
    for (let i = 0; i < W * H; i++) rgb[i * 3 + c] = bp[i];
  }
  return { rgb, w: W, h: H };
}

/** Apply a known similarity (ref pixel -> source pixel) with bilinear sampling. */
function warp(src, { tx, ty, scale, angleDeg }) {
  const a = scale * Math.cos((angleDeg * Math.PI) / 180);
  const b = scale * Math.sin((angleDeg * Math.PI) / 180);
  const rgb = new Float32Array(W * H * 3);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const sx = a * x - b * y + tx;
      const sy = b * x + a * y + ty;
      if (sx < 0 || sy < 0 || sx > W - 1 || sy > H - 1) continue;
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const x1 = Math.min(W - 1, x0 + 1);
      const y1 = Math.min(H - 1, y0 + 1);
      const fx = sx - x0;
      const fy = sy - y0;
      const d = (y * W + x) * 3;
      for (let c = 0; c < 3; c++) {
        const v00 = src.rgb[(y0 * W + x0) * 3 + c];
        const v10 = src.rgb[(y0 * W + x1) * 3 + c];
        const v01 = src.rgb[(y1 * W + x0) * 3 + c];
        const v11 = src.rgb[(y1 * W + x1) * 3 + c];
        rgb[d + c] = (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy;
      }
    }
  }
  return { rgb, w: W, h: H };
}

/** True pixel map of the warp: reference (x,y) -> moving source (qx,qy). */
function inverseMap(gt, x, y) {
  const s = gt.scale;
  const th = (gt.angleDeg * Math.PI) / 180;
  const ia = s * Math.cos(th);
  const ib = s * Math.sin(th);
  const det = s * s;
  return {
    x: (ia * (x - gt.tx) + ib * (y - gt.ty)) / det,
    y: (-ib * (x - gt.tx) + ia * (y - gt.ty)) / det,
  };
}

test('similarity registration recovers translation, scale and rotation', () => {
  const ref = texturedReference();
  const cases = [
    { tx: 4.5, ty: -3.2, scale: 1.004, angleDeg: 0.15 },
    { tx: -2, ty: 2, scale: 0.995, angleDeg: -0.2 },
  ];
  for (const gt of cases) {
    const mov = warp(ref, gt);
    const r = registerSimilarity(ref, mov, {});
    assert.ok(r.ncc > 0.95, `NCC ${r.ncc}`);
    // Recovered map must be the inverse of the applied warp.
    let sum = 0;
    let n = 0;
    for (let y = 20; y < H - 20; y += 8) {
      for (let x = 20; x < W - 20; x += 8) {
        const t = inverseMap(gt, x, y);
        const gx = r.params.a * x - r.params.b * y + r.params.tx;
        const gy = r.params.b * x + r.params.a * y + r.params.ty;
        sum += (t.x - gx) ** 2 + (t.y - gy) ** 2;
        n++;
      }
    }
    const rms = Math.sqrt(sum / n);
    assert.ok(rms < 1.0, `map RMS ${rms.toFixed(3)}px for ${JSON.stringify(gt)}`);
  }
});

test('resampling with the recovered transform leaves only noise', () => {
  const ref = texturedReference();
  const gt = { tx: 3.1, ty: 2.4, scale: 1.002, angleDeg: 0.1 };
  const mov = warp(ref, gt);
  const r = registerSimilarity(ref, mov, {});
  const s = sampleRegistered(mov, r.params, W, H);
  let sum = 0;
  let n = 0;
  for (let y = 12; y < H - 12; y++) {
    for (let x = 12; x < W - 12; x++) {
      const i = y * W + x;
      if (!s.valid[i]) continue;
      for (let c = 0; c < 3; c++) {
        sum += Math.abs(s.rgb[i * 3 + c] - ref.rgb[i * 3 + c]);
        n++;
      }
    }
  }
  const residual = sum / n;
  assert.ok(residual < 0.012, `mean abs residual ${residual}`);
});

test('pure translation is recovered in the correct direction', () => {
  const ref = texturedReference();
  // The moving image samples source at x+5, i.e. its content appears shifted
  // left by 5; the reference->moving map therefore has tx = -5.
  const mov = warp(ref, { tx: 5, ty: 0, scale: 1, angleDeg: 0 });
  const r = registerSimilarity(ref, mov, {});
  assert.ok(Math.abs(r.scale - 1) < 0.004, `scale ${r.scale}`);
  assert.ok(Math.abs(r.rotationDeg) < 0.1, `rotation ${r.rotationDeg}`);
  assert.ok(Math.abs(r.params.tx + 5) < 1, `tx ${r.params.tx}`);
  assert.ok(Math.abs(r.params.ty) < 1, `ty ${r.params.ty}`);
});
