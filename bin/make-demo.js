#!/usr/bin/env node
// Generate demo focal stacks as PNGs so the GUI can be tried without a
// camera. Two scenes are produced:
//   synthetic  - the controlled test chart (explicit geometry + depth)
//   stilllife  - the self-made tabletop arrangement
// Both are rendered through the same thin-lens circle-of-confusion model in
// src/core/synth.js, including focus breathing, drift and sensor noise.
//
// Usage: node bin/make-demo.js [out-dir] [--scene both|synthetic|still] [--size WxH]

import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { encodePNG } from '../src/core/png.js';
import { buildScene, renderSlice, renderSharp } from '../src/core/synth.js';
import { buildStillLife, renderStillLifeStack, stillLifeFocuses } from '../src/core/stilllife.js';

function parse(argv) {
  const out = { dir: 'demo', scene: 'both', width: 640, height: 480 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--scene') out.scene = argv[++i];
    else if (a === '--size') {
      const m = /^(\d+)x(\d+)$/.exec(argv[++i] || '');
      if (!m) throw new Error('--size expects WxH');
      out.width = Number(m[1]);
      out.height = Number(m[2]);
    } else if (!a.startsWith('--')) out.dir = a;
  }
  return out;
}

async function writePng(path, img) {
  const bytes = await encodePNG({
    width: img.width, height: img.height, channels: 4, bitDepth: 16, data: img.rgba,
    text: [{ keyword: 'Software', value: 'focus-fusion demo' }],
  });
  await writeFile(path, bytes);
  return bytes.length;
}

async function makeSynthetic(dir, w, h) {
  const scene = buildScene(w, h);
  const focuses = [0.5, 0.62, 0.85, 1.1];
  const slices = focuses.map((focus, k) => ({
    name: `chart-${String(k + 1).padStart(2, '0')}`,
    rgba: renderSlice(scene, {
      focus, aperture: 0.9, breathing: 0.004,
      shift: { x: (k - 1.5) * 1.2, y: k % 2 ? 1 : -1 }, noise: 0.006,
    }).rgba,
  }));
  let bytes = 0;
  for (const s of slices) bytes += await writePng(join(dir, `${s.name}.png`), { ...s, width: w, height: h });
  bytes += await writePng(join(dir, 'chart-sharp-reference.png'), renderSharp(scene));
  return { scene: 'synthetic', slices: slices.map((s) => `${s.name}.png`), focuses, bytes };
}

async function makeStill(dir, w, h) {
  const { scene, slices, focuses } = renderStillLifeStack(w, h, 8, { noise: 0.006 });
  let bytes = 0;
  for (const s of slices) bytes += await writePng(join(dir, `${s.name}.png`), s);
  // Ground-truth sharp render (no blur) at the same geometry, for reference.
  const sharp = new Uint16Array(w * h * 4);
  const { linearToSrgb16 } = await import('../src/core/image.js');
  for (let i = 0; i < w * h; i++) {
    sharp[i * 4] = linearToSrgb16(scene.rgb[i * 3]);
    sharp[i * 4 + 1] = linearToSrgb16(scene.rgb[i * 3 + 1]);
    sharp[i * 4 + 2] = linearToSrgb16(scene.rgb[i * 3 + 2]);
    sharp[i * 4 + 3] = 65535;
  }
  bytes += await writePng(join(dir, 'still-sharp-reference.png'), { width: w, height: h, rgba: sharp });
  return {
    scene: 'stilllife',
    slices: slices.map((s) => `${s.name}.png`),
    focuses: stillLifeFocuses(8),
    bytes,
  };
}

export async function main(argv = []) {
  const opts = parse(argv);
  await mkdir(opts.dir, { recursive: true });
  const manifest = { generated: new Date().toISOString(), width: opts.width, height: opts.height, scenes: [] };
  if (opts.scene === 'both' || opts.scene === 'synthetic') {
    const r = await makeSynthetic(opts.dir, opts.width, opts.height);
    manifest.scenes.push(r);
    console.log(`synthetic: ${r.slices.length} slices + sharp reference (${(r.bytes / 1024).toFixed(0)} KiB)`);
  }
  if (opts.scene === 'both' || opts.scene === 'still') {
    const r = await makeStill(opts.dir, opts.width, opts.height);
    manifest.scenes.push(r);
    console.log(`stilllife: ${r.slices.length} slices + sharp reference (${(r.bytes / 1024).toFixed(0)} KiB)`);
  }
  await writeFile(join(opts.dir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`written to ${opts.dir}/`);
  return manifest;
}

if (process.argv[1] && process.argv[1].endsWith('make-demo.js')) {
  main(process.argv.slice(2)).catch((err) => {
    console.error(`make-demo: ${err.message}`);
    process.exit(1);
  });
}
