#!/usr/bin/env node
// Generate representative imageio samples committed under test-samples/.
//
// The set deliberately covers every contract the new reader/writer must keep
// honest: legacy 16-bit PNG, FFI containers at u8/u16/u32/f32 with different
// numeric ranges, an FFI image carrying extra non-color channels, and an HDR
// float image whose values exceed 1.0 (which PNG cannot represent at all).
//
// Usage: node bin/make-samples.js [out-dir]

import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { writeImage, FFI_EXTENSION } from '../packages/imageio/src/index.js';

const W = 24;
const H = 16;

function pattern(max) {
  const n = W * H;
  const out = new Array(n * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const t = (x / (W - 1) + y / (H - 1)) * 0.5;
      out[i * 4] = Math.round(t * max);
      out[i * 4 + 1] = Math.round(((1 - t) * 0.8 + 0.1) * max);
      out[i * 4 + 2] = Math.round((((x * 7 + y * 3) % 11) / 11) * max);
      out[i * 4 + 3] = max;
    }
  }
  return out;
}

async function save(dir, name, tensor) {
  const ext = name.endsWith('.png') ? 'png' : 'ffi';
  const out = await writeImage(tensor, ext, { allowLoss: true });
  await writeFile(join(dir, name), out.bytes);
  return {
    file: name,
    kind: tensor.kind,
    channels: tensor.channels,
    channelOrder: tensor.channelOrder,
    transfer: tensor.transfer,
    valueRange: tensor.valueRange,
    losses: out.losses.map((l) => l.kind),
  };
}

export async function main(argv = []) {
  const dir = argv[0] || 'test-samples';
  await mkdir(dir, { recursive: true });
  const manifest = { generated: new Date(0).toISOString(), width: W, height: H, samples: [] };

  // 1. Legacy-compatible 16-bit sRGB RGBA PNG (the old contract's native form).
  const p16 = pattern(65535);
  manifest.samples.push(await save(dir, 'gradient-rgba16.png', {
    width: W, height: H, channels: 4, kind: 'u16', data: Uint16Array.from(p16),
    channelOrder: ['R', 'G', 'B', 'A'], valueRange: { min: 0, max: 65535 },
    transfer: 'srgb', metadata: { name: 'gradient-rgba16' },
  }));

  // 2. FFI u16, same numeric content as the PNG but lossless container + metadata.
  manifest.samples.push(await save(dir, `gradient-rgba16${FFI_EXTENSION}`, {
    width: W, height: H, channels: 4, kind: 'u16', data: Uint16Array.from(p16),
    channelOrder: ['R', 'G', 'B', 'A'], valueRange: { min: 0, max: 65535 },
    transfer: 'srgb', metadata: { name: 'gradient-rgba16-ffi' },
  }));

  // 3. FFI u8, standard 0..255 domain.
  manifest.samples.push(await save(dir, `gradient-rgba8${FFI_EXTENSION}`, {
    width: W, height: H, channels: 4, kind: 'u8', data: Uint8Array.from(pattern(255)),
    channelOrder: ['R', 'G', 'B', 'A'], valueRange: { min: 0, max: 255 },
    transfer: 'srgb', metadata: { name: 'gradient-rgba8' },
  }));

  // 4. FFI u32 high bit depth, wider than PNG's 16-bit ceiling.
  manifest.samples.push(await save(dir, `gradient-rgba32${FFI_EXTENSION}`, {
    width: W, height: H, channels: 4, kind: 'u32', data: Uint32Array.from(pattern(1_000_000)),
    channelOrder: ['R', 'G', 'B', 'A'], valueRange: { min: 0, max: 1_000_000 },
    transfer: 'srgb', metadata: { name: 'gradient-rgba32' },
  }));

  // 5. FFI f32 linear light, standard 0..1 photometric domain.
  const f01 = new Float32Array(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    const t = (i % W) / W;
    f01[i * 4] = t * t;
    f01[i * 4 + 1] = 0.25;
    f01[i * 4 + 2] = 1 - t;
    f01[i * 4 + 3] = 1;
  }
  manifest.samples.push(await save(dir, `linear-float-unit${FFI_EXTENSION}`, {
    width: W, height: H, channels: 4, kind: 'f32', data: f01,
    channelOrder: ['R', 'G', 'B', 'A'], valueRange: { min: 0, max: 1 },
    transfer: 'linear', metadata: { name: 'linear-float-unit' },
  }));

  // 6. FFI f32 HDR: values above 1.0 survive; PNG would have to clip them.
  const hdr = new Float32Array(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    hdr[i * 4] = 0.6 + (i / (W * H)) * 1.4;   // 0.6 .. 2.0
    hdr[i * 4 + 1] = 0.3;
    hdr[i * 4 + 2] = 0.1;
    hdr[i * 4 + 3] = 1;
  }
  manifest.samples.push(await save(dir, `linear-float-hdr${FFI_EXTENSION}`, {
    width: W, height: H, channels: 4, kind: 'f32', data: hdr,
    channelOrder: ['R', 'G', 'B', 'A'], valueRange: { min: 0, max: 1 },
    transfer: 'linear', metadata: { name: 'linear-float-hdr', hdr: true },
  }));

  // 7. FFI with two extra channels (depth + confidence); order must round-trip.
  const six = new Float32Array(W * H * 6);
  for (let i = 0; i < W * H; i++) {
    six[i * 6] = f01[i * 4];
    six[i * 6 + 1] = f01[i * 4 + 1];
    six[i * 6 + 2] = f01[i * 4 + 2];
    six[i * 6 + 3] = 1;
    six[i * 6 + 4] = 1 + (i % 5) * 0.7;    // Depth
    six[i * 6 + 5] = 0.5 + (i % 7) * 0.07; // Confidence
  }
  manifest.samples.push(await save(dir, `rgba-plus-depth${FFI_EXTENSION}`, {
    width: W, height: H, channels: 6, kind: 'f32', data: six,
    channelOrder: ['R', 'G', 'B', 'A', 'Depth', 'Confidence'],
    valueRange: { min: 0, max: 4 }, transfer: 'linear',
    metadata: { name: 'rgba-plus-depth' },
  }));

  await writeFile(join(dir, 'samples-manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}

if (process.argv[1] && process.argv[1].endsWith('make-samples.js')) {
  main(process.argv.slice(2)).then((m) => {
    console.log(`wrote ${m.samples.length} sample image(s) to ${process.argv[2] || 'test-samples'}/`);
  }).catch((err) => {
    console.error(`make-samples: ${err.message}`);
    process.exit(1);
  });
}
