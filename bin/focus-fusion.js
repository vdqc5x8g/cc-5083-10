#!/usr/bin/env node
// Auxiliary command-line front end for focus fusion.
//
// The graphical image-restoration window (bin/serve.js -> web UI) is the main
// entry point; this CLI exists for batch work and for the tests. It reads a
// set of PNG slices, registers them, solves the sharpness weights, writes the
// fused PNG (16-bit by default), the selection/weight map, the reliability
// overlay and a JSON report.
//
// Usage:
//   node bin/focus-fusion.js out-dir slice1.png slice2.png [...]
// Options:
//   --roi k:x,y,w,h      restrict slice k (0-based, in input order) to a
//                        normalised box; repeatable, use --roi k:none to clear
//   --exclude k          remove slice k from the automatic solve (repeatable)
//   --bit-depth 8|16     output depth (default 16)
//   --no-registration    disable the similarity registration stage
//   --report-only        do not write images
//   --quiet              suppress the progress/JSON summary on stderr

import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { decodePNG, encodePNG } from '../src/core/png.js';
import {
  fuseStack, renderFused, computePreview, encodeWeightMap, encodeReliability,
} from '../src/core/fuse.js';

function usage(code = 0) {
  const text = `focus-fusion — auxiliary CLI (the GUI is bin/serve.js)

usage: node bin/focus-fusion.js <out-dir> <slice.png>... [options]

options:
  --roi k:x,y,w,h   limit slice k to a normalised region (repeatable)
  --roi k:none      clear slice k's region
  --exclude k       exclude slice k from the automatic solve (repeatable)
  --bit-depth N     8 or 16 (default 16)
  --no-registration disable registration
  --report-only     skip image output
  --quiet           suppress progress output
`;
  (code === 0 ? process.stdout : process.stderr).write(text);
  process.exit(code);
}

function parseArgs(argv) {
  const out = { inputs: [], roi: {}, excluded: [], bitDepth: 16, registration: true, images: true, quiet: false };
  let outDir = null;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--help' || a === '-h') usage(0);
    else if (a === '--bit-depth') out.bitDepth = Number(argv[++i]) === 8 ? 8 : 16;
    else if (a === '--roi') {
      const spec = argv[++i] || '';
      const m = /^(\d+):(.*)$/.exec(spec);
      if (!m) usage(2);
      const k = Number(m[1]);
      if (m[2] === 'none') out.roi[k] = [];
      else {
        const nums = m[2].split(',').map(Number);
        if (nums.length !== 4 || nums.some((v) => !Number.isFinite(v))) usage(2);
        const [x, y, w, h] = nums;
        (out.roi[k] ||= []).push({ x, y, w, h });
      }
    } else if (a === '--exclude') out.excluded.push(Number(argv[++i]));
    else if (a === '--no-registration') out.registration = false;
    else if (a === '--report-only') out.images = false;
    else if (a === '--quiet') out.quiet = true;
    else if (a.startsWith('--')) { console.error(`unknown option: ${a}`); usage(2); }
    else if (!outDir) outDir = a;
    else out.inputs.push(a);
  }
  if (!outDir || !out.inputs.length) usage(2);
  out.outDir = outDir;
  return out;
}

async function loadSlice(path) {
  const bytes = new Uint8Array(await readFile(path));
  const img = await decodePNG(bytes);
  return { name: basename(path), width: img.width, height: img.height, rgba: img.data, bitDepth: img.bitDepth };
}

export async function main(argv) {
  const args = parseArgs(argv);
  if (process.env.FOCUS_FUSION_TEST === '1') return { args };
  const slices = [];
  for (const p of args.inputs) slices.push(await loadSlice(p));
  if (!args.quiet) process.stderr.write(`loaded ${slices.length} slice(s): ${slices.map((s) => `${s.name} ${s.width}x${s.height}`).join(', ')}\n`);
  const roiList = [];
  for (let k = 0; k < slices.length; k++) roiList.push(args.roi[k] ?? null);
  const t0 = Date.now();
  const result = fuseStack(slices, { registration: args.registration }, {
    roi: roiList,
    excludedSlices: new Set(args.excluded),
    onProgress: args.quiet ? undefined : (p) => process.stderr.write(`\r${(p * 100).toFixed(0)}%  `),
  });
  if (!args.quiet) process.stderr.write(`\rfused in ${Date.now() - t0} ms\n`);
  await mkdir(args.outDir, { recursive: true });
  const report = {
    inputs: slices.map((s, k) => ({ name: s.name, width: s.width, height: s.height, index: k })),
    referenceIndex: result.refIndex,
    skipped: result.skip,
    registration: result.registration.map((r, k) => (r ? {
      slice: k, ok: r.ok, ncc: r.ncc, scale: r.scale, rotationDeg: r.rotationDeg, shift: r.shiftPx,
    } : { slice: k, reference: true })),
    stats: result.stats,
    edgeWidth: result.edgeWidth,
    ghosting: result.ghosting,
    coverage: result.coverage,
    warnings: result.warnings,
  };
  if (args.images) {
    const fused = renderFused(result, { bitDepth: args.bitDepth });
    const fusedPng = await encodePNG({
      width: fused.width, height: fused.height, channels: 4, bitDepth: fused.bitDepth,
      data: fused.data,
      text: [{ keyword: 'Software', value: 'focus-fusion 1.0' }],
    });
    await writeFile(join(args.outDir, 'fused.png'), fusedPng);
    const preview = computePreview(result, 1400);
    const weightMap = encodeWeightMap(preview.weights, preview.width, preview.height, 'winner');
    await writeFile(join(args.outDir, 'selection.png'), await encodePNG({
      width: preview.width, height: preview.height, channels: 4, bitDepth: 8, data: weightMap,
    }));
    const relOverlay = encodeReliability({
      w: preview.width, h: preview.height, relClass: preview.relClass,
    });
    await writeFile(join(args.outDir, 'reliability.png'), await encodePNG({
      width: preview.width, height: preview.height, channels: 4, bitDepth: 8, data: relOverlay,
    }));
    // One weight heat image per slice, useful for inspecting the ramp.
    for (let k = 0; k < result.weights.length; k++) {
      const heat = encodeWeightMap(preview.weights, preview.width, preview.height, 'weight', k);
      await writeFile(join(args.outDir, `weight-${k}.png`), await encodePNG({
        width: preview.width, height: preview.height, channels: 4, bitDepth: 8, data: heat,
      }));
    }
    report.outputs = {
      fused: `fused.png (${fused.width}x${fused.height}, ${args.bitDepth}-bit)`,
      selection: 'selection.png', reliability: 'reliability.png',
      weights: result.weights.map((_, k) => `weight-${k}.png`),
    };
  }
  await writeFile(join(args.outDir, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
  if (!args.quiet) process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(import.meta.url.replace(/^file:\/\//, ''))) {
  main(process.argv.slice(2)).catch((err) => {
    console.error(`focus-fusion: ${err.message}`);
    process.exit(1);
  });
}
