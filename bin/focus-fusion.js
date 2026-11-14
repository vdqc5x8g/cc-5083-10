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

import { readFile, writeFile, mkdir, rename } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import {
  fuseStack, renderFused, computePreview, encodeWeightMap, encodeReliability,
} from '../src/core/fuse.js';
import {
  readImage, writeImage, tensorToSlice, fusedLinearToTensor,
  fromFocusGroup, parseProject,
} from '../packages/imageio/src/index.js';

function usage(code = 0) {
  const text = `focus-fusion — auxiliary CLI (the GUI is bin/serve.js)

usage: node bin/focus-fusion.js <out-dir> <slice.png|slice.ffi>... [options]

options:
  --roi k:x,y,w,h   limit slice k to a normalised region (repeatable)
  --roi k:none      clear slice k's region
  --exclude k       exclude slice k from the automatic solve (repeatable)
  --bit-depth N     8 or 16 for PNG output (default 16)
  --format png|ffi  fused output container; ffi is lossless float with the
                    full numeric range and metadata (default png)
  --project PATH    import a v2 project descriptor (order/ROI come from it)
  --write-project   also write the resolved project descriptor next to outputs
  --allow-loss      permit lossy PNG writes (8-bit depth, float quantisation)
  --no-registration disable registration
  --report-only     skip image output
  --quiet           suppress progress output
`;
  (code === 0 ? process.stdout : process.stderr).write(text);
  process.exit(code);
}

function parseArgs(argv) {
  const out = {
    inputs: [], roi: {}, excluded: [], bitDepth: 16, format: 'png', project: null,
    writeProject: false, allowLoss: false, registration: true, images: true, quiet: false,
  };
  let outDir = null;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--help' || a === '-h') usage(0);
    else if (a === '--bit-depth') out.bitDepth = Number(argv[++i]) === 8 ? 8 : 16;
    else if (a === '--format') {
      const f = argv[++i];
      if (f !== 'png' && f !== 'ffi') usage(2);
      out.format = f;
    }
    else if (a === '--project') out.project = argv[++i];
    else if (a === '--write-project') out.writeProject = true;
    else if (a === '--allow-loss') out.allowLoss = true;
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
  if (!outDir) usage(2);
  if (!out.project && !out.inputs.length) usage(2);
  out.outDir = outDir;
  return out;
}

// Every pixel read goes through the new imageio reader. The fusion engine's
// RGBA16 sRGB slice contract is reconstructed by the adapter, which also
// reports any conversion that could not be done losslessly.
async function loadSlice(path) {
  const bytes = new Uint8Array(await readFile(path));
  const tensor = await readImage(bytes);
  const slice = tensorToSlice(tensor, { allowLoss: true, name: basename(path).replace(/\.[^.]+$/, '') });
  if (slice.losses.length && !process.env.FOCUS_FUSION_QUIET_LOSS) {
    process.stderr.write(`note: ${basename(path)} imported with losses: ${JSON.stringify(slice.losses)}\n`);
  }
  return {
    name: slice.name,
    width: slice.width,
    height: slice.height,
    rgba: slice.rgba,
    bitDepth: 16,
    source: slice.info,
  };
}

async function writePNG8(path, data, w, h) {
  // Diagnostic overlays are natively 8-bit RGBA; this is an intended 8-bit
  // write, not a silent downgrade of a high-bit-depth result.
  const out = await writeImage(
    {
      width: w, height: h, channels: 4, kind: 'u8', data,
      channelOrder: ['R', 'G', 'B', 'A'], valueRange: { min: 0, max: 255 },
      transfer: 'srgb', metadata: {},
    },
    'png',
    { allowLoss: true },
  );
  await writeFile(path, out.bytes);
}

async function writeProjectAtomic(path, doc) {
  const tmp = `${path}.tmp`;
  await writeFile(tmp, `${JSON.stringify(doc, null, 2)}\n`);
  await rename(tmp, path);
}

export async function main(argv) {
  const args = parseArgs(argv);
  if (process.env.FOCUS_FUSION_TEST === '1') return { args };

  // Project import: the descriptor is the source of truth for slice order and
  // ROI. Flag-supplied ROIs still layer on top for ad-hoc CLI runs.
  let project = null;
  if (args.project) {
    project = parseProject(await readFile(args.project, 'utf8'));
    const baseDir = resolve(dirname(args.project));
    const ordered = project.slices
      .filter((s) => s.enabled !== false)
      .map((s) => resolve(baseDir, s.file));
    args.inputs = ordered;
  }

  const slices = [];
  for (const p of args.inputs) slices.push(await loadSlice(p));
  if (!args.quiet) process.stderr.write(`loaded ${slices.length} slice(s): ${slices.map((s) => `${s.name} ${s.width}x${s.height}`).join(', ')}\n`);

  const roiList = [];
  for (let k = 0; k < slices.length; k++) {
    const fromProject = project?.slices[k]?.roi?.length ? project.slices[k].roi : null;
    roiList.push(args.roi[k] ?? fromProject ?? null);
  }
  const excluded = new Set([...args.excluded, ...(project ? project.slices.map((s, k) => (s.enabled === false ? k : -1)).filter((k) => k >= 0) : [])]);

  const t0 = Date.now();
  const result = fuseStack(slices, { registration: args.registration }, {
    roi: roiList,
    excludedSlices: excluded,
    onProgress: args.quiet ? undefined : (p) => process.stderr.write(`\r${(p * 100).toFixed(0)}%  `),
  });
  if (!args.quiet) process.stderr.write(`\rfused in ${Date.now() - t0} ms\n`);
  await mkdir(args.outDir, { recursive: true });
  const report = {
    inputs: slices.map((s, k) => ({
      name: s.name, width: s.width, height: s.height, index: k, source: s.source ?? null,
    })),
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
    projectId: project?.id ?? null,
  };
  if (args.images) {
    const preview = computePreview(result, 1400);
    if (args.format === 'ffi') {
      // Lossless float path: the actual linear-light range and metadata are
      // carried in the FFI container; nothing is quantised or clipped.
      const tensor = fusedLinearToTensor(result, {
        metadata: {
          name: 'fused',
          projectId: project?.id ?? null,
          referenceIndex: result.refIndex,
          writer: '@focus-fusion/imageio',
        },
      });
      const out = await writeImage(tensor, 'ffi');
      await writeFile(join(args.outDir, 'fused.ffi'), out.bytes);
      report.outputs = { fused: `fused.ffi (f32 linear RGBA, range ${tensor.valueRange.min.toFixed(4)}..${tensor.valueRange.max.toFixed(4)})` };
    } else {
      const fused = renderFused(result, { bitDepth: args.bitDepth });
      const tensor = {
        width: fused.width, height: fused.height, channels: 4, kind: 'u16', data: fused.data,
        channelOrder: ['R', 'G', 'B', 'A'], valueRange: { min: 0, max: 65535 },
        transfer: 'srgb', metadata: { name: 'fused', projectId: project?.id ?? null },
      };
      // 16-bit is lossless; an explicit --bit-depth 8 is an acknowledged
      // lossy choice, recorded in the report instead of applied silently.
      const out = await writeImage(tensor, 'png', {
        bitDepth: args.bitDepth,
        allowLoss: args.allowLoss || args.bitDepth === 8,
      });
      if (out.losses.length) report.fusedWriteLosses = out.losses;
      await writeFile(join(args.outDir, 'fused.png'), out.bytes);
      report.outputs = {
        fused: `fused.png (${fused.width}x${fused.height}, ${out.bitDepth}-bit)`,
      };
    }
    const weightMap = encodeWeightMap(preview.weights, preview.width, preview.height, 'winner');
    await writePNG8(join(args.outDir, 'selection.png'), weightMap, preview.width, preview.height);
    const relOverlay = encodeReliability({
      w: preview.width, h: preview.height, relClass: preview.relClass,
    });
    await writePNG8(join(args.outDir, 'reliability.png'), relOverlay, preview.width, preview.height);
    // One weight heat image per slice, useful for inspecting the ramp.
    for (let k = 0; k < result.weights.length; k++) {
      const heat = encodeWeightMap(preview.weights, preview.width, preview.height, 'weight', k);
      await writePNG8(join(args.outDir, `weight-${k}.png`), heat, preview.width, preview.height);
    }
    report.outputs = {
      ...report.outputs,
      selection: 'selection.png', reliability: 'reliability.png',
      weights: result.weights.map((_, k) => `weight-${k}.png`),
    };
  }

  if (args.writeProject || args.project) {
    const doc = project ?? fromFocusGroup(
      slices.map((s, k) => ({ name: s.name, file: basename(args.inputs[k]), roi: roiList[k] ?? [] })),
      { width: slices[0].width, height: slices[0].height, source: 'cli' },
    );
    await writeProjectAtomic(join(args.outDir, 'project.json'), doc);
    report.projectDescriptor = 'project.json';
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
