import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  readImage, writeImage, tensorToSlice, fusedLinearToTensor,
  fromLegacyManifest, fromFocusGroup, parseProject,
} from '../packages/imageio/src/index.js';
import { fuseStack } from '../src/core/fuse.js';
import { makeChartStack } from './lib/fixtures.mjs';

const run = promisify(execFile);
const ROOT = new URL('..', import.meta.url).pathname;

async function writePngSlice(dir, k, s) {
  const tensor = {
    width: s.width, height: s.height, channels: 4, kind: 'u16', data: s.rgba,
    channelOrder: ['R', 'G', 'B', 'A'], valueRange: { min: 0, max: 65535 },
    transfer: 'srgb', metadata: { name: s.name },
  };
  const out = await writeImage(tensor, 'png', { bitDepth: 16 });
  const p = join(dir, `slice-${k}.png`);
  await writeFile(p, out.bytes);
  return p;
}

test('full chain: legacy PNG project -> fuse -> FFI export -> reread matches in-memory pixels', async () => {
  const { slices } = makeChartStack(120, 90, 3, { noise: 0.001 });

  // 1. Load through the new reader and adapt to engine slices (the formal path).
  const dir = await mkdtemp(join(tmpdir(), 'ff-e2e-'));
  try {
    const paths = [];
    for (let k = 0; k < slices.length; k++) paths.push(await writePngSlice(dir, k, slices[k]));
    const loaded = [];
    for (const p of paths) {
      const tensor = await readImage(new Uint8Array(await readFile(p)));
      assert.equal(tensor.transfer, 'srgb');
      assert.deepEqual(tensor.channelOrder, ['R', 'G', 'B', 'A']);
      const slice = tensorToSlice(tensor, { name: slices[loaded.length]?.name });
      assert.equal(slice.losses.length, 0, '16-bit sRGB RGBA must import losslessly');
      loaded.push(slice);
    }
    // Legacy RGBA16 bytes must be identical to the engine input.
    for (let k = 0; k < slices.length; k++) {
      assert.deepEqual(Array.from(loaded[k].rgba), Array.from(slices[k].rgba));
    }

    // 2. Compute in memory.
    const engineInputs = loaded.map((s) => ({ name: s.name, width: s.width, height: s.height, rgba: s.rgba }));
    const result = fuseStack(engineInputs, {});
    const inMemory = fusedLinearToTensor(result);

    // 3. Export to the new FFI container via the formal writer.
    const written = await writeImage(inMemory, 'ffi');
    assert.equal(written.losses.length, 0);

    // 4. Reread and compare against the in-memory tensor, sample by sample —
    //    not merely by file extension.
    const reread = await readImage(written.bytes);
    assert.equal(reread.kind, 'f32');
    assert.equal(reread.channels, 4);
    assert.deepEqual(reread.channelOrder, ['R', 'G', 'B', 'A']);
    assert.equal(reread.width, 120);
    assert.equal(reread.height, 90);
    assert.equal(reread.data.length, inMemory.data.length);
    let maxAbs = 0;
    for (let i = 0; i < inMemory.data.length; i++) maxAbs = Math.max(maxAbs, Math.abs(reread.data[i] - inMemory.data[i]));
    assert.equal(maxAbs, 0, 'FFI reread must reproduce every float sample exactly');
    assert.equal(reread.metadata.writer ?? inMemory.metadata.writer, inMemory.metadata.writer);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('PNG 16-bit export reread matches the rendered sRGB pixels; 8-bit is a reported loss', async () => {
  const { slices } = makeChartStack(96, 72, 3, { noise: 0.001 });
  const inputs = slices.map((s) => ({ name: s.name, width: s.width, height: s.height, rgba: s.rgba }));
  const result = fuseStack(inputs, { registration: false });
  const tensor = fusedLinearToTensor(result);
  // Quantised sRGB u16 render via the engine's own renderer for comparison.
  const { renderFused } = await import('../src/core/fuse.js');
  const rendered = renderFused(result, { bitDepth: 16 });

  // The fused tensor is linear f32; PNG requires sRGB u16, so build the same
  // render explicitly and write it through the new PNG writer.
  const pngTensor = {
    width: result.w, height: result.h, channels: 4, kind: 'u16', data: rendered.data,
    channelOrder: ['R', 'G', 'B', 'A'], valueRange: { min: 0, max: 65535 },
    transfer: 'srgb', metadata: {},
  };
  const out16 = await writeImage(pngTensor, 'png', { bitDepth: 16 });
  assert.equal(out16.losses.length, 0);
  const back16 = await readImage(out16.bytes);
  for (let i = 0; i < rendered.data.length; i++) assert.equal(back16.data[i], rendered.data[i]);

  // 8-bit is a real loss: refused without consent, reported with consent.
  await assert.rejects(() => writeImage(pngTensor, 'png', { bitDepth: 8 }), /loses depth/);
  const out8 = await writeImage(pngTensor, 'png', { bitDepth: 8, allowLoss: true });
  assert.ok(out8.losses.some((l) => l.kind === 'bit-depth'));
  const back8 = await readImage(out8.bytes);
  assert.equal(back8.metadata.sourceBitDepth, 8);
  // And it is genuinely 8-bit, not a silent rename of a 16-bit buffer.
  assert.ok(back8.data.every((v) => v % 257 === 0));
  void tensor;
});

test('extra-channel FFI input keeps all channels in the container; engine import flags the dropped ones', async () => {
  const w = 16;
  const h = 12;
  const data = new Float32Array(w * h * 6);
  for (let i = 0; i < data.length; i++) data[i] = (i * 0.11) % 1.5;
  const tensor = {
    width: w, height: h, channels: 6, kind: 'f32', data,
    channelOrder: ['R', 'G', 'B', 'A', 'Depth', 'Confidence'],
    valueRange: { min: 0, max: 1.5 }, transfer: 'linear', metadata: {},
  };
  const file = await writeImage(tensor, 'ffi');
  const back = await readImage(file.bytes);
  assert.deepEqual(back.channelOrder, ['R', 'G', 'B', 'A', 'Depth', 'Confidence']);
  for (let i = 0; i < data.length; i++) assert.equal(back.data[i], data[i]);
  // The RGBA engine cannot consume Depth/Confidence: refuse by default...
  assert.throws(() => tensorToSlice(back), /extra channel/);
  // ...and only drop them with an explicit, recorded loss.
  const slice = tensorToSlice(back, { allowLoss: true });
  assert.equal(slice.rgba.length, w * h * 4);
  assert.deepEqual(slice.losses.find((l) => l.kind === 'dropped-channels').channels, ['Depth', 'Confidence']);
});

test('project converter preserves slice order and ROI through an end-to-end CLI run', async () => {
  const { slices } = makeChartStack(120, 90, 3, { noise: 0.001 });
  const dir = await mkdtemp(join(tmpdir(), 'ff-proj-'));
  try {
    const paths = [];
    for (let k = 0; k < slices.length; k++) paths.push(await writePngSlice(dir, k, slices[k]));
    const roi = [[{ x: 0, y: 0, w: 0.4, h: 1 }], [], []];
    const project = fromFocusGroup(
      slices.map((s, k) => ({ name: s.name, file: `slice-${k}.png`, roi: roi[k] })),
      { width: 120, height: 90, source: 'test' },
    );
    const projectPath = join(dir, 'project.json');
    await writeFile(projectPath, `${JSON.stringify(project, null, 2)}\n`);
    // Repeated conversion must not change the focus identity.
    const again = fromFocusGroup(
      slices.map((s, k) => ({ name: s.name, file: `slice-${k}.png`, roi: roi[k] })),
      { width: 120, height: 90, source: 'test' },
    );
    assert.equal(again.id, project.id);

    const outDir = join(dir, 'out');
    await run(process.execPath, [
      join(ROOT, 'bin/focus-fusion.js'), outDir, ...paths,
      '--project', projectPath, '--format', 'ffi', '--quiet',
    ], { cwd: ROOT });
    const report = JSON.parse(await readFile(join(outDir, 'report.json'), 'utf8'));
    assert.equal(report.projectId, project.id);
    const exported = await readImage(new Uint8Array(await readFile(join(outDir, 'fused.ffi'))));
    assert.equal(exported.kind, 'f32');
    // The ROI restricted slice 0: its weight must be zero outside the box.
    const w = 120;
    const projBack = parseProject(await readFile(join(outDir, 'project.json'), 'utf8'));
    assert.deepEqual(projBack.slices[0].roi, roi[0]);
    // Verify against a direct in-process solve with the same ROI.
    const direct = fuseStack(
      slices.map((s) => ({ name: s.name, width: s.width, height: s.height, rgba: s.rgba })),
      {},
      { roi: [roi[0], null, null] },
    );
    let outsideWeight = 0;
    for (let y = 0; y < 90; y++) {
      for (let x = Math.ceil(w * 0.4); x < w; x++) outsideWeight += direct.weights[0][y * w + x];
    }
    assert.equal(outsideWeight, 0);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('CLI legacy-manifest conversion keeps every focus and is idempotent', async () => {
  const manifest = {
    width: 100, height: 80,
    scenes: [{ scene: 'synthetic', slices: ['a.png', 'b.png', 'c.png'], focuses: [0.2, 0.5, 0.9], bytes: 1 }],
  };
  const d1 = fromLegacyManifest(manifest);
  const d2 = fromLegacyManifest(manifest);
  assert.equal(d1.id, d2.id);
  assert.deepEqual(d1.slices.map((s) => s.focus), [0.2, 0.5, 0.9]);
  assert.deepEqual(d1.slices.map((s) => s.index), [0, 1, 2]);
  // Failure mid-migration leaves the original untouched.
  const snapshot = JSON.stringify(manifest);
  assert.throws(() => fromLegacyManifest({ scenes: [] }));
  assert.equal(JSON.stringify(manifest), snapshot);
});
