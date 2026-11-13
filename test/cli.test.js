import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { decodePNG, encodePNG } from '../src/core/png.js';
import { makeChartStack } from './lib/fixtures.mjs';

const run = promisify(execFile);
const ROOT = new URL('..', import.meta.url).pathname;

async function writeSlices(dir, slices) {
  const paths = [];
  for (let k = 0; k < slices.length; k++) {
    const s = slices[k];
    const bytes = await encodePNG({
      width: s.width, height: s.height, channels: 4, bitDepth: 16, data: s.rgba,
    });
    const p = join(dir, `slice-${k}.png`);
    await (await import('node:fs/promises')).writeFile(p, bytes);
    paths.push(p);
  }
  return paths;
}

test('CLI fuses a stack and writes 16-bit fused + selection + report', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'ff-cli-'));
  try {
    const { slices } = makeChartStack(160, 120, 3, { noise: 0.001 });
    assert.equal(slices.length, 3);
    const paths = await writeSlices(dir, slices);
    const outDir = join(dir, 'out');
    await run(process.execPath, [join(ROOT, 'bin/focus-fusion.js'), outDir, ...paths, '--quiet'], {
      cwd: ROOT,
    });
    const report = JSON.parse(await readFile(join(outDir, 'report.json'), 'utf8'));
    assert.equal(report.inputs.length, 3);
    assert.ok(report.stats.reliable > 0.5);
    assert.ok(Array.isArray(report.warnings));

    const fused = await decodePNG(new Uint8Array(await readFile(join(outDir, 'fused.png'))));
    assert.equal(fused.bitDepth, 16);
    assert.equal(fused.max, 65535);
    assert.equal(fused.width, 160);
    // Not effectively 8-bit.
    const low = new Set();
    for (let i = 0; i < 160 * 120; i++) low.add(fused.data[i * 4] & 255);
    assert.ok(low.size > 32, `distinct low bytes ${low.size}`);

    const sel = await decodePNG(new Uint8Array(await readFile(join(outDir, 'selection.png'))));
    assert.equal(sel.width, 160);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('CLI honours --bit-depth 8 and --report-only', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'ff-cli8-'));
  try {
    const { slices } = makeChartStack(120, 90, 3, { noise: 0.001 });
    const paths = await writeSlices(dir, slices);
    const outDir = join(dir, 'out');
    await run(process.execPath, [
      join(ROOT, 'bin/focus-fusion.js'), outDir, ...paths, '--quiet', '--bit-depth', '8',
    ], { cwd: ROOT });
    const fused = await decodePNG(new Uint8Array(await readFile(join(outDir, 'fused.png'))));
    // 8-bit file decoded to 16-bit nominal: every sample is a multiple of 257.
    for (let i = 0; i < 120 * 90; i++) assert.equal(fused.data[i * 4] % 257, 0);

    const outDir2 = join(dir, 'out2');
    await run(process.execPath, [
      join(ROOT, 'bin/focus-fusion.js'), outDir2, ...paths, '--quiet', '--report-only',
    ], { cwd: ROOT });
    const report = JSON.parse(await readFile(join(outDir2, 'report.json'), 'utf8'));
    assert.equal(report.outputs, undefined);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('CLI rejects size-mismatched inputs with a clear message', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'ff-cli-bad-'));
  try {
    const { slices } = makeChartStack(120, 90, 3, { noise: 0.001 });
    const paths = await writeSlices(dir, slices);
    // Overwrite slice 1 with a different size.
    const bad = await encodePNG({
      width: 121, height: 90, channels: 4, bitDepth: 16, data: new Uint16Array(121 * 90 * 4).fill(30000),
    });
    await (await import('node:fs/promises')).writeFile(paths[1], bad);
    await assert.rejects(
      run(process.execPath, [join(ROOT, 'bin/focus-fusion.js'), join(dir, 'out'), ...paths, '--quiet'], { cwd: ROOT }),
      /size mismatch/,
    );
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
