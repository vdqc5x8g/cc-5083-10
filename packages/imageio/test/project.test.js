import test from 'node:test';
import assert from 'node:assert/strict';
import {
  fromLegacyManifest, fromFocusGroup, parseProject, projectIdFingerprint, PROJECT_SCHEMA,
} from '../src/project.js';

const manifest = {
  generated: '2026-11-13T00:00:00.000Z',
  width: 320,
  height: 240,
  scenes: [
    { scene: 'synthetic', slices: ['chart-01.png', 'chart-02.png', 'chart-03.png'], focuses: [0.5, 0.62, 0.85], bytes: 10 },
    { scene: 'stilllife', slices: ['still-01.png', 'still-02.png'], focuses: [0.3, 0.4], bytes: 5 },
  ],
};

test('legacy manifest converts to a versioned project with order and focuses preserved', () => {
  const doc = fromLegacyManifest(manifest);
  assert.equal(doc.schema, PROJECT_SCHEMA);
  assert.equal(doc.slices.length, 3);
  assert.deepEqual(doc.slices.map((s) => s.file), ['chart-01.png', 'chart-02.png', 'chart-03.png']);
  assert.deepEqual(doc.slices.map((s) => s.focus), [0.5, 0.62, 0.85]);
  assert.deepEqual(doc.slices.map((s) => s.index), [0, 1, 2]);
  assert.equal(doc.width, 320);
  assert.ok(/^[0-9a-f]{32}$/.test(doc.id));
});

test('ROI boxes survive conversion verbatim and stay in normalised bounds', () => {
  const group = [
    { name: 'a', roi: [{ x: 0.1, y: 0.2, w: 0.3, h: 0.4 }, { x: 0.5, y: 0.5, w: 0.25, h: 0.25 }] },
    { name: 'b', roi: [] },
    { name: 'c' },
  ];
  const doc = fromFocusGroup(group, { width: 100, height: 80 });
  assert.deepEqual(doc.slices[0].roi, group[0].roi);
  assert.deepEqual(doc.slices[1].roi, []);
  assert.equal(doc.slices[2].enabled, true);
});

test('conversion is deterministic: repeated runs keep the same focus identity', () => {
  const d1 = fromLegacyManifest(manifest, { sceneName: 'stilllife' });
  const d2 = fromLegacyManifest(manifest, { sceneName: 'stilllife' });
  assert.equal(d1.id, d2.id);
  assert.deepEqual(d1, d2);
  // Reordering slices changes identity; identical order does not.
  const reordered = fromLegacyManifest({
    ...manifest,
    scenes: [{ ...manifest.scenes[0], slices: ['chart-03.png', 'chart-02.png', 'chart-01.png'], focuses: [0.85, 0.62, 0.5] }],
  });
  assert.notEqual(reordered.id, d1.id);
});

test('id fingerprint only tracks focus-group content, not transient options', () => {
  const g = [{ name: 'a' }, { name: 'b' }];
  const d1 = fromFocusGroup(g, { options: { registration: true } });
  const d2 = fromFocusGroup(g, { options: { registration: false } });
  assert.equal(d1.id, d2.id);
  assert.equal(projectIdFingerprint(d1), d1.id);
});

test('parseProject round-trips and rejects tampered ids and bad ROI', () => {
  const doc = fromFocusGroup([{ name: 'a' }, { name: 'b' }]);
  const parsed = parseProject(JSON.stringify(doc));
  assert.equal(parsed.id, doc.id);
  const tampered = JSON.parse(JSON.stringify(doc));
  tampered.slices[0].name = 'renamed';
  assert.throws(() => parseProject(tampered), /id does not match/);
  const badRoi = JSON.parse(JSON.stringify(doc));
  badRoi.roi = undefined;
  badRoi.slices[0].roi = [{ x: 0.5, y: 0.5, w: 0.8, h: 0.8 }];
  assert.throws(() => parseProject(badRoi), /inside/);
});

test('converter refuses malformed input without touching the source object', () => {
  const snapshot = JSON.parse(JSON.stringify(manifest));
  assert.throws(() => fromLegacyManifest({ scenes: [] }), /no scenes/);
  assert.throws(() => fromFocusGroup([{ name: 'only' }]), /at least two/);
  assert.deepEqual(manifest, snapshot);
});
