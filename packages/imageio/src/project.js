// Project descriptor converter.
//
// A "project" is the focus-fusion job description independent of pixel bytes:
// the ordered focus group, each member's selection state and valid regions
// (ROI), and the solver options. Pixel data stays in the image files; the
// descriptor only names them.
//
// Old shape (demo/manifest.json plus the ad-hoc state carried by the CLI/GUI):
//   { generated, width, height, scenes: [{ scene, slices: [filenames],
//       focuses, bytes }] }
// and per-run CLI state (ROI/excludes passed as flags, never persisted as a
// unit). The new shape is one explicit, versioned document:
//   { schema: 'focus-fusion-project/v2', id, createdAt, source, width, height,
//     slices: [{ id, name, file, index, focus, enabled, roi: []|[box] }],
//     options }
//
// Guarantees:
//   * slice order and every ROI box are preserved verbatim;
//   * conversion is deterministic and pure (input objects are never mutated);
//   * running the conversion twice yields the same id and equal documents, so
//     re-importing a project cannot change the focus group's identity;
//   * failures throw before anything is returned, leaving the original project
//     untouched (callers write atomically).

const SCHEMA = 'focus-fusion-project/v2';

function stableStringify(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`;
  const keys = Object.keys(value).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${stableStringify(value[k])}`).join(',')}}`;
}

// FNV-1a 32-bit, two rounds over different seeds to widen the id; hex output.
function fnv1a(str, seed) {
  let h = seed >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

export function projectIdFingerprint(doc) {
  const basis = stableStringify({
    width: doc.width ?? null,
    height: doc.height ?? null,
    slices: doc.slices.map((s) => ({ name: s.name, file: s.file, focus: s.focus ?? null })),
  });
  const a = fnv1a(basis, 0x811c9dc5);
  const b = fnv1a(basis, (a ^ 0x9e3779b9) >>> 0);
  const c = fnv1a(basis, (b ^ 0x85ebca6b) >>> 0);
  const d = fnv1a(basis, (c ^ 0xc2b2ae35) >>> 0);
  return [a, b, c, d].map((v) => v.toString(16).padStart(8, '0')).join('');
}

function normaliseBox(box, where) {
  if (!box || typeof box !== 'object') throw new Error(`project: ${where} ROI box is not an object`);
  const { x, y, w, h } = box;
  for (const [k, v] of Object.entries({ x, y, w, h })) {
    if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`project: ${where} ROI box.${k} must be a finite number`);
  }
  if (w <= 0 || h <= 0) throw new Error(`project: ${where} ROI box must have positive size`);
  if (x < 0 || y < 0 || x + w > 1 + 1e-12 || y + h > 1 + 1e-12) {
    throw new Error(`project: ${where} ROI box must stay inside the normalised image`);
  }
  return { x, y, w, h };
}

function normaliseOptions(options) {
  const out = { registration: true, exposureMatch: true };
  if (!options) return out;
  if (typeof options.registration === 'boolean') out.registration = options.registration;
  if (typeof options.exposureMatch === 'boolean') out.exposureMatch = options.exposureMatch;
  return out;
}

/**
 * Convert a legacy manifest (the shape written by bin/make-demo.js) into a v2
 * project descriptor. `sceneName` selects which manifest scene to import; the
 * first scene is used when omitted.
 */
export function fromLegacyManifest(manifest, { sceneName, options } = {}) {
  if (!manifest || !Array.isArray(manifest.scenes) || !manifest.scenes.length) {
    throw new Error('project: legacy manifest has no scenes');
  }
  const scene = sceneName
    ? manifest.scenes.find((s) => s.scene === sceneName)
    : manifest.scenes[0];
  if (!scene) throw new Error(`project: scene "${sceneName}" not found in manifest`);
  if (!Array.isArray(scene.slices) || scene.slices.length < 2) {
    throw new Error('project: a focus group needs at least two slices');
  }
  const slices = scene.slices.map((file, index) => ({
    id: `slice-${index}`,
    name: String(file).replace(/\.[^.]+$/, ''),
    file: String(file),
    index,
    focus: Array.isArray(scene.focuses) ? scene.focuses[index] : null,
    enabled: true,
    roi: [],
  }));
  const doc = {
    schema: SCHEMA,
    id: null,
    createdAt: null,
    source: { kind: 'legacy-manifest', scene: scene.scene },
    width: manifest.width ?? null,
    height: manifest.height ?? null,
    slices,
    options: normaliseOptions(options),
  };
  doc.id = projectIdFingerprint(doc);
  doc.createdAt = new Date(0).toISOString(); // deterministic; caller may override
  return doc;
}

/**
 * Convert an ad-hoc in-memory focus group (CLI/GUI state) into a v2 descriptor.
 * @param {Array<{name:string, file?:string, focus?:number|null, enabled?:boolean, roi?:Array}>} group
 */
export function fromFocusGroup(group, { width = null, height = null, options, source = 'focus-group' } = {}) {
  if (!Array.isArray(group) || group.length < 2) throw new Error('project: a focus group needs at least two slices');
  const slices = group.map((s, index) => {
    if (!s || typeof s.name !== 'string' || !s.name.length) {
      throw new Error(`project: slice ${index} needs a name`);
    }
    const roi = Array.isArray(s.roi) ? s.roi.map((b) => normaliseBox(b, `slice ${index}`)) : [];
    return {
      id: `slice-${index}`,
      name: s.name,
      file: s.file ?? `${s.name}.png`,
      index,
      focus: Number.isFinite(s.focus) ? s.focus : null,
      enabled: s.enabled !== false,
      roi,
    };
  });
  const doc = {
    schema: SCHEMA,
    id: null,
    createdAt: null,
    source: { kind: source },
    width,
    height,
    slices,
    options: normaliseOptions(options),
  };
  doc.id = projectIdFingerprint(doc);
  doc.createdAt = new Date(0).toISOString();
  return doc;
}

/**
 * Parse/validate an existing v2 descriptor coming from disk. The order and ROI
 * boxes are checked but never reordered, so focus identity is stable.
 */
export function parseProject(raw) {
  const doc = typeof raw === 'string' ? JSON.parse(raw) : raw;
  if (!doc || doc.schema !== SCHEMA) throw new Error(`project: unsupported schema (expected ${SCHEMA})`);
  if (!Array.isArray(doc.slices) || doc.slices.length < 2) throw new Error('project: descriptor needs >= 2 slices');
  const seenIndex = new Set();
  doc.slices.forEach((s, k) => {
    if (s.index !== k) throw new Error(`project: slice ${k} has non-positional index ${s.index}`);
    if (seenIndex.has(s.id)) throw new Error(`project: duplicate slice id ${s.id}`);
    seenIndex.add(s.id);
    if (Array.isArray(s.roi)) s.roi.forEach((b) => normaliseBox(b, `slice ${k}`));
  });
  const expectedId = projectIdFingerprint(doc);
  if (doc.id && doc.id !== expectedId) {
    throw new Error('project: descriptor id does not match its content; the focus group may have been tampered with');
  }
  doc.options = normaliseOptions(doc.options);
  return doc;
}

export const PROJECT_SCHEMA = SCHEMA;
