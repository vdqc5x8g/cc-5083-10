import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { createStaticServer } from '../bin/serve.js';

function fetchText(base, path) {
  return fetch(base + path);
}

// Raw request that does not normalise the path first (fetch/URL would turn
// "/../x" into "/x" and hide the traversal attempt).
function rawGet(port, path) {
  return new Promise((resolve, reject) => {
    const req = http.get({ host: '127.0.0.1', port, path }, (res) => {
      res.resume();
      res.on('end', () => resolve(res.statusCode));
    });
    req.on('error', reject);
  });
}

test('GUI entry point serves the image-restoration window and its modules', async () => {
  const server = createStaticServer({ root: new URL('..', import.meta.url).pathname, port: 0 });
  await new Promise((ok) => server.listen(0, '127.0.0.1', ok));
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    const index = await fetchText(base, '/');
    assert.equal(index.status, 200);
    const html = await index.text();
    assert.match(html, /焦点融合/);
    assert.match(html, /main\.js/);
    // The window must actually offer the required controls: focus group list,
    // ROI editing, weight distribution view, reliability view.
    assert.match(html, /焦点组/);
    assert.match(html, /有效区/);
    assert.match(html, /权重分布/);
    assert.match(html, /可靠度/);

    for (const path of [
      '/src/web/main.js',
      '/src/web/fusion-worker.js',
      '/src/web/style.css',
      '/src/core/fuse.js',
      '/src/core/register.js',
      '/src/core/focus.js',
      '/src/core/png.js',
    ]) {
      const res = await fetchText(base, path);
      assert.equal(res.status, 200, `${path} must be served`);
      assert.match(res.headers.get('content-type') || '', /javascript|text\/css/, `${path} mime`);
    }
  } finally {
    await new Promise((ok) => server.close(ok));
  }
});

test('the server serves the generated demo stack for the GUI', async () => {
  const server = createStaticServer({ root: new URL('..', import.meta.url).pathname, port: 0 });
  await new Promise((ok) => server.listen(0, '127.0.0.1', ok));
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    const manifest = await fetchText(base, '/demo/manifest.json');
    assert.equal(manifest.status, 200);
    const m = await manifest.json();
    assert.ok(m.scenes.length >= 1);
    // Each listed slice must be retrievable as a PNG.
    for (const scene of m.scenes) {
      const first = scene.slices[0];
      const res = await fetchText(base, `/demo/${first}`);
      assert.equal(res.status, 200, `${first} must be served`);
      assert.equal(res.headers.get('content-type'), 'image/png');
      const bytes = new Uint8Array(await res.arrayBuffer());
      // PNG signature.
      assert.deepEqual([...bytes.slice(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
    }
  } finally {
    await new Promise((ok) => server.close(ok));
  }
});

test('the server refuses path traversal and unknown routes', async () => {
  const server = createStaticServer({ root: new URL('..', import.meta.url).pathname, port: 0 });
  await new Promise((ok) => server.listen(0, '127.0.0.1', ok));
  const base = `http://127.0.0.1:${server.address().port}`;
  const port = server.address().port;
  try {
    // Encoded separators survive URL normalisation and reach safeJoin, which
    // must refuse to escape the root. (A literal "/../../x" is normalised to
    // "/x" by the URL parser first, so it can only ever name an in-root file.)
    for (const path of ['/..%2f..%2fetc%2fpasswd', '/%2e%2e%2f%2e%2e%2fpackage.json', '/nope.png']) {
      const status = await rawGet(port, path);
      assert.notEqual(status, 200, `${path} must not resolve`);
      assert.ok(status === 403 || status === 404, `${path} -> ${status}`);
    }
    assert.equal(await rawGet(port, '/src/web/index.html'), 200);
    void base;
  } finally {
    await new Promise((ok) => server.close(ok));
  }
});
