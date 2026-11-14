#!/usr/bin/env node
// Static server for the focus-fusion image-restoration window.
//
// Serves src/ and the demo/ folder over HTTP so the browser can load the ES
// modules directly (no bundler, no build step). All fusion computation happens
// in the browser using the shared src/core modules; this process only serves
// files.
//
// Usage: node bin/serve.js [--port 8080] [--host 127.0.0.1] [--root .]

import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, resolve, sep } from 'node:path';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.ffi': 'application/x-focus-fusion-image',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

function parse(argv) {
  const out = { port: 8080, host: '127.0.0.1', root: process.cwd() };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--port') out.port = Number(argv[++i]);
    else if (argv[i] === '--host') out.host = argv[++i];
    else if (argv[i] === '--root') out.root = resolve(argv[++i]);
  }
  return out;
}

function safeJoin(root, urlPath) {
  // Decoded slashes (%2f) turn into real separators here, so reject any ".."
  // segment before joining, then verify the result stayed inside the root.
  const segs = urlPath.split(/[/\\]+/).filter(Boolean);
  if (segs.some((s) => s === '..' || s === '.')) return null;
  const file = join(root, ...segs);
  const prefix = root.endsWith(sep) ? root : root + sep;
  if (file !== root && !file.startsWith(prefix)) return null;
  return file;
}

export function createStaticServer({ root = process.cwd(), host = '127.0.0.1', port = 8080 } = {}) {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, `http://${req.headers.host || host}`);
      let path = decodeURIComponent(url.pathname);
      if (path === '/' || path === '') path = '/src/web/index.html';
      const file = safeJoin(root, path);
      if (!file) { res.writeHead(403); res.end('forbidden'); return; }
      const info = await stat(file).catch(() => null);
      if (!info || !info.isFile()) {
        res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
        res.end(`not found: ${path}`);
        return;
      }
      const body = await readFile(file);
      res.writeHead(200, {
        'content-type': MIME[extname(file)] || 'application/octet-stream',
        'cache-control': 'no-cache',
        'content-length': body.length,
      });
      res.end(body);
    } catch (err) {
      res.writeHead(500, { 'content-type': 'text/plain; charset=utf-8' });
      res.end(`server error: ${err.message}`);
    }
  });
  return server;
}

export async function main(argv = []) {
  const opts = parse(argv);
  const server = createStaticServer(opts);
  await new Promise((ok) => server.listen(opts.port, opts.host, ok));
  const addr = server.address();
  const url = `http://${opts.host}:${addr.port}/`;
  console.log(`focus-fusion GUI (image-restoration window): ${url}`);
  console.log(`  serving ${opts.root}`);
  return server;
}

if (process.argv[1] && process.argv[1].endsWith('serve.js')) {
  main(process.argv.slice(2)).catch((err) => {
    console.error(`serve: ${err.message}`);
    process.exit(1);
  });
}
