import test from 'node:test';
import assert from 'node:assert/strict';
import { encodePNG, decodePNG } from '../src/core/png.js';

test('16-bit RGBA round-trips exactly', async () => {
  const w = 37;
  const h = 23;
  const data = new Uint16Array(w * h * 4);
  for (let i = 0; i < data.length; i++) data[i] = (i * 7919 + 13) % 65536;
  const png = await encodePNG({ width: w, height: h, channels: 4, bitDepth: 16, data });
  assert.ok(png.length > 8);
  assert.equal(png[0], 137);
  const img = await decodePNG(png);
  assert.equal(img.width, w);
  assert.equal(img.height, h);
  assert.equal(img.bitDepth, 16);
  assert.equal(img.max, 65535);
  for (let i = 0; i < data.length; i++) assert.equal(img.data[i], data[i], `pixel ${i}`);
});

test('8-bit RGBA is scaled to the 16-bit nominal range', async () => {
  const w = 8;
  const h = 4;
  const data = new Uint8Array(w * h * 4);
  for (let i = 0; i < data.length; i++) data[i] = (i * 53) % 256;
  const png = await encodePNG({ width: w, height: h, channels: 4, bitDepth: 8, data });
  const img = await decodePNG(png);
  for (let i = 0; i < data.length; i++) {
    assert.equal(img.data[i], Math.round((data[i] * 65535) / 255));
  }
});

test('8-bit RGB and gray decode to RGBA', async () => {
  const rgb = new Uint8Array(4 * 3 * 3);
  for (let i = 0; i < rgb.length; i++) rgb[i] = (i * 31) % 256;
  const p1 = await decodePNG(await encodePNG({ width: 4, height: 3, channels: 3, bitDepth: 8, data: rgb }));
  assert.equal(p1.data[3], 65535);
  assert.equal(p1.data[0], Math.round((rgb[0] * 65535) / 255));

  const gray = new Uint8Array(5 * 5).fill(200);
  const p2 = await decodePNG(await encodePNG({ width: 5, height: 5, channels: 1, bitDepth: 8, data: gray }));
  const v = Math.round((200 * 65535) / 255);
  assert.equal(p2.data[0], v);
  assert.equal(p2.data[1], v);
  assert.equal(p2.data[2], v);
  assert.equal(p2.data[3], 65535);
});

test('text chunks are written and don\'t break decoding', async () => {
  const data = new Uint16Array(2 * 2 * 4).fill(30000);
  const png = await encodePNG({
    width: 2, height: 2, channels: 4, bitDepth: 16, data,
    text: [{ keyword: 'Software', value: 'focus-fusion test' }],
  });
  const img = await decodePNG(png);
  assert.equal(img.width, 2);
  assert.equal(img.data[0], 30000);
});

test('a corrupted PNG is rejected', async () => {
  const data = new Uint16Array(4 * 4 * 4).fill(1234);
  const png = await encodePNG({ width: 4, height: 4, channels: 4, bitDepth: 16, data });
  const bad = Uint8Array.from(png);
  bad[bad.length - 6] ^= 0xff; // flip a byte inside IEND/IDAT area
  await assert.rejects(async () => decodePNG(bad));
});
