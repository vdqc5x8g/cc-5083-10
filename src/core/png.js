// Minimal dependency-free PNG codec (8/16-bit, gray/RGB/RGBA/palette, interlace).
// Uses Web CompressionStream/DecompressionStream ("deflate" = zlib wrapper),
// available in Node >=18 and modern browsers.

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf, start = 0, end = buf.length) {
  let c = 0xffffffff;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const PNG_SIG = [137, 80, 78, 71, 13, 10, 26, 10];

function u32(n) {
  return [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
}

function chunk(type, data) {
  const out = new Uint8Array(12 + data.length);
  out.set(u32(data.length), 0);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  const crc = crc32(out, 4, 8 + data.length);
  out.set(u32(crc), 8 + data.length);
  return out;
}

async function streamThrough(bytes, transform) {
  const stream = new Blob([bytes]).stream().pipeThrough(transform);
  const ab = await new Response(stream).arrayBuffer();
  return new Uint8Array(ab);
}

export function zlibDeflate(bytes) {
  return streamThrough(bytes, new CompressionStream('deflate'));
}

export function zlibInflate(bytes) {
  return streamThrough(bytes, new DecompressionStream('deflate'));
}

const FILTER_BPP = (channels, bitDepth) => {
  const bitsPerPixel = channels * bitDepth;
  return Math.max(1, (bitsPerPixel + 7) >> 3);
};

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

function filterRow(type, raw, prev, bpp, rowBytes) {
  const out = new Uint8Array(rowBytes);
  switch (type) {
    case 0:
      out.set(raw.subarray(0, rowBytes));
      break;
    case 1:
      for (let i = 0; i < rowBytes; i++) {
        out[i] = (raw[i] - (i >= bpp ? raw[i - bpp] : 0)) & 255;
      }
      break;
    case 2:
      for (let i = 0; i < rowBytes; i++) {
        out[i] = (raw[i] - (prev ? prev[i] : 0)) & 255;
      }
      break;
    case 3:
      for (let i = 0; i < rowBytes; i++) {
        const left = i >= bpp ? raw[i - bpp] : 0;
        const up = prev ? prev[i] : 0;
        out[i] = (raw[i] - ((left + up) >> 1)) & 255;
      }
      break;
    case 4:
      for (let i = 0; i < rowBytes; i++) {
        const left = i >= bpp ? raw[i - bpp] : 0;
        const up = prev ? prev[i] : 0;
        const ul = i >= bpp && prev ? prev[i - bpp] : 0;
        out[i] = (raw[i] - paeth(left, up, ul)) & 255;
      }
      break;
    default:
      throw new Error(`invalid PNG filter type ${type}`);
  }
  return out;
}

function filterScore(row) {
  let s = 0;
  for (let i = 0; i < row.length; i++) {
    s += row[i] < 128 ? row[i] : 256 - row[i];
  }
  return s;
}

function unfilterRow(type, src, dest, prev, bpp, rowBytes) {
  switch (type) {
    case 0:
      for (let i = 0; i < rowBytes; i++) dest[i] = src[i];
      break;
    case 1:
      for (let i = 0; i < rowBytes; i++) {
        dest[i] = (src[i] + (i >= bpp ? dest[i - bpp] : 0)) & 255;
      }
      break;
    case 2:
      for (let i = 0; i < rowBytes; i++) {
        dest[i] = (src[i] + (prev ? prev[i] : 0)) & 255;
      }
      break;
    case 3:
      for (let i = 0; i < rowBytes; i++) {
        const left = i >= bpp ? dest[i - bpp] : 0;
        const up = prev ? prev[i] : 0;
        dest[i] = (src[i] + ((left + up) >> 1)) & 255;
      }
      break;
    case 4:
      for (let i = 0; i < rowBytes; i++) {
        const left = i >= bpp ? dest[i - bpp] : 0;
        const up = prev ? prev[i] : 0;
        const ul = i >= bpp && prev ? prev[i - bpp] : 0;
        dest[i] = (src[i] + paeth(left, up, ul)) & 255;
      }
      break;
    default:
      throw new Error(`invalid PNG filter type ${type}`);
  }
}

const VALID_DEPTHS = {
  0: [1, 2, 4, 8, 16],
  2: [8, 16],
  3: [1, 2, 4, 8],
  4: [8, 16],
  6: [8, 16],
};

/**
 * Encode an image to PNG.
 * @param {{width:number,height:number,channels:1|2|3|4,bitDepth:8|16,data:Uint16Array|Uint8Array,text?:Array<{keyword:string,value:string}>}} img
 * @returns {Promise<Uint8Array>}
 */
export async function encodePNG(img) {
  const { width, height, channels, bitDepth, data } = img;
  if (!width || !height || width < 1 || height < 1) throw new Error('PNG: invalid dimensions');
  if (![1, 2, 3, 4].includes(channels)) throw new Error('PNG: channels must be 1..4');
  if (bitDepth !== 8 && bitDepth !== 16) throw new Error('PNG: bitDepth must be 8 or 16');
  const colorType = { 1: 0, 2: 4, 3: 2, 4: 6 }[channels];
  if (data.length < width * height * channels) throw new Error('PNG: data buffer too short');
  const max = (1 << bitDepth) - 1;
  const bytesPerSample = bitDepth >> 3;
  const rowBytes = width * channels * bytesPerSample;
  const bpp = FILTER_BPP(channels, bitDepth);
  const rows = new Uint8Array(height * (rowBytes + 1));
  const raw = new Uint8Array(rowBytes);
  const prev = new Uint8Array(rowBytes);
  const filtered = new Uint8Array(rowBytes);
  const candidates = [0, 1, 2, 3, 4].map(() => new Uint8Array(rowBytes));
  let p = 0;
  for (let y = 0; y < height; y++) {
    for (let i = 0; i < rowBytes; i++) raw[i] = 0;
    const base = y * width * channels;
    if (bitDepth === 16) {
      for (let x = 0, q = base, o = 0; x < width * channels; x++, q++, o += 2) {
        const v = Math.max(0, Math.min(65535, data[q] | 0));
        raw[o] = v >> 8;
        raw[o + 1] = v & 255;
      }
    } else {
      for (let x = 0, q = base; x < width * channels; x++, q++) {
        raw[x] = Math.max(0, Math.min(255, data[q] | 0));
      }
    }
    let bestType = 0;
    let bestScore = Infinity;
    for (let t = 0; t < 5; t++) {
      const f = filterRow(t, raw, y > 0 ? prev : null, bpp, rowBytes);
      candidates[t].set(f);
      const score = filterScore(f);
      if (score < bestScore) {
        bestScore = score;
        bestType = t;
      }
    }
    rows[p++] = bestType;
    rows.set(candidates[bestType], p);
    p += rowBytes;
    prev.set(raw);
  }
  const compressed = await zlibDeflate(rows);
  const ihdr = new Uint8Array(13);
  ihdr.set(u32(width), 0);
  ihdr.set(u32(height), 4);
  ihdr[8] = bitDepth;
  ihdr[9] = colorType;
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;
  const chunks = [chunk('IHDR', ihdr)];
  if (img.text) {
    for (const t of img.text) {
      const latin = `${t.keyword}\0${t.value}`;
      let bytes;
      try {
        bytes = new TextEncoder().encode(latin);
      } catch {
        continue;
      }
      chunks.push(chunk('tEXt', bytes));
    }
  }
  chunks.push(chunk('IDAT', compressed));
  chunks.push(chunk('IEND', new Uint8Array(0)));
  let total = 8;
  for (const c of chunks) total += c.length;
  const out = new Uint8Array(total);
  out.set(PNG_SIG, 0);
  let off = 8;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

/**
 * Decode a PNG into uniform RGBA.
 * @param {Uint8Array} bytes
 * @returns {Promise<{width:number,height:number,data:Uint16Array,bitDepth:8|16,max:255|65535,hasAlpha:boolean,info:string}>}
 */
export async function decodePNG(bytes) {
  if (bytes.length < 8 || PNG_SIG.some((v, i) => bytes[i] !== v)) throw new Error('PNG: bad signature');
  let off = 8;
  let header = null;
  const idat = [];
  let palette = null;
  let trns = null;
  while (off + 12 <= bytes.length) {
    const len = (bytes[off] << 24) | (bytes[off + 1] << 16) | (bytes[off + 2] << 8) | bytes[off + 3];
    const type = String.fromCharCode(bytes[off + 4], bytes[off + 5], bytes[off + 6], bytes[off + 7]);
    if (len < 0 || off + 12 + len > bytes.length) throw new Error('PNG: truncated chunk');
    const data = bytes.subarray(off + 8, off + 8 + len);
    const storedCrc = ((bytes[off + 8 + len] << 24) | (bytes[off + 9 + len] << 16) |
      (bytes[off + 10 + len] << 8) | bytes[off + 11 + len]) >>> 0;
    const calc = crc32(bytes, off + 4, off + 8 + len);
    if (storedCrc !== calc) throw new Error(`PNG: CRC mismatch in ${type}`);
    if (type === 'IHDR') {
      header = {
        width: (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3],
        height: (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7],
        bitDepth: data[8],
        colorType: data[9],
        compression: data[10],
        filter: data[11],
        interlace: data[12],
      };
    } else if (type === 'PLTE') {
      palette = new Uint8Array(data);
    } else if (type === 'tRNS') {
      trns = new Uint8Array(data);
    } else if (type === 'IDAT') {
      idat.push(data);
    } else if (type === 'IEND') {
      break;
    }
    off += 12 + len;
  }
  if (!header) throw new Error('PNG: missing IHDR');
  if (header.compression !== 0 || header.filter !== 0) throw new Error('PNG: unsupported compression/filter method');
  if (!VALID_DEPTHS[header.colorType]?.includes(header.bitDepth)) {
    throw new Error(`PNG: unsupported color type ${header.colorType} / depth ${header.bitDepth}`);
  }
  if (header.colorType === 3 && !palette) throw new Error('PNG: palette image without PLTE');
  const { width, height, bitDepth, colorType, interlace } = header;
  const channels = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[colorType];
  const idatBytes = new Uint8Array(idat.reduce((a, b) => a + b.length, 0));
  {
    let p = 0;
    for (const d of idat) { idatBytes.set(d, p); p += d.length; }
  }
  const inflated = await zlibInflate(idatBytes);
  const outMax = 65535;
  const out = new Uint16Array(width * height * 4);
  const hasAlpha = colorType === 4 || colorType === 6 || (colorType === 3 && trns != null) ||
    (colorType === 0 && trns != null) || (colorType === 2 && trns != null);
  const bpp = FILTER_BPP(channels, bitDepth);
  const trnsGray = trns && colorType === 0 ? (trns[0] << 8) | trns[1] : null;
  const trnsRGB = trns && colorType === 2 ? [(trns[0] << 8) | trns[1], (trns[2] << 8) | trns[3], (trns[4] << 8) | trns[5]] : null;
  const passes = interlace === 1
    ? [[0, 0, 8, 8], [4, 0, 8, 8], [0, 4, 4, 8], [2, 0, 4, 4], [0, 2, 2, 4], [1, 0, 2, 2], [0, 1, 1, 2]]
    : [[0, 0, 1, 1]];
  let cursor = 0;
  for (const [gx, gy, sx, sy] of passes) {
    const pw = Math.ceil((width - gx) / sx);
    const ph = Math.ceil((height - gy) / sy);
    if (pw <= 0 || ph <= 0) continue;
    const rowBytes = Math.ceil((pw * channels * bitDepth) / 8);
    let prev = null;
    for (let py = 0; py < ph; py++) {
      if (cursor + 1 + rowBytes > inflated.length) throw new Error('PNG: image data truncated');
      const ft = inflated[cursor++];
      const src = inflated.subarray(cursor, cursor + rowBytes);
      cursor += rowBytes;
      const row = new Uint8Array(rowBytes);
      unfilterRow(ft, src, row, prev, bpp, rowBytes);
      prev = row;
      const y = gy + py * sy;
      for (let px = 0; px < pw; px++) {
        const x = gx + px * sx;
        const dst = (y * width + x) * 4;
        writePixel(row, px, bitDepth, colorType, palette, trns, out, dst, outMax, trnsGray, trnsRGB, channels);
      }
    }
    if (interlace === 0) break;
  }
  return {
    width, height, data: out, bitDepth: 16, max: outMax, hasAlpha,
    info: `PNG ${width}x${height} ${bitDepth}-bit colorType ${colorType}${interlace ? ' interlace' : ''}`,
  };
}

function sampleAt(row, index, bitDepth) {
  if (bitDepth === 8) return row[index];
  if (bitDepth === 16) return (row[index * 2] << 8) | row[index * 2 + 1];
  const bit = index * bitDepth;
  const byte = row[bit >> 3];
  const shift = 8 - bitDepth - (bit & 7);
  return (byte >> shift) & ((1 << bitDepth) - 1);
}

function writePixel(row, px, bitDepth, colorType, palette, trns, out, dst, outMax, trnsGray, trnsRGB, channels) {
  let r, g, b, a = outMax;
  if (bitDepth < 8) {
    const s = sampleAt(row, px, bitDepth);
    const scale = (v, fromMax) => Math.round((v * outMax) / fromMax);
    if (colorType === 3) {
      const idx = s;
      const pi = idx * 3;
      if (!palette || pi + 2 >= palette.length) throw new Error('PNG: palette index out of range');
      r = scale(palette[pi], 255);
      g = scale(palette[pi + 1], 255);
      b = scale(palette[pi + 2], 255);
      if (trns && idx < trns.length) a = scale(trns[idx], 255);
    } else {
      r = g = b = scale(s, (1 << bitDepth) - 1);
    }
  } else if (bitDepth === 8) {
    const base = px * channels;
    if (colorType === 0) {
      r = g = b = Math.round((row[base] * 65535) / 255);
      if (trnsGray != null && row[base] === (trnsGray & 255)) a = 0;
    } else if (colorType === 3) {
      const idx = row[base];
      const pi = idx * 3;
      if (!palette || pi + 2 >= palette.length) throw new Error('PNG: palette index out of range');
      r = Math.round((palette[pi] * 65535) / 255);
      g = Math.round((palette[pi + 1] * 65535) / 255);
      b = Math.round((palette[pi + 2] * 65535) / 255);
      if (trns && idx < trns.length) a = Math.round((trns[idx] * 65535) / 255);
    } else if (colorType === 2) {
      r = Math.round((row[base] * 65535) / 255);
      g = Math.round((row[base + 1] * 65535) / 255);
      b = Math.round((row[base + 2] * 65535) / 255);
      if (trnsRGB && row[base] === (trnsRGB[0] & 255) && row[base + 1] === (trnsRGB[1] & 255) &&
          row[base + 2] === (trnsRGB[2] & 255)) a = 0;
    } else if (colorType === 4) {
      r = g = b = Math.round((row[base] * 65535) / 255);
      a = Math.round((row[base + 1] * 65535) / 255);
    } else {
      r = Math.round((row[base] * 65535) / 255);
      g = Math.round((row[base + 1] * 65535) / 255);
      b = Math.round((row[base + 2] * 65535) / 255);
      a = Math.round((row[base + 3] * 65535) / 255);
    }
    out[dst] = r; out[dst + 1] = g; out[dst + 2] = b; out[dst + 3] = a;
    return;
  } else {
    const base = px * channels * 2;
    if (colorType === 0) {
      r = g = b = (row[base] << 8) | row[base + 1];
      if (trnsGray != null && r === trnsGray) a = 0;
    } else if (colorType === 2) {
      r = (row[base] << 8) | row[base + 1];
      g = (row[base + 2] << 8) | row[base + 3];
      b = (row[base + 4] << 8) | row[base + 5];
      if (trnsRGB && r === trnsRGB[0] && g === trnsRGB[1] && b === trnsRGB[2]) a = 0;
    } else if (colorType === 4) {
      r = g = b = (row[base] << 8) | row[base + 1];
      a = (row[base + 2] << 8) | row[base + 3];
    } else {
      r = (row[base] << 8) | row[base + 1];
      g = (row[base + 2] << 8) | row[base + 3];
      b = (row[base + 4] << 8) | row[base + 5];
      a = (row[base + 6] << 8) | row[base + 7];
    }
    out[dst] = r; out[dst + 1] = g; out[dst + 2] = b; out[dst + 3] = a;
    return;
  }
  out[dst] = r; out[dst + 1] = g; out[dst + 2] = b; out[dst + 3] = a;
}
