/**
 * Minimal FITS writer: an empty primary HDU with header cards plus 2-D
 * float32 IMAGE extensions -- the layout simgenv2 writes (Primary + GT + LENSED).
 * Works in Node and the browser (returns a Uint8Array).
 */

export type CardValue = number | string | boolean;
export interface Card {
  key: string;
  value: CardValue;
  comment?: string;
  /** force a FITS integer (otherwise numbers are written as reals) */
  int?: boolean;
}

export interface ImageHDU {
  name: string;
  data: ArrayLike<number>; // row-major, rows = NAXIS2, cols = NAXIS1
  width: number;
  height: number;
  cards?: Card[];
}

const BLOCK = 2880;

/** toPrecision(p) rewritten as a FITS real: '.' always present, 'E+NN' exponent. */
function realAt(v: number, p: number): string {
  let s = v.toPrecision(p);
  let exp = "";
  if (s.includes("e")) {
    const [m, e] = s.split("e");
    const en = Number(e);
    s = m;
    exp = `E${en < 0 ? "-" : "+"}${String(Math.abs(en)).padStart(2, "0")}`;
  }
  if (s.includes(".")) s = s.replace(/0+$/, "").replace(/\.$/, ".0");
  else s = `${s}.0`;
  return s + exp;
}

/** Shortest representation that round-trips exactly within the 20-column value field. */
export function formatReal(v: number): string {
  if (!Number.isFinite(v)) throw new Error(`FITS cannot store non-finite value ${v}`);
  let best: string | null = null;
  for (let p = 1; p <= 17; p++) {
    const s = realAt(v, p);
    if (s.length > 20) continue;
    best = s;
    if (Number(s) === v) return s;
  }
  return best ?? realAt(v, 10);
}

function formatValue(c: Card): string {
  const v = c.value;
  if (typeof v === "boolean") return (v ? "T" : "F").padStart(20);
  if (typeof v === "string") {
    const esc = v.replace(/'/g, "''");
    return `'${esc.padEnd(8)}'`.padEnd(20);
  }
  if (c.int) {
    if (!Number.isInteger(v)) throw new Error(`card ${c.key}: ${v} is not an integer`);
    return String(v).padStart(20);
  }
  return formatReal(v).padStart(20);
}

function cardString(c: Card): string {
  const key = c.key.toUpperCase();
  if (key.length > 8) throw new Error(`FITS keyword too long: ${key}`);
  let s = `${key.padEnd(8)}= ${formatValue(c)}`;
  if (c.comment) s += ` / ${c.comment}`;
  if (s.length > 80) s = s.slice(0, 80);
  return s.padEnd(80);
}

function headerBytes(cards: Card[]): Uint8Array {
  const lines = cards.map(cardString);
  lines.push("END".padEnd(80));
  const text = lines.join("");
  const len = Math.ceil(text.length / BLOCK) * BLOCK;
  const out = new Uint8Array(len).fill(0x20);
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code < 0x20 || code > 0x7e) throw new Error(`non-ASCII character in FITS header: ${text[i]}`);
    out[i] = code;
  }
  return out;
}

function dataBytes(data: ArrayLike<number>): Uint8Array {
  const n = data.length * 4;
  const len = Math.ceil(n / BLOCK) * BLOCK;
  const out = new Uint8Array(len);
  const view = new DataView(out.buffer);
  for (let i = 0; i < data.length; i++) view.setFloat32(i * 4, data[i], false); // big-endian
  return out;
}

export function writeFits(primaryCards: Card[], images: ImageHDU[]): Uint8Array {
  const parts: Uint8Array[] = [];
  parts.push(
    headerBytes([
      { key: "SIMPLE", value: true, comment: "conforms to FITS standard" },
      { key: "BITPIX", value: 8, int: true, comment: "array data type" },
      { key: "NAXIS", value: 0, int: true, comment: "number of array dimensions" },
      { key: "EXTEND", value: true },
      ...primaryCards,
    ]),
  );
  for (const im of images) {
    if (im.data.length !== im.width * im.height) throw new Error(`HDU ${im.name}: data length mismatch`);
    parts.push(
      headerBytes([
        { key: "XTENSION", value: "IMAGE", comment: "Image extension" },
        { key: "BITPIX", value: -32, int: true, comment: "array data type" },
        { key: "NAXIS", value: 2, int: true, comment: "number of array dimensions" },
        { key: "NAXIS1", value: im.width, int: true },
        { key: "NAXIS2", value: im.height, int: true },
        { key: "PCOUNT", value: 0, int: true, comment: "number of parameters" },
        { key: "GCOUNT", value: 1, int: true, comment: "number of groups" },
        { key: "EXTNAME", value: im.name, comment: "extension name" },
        ...(im.cards ?? []),
      ]),
    );
    parts.push(dataBytes(im.data));
  }
  const total = parts.reduce((s, p) => s + p.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}
