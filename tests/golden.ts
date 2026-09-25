/** Helpers for reading the Python-generated golden files in tests/golden/. */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { SimParams } from "../src/params";

const HERE = dirname(fileURLToPath(import.meta.url));
export const GOLDEN_DIR = join(HERE, "golden");

interface Encoded {
  dtype: "float32" | "float64";
  shape: number[];
  b64: string;
}

export function decode(e: Encoded): Float64Array {
  const buf = Buffer.from(e.b64, "base64");
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const src = e.dtype === "float32" ? new Float32Array(ab) : new Float64Array(ab);
  return Float64Array.from(src);
}

export interface GoldenCase {
  name: string;
  seed?: number;
  params: SimParams;
  scalars: {
    theta_E_raw: number;
    src_mag_final: number;
    lens_mag_final: number;
    src_total_counts: number;
    lens_total_counts: number;
    ratio_history: number[];
    tries: number;
    psf_realized_fwhm_pix: number;
    psf_coma_applied: boolean;
    ll_center_x: number;
    ll_center_y: number;
    realized_fwhm_arcsec: number;
  };
  arrays: Record<string, Encoded>;
  truth?: {
    values: Record<string, number>;
    maskbits: number;
    clean: number;
    norm_m: number;
    ll_amp: number | null;
  };
}

export function loadManifest(): { cases: { file: string; coverage: unknown }[]; versions: Record<string, string> } {
  return JSON.parse(readFileSync(join(GOLDEN_DIR, "manifest.json"), "utf8"));
}

export function loadCases(): GoldenCase[] {
  return loadManifest().cases.map((c) => JSON.parse(readFileSync(join(GOLDEN_DIR, c.file), "utf8")));
}

export function loadUnits(): any {
  return JSON.parse(readFileSync(join(GOLDEN_DIR, "units.json"), "utf8"));
}

/** max |a - b| / max |b| -- error relative to the reference's dynamic range. */
export function relMaxDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) throw new Error(`length mismatch ${a.length} vs ${b.length}`);
  let d = 0;
  let m = 0;
  for (let i = 0; i < b.length; i++) {
    d = Math.max(d, Math.abs(a[i] - b[i]));
    m = Math.max(m, Math.abs(b[i]));
  }
  return m === 0 ? d : d / m;
}

export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) throw new Error(`length mismatch ${a.length} vs ${b.length}`);
  let d = 0;
  for (let i = 0; i < b.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
  return d;
}

/** Take the stride sample used by make_golden.py (start, step) from an n x n array. */
export function strided(arr: Float64Array, n: number, start: number, step: number): Float64Array {
  const idx: number[] = [];
  for (let i = start; i < n; i += step) idx.push(i);
  const out = new Float64Array(idx.length * idx.length);
  let k = 0;
  for (const r of idx) for (const c of idx) out[k++] = arr[r * n + c];
  return out;
}
