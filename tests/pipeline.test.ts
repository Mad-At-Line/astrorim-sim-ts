/**
 * Stage-by-stage comparison of the TS renderer against simgenv2 (via the
 * golden files written by reference/make_golden.py). Each case feeds the exact
 * parameters numpy drew into render() and compares every intermediate product.
 *
 * Tolerances are relative to each array's maximum. simgenv2 rounds to float32
 * at several points, so ~1e-7 is the floor; anything near 1e-3 or worse would
 * mean a real porting mistake, not rounding.
 */

import { describe, expect, it } from "vitest";
import * as C from "../src/config";
import { downsample } from "../src/image";
import { rayShoot } from "../src/lens";
import { render, renderSource, superGrid, lensComponents } from "../src/simulate";
import { decode, loadCases, maxAbsDiff, relMaxDiff, strided } from "./golden";

const cases = loadCases();
const S = C.SUPER_SIZE;

describe.each(cases.map((c) => [c.name, c] as const))("golden case %s", (_name, g) => {
  const r = render(g.params);

  it("ray-shooting matches lenstronomy (stride-8 sample, absolute arcsec)", () => {
    const grid = superGrid();
    const { xs, ys } = rayShoot(lensComponents(g.params), grid.X, grid.Y);
    expect(maxAbsDiff(strided(xs, S, 4, 8), decode(g.arrays.xs_s8))).toBeLessThan(1e-12);
    expect(maxAbsDiff(strided(ys, S, 4, 8), decode(g.arrays.ys_s8))).toBeLessThan(1e-12);
  });

  it("source pattern matches build_sources", () => {
    const src = renderSource(g.params.source);
    expect(relMaxDiff(strided(src, S, 2, 4), decode(g.arrays.src_pattern_s4))).toBeLessThan(1e-6);
  });

  it("lens light pattern matches", () => {
    expect(relMaxDiff(strided(r.lensLightSupPattern, S, 2, 4), decode(g.arrays.lens_light_sup_pattern_s4))).toBeLessThan(1e-6);
  });

  it("PSF kernel matches make_psf_kernel_fixed", () => {
    const ref = decode(g.arrays.psf_sup);
    expect(r.psf.kernel.length).toBe(ref.length);
    expect(relMaxDiff(r.psf.kernel, ref)).toBeLessThan(1e-6);
    expect(r.psf.realizedFwhmPix).toBeCloseTo(g.scalars.psf_realized_fwhm_pix, 12);
  });

  it("GT (unlensed, detector-sampled source) matches", () => {
    expect(relMaxDiff(r.GT, decode(g.arrays.GT))).toBeLessThan(1e-5);
  });

  it("lensed source on the super-grid matches (interpolation)", () => {
    expect(relMaxDiff(strided(r.lensedSrcCountsSup, S, 2, 4), decode(g.arrays.lensed_src_counts_sup_s4))).toBeLessThan(1e-5);
  });

  it("PSF-convolved super-grid image matches convolve_fft", () => {
    expect(relMaxDiff(strided(r.imageConvSup, S, 2, 4), decode(g.arrays.image_conv_sup_s4))).toBeLessThan(1e-5);
  });

  it("noiseless detector image LENSED_counts matches", () => {
    expect(relMaxDiff(r.lensedCounts, decode(g.arrays.LENSED_counts))).toBeLessThan(1e-5);
  });

  it("lens-light detector image matches", () => {
    const det = downsample(r.lensCountsSup, S, C.OVERSAMPLE).map(Math.fround);
    expect(relMaxDiff(det, decode(g.arrays.lens_light_det))).toBeLessThan(1e-5);
  });

  it("photometry and the peak-ratio loop match", () => {
    expect(r.tries).toBe(g.scalars.tries);
    expect(r.srcMagFinal).toBeCloseTo(g.scalars.src_mag_final, 12);
    expect(r.lensMagFinal).toBeCloseTo(g.scalars.lens_mag_final, 12);
    expect(r.srcTotalCounts / g.scalars.src_total_counts).toBeCloseTo(1, 12);
    expect(r.lensTotalCounts / g.scalars.lens_total_counts).toBeCloseTo(1, 12);
    expect(r.ratioHistory.length).toBe(g.scalars.ratio_history.length);
    r.ratioHistory.forEach((v, i) => expect(v / g.scalars.ratio_history[i]).toBeCloseTo(1, 5));
  });
});

it("golden set exercises the peak-ratio loop at least once", () => {
  expect(cases.some((c) => c.scalars.tries > 0)).toBe(true);
});
