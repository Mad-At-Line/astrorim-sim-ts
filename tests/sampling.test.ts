/**
 * sampleParams() must draw from the same distributions as simgenv2.
 * These checks use branch frequencies and ranges that follow directly from the
 * constants in generate_one; tests/golden/manifest.json records the Python side.
 */

import { describe, expect, it } from "vitest";
import * as C from "../src/config";
import { sampleParams } from "../src/params";
import { Rng } from "../src/rng";
import { applyNoise, render } from "../src/simulate";

const N = 4000;
const rng = new Rng(2026);
const samples = Array.from({ length: N }, () => sampleParams(rng));
const frac = (f: (p: (typeof samples)[number]) => boolean) => samples.filter(f).length / N;
const tol = (p: number) => 5 * Math.sqrt((p * (1 - p)) / N);

describe("parameter sampling matches simgenv2's priors", () => {
  it("branch probabilities", () => {
    expect(Math.abs(frac((p) => p.nfw !== null) - C.ADD_GROUP_HALO_PROB)).toBeLessThan(tol(0.18));
    expect(Math.abs(frac((p) => p.source.clumps !== null) - 0.75)).toBeLessThan(tol(0.75));
    expect(Math.abs(frac((p) => p.source.spiral_phase !== null) - 0.35)).toBeLessThan(tol(0.35));
    const out = samples.filter((p) => p.theta_E_clip.out_of_range);
    const kept = out.filter((p) => !p.theta_E_clip.applied).length / out.length;
    expect(Math.abs(kept - 0.08)).toBeLessThan(5 * Math.sqrt((0.08 * 0.92) / out.length));
  });

  it("subhalo count: P(any subhalo draw) * Poisson(1)", () => {
    // P(n = 0) = (1 - 0.7) + 0.7 * e^-1
    const p0 = 1 - C.ADD_SUBHALOS_PROB + C.ADD_SUBHALOS_PROB * Math.exp(-1);
    expect(Math.abs(frac((p) => p.subhalos.length === 0) - p0)).toBeLessThan(tol(p0));
  });

  it("parameter ranges", () => {
    for (const p of samples) {
      expect(p.z_l).toBeGreaterThanOrEqual(0.25);
      expect(p.z_l).toBeLessThan(0.7);
      expect(p.z_s).toBeGreaterThanOrEqual(Math.max(p.z_l + 0.05, 0.9));
      expect(p.z_s).toBeLessThan(3.0);
      expect(p.sie.e).toBeGreaterThanOrEqual(C.ELLIPTICITY_RANGE[0]);
      expect(p.sie.e).toBeLessThan(C.ELLIPTICITY_RANGE[1]);
      expect(p.shear.g).toBeLessThan(C.SHEAR_MAX);
      expect(p.source.extras.length).toBeLessThanOrEqual(2);
      expect(p.photometry.lens_mag - p.photometry.src_mag).toBeGreaterThanOrEqual(C.LENS_MAG_DELTA_RANGE[0]);
      if (p.theta_E_clip.applied) expect([0.9, 1.5]).toContain(p.sie.theta_E);
    }
  });

  it("documents the theta_E pile-up at the clip limits (behaviour inherited from simgenv2)", () => {
    // ~65% of (sigma, z_l, z_s) draws give theta_E outside [0.9, 1.5]; 92% of those
    // are clipped, so a majority of lenses sit exactly on 0.9" or 1.5".
    const atLimit = frac((p) => p.theta_E_clip.applied);
    expect(atLimit).toBeGreaterThan(0.5);
    expect(atLimit).toBeLessThan(0.7);
  });
});

describe("noise stage", () => {
  it("is unbiased: mean(LENSED - noiseless - sky) ~ 0 over many realisations", () => {
    const r = render(samples[0]);
    const nr = new Rng(7);
    let sum = 0;
    let count = 0;
    for (let k = 0; k < 20; k++) {
      const n = applyNoise(r.lensedCounts, samples[0].detector.read_noise, samples[0].detector.sky_adu, nr);
      if (n.cosmicHits.length) continue;
      for (let i = 0; i < n.LENSED.length; i++) {
        sum += n.LENSED[i] - (r.lensedCounts[i] + n.skyMap[i]);
        count++;
      }
    }
    const meanSignal = r.lensedCounts.reduce((a, b) => a + b, 0) / r.lensedCounts.length;
    expect(Math.abs(sum / count)).toBeLessThan(0.01 * Math.max(meanSignal, 1));
  });
});
