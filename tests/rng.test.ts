/**
 * The TS generator is not numpy's, so the random stages are checked
 * statistically: moments of each distribution against their analytic values.
 * Sample sizes are chosen so a correct implementation fails < 1e-6 of the time.
 */

import { describe, expect, it } from "vitest";
import { loggam, Rng } from "../src/rng";

function moments(xs: number[]): { mean: number; variance: number } {
  const n = xs.length;
  const mean = xs.reduce((a, b) => a + b, 0) / n;
  const variance = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1);
  return { mean, variance };
}

const draw = (n: number, f: () => number) => Array.from({ length: n }, f);

describe("Rng", () => {
  it("is reproducible for a fixed seed and differs between seeds", () => {
    const a = new Rng(42);
    const b = new Rng(42);
    const c = new Rng(43);
    const sa = draw(5, () => a.random());
    expect(draw(5, () => b.random())).toEqual(sa);
    expect(draw(5, () => c.random())).not.toEqual(sa);
  });

  it("uniform in [low, high) with the right moments", () => {
    const r = new Rng(1);
    const xs = draw(200_000, () => r.uniform(2, 5));
    expect(xs.reduce((a, b) => Math.min(a, b))).toBeGreaterThanOrEqual(2);
    expect(xs.reduce((a, b) => Math.max(a, b))).toBeLessThan(5);
    const m = moments(xs);
    expect(Math.abs(m.mean - 3.5)).toBeLessThan(5 * Math.sqrt(0.75 / 200_000));
    expect(m.variance).toBeCloseTo(0.75, 2);
  });

  it("randint covers [low, high) uniformly", () => {
    const r = new Rng(2);
    const counts = new Array(5).fill(0);
    for (let i = 0; i < 100_000; i++) counts[r.randint(1, 6) - 1]++;
    for (const c of counts) expect(Math.abs(c - 20_000)).toBeLessThan(5 * Math.sqrt(100_000 * 0.2 * 0.8));
  });

  it("normal(loc, scale)", () => {
    const r = new Rng(3);
    const m = moments(draw(200_000, () => r.normal(1.5, 0.8)));
    expect(Math.abs(m.mean - 1.5)).toBeLessThan(5 * (0.8 / Math.sqrt(200_000)));
    expect(m.variance / 0.64).toBeCloseTo(1, 1);
  });

  it.each([0.3, 1.0, 4.0, 9.99, 10.0, 37.5, 500.0, 25_000.0])("poisson(%f): mean = variance = lambda", (lam) => {
    const r = new Rng(4 + lam);
    const n = 100_000;
    const xs = draw(n, () => r.poisson(lam));
    expect(xs.every((x) => Number.isInteger(x) && x >= 0)).toBe(true);
    const m = moments(xs);
    expect(Math.abs(m.mean - lam)).toBeLessThan(5 * Math.sqrt(lam / n));
    expect(Math.abs(m.variance / lam - 1)).toBeLessThan(5 * Math.sqrt(2 / n) + 0.01);
  });

  it("beta(1.5, 3) has the right mean and variance", () => {
    const r = new Rng(5);
    const a = 1.5;
    const b = 3.0;
    const m = moments(draw(200_000, () => r.beta(a, b)));
    const mean = a / (a + b);
    const variance = (a * b) / ((a + b) ** 2 * (a + b + 1));
    expect(Math.abs(m.mean - mean)).toBeLessThan(5 * Math.sqrt(variance / 200_000));
    expect(m.variance / variance).toBeCloseTo(1, 1);
  });

  it("loggam matches known values of log Gamma", () => {
    expect(loggam(1)).toBe(0);
    expect(loggam(2)).toBe(0);
    expect(loggam(5)).toBeCloseTo(Math.log(24), 12);
    expect(loggam(0.5)).toBeCloseTo(0.5 * Math.log(Math.PI), 12);
    expect(loggam(101)).toBeCloseTo(363.73937555556347, 9);
  });
});
