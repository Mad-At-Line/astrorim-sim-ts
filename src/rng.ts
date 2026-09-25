/**
 * Seedable random numbers for the simulator.
 *
 * This is NOT numpy's Mersenne Twister, so a TS run with seed 7 does not draw
 * the same lens as `simgenv2` with seed 7. Bit-level agreement is tested on the
 * deterministic pipeline (fixed parameters in, images out); the random parts
 * are tested statistically (tests/rng.test.ts, tests/sampling.test.ts).
 *
 * Generator: xoshiro128** seeded through splitmix32. Doubles use 53 bits.
 * Distributions follow the algorithms numpy's legacy RandomState uses where it
 * matters (Poisson: multiplication method below 10, PTRS above; Beta via
 * Marsaglia-Tsang gammas).
 */

function splitmix32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
    return (z ^ (z >>> 16)) >>> 0;
  };
}

export class Rng {
  private s0: number;
  private s1: number;
  private s2: number;
  private s3: number;
  private spareNormal: number | null = null;

  constructor(seed: number = Date.now()) {
    const sm = splitmix32(Math.floor(seed) % 0x100000000);
    this.s0 = sm();
    this.s1 = sm();
    this.s2 = sm();
    this.s3 = sm();
    if ((this.s0 | this.s1 | this.s2 | this.s3) === 0) this.s0 = 1;
  }

  /** Uniform 32-bit unsigned integer. */
  nextUint32(): number {
    const result = Math.imul(rotl(Math.imul(this.s1, 5) >>> 0, 7), 9) >>> 0;
    const t = (this.s1 << 9) >>> 0;
    this.s2 ^= this.s0;
    this.s3 ^= this.s1;
    this.s1 ^= this.s2;
    this.s0 ^= this.s3;
    this.s2 ^= t;
    this.s3 = rotl(this.s3, 11);
    this.s0 >>>= 0;
    this.s1 >>>= 0;
    this.s2 >>>= 0;
    this.s3 >>>= 0;
    return result;
  }

  /** Uniform double in [0, 1) with 53 random bits (same construction as numpy). */
  random(): number {
    const a = this.nextUint32() >>> 5; // 27 bits
    const b = this.nextUint32() >>> 6; // 26 bits
    return (a * 67108864.0 + b) / 9007199254740992.0;
  }

  /** np.random.rand() */
  rand(): number {
    return this.random();
  }

  /** np.random.uniform(low, high) */
  uniform(low: number, high: number): number {
    return low + (high - low) * this.random();
  }

  /** np.random.randint(low, high): integer in [low, high). */
  randint(low: number, high: number): number {
    if (high <= low) throw new RangeError(`randint: high (${high}) must exceed low (${low})`);
    return low + Math.floor(this.random() * (high - low));
  }

  /** Standard normal via the Marsaglia polar method (numpy legacy gauss). */
  standardNormal(): number {
    if (this.spareNormal !== null) {
      const v = this.spareNormal;
      this.spareNormal = null;
      return v;
    }
    let x1: number, x2: number, r2: number;
    do {
      x1 = 2.0 * this.random() - 1.0;
      x2 = 2.0 * this.random() - 1.0;
      r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 === 0.0);
    const f = Math.sqrt((-2.0 * Math.log(r2)) / r2);
    this.spareNormal = f * x1;
    return f * x2;
  }

  /** np.random.normal(loc, scale) */
  normal(loc: number, scale: number): number {
    return loc + scale * this.standardNormal();
  }

  /** np.random.poisson(lam) */
  poisson(lam: number): number {
    if (!(lam >= 0)) throw new RangeError(`poisson: lam must be >= 0, got ${lam}`);
    if (lam === 0) return 0;
    if (lam >= 10) return this.poissonPtrs(lam);
    // multiplication method (numpy random_poisson_mult)
    const enlam = Math.exp(-lam);
    let x = 0;
    let prod = 1.0;
    for (;;) {
      prod *= this.random();
      if (prod > enlam) x += 1;
      else return x;
    }
  }

  /** Hormann's transformed rejection (PTRS), as in numpy random_poisson_ptrs. */
  private poissonPtrs(lam: number): number {
    const slam = Math.sqrt(lam);
    const loglam = Math.log(lam);
    const b = 0.931 + 2.53 * slam;
    const a = -0.059 + 0.02483 * b;
    const invalpha = 1.1239 + 1.1328 / (b - 3.4);
    const vr = 0.9277 - 3.6224 / (b - 2);
    for (;;) {
      const U = this.random() - 0.5;
      const V = this.random();
      const us = 0.5 - Math.abs(U);
      const k = Math.floor(((2 * a) / us + b) * U + lam + 0.43);
      if (us >= 0.07 && V <= vr) return k;
      if (k < 0 || (us < 0.013 && V > us)) continue;
      if (Math.log(V) + Math.log(invalpha) - Math.log(a / (us * us) + b) <= -lam + k * loglam - loggam(k + 1)) {
        return k;
      }
    }
  }

  /** Standard gamma (shape >= 1) via Marsaglia & Tsang. */
  standardGamma(shape: number): number {
    if (shape < 1) {
      // boost: G(a) = G(a+1) * U^(1/a)
      return this.standardGamma(shape + 1) * Math.pow(this.random(), 1 / shape);
    }
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x: number, v: number;
      do {
        x = this.standardNormal();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = this.random();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  /** np.random.beta(a, b) for a, b > 0 (simgenv2 only uses a=1.5, b=3). */
  beta(a: number, b: number): number {
    const x = this.standardGamma(a);
    const y = this.standardGamma(b);
    return x / (x + y);
  }
}

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

/** log Gamma(x) for x > 0 -- numpy's loggam (Stirling series with recurrence). */
export function loggam(x: number): number {
  const a = [
    8.333333333333333e-2, -2.777777777777778e-3, 7.936507936507937e-4, -5.952380952380952e-4,
    8.417508417508418e-4, -1.917526917526918e-3, 6.410256410256410e-3, -2.955065359477124e-2,
    1.796443723688307e-1, -1.3924322169059e0,
  ];
  if (x === 1.0 || x === 2.0) return 0.0;
  let x0 = x;
  let n = 0;
  if (x <= 7.0) {
    n = Math.floor(7 - x);
    x0 = x + n;
  }
  const x2 = 1.0 / (x0 * x0);
  const xp = 2 * Math.PI;
  let gl0 = a[9];
  for (let k = 8; k >= 0; k--) {
    gl0 *= x2;
    gl0 += a[k];
  }
  let gl = gl0 / x0 + 0.5 * Math.log(xp) + (x0 - 0.5) * Math.log(x0) - x0;
  if (x <= 7.0) {
    for (let k = 1; k <= n; k++) {
      gl -= Math.log(x0 - 1.0);
      x0 -= 1.0;
    }
  }
  return gl;
}
