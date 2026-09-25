/**
 * In-place radix-2 complex FFT and a zero-padded 2-D linear convolution that
 * reproduces astropy.convolution.convolve_fft(image, kernel,
 * normalize_kernel=True) with its defaults (boundary='fill', fill_value=0).
 *
 * Note on astropy's defaults: nan_treatment='interpolate' also divides by the
 * kernel convolved with a weight map, but with a finite fill_value the padding
 * counts as valid data, so that weight map is 1 everywhere and the division is
 * a no-op (up to FFT round-off). The result is a plain zero-padded convolution.
 */

interface Plan {
  n: number;
  rev: Uint32Array;
  cos: Float64Array; // cos(2 pi k / n), k < n/2
  sin: Float64Array; // sin(2 pi k / n)
}

const plans = new Map<number, Plan>();

function getPlan(n: number): Plan {
  let p = plans.get(n);
  if (p) return p;
  if (n & (n - 1)) throw new Error(`FFT size must be a power of two, got ${n}`);
  const bits = Math.round(Math.log2(n));
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    rev[i] = r;
  }
  const half = n >> 1;
  const cos = new Float64Array(Math.max(half, 1));
  const sin = new Float64Array(Math.max(half, 1));
  for (let k = 0; k < half; k++) {
    cos[k] = Math.cos((2 * Math.PI * k) / n);
    sin[k] = Math.sin((2 * Math.PI * k) / n);
  }
  p = { n, rev, cos, sin };
  plans.set(n, p);
  return p;
}

export function fft1d(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length;
  const { rev, cos, sin } = getPlan(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (i < j) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const stride = n / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const cr = cos[k * stride];
        const ci = sign * sin[k * stride];
        const a = i + k;
        const b = a + half;
        const xr = re[b] * cr - im[b] * ci;
        const xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

/** 2-D FFT of an N x N complex array stored row-major (N a power of two). */
export function fft2d(re: Float64Array, im: Float64Array, n: number, inverse: boolean): void {
  const rr = new Float64Array(n);
  const ri = new Float64Array(n);
  for (let r = 0; r < n; r++) {
    const o = r * n;
    for (let c = 0; c < n; c++) {
      rr[c] = re[o + c];
      ri[c] = im[o + c];
    }
    fft1d(rr, ri, inverse);
    for (let c = 0; c < n; c++) {
      re[o + c] = rr[c];
      im[o + c] = ri[c];
    }
  }
  for (let c = 0; c < n; c++) {
    for (let r = 0; r < n; r++) {
      rr[r] = re[r * n + c];
      ri[r] = im[r * n + c];
    }
    fft1d(rr, ri, inverse);
    for (let r = 0; r < n; r++) {
      re[r * n + c] = rr[r];
      im[r * n + c] = ri[r];
    }
  }
}

function nextPow2(v: number): number {
  let p = 1;
  while (p < v) p <<= 1;
  return p;
}

/**
 * A reusable convolver for one kernel: the kernel's FFT is computed once, so
 * convolving several images with the same PSF costs two FFTs each.
 */
export class Convolver {
  private readonly P: number;
  private readonly kRe: Float64Array;
  private readonly kIm: Float64Array;
  private readonly kc: number;

  constructor(
    kernel: Float64Array,
    private readonly ksize: number,
    private readonly imsize: number,
  ) {
    if (ksize % 2 !== 1) throw new Error("Convolver expects an odd kernel size");
    this.P = nextPow2(imsize + ksize - 1);
    this.kc = (ksize - 1) / 2;
    let ksum = 0;
    for (let i = 0; i < kernel.length; i++) ksum += kernel[i];
    const P = this.P;
    this.kRe = new Float64Array(P * P);
    this.kIm = new Float64Array(P * P);
    for (let r = 0; r < ksize; r++) {
      for (let c = 0; c < ksize; c++) this.kRe[r * P + c] = kernel[r * ksize + c] / ksum;
    }
    fft2d(this.kRe, this.kIm, P, false);
  }

  convolve(image: Float64Array): Float64Array {
    const { P, imsize: N, kc } = this;
    const re = new Float64Array(P * P);
    const im = new Float64Array(P * P);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) re[r * P + c] = image[r * N + c];
    }
    fft2d(re, im, P, false);
    for (let i = 0; i < P * P; i++) {
      const a = re[i];
      const b = im[i];
      re[i] = a * this.kRe[i] - b * this.kIm[i];
      im[i] = a * this.kIm[i] + b * this.kRe[i];
    }
    fft2d(re, im, P, true);
    // full linear convolution F[u] = sum_a img[u-a] ker[a]; 'same' output is F[i + kc]
    const out = new Float64Array(N * N);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) out[r * N + c] = re[(r + kc) * P + (c + kc)];
    }
    return out;
  }
}

export function convolveFFT(image: Float64Array, imsize: number, kernel: Float64Array, ksize: number): Float64Array {
  return new Convolver(kernel, ksize, imsize).convolve(image);
}
