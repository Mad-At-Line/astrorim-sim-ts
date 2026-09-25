/**
 * Port of simgen_truth.make_psf_kernel_fixed ([FIX S1] version): Gaussian and
 * Moffat kernels on the true elliptical pixel radius, optional coma, kernel
 * normalised to unit sum and rounded to float32 like the Python version.
 * The AIRY branch is not ported yet (simgenv2 only uses GAUSSIAN).
 */

export type PsfType = "GAUSSIAN" | "MOFFAT" | "AIRY";

export interface PsfArgs {
  fwhm_arcsec: number;
  pixel_scale: number; // arcsec per pixel of the grid the kernel lives on
  type: PsfType;
  ellip: number;
  angle: number;
  beta: number;
  coma_prob: number;
  /** the uniform draw make_psf_kernel_fixed compares against coma_prob */
  coma_u: number;
}

export interface PsfKernel {
  kernel: Float64Array; // row-major size x size, values rounded to float32
  size: number;
  realizedFwhmPix: number;
  comaApplied: boolean;
}

export function makePsfKernel(a: PsfArgs): PsfKernel {
  const fwhmPix = a.fwhm_arcsec / a.pixel_scale;
  const sigmaPix = fwhmPix / 2.3548;
  let k = Math.trunc(Math.max(21, Math.ceil(fwhmPix * 6)));
  if (k % 2 === 0) k += 1;
  const half = Math.floor(k / 2);
  const kern = new Float64Array(k * k);
  const type = a.type.toUpperCase() as PsfType;
  if (type === "AIRY") {
    throw new Error("AIRY PSF not ported yet (simgenv2 uses GAUSSIAN); see README roadmap");
  }
  const q = Math.max(0.001, 1.0 - a.ellip);
  const ca = Math.cos(a.angle);
  const sa = Math.sin(a.angle);
  const alpha = fwhmPix / (2.0 * Math.sqrt(Math.pow(2.0, 1.0 / a.beta) - 1.0));
  for (let r = 0; r < k; r++) {
    const y = r - half; // np.mgrid[:k, :k] - k//2 -> first index is y (rows)
    for (let c = 0; c < k; c++) {
      const x = c - half;
      const xRot = ca * x + sa * y;
      const yRot = -sa * x + ca * y;
      const r2 = q * xRot ** 2 + yRot ** 2 / q;
      kern[r * k + c] =
        type === "GAUSSIAN"
          ? Math.exp((-0.5 * r2) / (sigmaPix ** 2 + 1e-12))
          : Math.pow(1.0 + r2 / alpha ** 2, -a.beta);
    }
  }
  let comaApplied = false;
  if (a.coma_u < a.coma_prob) {
    const coma = new Float64Array(k * k);
    let mx = 0;
    for (let r = 0; r < k; r++) {
      const y = r - half;
      for (let c = 0; c < k; c++) {
        const x = c - half;
        const v = x * Math.exp(-(x ** 2 + y ** 2) / (2 * (sigmaPix * 1.5) ** 2));
        coma[r * k + c] = v;
        if (Math.abs(v) > mx) mx = Math.abs(v);
      }
    }
    if (mx > 0) {
      for (let i = 0; i < kern.length; i++) kern[i] *= 1 + (0.05 * coma[i]) / mx;
      comaApplied = true;
    }
  }
  let s = 0;
  for (let i = 0; i < kern.length; i++) s += kern[i];
  for (let i = 0; i < kern.length; i++) kern[i] = Math.fround(kern[i] / s);
  return { kernel: kern, size: k, realizedFwhmPix: fwhmPix, comaApplied };
}
