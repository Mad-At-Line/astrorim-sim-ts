/**
 * Elliptical Sersic surface brightness, ported from lenstronomy 1.14
 * (LightModel/Profiles/sersic.py SersicElliptic + LensModel/Profiles/sersic_utils.py).
 * lenstronomy's default convention is the *product-average* radius
 * (sersic_major_axis=False), implemented by param_util.transform_e1e2_product_average.
 */

export interface SersicKwargs {
  amp: number;
  R_sersic: number;
  n_sersic: number;
  e1: number;
  e2: number;
  center_x: number;
  center_y: number;
}

const SMOOTHING = 0.0001; // SersicUtil default
const MAX_R_FRAC = 1000.0;

/** SersicUtil.b_n: the 1.9992 n - 0.3271 approximation, floored at 1e-5. */
export function sersicBn(n: number): number {
  return Math.max(1.9992 * n - 0.3271, 0.00001);
}

/**
 * Evaluate the profile at every (x[i], y[i]) and write amp * I(R) into out
 * (or add it when `accumulate` is true).
 */
export function sersicEllipse(
  kw: SersicKwargs,
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  out: Float64Array,
  accumulate = false,
): Float64Array {
  const Rs = Math.max(SMOOTHING, Math.max(0, kw.R_sersic));
  const n = kw.n_sersic;
  const bn = sersicBn(n);
  const invN = 1.0 / n;
  const { e1, e2 } = kw;
  const norm = Math.sqrt(Math.max(Math.abs(1 - e1 * e1 - e2 * e2), 0.000001));
  for (let i = 0; i < x.length; i++) {
    const xs = x[i] - kw.center_x;
    const ys = y[i] - kw.center_y;
    const x_ = ((1 - e1) * xs - e2 * ys) / norm;
    const y_ = (-e2 * xs + (1 + e1) * ys) / norm;
    const R = Math.max(SMOOTHING, Math.sqrt(x_ * x_ + y_ * y_));
    const Rfrac = R / Rs;
    let v = 0;
    if (Rfrac <= MAX_R_FRAC) v = Math.exp(-bn * (Math.pow(Rfrac, invN) - 1.0));
    if (!Number.isFinite(v)) v = Number.isNaN(v) ? 0 : v > 0 ? Number.MAX_VALUE : -Number.MAX_VALUE; // np.nan_to_num
    const val = kw.amp * v;
    if (accumulate) out[i] += val;
    else out[i] = val;
  }
  return out;
}
