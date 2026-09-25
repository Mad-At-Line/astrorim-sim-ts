/**
 * Lens mass models used by simgenv2, ported from lenstronomy 1.14
 * (LensModel/Profiles/{sie,nie,shear,nfw,sis}.py). Each returns the deflection
 * (alpha_x, alpha_y) in arcsec; rayShoot returns beta = theta - sum(alpha),
 * exactly like lenstronomy's SinglePlane.ray_shooting.
 */

export interface SIEKwargs {
  theta_E: number;
  e1: number;
  e2: number;
  center_x: number;
  center_y: number;
}
export interface ShearKwargs {
  gamma1: number;
  gamma2: number;
}
export interface NFWKwargs {
  alpha_Rs: number;
  Rs: number;
  center_x: number;
  center_y: number;
}
export interface SISKwargs {
  theta_E: number;
  center_x: number;
  center_y: number;
}

export type LensComponent =
  | { type: "SIE"; kw: SIEKwargs }
  | { type: "SHEAR"; kw: ShearKwargs }
  | { type: "NFW"; kw: NFWKwargs }
  | { type: "SIS"; kw: SISKwargs };

/** lenstronomy param_util.ellipticity2phi_q */
export function ellipticity2phiQ(e1: number, e2: number): { phi: number; q: number } {
  const phi = Math.atan2(e2, e1) / 2;
  let c = Math.sqrt(e1 * e1 + e2 * e2);
  c = Math.min(c, 0.9999);
  const q = (1 - c) / (1 + c);
  return { phi, q };
}

const SIE_S_SCALE = 0.0000000001;

/** Precomputed SIE constants (lenstronomy NIE._param_conv with s_scale = 1e-10). */
function siePrep(kw: SIEKwargs) {
  const { phi, q: q0 } = ellipticity2phiQ(kw.e1, kw.e2);
  const thetaEMajor = kw.theta_E / Math.sqrt((1.0 + q0 * q0) / (2.0 * q0));
  const b = thetaEMajor * Math.sqrt((1 + q0 * q0) / 2);
  const s = SIE_S_SCALE / Math.sqrt(q0);
  const q = q0 >= 1 ? 0.99999999 : q0; // NIEMajorAxis.derivatives guard
  const sq = Math.sqrt(1.0 - q * q);
  return { b, s, q, sq, cos: Math.cos(phi), sin: Math.sin(phi) };
}

function nfwG(X: number): number {
  const c = 0.000001;
  if (X <= c) X = c;
  if (X < 1) return Math.log(X / 2.0) + (1 / Math.sqrt(1 - X * X)) * Math.acosh(1.0 / X);
  if (X === 1) return 1 + Math.log(1.0 / 2.0);
  return Math.log(X / 2) + (1 / Math.sqrt(X * X - 1)) * Math.acos(1.0 / X);
}

/**
 * Deflection of one component at many points. Accumulates into (ax, ay) so the
 * summation order matches lenstronomy's `f_x += f_x_i`.
 */
export function addDeflection(
  comp: LensComponent,
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  ax: Float64Array,
  ay: Float64Array,
): void {
  const n = x.length;
  switch (comp.type) {
    case "SIE": {
      const kw = comp.kw;
      const { b, s, q, sq, cos, sin } = siePrep(kw);
      const q2 = q * q;
      const bOverSq = b / sq;
      for (let i = 0; i < n; i++) {
        const x_ = x[i] - kw.center_x;
        const y_ = y[i] - kw.center_y;
        // util.rotate(x_, y_, phi)
        const xr = x_ * cos + y_ * sin;
        const yr = -x_ * sin + y_ * cos;
        const psi = Math.sqrt(q2 * (s * s + xr * xr) + yr * yr);
        const fx = bOverSq * Math.atan((sq * xr) / (psi + s));
        const fy = bOverSq * Math.atanh((sq * yr) / (psi + q2 * s));
        // rotate back by -phi: cos(-phi) = cos, sin(-phi) = -sin
        ax[i] += fx * cos + fy * -sin;
        ay[i] += -fx * -sin + fy * cos;
      }
      return;
    }
    case "SHEAR": {
      const { gamma1, gamma2 } = comp.kw;
      for (let i = 0; i < n; i++) {
        ax[i] += gamma1 * x[i] + gamma2 * y[i];
        ay[i] += +gamma2 * x[i] - gamma1 * y[i];
      }
      return;
    }
    case "NFW": {
      const kw = comp.kw;
      const rho0 = kw.alpha_Rs / (4.0 * kw.Rs ** 2 * (1.0 + Math.log(1.0 / 2.0)));
      const Rs = kw.Rs < 0.0000001 ? 0.0000001 : kw.Rs;
      for (let i = 0; i < n; i++) {
        const x_ = x[i] - kw.center_x;
        const y_ = y[i] - kw.center_y;
        const R = Math.max(Math.sqrt(x_ * x_ + y_ * y_), 0.00000001);
        const X = R / Rs;
        const a = (4 * rho0 * Rs * nfwG(X)) / (X * X);
        ax[i] += a * x_;
        ay[i] += a * y_;
      }
      return;
    }
    case "SIS": {
      const kw = comp.kw;
      for (let i = 0; i < n; i++) {
        const x_ = x[i] - kw.center_x;
        const y_ = y[i] - kw.center_y;
        const R = Math.sqrt(x_ * x_ + y_ * y_);
        const a = R > 0 ? kw.theta_E / R : 0;
        ax[i] += a * x_;
        ay[i] += a * y_;
      }
      return;
    }
  }
}

/** lenstronomy LensModel(...).ray_shooting(x, y, kwargs) for a single plane. */
export function rayShoot(
  components: LensComponent[],
  x: ArrayLike<number>,
  y: ArrayLike<number>,
): { xs: Float64Array; ys: Float64Array } {
  const n = x.length;
  const ax = new Float64Array(n);
  const ay = new Float64Array(n);
  for (const c of components) addDeflection(c, x, y, ax, ay);
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = x[i] - ax[i];
    ys[i] = y[i] - ay[i];
  }
  return { xs, ys };
}
