/**
 * Port of simgen_truth.py: truth labels in the model's parameter convention
 * (TRUTH_ORDER), the TR_MASK bitmask and TR_CLEAN flag, and the lens-light
 * amplitude measurement. See the module docstring of simgen_truth.py for the
 * derivations; this file only re-expresses them.
 */

import type { LensComponent } from "./lens";
import { ellipticity2phiQ } from "./lens";

export const TRUTH_ORDER = [
  "b", "q", "phi", "x0", "y0", "gamma", "gamma_phi", "kappa_s", "rs",
  "psf_fwhm", "psf_beta", "psf_e1", "psf_e2",
  "lens_flux", "lens_Re", "lens_n",
] as const;
export type TruthName = (typeof TRUTH_ORDER)[number];

export const TRUTH_FITS_KEYS: Record<TruthName, string> = {
  b: "TR_B", q: "TR_Q", phi: "TR_PHI", x0: "TR_X0", y0: "TR_Y0",
  gamma: "TR_GAM", gamma_phi: "TR_GPHI", kappa_s: "TR_KS", rs: "TR_RS",
  psf_fwhm: "TR_PSFW", psf_beta: "TR_PSFB", psf_e1: "TR_PSFE1", psf_e2: "TR_PSFE2",
  lens_flux: "TR_LFLX", lens_Re: "TR_LRE", lens_n: "TR_LN",
};

export const TRUTH_COMMENTS: Record<TruthName, string> = {
  b: "true theta_E in normalized units",
  q: "true SIE axis ratio",
  phi: "true SIE PA [rad], model frame, mod pi",
  x0: "true lens center x [norm]",
  y0: "true lens center y [norm]",
  gamma: "true external shear amplitude",
  gamma_phi: "true shear PA [rad], mod pi",
  kappa_s: "true NFW kappa_s (0 if no halo)",
  rs: "true NFW rs [norm]",
  psf_fwhm: "realized PSF FWHM [detector px]",
  psf_beta: "true Moffat beta (Moffat PSFs only)",
  psf_e1: "true PSF e1 (model stretch conv.)",
  psf_e2: "true PSF e2 (model stretch conv.)",
  lens_flux: "lens light amp at Re [norm units]",
  lens_Re: "lens light Re [norm]",
  lens_n: "lens light Sersic n",
};

export type Truth = Record<TruthName, number>;
export interface TruthResult {
  truth: Truth;
  maskbits: number;
  clean: number;
}

/** Python's floor-mod: the result takes the sign of the divisor (JS % does not). */
export function pymod(a: number, m: number): number {
  const r = a % m;
  return r !== 0 && (r < 0) !== (m < 0) ? r + m : r;
}

/** _wrap_axis_angle: wrap a period-pi angle into (-pi/2, pi/2]. */
export function wrapAxisAngle(a: number): number {
  a = pymod(a + Math.PI / 2.0, Math.PI) - Math.PI / 2.0;
  if (a <= -Math.PI / 2.0) a += Math.PI;
  return a;
}

/** sim_psf_e_to_model */
export function simPsfEToModel(ellip: number, angle: number): [number, number] {
  const qp = Math.max(0.001, 1.0 - ellip);
  const eMod = (1.0 - qp) / (1.0 + qp);
  return [eMod * Math.cos(2.0 * angle), -eMod * Math.sin(2.0 * angle)];
}

export interface PsfInfo {
  realized_fwhm_pix: number;
  psf_type?: string;
  beta?: number;
  ellip?: number;
  angle?: number;
}
export interface LensLightInfo {
  Re_arcsec?: number | null;
  n_sersic?: number | null;
  amp_at_Re_normalized?: number | null;
}

export function truthFromLenstronomy(
  comps: LensComponent[],
  pixelScale: number,
  imageSize: number,
  psfInfo?: PsfInfo | null,
  lensLightInfo?: LensLightInfo | null,
): TruthResult {
  const halfFov = (imageSize / 2.0) * pixelScale;
  const truth = Object.fromEntries(TRUTH_ORDER.map((k) => [k, 0.0])) as Truth;
  let maskbits = 0;
  let clean = 1;
  const setOk = (name: TruthName, value: number) => {
    truth[name] = value;
    maskbits |= 1 << TRUTH_ORDER.indexOf(name);
  };

  let nSie = 0;
  let nNfw = 0;
  let nOther = 0;
  let nfwCocentred = false;
  let primary: { theta_E: number; e1: number; e2: number; center_x: number; center_y: number } | null = null;

  for (const c of comps) {
    if (c.type === "SIE") {
      nSie += 1;
      if (primary === null) primary = c.kw;
    } else if (c.type === "SHEAR") {
      const { gamma1: g1, gamma2: g2 } = c.kw;
      setOk("gamma", Math.hypot(g1, g2));
      setOk("gamma_phi", wrapAxisAngle(0.5 * Math.atan2(g2, g1)));
    } else if (c.type === "NFW") {
      nNfw += 1;
      const RsArc = c.kw.Rs;
      truth.kappa_s = c.kw.alpha_Rs / (4.0 * RsArc * (1.0 + Math.log(0.5)));
      truth.rs = RsArc / halfFov;
      if (primary !== null) {
        const dxc = c.kw.center_x - primary.center_x;
        const dyc = c.kw.center_y - primary.center_y;
        nfwCocentred = Math.hypot(dxc, dyc) < 1e-6;
      }
    } else {
      nOther += 1;
    }
  }

  if (primary !== null) {
    const { phi, q } = ellipticity2phiQ(primary.e1, primary.e2);
    setOk("b", primary.theta_E / halfFov);
    setOk("q", q);
    setOk("phi", wrapAxisAngle(phi));
    setOk("x0", primary.center_x / halfFov);
    setOk("y0", primary.center_y / halfFov);
  }

  const gammaAlreadySet = ((maskbits >> TRUTH_ORDER.indexOf("gamma")) & 1) === 1;
  if (!gammaAlreadySet) setOk("gamma", 0.0);

  if (nNfw === 0) {
    setOk("kappa_s", 0.0);
    truth.rs = 0.0;
  } else if (nNfw === 1 && nfwCocentred && nOther === 0 && nSie === 1) {
    setOk("kappa_s", truth.kappa_s);
    setOk("rs", truth.rs);
  } else {
    clean = 0;
  }
  if (nSie !== 1 || nOther > 0) clean = 0;

  if (psfInfo) {
    setOk("psf_fwhm", psfInfo.realized_fwhm_pix);
    const ptype = String(psfInfo.psf_type ?? "MOFFAT").toUpperCase();
    if (ptype === "MOFFAT") setOk("psf_beta", psfInfo.beta ?? 3.5);
    else truth.psf_beta = 0.0;
    if (ptype === "AIRY") {
      setOk("psf_e1", 0.0);
      setOk("psf_e2", 0.0);
    } else {
      const [e1m, e2m] = simPsfEToModel(psfInfo.ellip ?? 0.0, psfInfo.angle ?? 0.0);
      setOk("psf_e1", e1m);
      setOk("psf_e2", e2m);
    }
  }

  if (lensLightInfo) {
    if (lensLightInfo.Re_arcsec != null) setOk("lens_Re", lensLightInfo.Re_arcsec / halfFov);
    if (lensLightInfo.n_sersic != null) setOk("lens_n", lensLightInfo.n_sersic);
    if (lensLightInfo.amp_at_Re_normalized != null) setOk("lens_flux", lensLightInfo.amp_at_Re_normalized);
  }
  return { truth, maskbits, clean };
}

/** measure_lens_light_amp_at_Re: median of an annulus at r = Re, divided by norm. */
export function measureLensLightAmpAtRe(
  lensLightDet: Float64Array,
  center: [number, number],
  ReArcsec: number,
  pixelScale: number,
  normFactor: number,
  size = Math.round(Math.sqrt(lensLightDet.length)),
): number | null {
  const H = size;
  const W = size;
  const cx = center[0] / pixelScale + (W - 1) / 2.0;
  const cy = center[1] / pixelScale + (H - 1) / 2.0;
  const RePix = ReArcsec / pixelScale;
  if (!(cx >= 0 && cx < W && cy >= 0 && cy < H) || RePix <= 0.5) return null;
  const vals: number[] = [];
  for (let yy = 0; yy < H; yy++) {
    for (let xx = 0; xx < W; xx++) {
      const r = Math.sqrt((xx - cx) ** 2 + (yy - cy) ** 2);
      if (r > RePix - 0.75 && r < RePix + 0.75) vals.push(lensLightDet[yy * W + xx]);
    }
  }
  if (vals.length < 4) return null;
  vals.sort((a, b) => a - b);
  const m = vals.length;
  // np.median of a float32 array averages the middle pair in float32
  const med = m % 2 ? vals[(m - 1) / 2] : Math.fround(Math.fround(vals[m / 2 - 1] + vals[m / 2]) / 2);
  if (normFactor <= 0) return null;
  return med / normFactor;
}
