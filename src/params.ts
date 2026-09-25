/**
 * Simulation parameters. The JSON shape is shared with reference/harness.py
 * (schema "astrorim-sim-params/1"): the Python harness records numpy's draws in
 * this shape, and the golden tests feed those exact parameters to the TS renderer.
 *
 * sampleParams() draws from the same distributions as simgenv2.generate_one,
 * in the same order, but with the TS Rng (so values differ from numpy's).
 */

import * as C from "./config";
import { sigmaToThetaEArcsec } from "./cosmology";
import type { PsfType } from "./psf";
import { Rng } from "./rng";
import type { SersicKwargs } from "./light";

export const SCHEMA = "astrorim-sim-params/1";

export interface Clump {
  cx: number; // super-pixel column
  cy: number; // super-pixel row
  amp_rel: number; // amplitude relative to the running max of the source
  sigma: number; // super-pixels
}

export interface SourceParams {
  frac_bulge: number;
  bulge: SersicKwargs;
  disk: SersicKwargs;
  extras: SersicKwargs[];
  clumps: { max_rel: number; items: Clump[] } | null;
  spiral_phase: number | null;
}

export interface SimParams {
  schema: typeof SCHEMA;
  z_l: number;
  z_s: number;
  sigma_kms: number;
  theta_E_raw: number;
  theta_E_clip: { out_of_range: boolean; applied: boolean };
  sie: { theta_E: number; e: number; phi: number; center_x: number; center_y: number };
  shear: { g: number; pa: number };
  nfw: { alpha_Rs: number; Rs: number; dx: number; dy: number } | null;
  subhalos: { theta_E: number; center_x: number; center_y: number }[];
  source: SourceParams;
  psf: {
    type: PsfType;
    fwhm_arcsec: number;
    ellip: number;
    angle: number;
    beta: number;
    coma_prob: number;
    coma_u: number;
  };
  photometry: { exptime: number; zp: number; src_mag: number; lens_mag: number };
  detector: { read_noise: number; sky_adu: number };
  lens_light: {
    R_sersic: number;
    n_sersic: number;
    dx: number;
    dy: number;
    e1_scale: number;
    e1_add: number;
    e2_scale: number;
    e2_add: number;
  };
}

export function sampleParams(rng: Rng): SimParams {
  const [lo, hi] = C.THETA_E_VIS_RANGE;
  const z_l = rng.uniform(0.25, 0.7);
  const z_s = rng.uniform(Math.max(z_l + 0.05, 0.9), 3.0);
  const sigma_kms = rng.uniform(...C.SIGMA_KMS_RANGE);
  const theta_E_raw = sigmaToThetaEArcsec(sigma_kms, z_l, z_s);
  let theta_E = theta_E_raw;
  const clip = { out_of_range: false, applied: false };
  if (theta_E < lo || theta_E > hi) {
    clip.out_of_range = true;
    if (rng.rand() > 0.08) {
      theta_E = Math.min(Math.max(theta_E, lo), hi);
      clip.applied = true;
    }
  }

  const e = rng.uniform(...C.ELLIPTICITY_RANGE);
  const phi = rng.uniform(0, Math.PI);
  const shear_g = rng.uniform(0.0, C.SHEAR_MAX);
  const shear_pa = rng.uniform(0, Math.PI);
  const sie_cx = rng.uniform(-0.04, 0.04);
  const sie_cy = rng.uniform(-0.04, 0.04);

  let nfw: SimParams["nfw"] = null;
  if (rng.rand() < C.ADD_GROUP_HALO_PROB) {
    const Rs = rng.uniform(5.0, 25.0);
    const dx = rng.uniform(-0.3, 0.3);
    const dy = rng.uniform(-0.3, 0.3);
    nfw = { alpha_Rs: 0.5, Rs, dx, dy };
  }

  const subhalos: SimParams["subhalos"] = [];
  if (rng.rand() < C.ADD_SUBHALOS_PROB) {
    const nSub = rng.poisson(1.0);
    for (let i = 0; i < nSub; i++) {
      subhalos.push({
        theta_E: rng.uniform(0.01, 0.06),
        center_x: rng.uniform(-0.5 * theta_E, 0.5 * theta_E),
        center_y: rng.uniform(-0.5 * theta_E, 0.5 * theta_E),
      });
    }
  }

  const source = sampleSource(rng, theta_E);

  const exptime = rng.uniform(...C.EXPTIME_RANGE);
  const psf_fwhm = rng.uniform(...C.PSF_FWHM_ARCSEC);
  const psf_ellip = rng.uniform(0.0, C.PSF_ELLIP_MAX);
  const psf_angle = rng.uniform(0, 2 * Math.PI);
  const coma_u = rng.rand();
  const read_noise = rng.uniform(...C.READ_NOISE_RANGE);
  const sky_adu = rng.uniform(...C.SKY_ADU_RANGE);
  const src_mag = rng.uniform(...C.SRC_MAG_RANGE);
  const lens_mag = src_mag + rng.uniform(...C.LENS_MAG_DELTA_RANGE);

  const ll = {
    R_sersic: rng.uniform(0.25, 1.0),
    n_sersic: rng.uniform(3.0, 5.0),
    dx: rng.uniform(-0.02, 0.02),
    dy: rng.uniform(-0.02, 0.02),
    e1_scale: rng.uniform(0.6, 1.0),
    e1_add: rng.uniform(-0.05, 0.05),
    e2_scale: rng.uniform(0.6, 1.0),
    e2_add: rng.uniform(-0.05, 0.05),
  };

  return {
    schema: SCHEMA,
    z_l,
    z_s,
    sigma_kms,
    theta_E_raw,
    theta_E_clip: clip,
    sie: { theta_E, e, phi, center_x: sie_cx, center_y: sie_cy },
    shear: { g: shear_g, pa: shear_pa },
    nfw,
    subhalos,
    source,
    psf: {
      type: C.PSF_TYPE,
      fwhm_arcsec: psf_fwhm,
      ellip: psf_ellip,
      angle: psf_angle,
      beta: 3.5,
      coma_prob: 0.0,
      coma_u,
    },
    photometry: { exptime, zp: C.DEFAULT_ZP, src_mag, lens_mag },
    detector: { read_noise, sky_adu },
    lens_light: ll,
  };
}

function sampleSource(rng: Rng, theta_E: number): SourceParams {
  const cx = rng.uniform(-0.15 * theta_E, 0.15 * theta_E);
  const cy = rng.uniform(-0.15 * theta_E, 0.15 * theta_E);
  const frac_bulge = rng.beta(1.5, 3.0);
  const Rb = rng.uniform(0.03, 0.12);
  const Rd = Rb * rng.uniform(1.6, 3.5);
  const n_bulge = rng.uniform(2.5, 4.5);
  const amp_b = rng.uniform(1.0, 4.0);
  const amp_d = amp_b * rng.uniform(0.3, 1.2);
  const phi = rng.uniform(0, Math.PI);
  const e = rng.uniform(0.0, 0.6);
  const e1 = e * Math.cos(2 * phi);
  const e2 = e * Math.sin(2 * phi);
  const bulge: SersicKwargs = { amp: amp_b, R_sersic: Rb, n_sersic: n_bulge, e1, e2, center_x: cx, center_y: cy };
  const disk: SersicKwargs = {
    amp: amp_d,
    R_sersic: Rd,
    n_sersic: 1.0,
    e1: e1 * rng.uniform(0.8, 1.0),
    e2: e2 * rng.uniform(0.8, 1.0),
    center_x: cx + rng.uniform(-0.02, 0.02),
    center_y: cy + rng.uniform(-0.02, 0.02),
  };

  const extras: SersicKwargs[] = [];
  const nExtra = rng.randint(C.N_EXTRA_SOURCES_RANGE[0], C.N_EXTRA_SOURCES_RANGE[1] + 1);
  for (let i = 0; i < nExtra; i++) {
    const cx2 = rng.uniform(-0.9 * theta_E, 0.9 * theta_E);
    const cy2 = rng.uniform(-0.9 * theta_E, 0.9 * theta_E);
    extras.push({
      amp: rng.uniform(0.3, 2.0),
      R_sersic: rng.uniform(0.02, 0.1),
      n_sersic: rng.uniform(0.8, 3.5),
      e1: rng.uniform(-0.4, 0.4),
      e2: rng.uniform(-0.4, 0.4),
      center_x: cx2,
      center_y: cy2,
    });
  }

  let clumps: SourceParams["clumps"] = null;
  if (rng.rand() < 0.75) {
    const n = rng.randint(1, 6);
    const max_rel = rng.uniform(0.2, 0.8);
    const sig: [number, number] = [C.OVERSAMPLE * 0.4, C.OVERSAMPLE * 4.0];
    const S = C.SUPER_SIZE;
    const items: Clump[] = [];
    for (let i = 0; i < n; i++) {
      items.push({
        cx: rng.uniform(0.25 * S, 0.75 * S),
        cy: rng.uniform(0.25 * S, 0.75 * S),
        amp_rel: rng.uniform(0.02, max_rel),
        sigma: rng.uniform(...sig),
      });
    }
    clumps = { max_rel, items };
  }

  let spiral_phase: number | null = null;
  if (rng.rand() < 0.35) spiral_phase = rng.uniform(0, 2 * Math.PI);

  return { frac_bulge, bulge, disk, extras, clumps, spiral_phase };
}
