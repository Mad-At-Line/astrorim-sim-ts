/**
 * Build a FITS file with the same HDUs and header keys simgenv2 writes, so the
 * AstroRIM training code can read TS-generated simulations unchanged.
 * reference/check_ts_fits.py verifies this against a real simgenv2 file.
 */

import * as C from "./config";
import { writeFits, type Card } from "./fits";
import type { SimulationResult } from "./simulate";
import { TRUTH_COMMENTS, TRUTH_FITS_KEYS, TRUTH_ORDER } from "./truth";

export const GENERATOR = "astrorim-sim-ts 0.1.0";

export function simulationToFits(sim: SimulationResult, date: Date = new Date()): Uint8Array {
  const p = sim.params;
  const extCards: Card[] = [
    { key: "PIXSCALE", value: C.PIXEL_SCALE, comment: "arcsec/pixel" },
    { key: "BUNIT", value: "ADU", comment: "Brightness unit" },
    { key: "OVERSAMP", value: C.OVERSAMPLE, int: true, comment: "super-sampling factor" },
    { key: "ZP", value: C.DEFAULT_ZP, comment: "AB zeropoint (ADU/s)" },
    { key: "EXPTIME", value: p.photometry.exptime, comment: "seconds" },
  ];
  const primary: Card[] = [
    { key: "DATE", value: date.toISOString().replace("Z", ""), comment: "UTC creation time" },
    { key: "GAIN", value: C.GAIN, comment: "e-/ADU" },
    { key: "RN_E", value: p.detector.read_noise, comment: "read noise e- RMS" },
    { key: "SKYADU", value: p.detector.sky_adu, comment: "median sky level (ADU/pixel)" },
    { key: "PSF_FWH", value: p.psf.fwhm_arcsec, comment: "PSF FWHM (arcsec)" },
    { key: "PSF_TYP", value: p.psf.type, comment: "PSF model type" },
    { key: "THETA_E", value: p.sie.theta_E, comment: "Einstein radius (arcsec)" },
    { key: "SIGMA", value: p.sigma_kms, comment: "velocity dispersion km/s used to set thetaE" },
    { key: "ELLIP", value: p.sie.e, comment: "SIE ellipticity" },
    { key: "SHEAR", value: p.shear.g, comment: "external shear amplitude" },
    { key: "SRCMAG", value: sim.srcMagFinal, comment: "source integrated AB mag" },
    { key: "LENSMAG", value: sim.lensMagFinal, comment: "lens integrated AB mag" },
    { key: "ZLENS", value: p.z_l, comment: "lens redshift" },
    { key: "ZSRC", value: p.z_s, comment: "source redshift" },
    { key: "PRNU", value: C.PRNU_RMS, comment: "PRNU RMS (fractional)" },
    { key: "SKYGRAD", value: C.SKY_GRADIENT_MAX, comment: "max fractional sky gradient" },
  ];
  for (const name of TRUTH_ORDER) {
    primary.push({ key: TRUTH_FITS_KEYS[name], value: sim.truth.truth[name], comment: TRUTH_COMMENTS[name] });
  }
  primary.push(
    { key: "TR_MASK", value: sim.truth.maskbits, int: true, comment: "bitmask: TRUTH_ORDER[i] supervisable" },
    { key: "TR_CLEAN", value: sim.truth.clean, int: true, comment: "1 = lens exactly in model family" },
    { key: "TR_SET", value: "simgenv2", comment: "simulator variant" },
    { key: "TR_PS", value: C.PIXEL_SCALE, comment: "sim pixel scale [arcsec/px]" },
    { key: "TR_COMA", value: sim.psf.comaApplied ? 1 : 0, int: true, comment: "coma perturbation applied to PSF" },
    { key: "TR_VER", value: "2.1.0", comment: "truth header schema version" },
    { key: "SIMCODE", value: GENERATOR, comment: "generator (TypeScript port)" },
  );
  return writeFits(primary, [
    { name: "GT", data: sim.GT, width: C.IMAGE_SIZE, height: C.IMAGE_SIZE, cards: extCards },
    { name: "LENSED", data: sim.LENSED, width: C.IMAGE_SIZE, height: C.IMAGE_SIZE, cards: extCards },
  ]);
}
