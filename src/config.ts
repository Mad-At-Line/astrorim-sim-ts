/**
 * Constants of the simgenv2 regime (reference/simgenv2.py, lines 41-88).
 * Keep these in sync with the Python file; tests/config.test.ts checks them
 * against values parsed out of simgenv2.py.
 */

export const IMAGE_SIZE = 96;
export const OVERSAMPLE = 4;
export const PIXEL_SCALE = 0.04; // arcsec / detector pixel
export const DEFAULT_ZP = 25.94;

export const THETA_E_VIS_RANGE: readonly [number, number] = [0.9, 1.5];
export const SIGMA_KMS_RANGE: readonly [number, number] = [160.0, 320.0];
export const ELLIPTICITY_RANGE: readonly [number, number] = [0.05, 0.45];
export const SHEAR_MAX = 0.08;

export const EXPTIME_RANGE: readonly [number, number] = [800.0, 1600.0];
export const READ_NOISE_RANGE: readonly [number, number] = [0.8, 1.8];
export const GAIN = 1.5;
export const SKY_ADU_RANGE: readonly [number, number] = [0.06, 0.25];

export const SRC_MAG_RANGE: readonly [number, number] = [21.0, 23.2];
export const LENS_MAG_DELTA_RANGE: readonly [number, number] = [-0.3, 1.2];

export const PSF_TYPE = "GAUSSIAN" as const;
export const PSF_FWHM_ARCSEC: readonly [number, number] = [0.06, 0.1];
export const PSF_ELLIP_MAX = 0.08;

export const N_EXTRA_SOURCES_RANGE: readonly [number, number] = [0, 2];

export const PRNU_RMS = 0.01;
export const SKY_GRADIENT_MAX = 0.02;
export const COSMIC_RAY_PROB = 0.02;
export const COSMIC_RAY_INTENSITY: readonly [number, number] = [50.0, 300.0];

export const ADD_GROUP_HALO_PROB = 0.18;
export const ADD_SUBHALOS_PROB = 0.7;

export const SUPER_SIZE = IMAGE_SIZE * OVERSAMPLE;
export const SUP_PIXEL_SCALE = PIXEL_SCALE / OVERSAMPLE;
export const SUPER_PIX_AREA = SUP_PIXEL_SCALE ** 2;

/** Peak-ratio loop settings (generate_one, lines 343-364). */
export const TARGET_PEAK_RATIO = 0.35;
export const MAX_RATIO_TRIES = 2;
