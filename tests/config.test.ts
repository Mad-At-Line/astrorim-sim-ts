/**
 * Guards against the TS constants drifting from the Python file: parses the
 * module-level assignments in reference/simgenv2.py and compares.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, it } from "vitest";
import * as C from "../src/config";

const py = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "reference", "simgenv2.py"), "utf8");

function pyValue(name: string): unknown {
  const m = py.match(new RegExp(`^${name}\\s*=\\s*(.+?)\\s*(#.*)?$`, "m"));
  if (!m) throw new Error(`${name} not found in simgenv2.py`);
  const src = m[1].replace(/\(/g, "[").replace(/\)/g, "]").replace(/'/g, '"').replace(/\+(\d)/g, "$1");
  return JSON.parse(src);
}

const NAMES = [
  "IMAGE_SIZE", "OVERSAMPLE", "PIXEL_SCALE", "DEFAULT_ZP", "THETA_E_VIS_RANGE", "SIGMA_KMS_RANGE",
  "ELLIPTICITY_RANGE", "SHEAR_MAX", "EXPTIME_RANGE", "READ_NOISE_RANGE", "GAIN", "SKY_ADU_RANGE",
  "SRC_MAG_RANGE", "LENS_MAG_DELTA_RANGE", "PSF_TYPE", "PSF_FWHM_ARCSEC", "PSF_ELLIP_MAX",
  "N_EXTRA_SOURCES_RANGE", "PRNU_RMS", "SKY_GRADIENT_MAX", "COSMIC_RAY_PROB", "COSMIC_RAY_INTENSITY",
  "ADD_GROUP_HALO_PROB", "ADD_SUBHALOS_PROB",
] as const;

it.each(NAMES)("%s matches simgenv2.py", (name) => {
  expect((C as Record<string, unknown>)[name]).toEqual(pyValue(name));
});
