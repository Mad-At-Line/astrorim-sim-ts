"""
check_ts_fits.py -- confirms FITS files written by the TypeScript generator
(npm run generate) are drop-in compatible with simgenv2's output: same HDU
names, shapes, dtypes and header keys, and physically sane values.

usage:  python reference/check_ts_fits.py sims_ts/
"""

import contextlib
import glob
import io
import os
import sys
import tempfile

import numpy as np
from astropy.io import fits

import harness as H


def reference_file():
    tmp = tempfile.mkdtemp(prefix="simgenv2_ref_")
    H.sg.set_seed(1)
    with contextlib.redirect_stdout(io.StringIO()):
        return H.sg.generate_one(0, tmp)


def describe(path):
    with fits.open(path) as hdul:
        return {
            "names": [h.name for h in hdul],
            "shapes": [None if h.data is None else h.data.shape for h in hdul],
            "dtypes": [None if h.data is None else h.data.dtype.name for h in hdul],
            "primary_keys": set(hdul[0].header.keys()),
            "ext_keys": set(hdul["GT"].header.keys()),
            "hdr": dict(hdul[0].header),
            "gt": hdul["GT"].data.astype(np.float64),
            "lensed": hdul["LENSED"].data.astype(np.float64),
        }


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "sims_ts"
    files = sorted(glob.glob(os.path.join(folder, "*.fits")))
    if not files:
        sys.exit(f"no FITS files in {folder}")
    ref = describe(reference_file())
    problems = 0
    for f in files:
        d = describe(f)
        issues = []
        if d["names"] != ref["names"]:
            issues.append(f"HDU names {d['names']} != {ref['names']}")
        if d["shapes"] != ref["shapes"]:
            issues.append(f"shapes {d['shapes']} != {ref['shapes']}")
        if [t.replace(">", "") if t else t for t in d["dtypes"]] != ref["dtypes"]:
            issues.append(f"dtypes {d['dtypes']} != {ref['dtypes']}")
        missing = ref["primary_keys"] - d["primary_keys"]
        if missing:
            issues.append(f"missing primary keys {sorted(missing)}")
        missing_ext = ref["ext_keys"] - d["ext_keys"]
        if missing_ext:
            issues.append(f"missing extension keys {sorted(missing_ext)}")
        h = d["hdr"]
        if not (0.2 <= h["THETA_E"] <= 3.0):
            issues.append(f"THETA_E out of range: {h['THETA_E']}")
        if not np.all(np.isfinite(d["lensed"])) or not np.all(np.isfinite(d["gt"])):
            issues.append("non-finite pixels")
        if d["gt"].min() < 0:
            issues.append("negative GT pixels")
        if abs(h["TR_B"] * 48 * 0.04 - h["THETA_E"]) > 1e-9:
            issues.append("TR_B inconsistent with THETA_E")
        status = "OK  " if not issues else "FAIL"
        problems += bool(issues)
        print(f"{status} {os.path.basename(f)}  THETA_E={h['THETA_E']:.3f}  "
              f"LENSED[min,max]=[{d['lensed'].min():.2f},{d['lensed'].max():.2f}]  {'; '.join(issues)}")
    extra = sorted(d["primary_keys"] - ref["primary_keys"])
    print(f"\n{len(files) - problems}/{len(files)} files match simgenv2's layout"
          f"{' (extra keys: ' + ', '.join(extra) + ')' if extra else ''}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
