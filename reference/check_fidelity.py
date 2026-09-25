"""
check_fidelity.py -- proves harness.py is a faithful decomposition of simgenv2.

For each seed it runs the UNMODIFIED simgenv2.generate_one (which writes a FITS
file) and, separately, harness.simulate_seeded(seed). The GT and LENSED arrays
must be bit-identical and every header value must match exactly. If this passes,
the golden files produced by make_golden.py describe what simgenv2 really does.

usage:  python reference/check_fidelity.py [n_seeds]
"""

import contextlib
import io
import os
import sys
import tempfile

import numpy as np
from astropy.io import fits

import harness as H

sg = H.sg


def run_original(seed: int, outdir: str):
    sg.set_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        path = sg.generate_one(0, outdir)
    with fits.open(path) as hdul:
        hdr = dict(hdul[0].header)
        gt = hdul["GT"].data.copy()
        lensed = hdul["LENSED"].data.copy()
    return hdr, gt, lensed


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seeds = list(range(1, n + 1))
    outdir = tempfile.mkdtemp(prefix="fidelity_")
    failures = 0
    for seed in seeds:
        hdr, gt, lensed = run_original(seed, outdir)
        p, st, LENSED, _, tr = H.simulate_seeded(seed)
        ok_gt = np.array_equal(gt, st["GT"])
        ok_lensed = np.array_equal(lensed, LENSED)
        bad_keys = []
        expect = {
            "THETA_E": p["sie"]["theta_E"], "SIGMA": p["sigma_kms"], "ELLIP": p["sie"]["e"],
            "SHEAR": p["shear"]["g"], "SRCMAG": st["src_mag_final"], "LENSMAG": st["lens_mag_final"],
            "ZLENS": p["z_l"], "ZSRC": p["z_s"], "RN_E": p["detector"]["read_noise"],
            "SKYADU": p["detector"]["sky_adu"], "PSF_FWH": p["psf"]["fwhm_arcsec"],
            "TR_MASK": tr["maskbits"], "TR_CLEAN": tr["clean"],
        }
        for name, key in H.sgt.TRUTH_FITS_KEYS.items():
            expect[key] = float(tr["truth"].get(name, 0.0))
        for k, v in expect.items():
            # FITS cards hold at most 20 characters of value, so compare against
            # the harness value after the same card round-trip astropy applies.
            v_card = fits.Card.fromstring(str(fits.Card(k, v))).value
            if hdr[k] != v_card:
                bad_keys.append((k, hdr[k], v))
        status = "OK " if (ok_gt and ok_lensed and not bad_keys) else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"seed {seed:3d}: {status}  GT={'=' if ok_gt else 'x'} LENSED={'=' if ok_lensed else 'x'} "
              f"tries={st['tries']} nfw={p['nfw'] is not None} subhalos={len(p['subhalos'])} "
              f"{'bad=' + str(bad_keys[:3]) if bad_keys else ''}")
    print(f"\n{len(seeds) - failures}/{len(seeds)} seeds reproduce simgenv2.generate_one exactly")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    os.chdir(H.HERE)
    main()
