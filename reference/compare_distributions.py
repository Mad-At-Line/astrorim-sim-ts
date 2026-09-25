"""
compare_distributions.py -- population-level check of the random stages.

The golden tests prove the deterministic pipeline matches simgenv2 for fixed
parameters. The TS port uses its own RNG, so parameter sampling and detector
noise can only be compared statistically: this script simulates N lenses with
the real simgenv2 (via harness.simulate_seeded) and compares them with N FITS
files from the TS generator using two-sample Kolmogorov-Smirnov tests.

usage:
    npm run generate -- --n 300 --seed 12345 --out sims_ts
    python reference/compare_distributions.py sims_ts [n_python]
"""

import glob
import os
import sys

import numpy as np
from astropy.io import fits
from scipy.stats import ks_2samp

import harness as H


def stats_from(hdr, gt, lensed):
    corner = np.concatenate([lensed[:12, :12].ravel(), lensed[-12:, -12:].ravel()])
    return {
        "THETA_E": hdr["THETA_E"],
        "TR_Q": hdr["TR_Q"],
        "TR_GAM": hdr["TR_GAM"],
        "TR_PSFW": hdr["TR_PSFW"],
        "SRCMAG": hdr["SRCMAG"],
        "LENSMAG": hdr["LENSMAG"],
        "TR_LFLX": hdr["TR_LFLX"],
        "TR_CLEAN": hdr["TR_CLEAN"],
        "GT sum": float(gt.sum()),
        "LENSED sum": float(lensed.sum()),
        "LENSED p99": float(np.percentile(lensed, 99)),
        "corner std (noise)": float(np.std(corner)),
    }


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "sims_ts"
    n_py = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    ts = []
    for f in sorted(glob.glob(os.path.join(folder, "*.fits"))):
        with fits.open(f) as h:
            ts.append(stats_from(dict(h[0].header), h["GT"].data.astype(float), h["LENSED"].data.astype(float)))
    py = []
    for seed in range(100_000, 100_000 + n_py):
        p, st, LENSED, _, tr = H.simulate_seeded(seed)
        hdr = {"THETA_E": p["sie"]["theta_E"], "SRCMAG": st["src_mag_final"], "LENSMAG": st["lens_mag_final"],
               "TR_CLEAN": tr["clean"]}
        for name, key in H.sgt.TRUTH_FITS_KEYS.items():
            hdr[key] = float(tr["truth"].get(name, 0.0))
        py.append(stats_from(hdr, st["GT"].astype(float), LENSED.astype(float)))

    print(f"TS files: {len(ts)}   Python simulations: {len(py)}\n")
    print(f"{'quantity':22s} {'median TS':>12s} {'median Py':>12s} {'KS p-value':>11s}")
    worst = 1.0
    for k in ts[0]:
        a = np.array([r[k] for r in ts])
        b = np.array([r[k] for r in py])
        pval = ks_2samp(a, b).pvalue
        worst = min(worst, pval)
        flag = "" if pval > 0.001 else "  <-- differs"
        print(f"{k:22s} {np.median(a):12.5g} {np.median(b):12.5g} {pval:11.3g}{flag}")
    n_tests = len(ts[0])
    print(f"\nsmallest p-value {worst:.3g} over {n_tests} quantities "
          f"(Bonferroni threshold at alpha=0.01: {0.01 / n_tests:.3g})")
    sys.exit(0 if worst > 0.01 / n_tests else 1)


if __name__ == "__main__":
    main()
