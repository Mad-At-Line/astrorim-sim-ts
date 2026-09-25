"""
harness.py -- splits simgenv2.generate_one into three pieces so the
TypeScript port can be checked stage by stage:

    sample_params()          every random draw generate_one makes, in the same
                             order, recorded into a JSON-serialisable dict
    render(params)           the deterministic part (no RNG): source, lensing,
                             lens light, PSF, detector sampling, peak-ratio loop
    apply_noise_numpy(...)   the detector-noise stage, using np.random exactly
                             like generate_one

`check_fidelity.py` proves that seeding numpy and running
sample_params -> render -> apply_noise_numpy reproduces the FITS files written
by the untouched simgenv2.generate_one bit-for-bit. Only after that check passes
are the golden files (make_golden.py) trusted as a reference for the TS port.

simgenv2.py and simgen_truth.py sit next to this file, unmodified.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _import_simgenv2():
    # simgenv2 creates its hard-coded Windows OUTPUT_DIR at import time; do that
    # inside a throwaway directory so the repo is not polluted.
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="simgenv2_import_")
    os.chdir(tmp)
    try:
        import simgenv2  # noqa: F401
    finally:
        os.chdir(cwd)
    return sys.modules["simgenv2"]


sg = _import_simgenv2()
import simgen_truth as sgt  # noqa: E402
from lenstronomy.LensModel.lens_model import LensModel  # noqa: E402
from lenstronomy.LightModel.light_model import LightModel  # noqa: E402

SCHEMA = "astrorim-sim-params/1"


# ---------------------------------------------------------------------------
# 1. Parameter sampling (mirrors the RNG call order of simgenv2.generate_one)
# ---------------------------------------------------------------------------

def sample_params() -> dict:
    R = np.random
    lo, hi = sg.THETA_E_VIS_RANGE

    z_l = float(R.uniform(0.25, 0.7))
    z_s = float(R.uniform(max(z_l + 0.05, 0.9), 3.0))
    sigma_kms = float(R.uniform(*sg.SIGMA_KMS_RANGE))
    theta_E_raw = float(sg.sigma_to_thetaE_arcsec(sigma_kms, z_l, z_s))
    theta_E = theta_E_raw
    clip = {"out_of_range": False, "applied": False}
    if theta_E < lo or theta_E > hi:
        clip["out_of_range"] = True
        if R.rand() > 0.08:
            theta_E = float(np.clip(theta_E, lo, hi))
            clip["applied"] = True

    e = float(R.uniform(*sg.ELLIPTICITY_RANGE))
    phi = float(R.uniform(0, np.pi))
    shear_g = float(R.uniform(0.0, sg.SHEAR_MAX))
    shear_pa = float(R.uniform(0, np.pi))
    sie_cx = float(R.uniform(-0.04, 0.04))
    sie_cy = float(R.uniform(-0.04, 0.04))

    nfw = None
    if R.rand() < sg.ADD_GROUP_HALO_PROB:
        Rs = float(R.uniform(5.0, 25.0))
        dx = float(R.uniform(-0.3, 0.3))
        dy = float(R.uniform(-0.3, 0.3))
        nfw = {"alpha_Rs": 0.5, "Rs": Rs, "dx": dx, "dy": dy}

    subhalos = []
    if R.rand() < sg.ADD_SUBHALOS_PROB:
        n_sub = int(R.poisson(1.0))
        for _ in range(max(0, n_sub)):
            subhalos.append({
                "theta_E": float(R.uniform(0.01, 0.06)),
                "center_x": float(R.uniform(-0.5 * theta_E, 0.5 * theta_E)),
                "center_y": float(R.uniform(-0.5 * theta_E, 0.5 * theta_E)),
            })

    source = _sample_source(theta_E)

    exptime = float(R.uniform(*sg.EXPTIME_RANGE))
    psf_fwhm = float(R.uniform(*sg.PSF_FWHM_ARCSEC))
    psf_ellip = float(R.uniform(0.0, sg.PSF_ELLIP_MAX))
    psf_angle = float(R.uniform(0, 2 * np.pi))
    coma_u = float(R.rand())  # make_psf_kernel_fixed always draws this (coma_prob=0 in v2)
    read_noise = float(R.uniform(*sg.READ_NOISE_RANGE))
    sky_adu = float(R.uniform(*sg.SKY_ADU_RANGE))
    src_mag = float(R.uniform(*sg.SRC_MAG_RANGE))
    lens_mag = src_mag + float(R.uniform(*sg.LENS_MAG_DELTA_RANGE))

    ll_Re = float(R.uniform(0.25, 1.0))
    ll_n = float(R.uniform(3.0, 5.0))
    ll_dx = float(R.uniform(-0.02, 0.02))
    ll_dy = float(R.uniform(-0.02, 0.02))
    e1_scale = float(R.uniform(0.6, 1.0))
    e1_add = float(R.uniform(-0.05, 0.05))
    e2_scale = float(R.uniform(0.6, 1.0))
    e2_add = float(R.uniform(-0.05, 0.05))

    return {
        "schema": SCHEMA,
        "z_l": z_l, "z_s": z_s, "sigma_kms": sigma_kms,
        "theta_E_raw": theta_E_raw, "theta_E_clip": clip,
        "sie": {"theta_E": theta_E, "e": e, "phi": phi,
                "center_x": sie_cx, "center_y": sie_cy},
        "shear": {"g": shear_g, "pa": shear_pa},
        "nfw": nfw,
        "subhalos": subhalos,
        "source": source,
        "psf": {"type": sg.PSF_TYPE, "fwhm_arcsec": psf_fwhm, "ellip": psf_ellip,
                "angle": psf_angle, "beta": 3.5, "coma_prob": 0.0, "coma_u": coma_u},
        "photometry": {"exptime": exptime, "zp": sg.DEFAULT_ZP,
                       "src_mag": src_mag, "lens_mag": lens_mag},
        "detector": {"read_noise": read_noise, "sky_adu": sky_adu},
        "lens_light": {"R_sersic": ll_Re, "n_sersic": ll_n, "dx": ll_dx, "dy": ll_dy,
                       "e1_scale": e1_scale, "e1_add": e1_add,
                       "e2_scale": e2_scale, "e2_add": e2_add},
    }


def _sample_source(theta_E: float) -> dict:
    """Mirrors the draws inside simgenv2.build_sources (and add_clumps)."""
    R = np.random
    cx = float(R.uniform(-0.15 * theta_E, 0.15 * theta_E))
    cy = float(R.uniform(-0.15 * theta_E, 0.15 * theta_E))
    frac_bulge = float(R.beta(1.5, 3.0))
    Rb = float(R.uniform(0.03, 0.12))
    Rd = Rb * float(R.uniform(1.6, 3.5))
    n_bulge = float(R.uniform(2.5, 4.5))
    amp_b = float(R.uniform(1.0, 4.0))
    amp_d = amp_b * float(R.uniform(0.3, 1.2))
    phi = float(R.uniform(0, np.pi))
    e = float(R.uniform(0.0, 0.6))
    e1 = float(e * np.cos(2 * phi))
    e2 = float(e * np.sin(2 * phi))
    bulge = {"amp": amp_b, "R_sersic": Rb, "n_sersic": n_bulge,
             "e1": e1, "e2": e2, "center_x": cx, "center_y": cy}
    disk = {"amp": amp_d, "R_sersic": Rd, "n_sersic": 1.0,
            "e1": e1 * float(R.uniform(0.8, 1.0)), "e2": e2 * float(R.uniform(0.8, 1.0)),
            "center_x": cx + float(R.uniform(-0.02, 0.02)),
            "center_y": cy + float(R.uniform(-0.02, 0.02))}

    extras = []
    n_extra = int(R.randint(sg.N_EXTRA_SOURCES_RANGE[0], sg.N_EXTRA_SOURCES_RANGE[1] + 1))
    for _ in range(n_extra):
        cx2 = float(R.uniform(-0.9 * theta_E, 0.9 * theta_E))
        cy2 = float(R.uniform(-0.9 * theta_E, 0.9 * theta_E))
        extras.append({
            "amp": float(R.uniform(0.3, 2.0)),
            "R_sersic": float(R.uniform(0.02, 0.1)),
            "n_sersic": float(R.uniform(0.8, 3.5)),
            "e1": float(R.uniform(-0.4, 0.4)),
            "e2": float(R.uniform(-0.4, 0.4)),
            "center_x": cx2, "center_y": cy2,
        })

    clumps = None
    if R.rand() < 0.75:
        n = int(R.randint(1, 6))
        max_rel = float(R.uniform(0.2, 0.8))
        sig_pix = (sg.OVERSAMPLE * 0.4, sg.OVERSAMPLE * 4.0)
        S = sg.SUPER_SIZE
        items = []
        for _ in range(n):
            items.append({
                "cx": float(R.uniform(0.25 * S, 0.75 * S)),
                "cy": float(R.uniform(0.25 * S, 0.75 * S)),
                "amp_rel": float(R.uniform(0.02, max_rel)),
                "sigma": float(R.uniform(*sig_pix)),
            })
        clumps = {"max_rel": max_rel, "items": items}

    spiral_phase = None
    if R.rand() < 0.35:
        spiral_phase = float(R.uniform(0, 2 * np.pi))

    return {"frac_bulge": frac_bulge, "bulge": bulge, "disk": disk,
            "extras": extras, "clumps": clumps, "spiral_phase": spiral_phase}


# ---------------------------------------------------------------------------
# helpers shared by render() and make_golden.py
# ---------------------------------------------------------------------------

def lens_model_from_params(p: dict):
    sie = p["sie"]
    e1 = sie["e"] * np.cos(2 * sie["phi"])
    e2 = sie["e"] * np.sin(2 * sie["phi"])
    lens_model_list = ["SIE"]
    kwargs_lens = [{"theta_E": sie["theta_E"], "e1": e1, "e2": e2,
                    "center_x": sie["center_x"], "center_y": sie["center_y"]}]
    g = p["shear"]["g"]
    if g > 0.0:
        pa = p["shear"]["pa"]
        lens_model_list.append("SHEAR")
        kwargs_lens.append({"gamma1": g * np.cos(2 * pa), "gamma2": g * np.sin(2 * pa)})
    if p["nfw"] is not None:
        n = p["nfw"]
        lens_model_list.append("NFW")
        kwargs_lens.append({"alpha_Rs": n["alpha_Rs"], "Rs": n["Rs"],
                            "center_x": sie["center_x"] + n["dx"],
                            "center_y": sie["center_y"] + n["dy"]})
    for s in p["subhalos"]:
        lens_model_list.append("SIS")
        kwargs_lens.append(dict(s))
    return lens_model_list, kwargs_lens, e1, e2


def super_grid():
    grid_lin_sup = np.linspace(-0.5 * sg.IMAGE_SIZE * sg.PIXEL_SCALE,
                               0.5 * sg.IMAGE_SIZE * sg.PIXEL_SCALE, sg.SUPER_SIZE)
    return np.meshgrid(grid_lin_sup, grid_lin_sup)


class _FixedDraw:
    """Stands in for np.random inside make_psf_kernel_fixed so render() is RNG-free."""

    def __init__(self, u):
        self.u = u

    def rand(self):
        return self.u


def render_source(src: dict, x_sup, y_sup) -> np.ndarray:
    """Deterministic body of simgenv2.build_sources (same dtypes/casts)."""
    lm = LightModel(["SERSIC_ELLIPSE"])
    bulge = lm.surface_brightness(x_sup, y_sup, [src["bulge"]]).astype(np.float32)
    disk = lm.surface_brightness(x_sup, y_sup, [src["disk"]]).astype(np.float32)
    frac_bulge = src["frac_bulge"]
    main = frac_bulge * bulge + (1 - frac_bulge) * disk
    for kw in src["extras"]:
        main += lm.surface_brightness(x_sup, y_sup, [kw]).astype(np.float32)
    if src["clumps"] is not None:
        sup = main.copy()
        S = sup.shape[0]
        for c in src["clumps"]["items"]:
            amp = c["amp_rel"] * (sup.max() if sup.max() > 0 else 1.0)
            X, Y = np.meshgrid(np.arange(S), np.arange(S))
            sup += amp * np.exp(-0.5 * (((X - c["cx"]) ** 2 + (Y - c["cy"]) ** 2) / (c["sigma"] ** 2)))
        main = np.clip(sup, 0.0, None)
    if src["spiral_phase"] is not None:
        X, Y = np.meshgrid(np.linspace(-1, 1, main.shape[0]), np.linspace(-1, 1, main.shape[1]))
        spiral = 1.0 + 0.08 * np.sin(3.0 * np.arctan2(Y, X) + src["spiral_phase"])
        main *= spiral
    return np.clip(main, 0.0, None).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Deterministic render (mirrors generate_one lines 260-364)
# ---------------------------------------------------------------------------

def render(p: dict) -> dict:
    SUPER_SIZE, OVERSAMPLE = sg.SUPER_SIZE, sg.OVERSAMPLE
    SUP_PIXEL_SCALE, SUPER_PIX_AREA = sg.SUP_PIXEL_SCALE, sg.SUPER_PIX_AREA
    lens_model_list, kwargs_lens, e1, e2 = lens_model_from_params(p)
    lens = LensModel(lens_model_list)
    x_sup, y_sup = super_grid()

    src_pattern = render_source(p["source"], x_sup, y_sup)

    ph = p["photometry"]
    exptime, zp = ph["exptime"], ph["zp"]
    src_mag, lens_mag = ph["src_mag"], ph["lens_mag"]
    ps = p["psf"]
    psf_sup, psf_kernel_info = sgt.make_psf_kernel_fixed(
        ps["fwhm_arcsec"], SUP_PIXEL_SCALE, psf_type=ps["type"], ellip=ps["ellip"],
        angle=ps["angle"], beta=ps["beta"], rng=_FixedDraw(ps["coma_u"]), coma_prob=ps["coma_prob"])

    sb_src_sup, src_total_counts = sg.normalize_to_sb_per_arcsec2(src_pattern, src_mag, zp, exptime, SUP_PIXEL_SCALE)
    src_counts_sup = (sb_src_sup * SUPER_PIX_AREA).astype(np.float32)
    GT = sg.downsample(src_counts_sup, OVERSAMPLE).astype(np.float32)

    x_flat, y_flat = x_sup.ravel(), y_sup.ravel()
    xs, ys = lens.ray_shooting(x_flat, y_flat, kwargs_lens)
    xs = xs.reshape(SUPER_SIZE, SUPER_SIZE)
    ys = ys.reshape(SUPER_SIZE, SUPER_SIZE)

    interp_sb = sg.rg_interpolator(x_sup, y_sup, sb_src_sup, method="linear")
    pts = np.vstack([ys.ravel(), xs.ravel()]).T
    sb_mapped = interp_sb(pts).reshape(SUPER_SIZE, SUPER_SIZE)
    lensed_src_counts_sup = (sb_mapped * SUPER_PIX_AREA).astype(np.float32)

    ll = p["lens_light"]
    ll_center_x = kwargs_lens[0]["center_x"] + ll["dx"]
    ll_center_y = kwargs_lens[0]["center_y"] + ll["dy"]
    lens_light_sup_pattern = LightModel(["SERSIC_ELLIPSE"]).surface_brightness(x_sup, y_sup, [{
        "amp": 1.0, "R_sersic": ll["R_sersic"], "n_sersic": ll["n_sersic"],
        "e1": e1 * ll["e1_scale"] + ll["e1_add"],
        "e2": e2 * ll["e2_scale"] + ll["e2_add"],
        "center_x": ll_center_x, "center_y": ll_center_y,
    }]).astype(np.float32)
    sb_lens_sup, lens_total_counts = sg.normalize_to_sb_per_arcsec2(lens_light_sup_pattern, lens_mag, zp, exptime, SUP_PIXEL_SCALE)
    lens_counts_sup = (sb_lens_sup * SUPER_PIX_AREA).astype(np.float32)

    image_sup = lensed_src_counts_sup + lens_counts_sup
    image_conv_sup = sg.convolve_fft(image_sup, psf_sup, normalize_kernel=True, allow_huge=True)
    LENSED_counts = sg.downsample(image_conv_sup, OVERSAMPLE).astype(np.float32)

    def peak99(arr):
        return float(np.percentile(arr, 99))

    peak_src = peak99(sg.downsample(sg.convolve_fft(lensed_src_counts_sup, psf_sup, normalize_kernel=True), OVERSAMPLE))
    peak_lens = peak99(sg.downsample(sg.convolve_fft(lens_counts_sup, psf_sup, normalize_kernel=True), OVERSAMPLE))
    ratio = (peak_src + 1e-9) / (peak_lens + 1e-9)
    ratio_history = [ratio]
    target_ratio = 0.35
    tries = 0
    while ratio < target_ratio and tries < 2:
        src_mag -= 0.25
        lens_mag += 0.12
        sb_src_sup, src_total_counts = sg.normalize_to_sb_per_arcsec2(src_pattern, src_mag, zp, exptime, SUP_PIXEL_SCALE)
        src_counts_sup = (sb_src_sup * SUPER_PIX_AREA).astype(np.float32)
        interp_sb = sg.rg_interpolator(x_sup, y_sup, sb_src_sup, method="linear")
        sb_mapped = interp_sb(pts).reshape(SUPER_SIZE, SUPER_SIZE)
        lensed_src_counts_sup = (sb_mapped * SUPER_PIX_AREA).astype(np.float32)
        sb_lens_sup, lens_total_counts = sg.normalize_to_sb_per_arcsec2(lens_light_sup_pattern, lens_mag, zp, exptime, SUP_PIXEL_SCALE)
        lens_counts_sup = (sb_lens_sup * SUPER_PIX_AREA).astype(np.float32)
        image_sup = lensed_src_counts_sup + lens_counts_sup
        image_conv_sup = sg.convolve_fft(image_sup, psf_sup, normalize_kernel=True, allow_huge=True)
        LENSED_counts = sg.downsample(image_conv_sup, OVERSAMPLE).astype(np.float32)
        peak_src = peak99(sg.downsample(sg.convolve_fft(lensed_src_counts_sup, psf_sup, normalize_kernel=True), OVERSAMPLE))
        peak_lens = peak99(sg.downsample(sg.convolve_fft(lens_counts_sup, psf_sup, normalize_kernel=True), OVERSAMPLE))
        ratio = (peak_src + 1e-9) / (peak_lens + 1e-9)
        ratio_history.append(ratio)
        tries += 1

    return {
        "lens_model_list": lens_model_list, "kwargs_lens": kwargs_lens,
        "x_sup": x_sup, "y_sup": y_sup, "xs": xs, "ys": ys,
        "src_pattern": src_pattern, "psf_sup": psf_sup, "psf_kernel_info": psf_kernel_info,
        "GT": GT, "lensed_src_counts_sup": lensed_src_counts_sup,
        "lens_light_sup_pattern": lens_light_sup_pattern, "lens_counts_sup": lens_counts_sup,
        "image_conv_sup": image_conv_sup, "LENSED_counts": LENSED_counts,
        "src_mag_final": src_mag, "lens_mag_final": lens_mag,
        "src_total_counts": src_total_counts, "lens_total_counts": lens_total_counts,
        "ratio": ratio, "ratio_history": ratio_history, "tries": tries,
        "ll_center_x": ll_center_x, "ll_center_y": ll_center_y,
    }


# ---------------------------------------------------------------------------
# 3. Detector noise (mirrors generate_one lines 366-392, np.random draws)
# ---------------------------------------------------------------------------

def apply_noise_numpy(LENSED_counts, read_noise, sky_adu):
    IMAGE_SIZE, GAIN = sg.IMAGE_SIZE, sg.GAIN
    xg = np.linspace(-0.5, 0.5, IMAGE_SIZE)
    sky_gradient = (1.0 + sg.SKY_GRADIENT_MAX * (xg - xg.mean()) / (xg.max() - xg.min()))
    sky_map = sky_adu * sky_gradient[np.newaxis, :]
    LENSED_counts = LENSED_counts + sky_map
    prnu_map = 1.0 + np.random.normal(0.0, sg.PRNU_RMS, size=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    LENSED_counts_prnu = LENSED_counts * prnu_map
    e_image = np.clip(LENSED_counts_prnu * GAIN, 0.0, None)
    noisy_e = np.random.poisson(e_image).astype(np.float32)
    noisy_e += np.random.normal(0.0, read_noise, noisy_e.shape).astype(np.float32)
    cosmic_hits = []
    if np.random.rand() < sg.COSMIC_RAY_PROB:
        n_hits = np.random.randint(1, 6)
        for _ in range(n_hits):
            cx = np.random.randint(0, IMAGE_SIZE)
            cy = np.random.randint(0, IMAGE_SIZE)
            intensity = np.random.uniform(*sg.COSMIC_RAY_INTENSITY)
            noisy_e[cy, cx] += intensity * GAIN
            cosmic_hits.append((int(cx), int(cy), float(intensity)))
    LENSED = (noisy_e / GAIN).astype(np.float32)
    return LENSED, {"sky_map": sky_map, "cosmic_hits": cosmic_hits}


# ---------------------------------------------------------------------------
# 4. Truth labels (mirrors generate_one lines 434-454)
# ---------------------------------------------------------------------------

def truth_labels(p: dict, st: dict, GT_out, LENSED):
    PIXEL_SCALE, IMAGE_SIZE, OVERSAMPLE = sg.PIXEL_SCALE, sg.IMAGE_SIZE, sg.OVERSAMPLE
    realized_fwhm_arcsec = st["psf_kernel_info"]["realized_fwhm_pix"] * sg.SUP_PIXEL_SCALE
    lens_light_det = sg.downsample(st["lens_counts_sup"], OVERSAMPLE).astype(np.float32)
    norm_m = float(max(GT_out.max(), LENSED.max(), 1e-8))
    ll_amp = sgt.measure_lens_light_amp_at_Re(
        lens_light_det, (st["ll_center_x"], st["ll_center_y"]), p["lens_light"]["R_sersic"],
        PIXEL_SCALE, norm_m)
    truth, maskbits, clean = sgt.truth_from_lenstronomy(
        st["lens_model_list"], st["kwargs_lens"],
        pixel_scale=PIXEL_SCALE, image_size=IMAGE_SIZE,
        psf_info={
            "realized_fwhm_pix": realized_fwhm_arcsec / PIXEL_SCALE,
            "psf_type": sg.PSF_TYPE, "beta": 3.5,
            "ellip": p["psf"]["ellip"], "angle": p["psf"]["angle"],
            "coma_applied": False,
        },
        lens_light_info={
            "Re_arcsec": p["lens_light"]["R_sersic"], "n_sersic": p["lens_light"]["n_sersic"],
            "amp_at_Re_normalized": ll_amp,
        })
    return {"truth": truth, "maskbits": maskbits, "clean": clean,
            "lens_light_det": lens_light_det, "norm_m": norm_m, "ll_amp": ll_amp}


def simulate_seeded(seed: int):
    """sample -> render -> noise -> truth using numpy's global RNG, like simgenv2.main()."""
    sg.set_seed(seed)
    p = sample_params()
    st = render(p)
    LENSED, noise_info = apply_noise_numpy(st["LENSED_counts"], p["detector"]["read_noise"], p["detector"]["sky_adu"])
    tr = truth_labels(p, st, st["GT"], LENSED)
    return p, st, LENSED, noise_info, tr
