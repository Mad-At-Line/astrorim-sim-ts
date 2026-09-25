# Verification

How the TypeScript port was checked against `simgenv2.py` / `simgen_truth.py`, with the numbers from the
last full run. Environment for the reference data: Python 3.11, numpy 2.4.6, lenstronomy 1.14.2, astropy 8.0.1,
scipy 1.17.1 (recorded in `tests/golden/manifest.json`).

## 1. The Python harness reproduces simgenv2 exactly

To compare stages, the monolithic `generate_one` has to be split into *sample parameters → render → add noise*.
A split that silently changed behaviour would make every later comparison meaningless, so this is checked first.

`reference/check_fidelity.py` seeds numpy, runs the **unmodified** `simgenv2.generate_one`, then re-seeds and runs
the harness. Both GT and LENSED arrays must be identical (`np.array_equal`), and every header value must match:

```
40/40 seeds reproduce simgenv2.generate_one exactly
```

That includes the noise, which means the harness consumes numpy's random stream in exactly the same order as
`generate_one`. One subtlety it had to reproduce: `make_psf_kernel_fixed` always draws one uniform for the coma test,
even when `coma_prob = 0`.

## 2. Stage-by-stage golden comparison

`reference/make_golden.py` writes 9 cases (8 seeds chosen for branch coverage plus 1 forced case) with numpy's
parameter draws and every intermediate array. `tests/pipeline.test.ts` renders the same parameters in TypeScript.
Errors are max |TS − Python| / max |Python| (ray-shooting: absolute, arcsec).

| Stage | Worst over 9 cases | Test tolerance |
| --- | --- | --- |
| Ray-shooting (SIE + shear + NFW + SIS) | 2.8 × 10⁻¹³ arcsec | 10⁻¹² |
| Source pattern (`build_sources`) | 1.0 × 10⁻⁷ | 10⁻⁶ |
| Lens-light pattern | 0 (bit-identical) | 10⁻⁶ |
| PSF kernel | 0 (bit-identical) | 10⁻⁶ |
| GT (unlensed source, detector) | 1.2 × 10⁻⁷ | 10⁻⁵ |
| Lensed source (interpolation) | 1.1 × 10⁻⁷ | 10⁻⁵ |
| PSF-convolved super-grid image | 4.6 × 10⁻⁸ | 10⁻⁵ |
| Noiseless detector image | 1.1 × 10⁻⁷ | 10⁻⁵ |
| Lens-light detector image | 1.1 × 10⁻⁷ | 10⁻⁵ |

Errors of ~10⁻⁷ are float32 rounding: `simgenv2` casts to float32 at several points and uses float32 arithmetic
inside `build_sources`. The port mirrors the explicit `.astype(np.float32)` casts.

Branch coverage of the golden set: NFW halo (3 cases), 0–4 subhalos, 0–2 extra sources, with/without clumps,
with/without spiral modulation, θ<sub>E</sub> clipped / out of range but kept / in range, and the peak-ratio loop
(forced case, 2 tries).

Unit-level references (`tests/golden/units.json`) cover pieces the seeded cases don't reach: Moffat PSFs and the coma
branch, three hand-built lens configurations for `truth_from_lenstronomy` (co-centred NFW → `TR_CLEAN = 1`; offset halo +
subhalo → masked), `_wrap_axis_angle` on negative and large angles, and Planck15 distances. Cosmology agrees with astropy
to better than 10⁻¹⁰ relative.

## 3. Random stages: statistics, not bits

The TS generator (xoshiro128\*\*) is not numpy's Mersenne Twister, so a TS seed does not draw the same lens as a numpy
seed. The random stages are checked in two ways:

- **Unit statistics** (`tests/rng.test.ts`, `tests/sampling.test.ts`): moments of uniform, normal, Poisson
  (λ from 0.3 to 25,000, covering both of numpy's algorithms), Beta(1.5, 3); branch probabilities of the priors
  (group halo 0.18, clumps 0.75, spiral 0.35, kept-extreme θ<sub>E</sub> 0.08, subhalo count); noise is unbiased.
- **Population comparison** (`reference/compare_distributions.py`): 300 TS simulations vs 300 `simgenv2` simulations,
  two-sample KS test on 12 quantities:

```
quantity                  median TS    median Py  KS p-value
THETA_E                      1.1136       1.0413       0.249
TR_Q                        0.60135      0.60312       0.788
TR_GAM                     0.039958     0.042222       0.455
TR_PSFW                      1.9983       1.9971       0.848
SRCMAG                       22.085       22.031       0.585
LENSMAG                      22.553        22.52       0.455
TR_LFLX                   0.0080132    0.0073223       0.342
TR_CLEAN                          0            0           1
GT sum                       2488.9       2759.4       0.176
LENSED sum                    19403        18982       0.518
LENSED p99                   18.821        20.62       0.293
corner std (noise)          0.99077       1.0225       0.654
```

No quantity differs (smallest p = 0.18; Bonferroni threshold at α = 0.01 is 8 × 10⁻⁴).

## 4. The tests themselves are tested

`npm run mutation-check` injects typical porting bugs one at a time and confirms the suite fails:

| Injected bug | Result |
| --- | --- |
| SIE deflection rotated back with the wrong sign | caught (45 tests fail) |
| SIE θ<sub>E</sub> used as major-axis *b* instead of lenstronomy's product average | caught (45) |
| Sérsic elliptical radius with e1 sign flipped | caught (80) |
| Sérsic b<sub>n</sub> = 2n − 1/3 instead of lenstronomy's 1.9992n − 0.3271 | caught (80) |
| Interpolation with x and y swapped | caught (36) |
| Convolution output shifted by one pixel | caught (27) |
| Super-grid spacing exactly 0.01″ instead of `np.linspace` endpoints | caught (89) |
| GT normalised with the post-loop source magnitude | caught (1: the forced ratio-loop case) |
| JS `%` instead of Python's floor-mod in `_wrap_axis_angle` | equivalent mutant |

The last one survives for a good reason: `_wrap_axis_angle` follows the modulo with
`if a <= -pi/2: a += pi`, which absorbs the one-period difference between JS's truncated remainder and Python's
floor-mod. The two versions agree on every input (up to floating-point rounding), so no test can tell them apart.
The port keeps a real floor-mod (`pymod`) anyway, so it reads the same as the Python code.

The GT mutation is only caught by the forced case. That is why the forced case exists: under the `simgenv2` priors the
peak-ratio loop never runs (see findings below).

## 5. FITS compatibility

`reference/check_ts_fits.py` opens files from `npm run generate` with astropy and compares them with a real
`simgenv2` file: same HDU names (PRIMARY, GT, LENSED), shapes (96 × 96), dtype (float32), and every primary and
extension header key. The TS files add one key, `SIMCODE`, naming the generator. Header reals are shortened to 20
characters exactly as astropy does. For example, `TR_X0 = -0.00430135524038875` loses its 17th digit in both.

## Findings in simgenv2

These came up while porting. The TS port reproduces all of them on purpose (it has to match the reference first).
They're listed here to discuss before changing anything in either version.

1. **θ<sub>E</sub> piles up at the clip limits.** Drawing σ ∈ [160, 320] km/s and the redshift ranges puts about 62–65%
   of raw Einstein radii outside [0.9″, 1.5″], and 92% of those are clipped. So **about 57–59% of all simulated lenses
   have θ<sub>E</sub> exactly 0.9″ or 1.5″** (Python: 234/400 seeds; TS: 57.0% of 20,000). The distribution is two spikes
   plus a thin continuum, not a smooth prior. It's worth checking whether the encoder's θ<sub>E</sub> accuracy on v2 data
   partly reflects those spikes. Also, after clipping, the `SIGMA`, `ZLENS` and `ZSRC` header values no longer produce
   `THETA_E`.
2. **Super-grid spacing is 0.0100261″, not 0.01″.** `np.linspace(-1.92, 1.92, 384)` includes both endpoints, so the
   step is 3.84/383. Detector pixels therefore span 0.040104″ while `PIXSCALE` and the truth conversion use 0.04″. Lens
   features appear 0.26% smaller in pixels than their labels imply. Small, but systematic in every image.
3. **The peak-ratio loop never runs under the v2 priors**, and it would desynchronise GT if it did. It didn't run for
   any of the 40 fidelity seeds. Among the 25 seeds (out of 400) with the brightest lens relative to the source, the
   smallest src/lens peak ratio was 1.6, against a threshold of 0.35. If the loop does run (other variants or changed priors), it brightens the source by
   0.25 mag per try but GT keeps the original normalisation. GT would then be 26% (1 try) or 58% (2 tries) fainter than
   the source in LENSED.
4. **`angular_diameter_distance_z1z2` is deprecated in astropy 8** (it emits `AstropyDeprecationWarning`). The
   replacement is `angular_diameter_distance(z1, z2)`.
5. **Minor:** `simgenv2` creates its hard-coded Windows `OUTPUT_DIR` at import time, so importing the module anywhere
   creates that folder. `datetime.utcnow()` is deprecated from Python 3.12. The docstring says the files are named
   `rim_simv2_*` but the code writes `rim_sim2_*`.

## Re-running everything

```bash
npm test                                   # golden, unit and statistical tests
npm run mutation-check                     # ~2 minutes
python reference/check_fidelity.py 40      # needs reference/requirements.txt
python reference/make_golden.py && npm test   # re-derive the goldens from your installed lenstronomy/astropy
npm run generate -- --n 300 --seed 12345 --out sims_ts
python reference/check_ts_fits.py sims_ts
python reference/compare_distributions.py sims_ts 300
```
