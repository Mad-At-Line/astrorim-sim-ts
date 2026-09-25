/**
 * The simgenv2 pipeline (generate_one), split like reference/harness.py:
 *
 *   render(params)            deterministic: source, lensing, lens light, PSF,
 *                             detector sampling, peak-ratio loop
 *   applyNoise(...)           sky gradient, PRNU, Poisson, read noise, cosmic rays
 *   simulate(params, rng)     both, plus the TR_* truth labels
 *
 * Where simgenv2 calls .astype(np.float32) the port rounds with Math.fround, so
 * intermediate precision matches the Python file.
 *
 * One deliberate structural difference: the peak-ratio loop. simgenv2 redoes
 * the interpolation and three FFT convolutions on every try. Every one of those
 * steps is linear in the source/lens flux, so this port convolves unit-flux
 * images once and rescales. The result is the same up to float rounding
 * (checked by the forced_ratio_loop golden case) and ~4x faster in the browser.
 */

import * as C from "./config";
import { Convolver } from "./fft";
import { bilinearOnGrid, downsample, linspace, maxOf, meshgrid, percentile, sumOf } from "./image";
import type { LensComponent } from "./lens";
import { rayShoot } from "./lens";
import { sersicEllipse } from "./light";
import type { SimParams, SourceParams } from "./params";
import { makePsfKernel, type PsfKernel } from "./psf";
import type { Rng } from "./rng";
import { measureLensLightAmpAtRe, truthFromLenstronomy, type TruthResult } from "./truth";

export interface Grid {
  lin: Float64Array; // 1-D super-grid coordinates (arcsec)
  X: Float64Array;
  Y: Float64Array;
}

let cachedGrid: Grid | null = null;

/** simgenv2's super-resolution grid: linspace(-N*ps/2, N*ps/2, SUPER_SIZE). */
export function superGrid(): Grid {
  if (!cachedGrid) {
    const half = 0.5 * C.IMAGE_SIZE * C.PIXEL_SCALE;
    const lin = linspace(-half, half, C.SUPER_SIZE);
    const { X, Y } = meshgrid(lin);
    cachedGrid = { lin, X, Y };
  }
  return cachedGrid;
}

export function massEllipticity(p: SimParams): { e1: number; e2: number } {
  return { e1: p.sie.e * Math.cos(2 * p.sie.phi), e2: p.sie.e * Math.sin(2 * p.sie.phi) };
}

/** Same component list and order as generate_one builds lens_model_list. */
export function lensComponents(p: SimParams): LensComponent[] {
  const { e1, e2 } = massEllipticity(p);
  const comps: LensComponent[] = [
    { type: "SIE", kw: { theta_E: p.sie.theta_E, e1, e2, center_x: p.sie.center_x, center_y: p.sie.center_y } },
  ];
  if (p.shear.g > 0.0) {
    comps.push({
      type: "SHEAR",
      kw: { gamma1: p.shear.g * Math.cos(2 * p.shear.pa), gamma2: p.shear.g * Math.sin(2 * p.shear.pa) },
    });
  }
  if (p.nfw) {
    comps.push({
      type: "NFW",
      kw: {
        alpha_Rs: p.nfw.alpha_Rs,
        Rs: p.nfw.Rs,
        center_x: p.sie.center_x + p.nfw.dx,
        center_y: p.sie.center_y + p.nfw.dy,
      },
    });
  }
  for (const s of p.subhalos) comps.push({ type: "SIS", kw: { ...s } });
  return comps;
}

/** Deterministic body of simgenv2.build_sources (super-grid, float32-rounded). */
export function renderSource(src: SourceParams, grid: Grid = superGrid()): Float64Array {
  const { X, Y } = grid;
  const S = C.SUPER_SIZE;
  const n = S * S;
  const bulge = sersicEllipse(src.bulge, X, Y, new Float64Array(n));
  const disk = sersicEllipse(src.disk, X, Y, new Float64Array(n));
  const main = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    main[i] = src.frac_bulge * Math.fround(bulge[i]) + (1 - src.frac_bulge) * Math.fround(disk[i]);
  }
  const tmp = new Float64Array(n);
  for (const kw of src.extras) {
    sersicEllipse(kw, X, Y, tmp);
    for (let i = 0; i < n; i++) main[i] += Math.fround(tmp[i]);
  }
  if (src.clumps) {
    for (const c of src.clumps.items) {
      const mx = maxOf(main);
      const amp = c.amp_rel * (mx > 0 ? mx : 1.0);
      const inv2s2 = 1 / (c.sigma * c.sigma);
      for (let r = 0; r < S; r++) {
        const dy2 = (r - c.cy) ** 2;
        for (let col = 0; col < S; col++) {
          main[r * S + col] += amp * Math.exp(-0.5 * (((col - c.cx) ** 2 + dy2) * inv2s2));
        }
      }
    }
    for (let i = 0; i < n; i++) if (main[i] < 0) main[i] = 0;
  }
  if (src.spiral_phase !== null) {
    const g = linspace(-1, 1, S);
    for (let r = 0; r < S; r++) {
      for (let col = 0; col < S; col++) {
        main[r * S + col] *= 1.0 + 0.08 * Math.sin(3.0 * Math.atan2(g[r], g[col]) + src.spiral_phase);
      }
    }
  }
  for (let i = 0; i < n; i++) main[i] = Math.fround(main[i] < 0 ? 0 : main[i]);
  return main;
}

/** exptime * 10^(-0.4 (mag - zp)) -- total counts in normalize_to_sb_per_arcsec2 */
export function totalCounts(mag: number, zp: number, exptime: number): number {
  return exptime * Math.pow(10, -0.4 * (mag - zp));
}

/**
 * normalize_to_sb_per_arcsec2 followed by `* SUPER_PIX_AREA` and the float32
 * casts, i.e. counts per super-pixel for a given magnitude.
 * Returns the unit-flux surface brightness too (sb per total count).
 */
function normalisedUnit(img: Float64Array): Float64Array {
  let s = 0;
  for (let i = 0; i < img.length; i++) s += img[i] > 0 ? img[i] : 0;
  const out = new Float64Array(img.length);
  if (s <= 0) return out;
  for (let i = 0; i < img.length; i++) out[i] = (img[i] > 0 ? img[i] : 0) / s;
  return out;
}

function countsSup(unit: Float64Array, total: number): Float64Array {
  const out = new Float64Array(unit.length);
  for (let i = 0; i < unit.length; i++) {
    const sb = Math.fround((unit[i] * total) / C.SUPER_PIX_AREA);
    out[i] = Math.fround(sb * C.SUPER_PIX_AREA);
  }
  return out;
}

export interface RenderResult {
  lens: LensComponent[];
  xs: Float64Array; // source-plane x of every super-pixel
  ys: Float64Array;
  srcPattern: Float64Array; // 384^2
  psf: PsfKernel;
  GT: Float64Array; // 96^2, unlensed source, detector-sampled, not PSF-convolved
  lensedSrcCountsSup: Float64Array; // 384^2
  lensLightSupPattern: Float64Array; // 384^2 (amp = 1)
  lensCountsSup: Float64Array; // 384^2
  imageConvSup: Float64Array; // 384^2
  lensedCounts: Float64Array; // 96^2 noiseless LENSED_counts
  srcDet: Float64Array; // 96^2 lensed source only (PSF-convolved, detector-sampled)
  lensDet: Float64Array; // 96^2 lens light only (PSF-convolved, detector-sampled)
  srcMagFinal: number;
  lensMagFinal: number;
  srcTotalCounts: number;
  lensTotalCounts: number;
  ratioHistory: number[];
  tries: number;
  llCenter: [number, number];
  timingsMs: Record<string, number>;
}

export function render(p: SimParams): RenderResult {
  const t0 = now();
  const timings: Record<string, number> = {};
  const lap = (name: string, since: number) => {
    const t = now();
    timings[name] = t - since;
    return t;
  };
  const grid = superGrid();
  const S = C.SUPER_SIZE;
  const lens = lensComponents(p);
  const { e1, e2 } = massEllipticity(p);

  let t = t0;
  const srcPattern = renderSource(p.source, grid);
  t = lap("source", t);

  const { exptime, zp } = p.photometry;
  let srcMag = p.photometry.src_mag;
  let lensMag = p.photometry.lens_mag;
  const psf = makePsfKernel({ ...p.psf, pixel_scale: C.SUP_PIXEL_SCALE });

  // source normalisation; GT uses the INITIAL src_mag (simgenv2 line 289)
  const srcUnit = normalisedUnit(srcPattern);
  const srcCountsInitial = countsSup(srcUnit, totalCounts(srcMag, zp, exptime));
  const GT = downsample(srcCountsInitial, S, C.OVERSAMPLE).map(Math.fround);

  // ray shooting + interpolation of the unit-flux surface brightness
  const { xs, ys } = rayShoot(lens, grid.X, grid.Y);
  t = lap("rayShoot", t);
  const sbUnit = new Float64Array(srcUnit.length);
  for (let i = 0; i < sbUnit.length; i++) sbUnit[i] = srcUnit[i] / C.SUPER_PIX_AREA;
  const sbMappedUnit = bilinearOnGrid(grid.lin, sbUnit, xs, ys);
  t = lap("interpolate", t);

  // lens light (image plane, not lensed), aligned with the mass ellipticity
  const ll = p.lens_light;
  const llCenter: [number, number] = [p.sie.center_x + ll.dx, p.sie.center_y + ll.dy];
  const lensLightSupPattern = sersicEllipse(
    {
      amp: 1.0,
      R_sersic: ll.R_sersic,
      n_sersic: ll.n_sersic,
      e1: e1 * ll.e1_scale + ll.e1_add,
      e2: e2 * ll.e2_scale + ll.e2_add,
      center_x: llCenter[0],
      center_y: llCenter[1],
    },
    grid.X,
    grid.Y,
    new Float64Array(S * S),
  ).map(Math.fround);
  const lensUnit = normalisedUnit(lensLightSupPattern);
  t = lap("lensLight", t);

  // unit-flux lensed source counts and lens counts, then two convolutions
  const lensedSrcUnitSup = new Float64Array(S * S);
  for (let i = 0; i < lensedSrcUnitSup.length; i++) lensedSrcUnitSup[i] = sbMappedUnit[i] * C.SUPER_PIX_AREA;
  const conv = new Convolver(psf.kernel, psf.size, S);
  const convSrcUnit = conv.convolve(lensedSrcUnitSup);
  const convLensUnit = conv.convolve(lensUnit);
  const dSrcUnit = downsample(convSrcUnit, S, C.OVERSAMPLE);
  const dLensUnit = downsample(convLensUnit, S, C.OVERSAMPLE);
  t = lap("convolve", t);

  // peak-ratio loop (closed form thanks to linearity; percentile is homogeneous)
  const p99Src = percentile(dSrcUnit, 99);
  const p99Lens = percentile(dLensUnit, 99);
  const ratioAt = (sm: number, lm: number) =>
    (totalCounts(sm, zp, exptime) * p99Src + 1e-9) / (totalCounts(lm, zp, exptime) * p99Lens + 1e-9);
  let ratio = ratioAt(srcMag, lensMag);
  const ratioHistory = [ratio];
  let tries = 0;
  while (ratio < C.TARGET_PEAK_RATIO && tries < C.MAX_RATIO_TRIES) {
    srcMag -= 0.25;
    lensMag += 0.12;
    ratio = ratioAt(srcMag, lensMag);
    ratioHistory.push(ratio);
    tries += 1;
  }

  const srcTotal = totalCounts(srcMag, zp, exptime);
  const lensTotal = totalCounts(lensMag, zp, exptime);
  const lensedSrcCountsSup = new Float64Array(S * S);
  for (let i = 0; i < lensedSrcCountsSup.length; i++) {
    lensedSrcCountsSup[i] = Math.fround(Math.fround(sbMappedUnit[i] * srcTotal) * C.SUPER_PIX_AREA);
  }
  const lensCountsSup = countsSup(lensUnit, lensTotal);
  const imageConvSup = new Float64Array(S * S);
  for (let i = 0; i < imageConvSup.length; i++) {
    imageConvSup[i] = srcTotal * convSrcUnit[i] + lensTotal * convLensUnit[i];
  }
  const nDet = C.IMAGE_SIZE * C.IMAGE_SIZE;
  const lensedCounts = new Float64Array(nDet);
  const srcDet = new Float64Array(nDet);
  const lensDet = new Float64Array(nDet);
  for (let i = 0; i < nDet; i++) {
    srcDet[i] = srcTotal * dSrcUnit[i];
    lensDet[i] = lensTotal * dLensUnit[i];
    lensedCounts[i] = Math.fround(srcDet[i] + lensDet[i]);
  }
  lap("assemble", t);
  timings.total = now() - t0;

  return {
    lens,
    xs,
    ys,
    srcPattern,
    psf,
    GT,
    lensedSrcCountsSup,
    lensLightSupPattern,
    lensCountsSup,
    imageConvSup,
    lensedCounts,
    srcDet,
    lensDet,
    srcMagFinal: srcMag,
    lensMagFinal: lensMag,
    srcTotalCounts: srcTotal,
    lensTotalCounts: lensTotal,
    ratioHistory,
    tries,
    llCenter,
    timingsMs: timings,
  };
}

export interface NoiseResult {
  LENSED: Float64Array; // 96^2 final image in ADU (float32-rounded)
  skyMap: Float64Array;
  cosmicHits: { x: number; y: number; intensity: number }[];
}

/** generate_one lines 366-392 with the TS Rng. */
export function applyNoise(lensedCounts: Float64Array, readNoise: number, skyAdu: number, rng: Rng): NoiseResult {
  const N = C.IMAGE_SIZE;
  const xg = linspace(-0.5, 0.5, N);
  const mean = sumOf(xg) / N;
  const span = xg[N - 1] - xg[0];
  const skyMap = new Float64Array(N * N);
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) skyMap[r * N + c] = skyAdu * (1.0 + (C.SKY_GRADIENT_MAX * (xg[c] - mean)) / span);
  }
  // PRNU on counts before Poisson (small multiplicative fixed pattern)
  const prnu = new Float64Array(N * N);
  for (let i = 0; i < N * N; i++) prnu[i] = Math.fround(1.0 + Math.fround(rng.normal(0.0, C.PRNU_RMS)));
  // ADU -> electrons, Poisson noise, then read noise (electrons)
  const noisyE = new Float64Array(N * N);
  for (let i = 0; i < N * N; i++) {
    const eImage = Math.max((lensedCounts[i] + skyMap[i]) * prnu[i] * C.GAIN, 0.0);
    noisyE[i] = rng.poisson(eImage);
  }
  for (let i = 0; i < N * N; i++) noisyE[i] = Math.fround(noisyE[i] + Math.fround(rng.normal(0.0, readNoise)));
  const cosmicHits: NoiseResult["cosmicHits"] = [];
  if (rng.rand() < C.COSMIC_RAY_PROB) {
    const nHits = rng.randint(1, 6);
    for (let h = 0; h < nHits; h++) {
      const x = rng.randint(0, N);
      const y = rng.randint(0, N);
      const intensity = rng.uniform(...C.COSMIC_RAY_INTENSITY);
      noisyE[y * N + x] = Math.fround(noisyE[y * N + x] + intensity * C.GAIN);
      cosmicHits.push({ x, y, intensity });
    }
  }
  const LENSED = new Float64Array(N * N);
  for (let i = 0; i < N * N; i++) LENSED[i] = Math.fround(noisyE[i] / C.GAIN);
  return { LENSED, skyMap, cosmicHits };
}

export interface SimulationResult extends RenderResult {
  params: SimParams;
  LENSED: Float64Array;
  noise: NoiseResult;
  truth: TruthResult;
  lensLightDet: Float64Array;
  normM: number;
  llAmp: number | null;
}

/** Truth labels exactly as generate_one computes them (lines 434-454). */
export function truthFor(p: SimParams, r: RenderResult, LENSED: Float64Array) {
  const lensLightDet = downsample(r.lensCountsSup, C.SUPER_SIZE, C.OVERSAMPLE).map(Math.fround);
  const normM = Math.max(maxOf(r.GT), maxOf(LENSED), 1e-8);
  const llAmp = measureLensLightAmpAtRe(lensLightDet, r.llCenter, p.lens_light.R_sersic, C.PIXEL_SCALE, normM);
  const realizedFwhmArcsec = r.psf.realizedFwhmPix * C.SUP_PIXEL_SCALE;
  const truth = truthFromLenstronomy(r.lens, C.PIXEL_SCALE, C.IMAGE_SIZE, {
    realized_fwhm_pix: realizedFwhmArcsec / C.PIXEL_SCALE,
    psf_type: C.PSF_TYPE,
    beta: 3.5,
    ellip: p.psf.ellip,
    angle: p.psf.angle,
  }, {
    Re_arcsec: p.lens_light.R_sersic,
    n_sersic: p.lens_light.n_sersic,
    amp_at_Re_normalized: llAmp,
  });
  return { truth, lensLightDet, normM, llAmp };
}

export function simulate(p: SimParams, rng: Rng): SimulationResult {
  const r = render(p);
  const noise = applyNoise(r.lensedCounts, p.detector.read_noise, p.detector.sky_adu, rng);
  const { truth, lensLightDet, normM, llAmp } = truthFor(p, r, noise.LENSED);
  return { ...r, params: p, LENSED: noise.LENSED, noise, truth, lensLightDet, normM, llAmp };
}

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}
