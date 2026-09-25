import { ellipticity2phiQ } from "../src/lens";
import { sampleParams, type SimParams } from "../src/params";
import { Rng } from "../src/rng";
import { TRUTH_COMMENTS, TRUTH_FITS_KEYS, TRUTH_ORDER, type TruthResult } from "../src/truth";
import { handle } from "./engine";
import SimWorker from "./worker?worker&inline";

/** Set this to your GitHub repository URL after publishing to show a link in the header. */
const REPO_URL = "";

const DEG = Math.PI / 180;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// ---------------------------------------------------------------------------
// state

let sampled: SimParams; // what the prior drew for the current seed
let current: SimParams; // sampled + slider overrides
let seed = 1;
let busy = false;
let dirty = false;
let lastImages: Record<string, Float32Array> | null = null;
let requestId = 0;

interface SliderDef {
  id: string;
  label: string;
  min: number;
  max: number;
  step: number;
  get: (p: SimParams) => number;
  set: (p: SimParams, v: number) => void;
  fmt: (v: number) => string;
}

const sliders: SliderDef[] = [
  {
    id: "thetaE", label: "Einstein radius θ<sub>E</sub>", min: 0.2, max: 2.4, step: 0.01,
    get: (p) => p.sie.theta_E, set: (p, v) => (p.sie.theta_E = v), fmt: (v) => `${v.toFixed(2)}″`,
  },
  {
    id: "e", label: "Mass ellipticity e", min: 0, max: 0.6, step: 0.005,
    get: (p) => p.sie.e, set: (p, v) => (p.sie.e = v),
    fmt: (v) => `${v.toFixed(3)} (q=${ellipticity2phiQ(v, 0).q.toFixed(2)})`,
  },
  {
    id: "phi", label: "Mass position angle", min: 0, max: 180, step: 1,
    get: (p) => p.sie.phi / DEG, set: (p, v) => (p.sie.phi = v * DEG), fmt: (v) => `${v.toFixed(0)}°`,
  },
  {
    id: "shear", label: "External shear γ", min: 0, max: 0.15, step: 0.001,
    get: (p) => p.shear.g, set: (p, v) => (p.shear.g = v), fmt: (v) => v.toFixed(3),
  },
  {
    id: "shearpa", label: "Shear angle", min: 0, max: 180, step: 1,
    get: (p) => p.shear.pa / DEG, set: (p, v) => (p.shear.pa = v * DEG), fmt: (v) => `${v.toFixed(0)}°`,
  },
  {
    id: "fwhm", label: "PSF FWHM", min: 0.04, max: 0.2, step: 0.005,
    get: (p) => p.psf.fwhm_arcsec, set: (p, v) => (p.psf.fwhm_arcsec = v), fmt: (v) => `${v.toFixed(3)}″`,
  },
  {
    id: "srcmag", label: "Source magnitude", min: 19, max: 25, step: 0.05,
    get: (p) => p.photometry.src_mag, set: (p, v) => (p.photometry.src_mag = v), fmt: (v) => v.toFixed(2),
  },
  {
    id: "lensmag", label: "Lens magnitude", min: 18, max: 25, step: 0.05,
    get: (p) => p.photometry.lens_mag, set: (p, v) => (p.photometry.lens_mag = v), fmt: (v) => v.toFixed(2),
  },
  {
    id: "exptime", label: "Exposure time", min: 200, max: 4000, step: 50,
    get: (p) => p.photometry.exptime, set: (p, v) => (p.photometry.exptime = v), fmt: (v) => `${v.toFixed(0)} s`,
  },
];

// ---------------------------------------------------------------------------
// colour map (matplotlib 'magma', 6th-order polynomial fit) and stretches

const MAGMA: [number, number, number][] = [
  [-0.002136485053939582, -0.000749655052795221, -0.005386127855323933],
  [0.2516605407371642, 0.6775232436837668, 2.494026599312351],
  [8.353717279216625, -3.577719514958484, 0.3144679030132573],
  [-27.66873308576866, 14.26473078096533, -13.64921318813922],
  [52.17613981234068, -27.94360607168351, 12.94416944238394],
  [-50.76852536473588, 29.04658282127291, 4.23415299384598],
  [18.65570506591883, -11.48977351997711, -5.601961508734096],
];
const LUT = new Uint8ClampedArray(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = i / 255;
  for (let ch = 0; ch < 3; ch++) {
    let v = 0;
    for (let k = MAGMA.length - 1; k >= 0; k--) v = v * t + MAGMA[k][ch];
    LUT[i * 3 + ch] = Math.round(Math.min(1, Math.max(0, v)) * 255);
  }
}

function quantile(sorted: Float32Array, q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function drawImage(canvas: HTMLCanvasElement, img: Float32Array, stretch: string): void {
  const n = 96;
  const sorted = Float32Array.from(img).sort();
  const lo = quantile(sorted, 0.005);
  const hi = Math.max(quantile(sorted, 0.998), lo + 1e-12);
  const soft = 0.05 * (hi - lo);
  const ctx = canvas.getContext("2d")!;
  const out = ctx.createImageData(n, n);
  for (let i = 0; i < n * n; i++) {
    let t = Math.min(1, Math.max(0, (img[i] - lo) / (hi - lo)));
    if (stretch === "asinh") t = Math.asinh(t * ((hi - lo) / soft)) / Math.asinh((hi - lo) / soft);
    else if (stretch === "log") t = Math.log10(1 + 999 * t) / 3;
    const k = Math.round(t * 255) * 3;
    out.data[i * 4] = LUT[k];
    out.data[i * 4 + 1] = LUT[k + 1];
    out.data[i * 4 + 2] = LUT[k + 2];
    out.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}

function drawColorbar(): void {
  const c = $<HTMLCanvasElement>("colorbar");
  const ctx = c.getContext("2d")!;
  const d = ctx.createImageData(256, 1);
  for (let i = 0; i < 256; i++) {
    d.data[i * 4] = LUT[i * 3];
    d.data[i * 4 + 1] = LUT[i * 3 + 1];
    d.data[i * 4 + 2] = LUT[i * 3 + 2];
    d.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(d, 0, 0);
}

// ---------------------------------------------------------------------------
// UI

function buildSliders(): void {
  const host = $("sliders");
  host.innerHTML = "<h2>Parameters</h2>";
  for (const s of sliders) {
    const row = document.createElement("div");
    row.className = "slider";
    row.innerHTML = `<label for="s-${s.id}">${s.label}</label><output id="o-${s.id}" for="s-${s.id}"></output>
      <input id="s-${s.id}" type="range" min="${s.min}" max="${s.max}" step="${s.step}" />`;
    host.appendChild(row);
    const input = row.querySelector("input")!;
    input.addEventListener("input", () => {
      const v = Number(input.value);
      s.set(current, v);
      $(`o-${s.id}`).textContent = s.fmt(v);
      requestRender();
    });
  }
}

function syncSliders(): void {
  for (const s of sliders) {
    const v = s.get(current);
    const input = $<HTMLInputElement>(`s-${s.id}`);
    input.value = String(v);
    $(`o-${s.id}`).textContent = s.fmt(v);
  }
}

function effectiveParams(): SimParams {
  const p: SimParams = structuredClone(current);
  if (!$<HTMLInputElement>("t-nfw").checked) p.nfw = null;
  if (!$<HTMLInputElement>("t-sub").checked) p.subhalos = [];
  return p;
}

function newLens(): void {
  seed = Math.max(0, Math.floor(Number($<HTMLInputElement>("seed").value) || 0));
  sampled = sampleParams(new Rng(seed));
  current = structuredClone(sampled);
  const nfw = $<HTMLInputElement>("t-nfw");
  const sub = $<HTMLInputElement>("t-sub");
  nfw.disabled = sampled.nfw === null;
  nfw.checked = sampled.nfw !== null;
  sub.disabled = sampled.subhalos.length === 0;
  sub.checked = sampled.subhalos.length > 0;
  const clip = sampled.theta_E_clip.applied
    ? `, clipped to ${sampled.sie.theta_E.toFixed(2)}″ as in simgenv2`
    : sampled.theta_E_clip.out_of_range
      ? " (outside 0.9–1.5″, kept: the 8% case)"
      : "";
  $("lens-origin").textContent =
    `Drawn from the simgenv2 priors: z_l = ${sampled.z_l.toFixed(2)}, z_s = ${sampled.z_s.toFixed(2)}, ` +
    `σ = ${sampled.sigma_kms.toFixed(0)} km/s → θ_E = ${sampled.theta_E_raw.toFixed(2)}″ (Planck15)${clip}. ` +
    `${sampled.subhalos.length} subhalo(s), ${sampled.nfw ? "with" : "no"} group halo, ` +
    `${sampled.source.extras.length} extra source(s).`;
  syncSliders();
  requestRender();
}

function requestRender(): void {
  if (busy) {
    dirty = true;
    return;
  }
  busy = true;
  dirty = false;
  const noise = $<HTMLInputElement>("t-noise").checked;
  post({ type: "render", id: ++requestId, params: effectiveParams(), noiseSeed: seed * 7919 + 17, noise });
}

function redraw(): void {
  if (!lastImages) return;
  const stretch = $<HTMLSelectElement>("stretch").value;
  const noise = $<HTMLInputElement>("t-noise").checked;
  drawImage($("c-GT"), lastImages.GT, stretch);
  drawImage($("c-srcDet"), lastImages.srcDet, stretch);
  drawImage($("c-noiseless"), lastImages.noiseless, stretch);
  drawImage($("c-LENSED"), noise ? lastImages.LENSED : lastImages.noiseless, stretch);
}

function showTruth(t: TruthResult): void {
  const body = $("truth-body");
  body.innerHTML = "";
  TRUTH_ORDER.forEach((name, i) => {
    const on = (t.maskbits >> i) & 1;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td title="${TRUTH_COMMENTS[name]}">${TRUTH_FITS_KEYS[name]}</td><td class="desc">${TRUTH_COMMENTS[name]}</td>
      <td class="num">${t.truth[name].toPrecision(5)}</td>
      <td class="${on ? "yes" : "no"}">${on ? "yes" : "masked"}</td>`;
    body.appendChild(tr);
  });
  $("clean-flag").textContent = t.clean
    ? "TR_CLEAN = 1: the lens is exactly inside the forward-operator family."
    : "TR_CLEAN = 0: subhalos or an offset halo put this lens outside the forward-operator family.";
}

function showDetails(m: any): void {
  const rows: [string, string][] = [
    ["Render time", `${m.timings.total.toFixed(0)} ms`],
    ["Ray-shooting", `${m.timings.rayShoot.toFixed(0)} ms`],
    ["FFT convolution", `${m.timings.convolve.toFixed(0)} ms`],
    ["Source mag (final)", m.srcMagFinal.toFixed(2)],
    ["Lens mag (final)", m.lensMagFinal.toFixed(2)],
    ["Peak ratio src/lens", m.ratio.toFixed(2)],
    ["Ratio-loop tries", String(m.tries)],
    ["Cosmic-ray hits", String(m.cosmicHits)],
  ];
  $("details").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

function onReply(m: any): void {
  if (m.type === "rendered") {
    busy = false;
    pending.length = 0;
    lastImages = m.images;
    redraw();
    showTruth(m.truth);
    showDetails(m);
    $("status").textContent = "";
    if (dirty) requestRender();
  } else if (m.type === "fits") {
    const blob = new Blob([m.bytes], { type: "application/fits" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `astrorim_sim_seed${seed}.fits`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    $("status").textContent = "FITS file downloaded (Primary + GT + LENSED, simgenv2 header keys).";
  } else if (m.type === "error") {
    busy = false;
    $("status").textContent = `Error: ${m.message}`;
  }
}

// Transport: a Web Worker when the browser allows it, otherwise the same code
// on the main thread (module workers are blocked on opaque origins like file://).
let worker: Worker | null = null;
const pending: any[] = [];
try {
  worker = new SimWorker();
  worker.onmessage = (ev: MessageEvent) => onReply(ev.data);
  worker.onerror = () => {
    worker = null;
    $("status").textContent = "Running on the main thread (Web Workers unavailable here).";
    for (const msg of pending.splice(0)) post(msg);
  };
} catch {
  worker = null;
}

function post(msg: any): void {
  if (worker) {
    if (msg.type === "render") pending.splice(0, pending.length, msg);
    worker.postMessage(msg);
  } else {
    setTimeout(() => onReply(handle(msg).message), 0);
  }
}

$("new-lens").addEventListener("click", () => {
  $<HTMLInputElement>("seed").value = String(seed + 1);
  newLens();
});
$("seed").addEventListener("change", newLens);
for (const id of ["t-nfw", "t-sub"]) $(id).addEventListener("change", requestRender);
$("t-noise").addEventListener("change", redraw);
$("stretch").addEventListener("change", redraw);
$("download").addEventListener("click", () => post({ type: "fits", id: ++requestId }));
$("copy-params").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(JSON.stringify(effectiveParams(), null, 2));
    $("status").textContent = "Parameters copied (schema astrorim-sim-params/1).";
  } catch {
    $("status").textContent = "Clipboard not available in this context.";
  }
});

if (REPO_URL) {
  const a = $<HTMLAnchorElement>("repo-link");
  a.href = REPO_URL;
  a.hidden = false;
}

buildSliders();
drawColorbar();
$<HTMLInputElement>("seed").value = "1";
newLens();
