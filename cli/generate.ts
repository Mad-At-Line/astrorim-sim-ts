/**
 * Generate simgenv2-regime simulations as FITS files from Node.
 *
 *   npm run generate -- --n 20 --seed 1 --out sims_ts
 *
 * Files are named rim_sim2_ts_{i:05d}.fits and use the same HDUs and header
 * keys as simgenv2 (Primary + GT + LENSED), plus SIMCODE naming this generator.
 * A params JSON sidecar is written next to each file with --params.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { simulationToFits } from "../src/output";
import { sampleParams } from "../src/params";
import { Rng } from "../src/rng";
import { simulate } from "../src/simulate";

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const n = Number(arg("n", "20"));
const seed = Number(arg("seed", String(Date.now() % 2 ** 31)));
const out = arg("out", "sims_ts");
const withParams = process.argv.includes("--params");

mkdirSync(out, { recursive: true });
const rng = new Rng(seed);
const t0 = performance.now();
for (let i = 0; i < n; i++) {
  const p = sampleParams(rng);
  const sim = simulate(p, rng);
  const file = join(out, `rim_sim2_ts_${String(i).padStart(5, "0")}.fits`);
  writeFileSync(file, simulationToFits(sim));
  if (withParams) writeFileSync(file.replace(/\.fits$/, ".params.json"), JSON.stringify(p, null, 2));
  const ratio = sim.ratioHistory[sim.ratioHistory.length - 1];
  console.log(`[${i}] Saved ${file}  | theta_E=${p.sie.theta_E.toFixed(3)}" peak(src/lens)=${ratio.toFixed(2)} ${sim.timingsMs.total.toFixed(0)} ms`);
}
console.log(`\n${n} simulations in ${((performance.now() - t0) / 1000).toFixed(1)} s (seed ${seed})`);
