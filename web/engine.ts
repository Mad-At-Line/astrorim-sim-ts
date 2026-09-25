/**
 * Message handler shared by the Web Worker and the main-thread fallback
 * (used when the page runs from an opaque origin such as file://, where
 * module workers are blocked).
 */

import { simulationToFits } from "../src/output";
import type { SimParams } from "../src/params";
import { Rng } from "../src/rng";
import { applyNoise, render, truthFor, type SimulationResult } from "../src/simulate";

let last: SimulationResult | null = null;

export interface Reply {
  message: any;
  transfer: Transferable[];
}

export function handle(msg: any): Reply {
  try {
    if (msg.type === "render") {
      const p: SimParams = msg.params;
      const r = render(p);
      const noise = applyNoise(r.lensedCounts, p.detector.read_noise, p.detector.sky_adu, new Rng(msg.noiseSeed));
      const { truth, lensLightDet, normM, llAmp } = truthFor(p, r, noise.LENSED);
      last = { ...r, params: p, LENSED: noise.LENSED, noise, truth, lensLightDet, normM, llAmp };
      const f32 = (a: Float64Array) => Float32Array.from(a);
      const images = {
        GT: f32(r.GT),
        srcDet: f32(r.srcDet),
        lensDet: f32(r.lensDet),
        noiseless: f32(r.lensedCounts),
        LENSED: f32(noise.LENSED),
      };
      return {
        message: {
          type: "rendered",
          id: msg.id,
          images,
          truth,
          timings: r.timingsMs,
          srcMagFinal: r.srcMagFinal,
          lensMagFinal: r.lensMagFinal,
          tries: r.tries,
          ratio: r.ratioHistory[r.ratioHistory.length - 1],
          cosmicHits: noise.cosmicHits.length,
          llAmp,
        },
        transfer: Object.values(images).map((a) => a.buffer),
      };
    }
    if (msg.type === "fits") {
      if (!last) throw new Error("nothing rendered yet");
      const bytes = simulationToFits(last);
      return { message: { type: "fits", id: msg.id, bytes }, transfer: [bytes.buffer] };
    }
    throw new Error(`unknown message type ${msg.type}`);
  } catch (e) {
    return { message: { type: "error", id: msg.id, message: String((e as Error).message ?? e) }, transfer: [] };
  }
}
