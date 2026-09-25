import { describe, expect, it } from "vitest";
import * as C from "../src/config";
import type { LensComponent } from "../src/lens";
import { makePsfKernel } from "../src/psf";
import { render } from "../src/simulate";
import { downsample } from "../src/image";
import { measureLensLightAmpAtRe, pymod, truthFromLenstronomy, TRUTH_ORDER, wrapAxisAngle } from "../src/truth";
import { decode, loadCases, loadUnits, relMaxDiff } from "./golden";

const units = loadUnits();

function toComponents(list: string[], kwargs: any[]): LensComponent[] {
  return list.map((type, i) => ({ type, kw: kwargs[i] }) as LensComponent);
}

describe("truth labels (simgen_truth.py)", () => {
  it("Python floor-mod semantics", () => {
    expect(pymod(-1, 3)).toBe(2);
    expect(pymod(5.5, Math.PI)).toBeCloseTo(5.5 - Math.PI, 15);
    expect(pymod(-0.5, Math.PI)).toBeCloseTo(Math.PI - 0.5, 15);
  });

  it.each(units.wrap_axis_angle as [number, number][])("_wrap_axis_angle(%f)", (a, expected) => {
    expect(wrapAxisAngle(a)).toBeCloseTo(expected, 14);
  });

  it.each(units.truth.map((t: any, i: number) => [i, t]) as [number, any][])("truth_from_lenstronomy config %i", (_i, t: any) => {
    const res = truthFromLenstronomy(toComponents(t.lens_model_list, t.kwargs_lens), C.PIXEL_SCALE, C.IMAGE_SIZE, t.psf_info, {
      Re_arcsec: 0.5,
      n_sersic: 4.0,
      amp_at_Re_normalized: 0.02,
    });
    expect(res.maskbits).toBe(t.maskbits);
    expect(res.clean).toBe(t.clean);
    for (const k of TRUTH_ORDER) expect(res.truth[k]).toBeCloseTo(t.truth[k], 14);
  });
});

describe("truth labels on the golden cases", () => {
  for (const g of loadCases().filter((c) => c.truth)) {
    it(`${g.name}: lens_flux measurement and all TR_* values`, () => {
      const r = render(g.params);
      const det = downsample(r.lensCountsSup, C.SUPER_SIZE, C.OVERSAMPLE).map(Math.fround);
      const llAmp = measureLensLightAmpAtRe(det, r.llCenter, g.params.lens_light.R_sersic, C.PIXEL_SCALE, g.truth!.norm_m);
      if (g.truth!.ll_amp === null) expect(llAmp).toBeNull();
      else expect(llAmp! / g.truth!.ll_amp).toBeCloseTo(1, 5);

      const res = truthFromLenstronomy(
        r.lens,
        C.PIXEL_SCALE,
        C.IMAGE_SIZE,
        {
          realized_fwhm_pix: (r.psf.realizedFwhmPix * C.SUP_PIXEL_SCALE) / C.PIXEL_SCALE,
          psf_type: C.PSF_TYPE,
          beta: 3.5,
          ellip: g.params.psf.ellip,
          angle: g.params.psf.angle,
        },
        { Re_arcsec: g.params.lens_light.R_sersic, n_sersic: g.params.lens_light.n_sersic, amp_at_Re_normalized: g.truth!.ll_amp },
      );
      expect(res.maskbits).toBe(g.truth!.maskbits);
      expect(res.clean).toBe(g.truth!.clean);
      for (const k of TRUTH_ORDER) expect(res.truth[k]).toBeCloseTo(g.truth!.values[k], 13);
    });
  }
});

describe("PSF kernels (incl. Moffat and coma branches not used by simgenv2)", () => {
  it.each(units.psf.map((p: any) => [p.args.type, p.args.fwhm_arcsec, p]) as [string, number, any][])("%s fwhm=%f", (_t, _f, p: any) => {
    const k = makePsfKernel(p.args);
    const ref = decode(p.kernel);
    expect(k.kernel.length).toBe(ref.length);
    expect(relMaxDiff(k.kernel, ref)).toBeLessThan(1e-6);
    expect(k.comaApplied).toBe(p.info.coma_applied);
    expect(k.realizedFwhmPix).toBeCloseTo(p.info.realized_fwhm_pix, 12);
  });
});
