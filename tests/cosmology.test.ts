import { describe, expect, it } from "vitest";
import { planck15, sigmaToThetaEArcsec } from "../src/cosmology";
import { loadCases, loadUnits } from "./golden";

const ref = loadUnits().cosmology;

describe("Planck15 cosmology vs astropy", () => {
  it("derived density parameters", () => {
    expect(planck15.Ogamma0 / ref.Ogamma0).toBeCloseTo(1, 12);
    expect(planck15.Onu0 / ref.Onu0).toBeCloseTo(1, 12);
    expect(planck15.Ode0 / ref.Ode0).toBeCloseTo(1, 12);
  });

  it.each(ref.D_C as [number, number][])("comoving distance z=%f", (z, dc) => {
    expect(planck15.comovingDistance(z) / dc).toBeCloseTo(1, 10);
  });

  it.each(ref.D_A as [number, number][])("angular diameter distance z=%f", (z, da) => {
    expect(planck15.angularDiameterDistance(z) / da).toBeCloseTo(1, 10);
  });

  it.each(ref.D_A12 as [number, number, number][])("D_A(z1=%f, z2=%f)", (z1, z2, d) => {
    expect(planck15.angularDiameterDistanceZ1Z2(z1, z2) / d).toBeCloseTo(1, 10);
  });

  it.each(ref.thetaE as [number, number, number, number][])(
    "sigma_to_thetaE_arcsec(sigma=%f, zl=%f, zs=%f)",
    (s, zl, zs, th) => {
      expect(sigmaToThetaEArcsec(s, zl, zs) / th).toBeCloseTo(1, 10);
    },
  );

  it("reproduces theta_E_raw of every golden case", () => {
    for (const g of loadCases()) {
      const p = g.params;
      expect(sigmaToThetaEArcsec(p.sigma_kms, p.z_l, p.z_s) / p.theta_E_raw).toBeCloseTo(1, 10);
    }
  });
});
