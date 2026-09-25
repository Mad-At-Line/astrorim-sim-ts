import { describe, expect, it } from "vitest";
import { formatReal, writeFits } from "../src/fits";
import { simulationToFits } from "../src/output";
import { sampleParams } from "../src/params";
import { Rng } from "../src/rng";
import { simulate } from "../src/simulate";
import { TRUTH_FITS_KEYS } from "../src/truth";

function parseHeader(bytes: Uint8Array, offset: number): { cards: Map<string, string>; end: number } {
  const cards = new Map<string, string>();
  let pos = offset;
  for (;;) {
    const card = new TextDecoder("ascii").decode(bytes.subarray(pos, pos + 80));
    pos += 80;
    const key = card.slice(0, 8).trim();
    if (key === "END") break;
    if (card[8] === "=") cards.set(key, card.slice(10).split(" / ")[0].trim());
  }
  const end = Math.ceil((pos - offset) / 2880) * 2880 + offset;
  return { cards, end };
}

describe("FITS writer", () => {
  it.each([0.04, 25.94, 1e-30, 123456789.123, 5.7e-4, 1, -2.5e17])(
    "formatReal(%f) round-trips within 20 characters",
    (v) => {
      const s = formatReal(v);
      expect(s.length).toBeLessThanOrEqual(20);
      expect(s).toMatch(/[.E]/);
      expect(Number(s)).toBe(v);
    },
  );

  it("values needing more than 20 characters are shortened exactly like astropy", () => {
    // simgenv2 wrote TR_X0 = -0.00430135524038875 for this value (seed 1); a FITS
    // card has 20 columns for the value, so 1 digit is lost in both implementations.
    const v = -0.004301355240388752;
    expect(formatReal(v)).toBe("-0.00430135524038875");
  });

  it("writes 2880-byte blocks with a valid primary header and image extensions", () => {
    const data = Float32Array.from({ length: 6 }, (_, i) => i + 0.5);
    const out = writeFits([{ key: "FOO", value: 1.25, comment: "test" }], [{ name: "IMG", data, width: 3, height: 2 }]);
    expect(out.length % 2880).toBe(0);
    const h0 = parseHeader(out, 0);
    expect(h0.cards.get("SIMPLE")).toBe("T");
    expect(h0.cards.get("NAXIS")).toBe("0");
    expect(h0.cards.get("FOO")).toBe("1.25");
    const h1 = parseHeader(out, h0.end);
    expect(h1.cards.get("XTENSION")).toBe("'IMAGE   '");
    expect(h1.cards.get("BITPIX")).toBe("-32");
    expect(h1.cards.get("NAXIS1")).toBe("3");
    expect(h1.cards.get("NAXIS2")).toBe("2");
    expect(h1.cards.get("EXTNAME")).toBe("'IMG     '");
    const view = new DataView(out.buffer, h1.end, 24);
    for (let i = 0; i < 6; i++) expect(view.getFloat32(i * 4, false)).toBe(i + 0.5);
  });

  it("simulation files carry every simgenv2 header key", () => {
    const rng = new Rng(11);
    const sim = simulate(sampleParams(rng), rng);
    const out = simulationToFits(sim);
    const h0 = parseHeader(out, 0);
    const expected = [
      "DATE", "GAIN", "RN_E", "SKYADU", "PSF_FWH", "PSF_TYP", "THETA_E", "SIGMA", "ELLIP", "SHEAR",
      "SRCMAG", "LENSMAG", "ZLENS", "ZSRC", "PRNU", "SKYGRAD", ...Object.values(TRUTH_FITS_KEYS),
      "TR_MASK", "TR_CLEAN", "TR_SET", "TR_PS", "TR_COMA", "TR_VER",
    ];
    for (const k of expected) expect(h0.cards.has(k), k).toBe(true);
    const gt = parseHeader(out, h0.end);
    expect(gt.cards.get("EXTNAME")).toBe("'GT      '");
    const lensedStart = gt.end + Math.ceil((96 * 96 * 4) / 2880) * 2880;
    expect(parseHeader(out, lensedStart).cards.get("EXTNAME")).toBe("'LENSED  '");
  });
});
