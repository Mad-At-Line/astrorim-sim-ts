// Mutation check: inject typical porting bugs one at a time and confirm the
// test suite fails for each. A mutation that survives is either a gap in the
// tests or an "equivalent mutant" (a change that cannot alter the output).
//
//   npm run mutation-check
//
// Works on Windows, macOS and Linux. Files are always restored, even on Ctrl-C.

import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";

const MUTATIONS = [
  {
    label: "Python floor-mod replaced by JS % in _wrap_axis_angle",
    file: "src/truth.ts",
    find: "a = pymod(a + Math.PI / 2.0, Math.PI) - Math.PI / 2.0;",
    replace: "a = ((a + Math.PI / 2.0) % Math.PI) - Math.PI / 2.0;",
    expectEquivalent: true, // the following `if (a <= -pi/2) a += pi` absorbs the difference
  },
  {
    label: "SIE: wrong sign when rotating the deflection back",
    file: "src/lens.ts",
    find: "ax[i] += fx * cos + fy * -sin;",
    replace: "ax[i] += fx * cos + fy * sin;",
  },
  {
    label: "SIE: theta_E used as the major-axis b (lenstronomy uses the product average)",
    file: "src/lens.ts",
    find: "const b = thetaEMajor * Math.sqrt((1 + q0 * q0) / 2);",
    replace: "const b = kw.theta_E;",
  },
  {
    label: "Sersic: sign of e1 flipped in the elliptical radius",
    file: "src/light.ts",
    find: "const x_ = ((1 - e1) * xs - e2 * ys) / norm;",
    replace: "const x_ = ((1 + e1) * xs - e2 * ys) / norm;",
  },
  {
    label: "Sersic: exact-ish b_n = 2n - 1/3 instead of lenstronomy's 1.9992n - 0.3271",
    file: "src/light.ts",
    find: "return Math.max(1.9992 * n - 0.3271, 0.00001);",
    replace: "return Math.max(2 * n - 1 / 3, 0.00001);",
  },
  {
    label: "Interpolation: x and y swapped (row/column confusion)",
    file: "src/simulate.ts",
    find: "const sbMappedUnit = bilinearOnGrid(grid.lin, sbUnit, xs, ys);",
    replace: "const sbMappedUnit = bilinearOnGrid(grid.lin, sbUnit, ys, xs);",
  },
  {
    label: "Convolution output shifted by one pixel",
    file: "src/fft.ts",
    find: "out[r * N + c] = re[(r + kc) * P + (c + kc)];",
    replace: "out[r * N + c] = re[(r + kc) * P + (c + kc + 1)];",
  },
  {
    label: "Super-grid spacing 0.01 exactly instead of numpy.linspace endpoints",
    file: "src/image.ts",
    find: "for (let i = 0; i < num; i++) out[i] = i * step + start;\n  out[num - 1] = stop;",
    replace: "for (let i = 0; i < num; i++) out[i] = start + 0.01 * i;",
  },
  {
    label: "GT normalised with the post-loop source magnitude",
    file: "src/simulate.ts",
    find: "const GT = downsample(srcCountsInitial, S, C.OVERSAMPLE).map(Math.fround);",
    replace: "let GT = downsample(srcCountsInitial, S, C.OVERSAMPLE).map(Math.fround);",
    also: [
      "const srcTotal = totalCounts(srcMag, zp, exptime);",
      "const srcTotal = totalCounts(srcMag, zp, exptime);\n  GT = downsample(countsSup(srcUnit, srcTotal), S, C.OVERSAMPLE).map(Math.fround);",
    ],
  },
];

const originals = new Map();
function restoreAll() {
  for (const [file, text] of originals) writeFileSync(file, text);
}
process.on("SIGINT", () => {
  restoreAll();
  process.exit(130);
});

let caught = 0;
let equivalent = 0;
let survived = 0;
for (const m of MUTATIONS) {
  const src = readFileSync(m.file, "utf8");
  if (!src.includes(m.find)) {
    console.error(`!! could not apply "${m.label}" (pattern not found in ${m.file})`);
    process.exitCode = 2;
    continue;
  }
  originals.set(m.file, src);
  let mutated = src.replace(m.find, m.replace);
  if (m.also) mutated = mutated.replace(m.also[0], m.also[1]);
  writeFileSync(m.file, mutated);
  const res = spawnSync("npx", ["vitest", "run", "--reporter=dot"], { shell: true, encoding: "utf8" });
  writeFileSync(m.file, src);
  originals.delete(m.file);
  const failedLine = (res.stdout + res.stderr).split("\n").find((l) => /Tests\s+\d+ failed/.test(l)) ?? "";
  if (res.status !== 0) {
    caught++;
    console.log(`caught      ${m.label}\n            ${failedLine.trim()}`);
  } else if (m.expectEquivalent) {
    equivalent++;
    console.log(`equivalent  ${m.label}`);
  } else {
    survived++;
    console.log(`SURVIVED    ${m.label}   <-- the tests do not detect this bug`);
  }
}
restoreAll();
console.log(`\n${caught} caught, ${equivalent} equivalent, ${survived} survived (of ${MUTATIONS.length})`);
if (survived > 0) process.exitCode = 1;
