/**
 * Flat LambdaCDM with photons and one massive neutrino species -- a port of
 * astropy's `Planck15` realisation, which simgenv2 uses to turn a velocity
 * dispersion into an Einstein radius (sigma_to_thetaE_arcsec).
 *
 * Derived densities are computed from CODATA constants the same way astropy
 * does (astropy/cosmology/_src/flrw/base.py) and checked against astropy's own
 * numbers in tests/cosmology.test.ts.
 */

// Physical constants (CODATA 2018 as shipped with astropy)
const SIGMA_SB = 5.6703744191844314e-8; // W m^-2 K^-4
const C_M_S = 299792458.0; // m/s
const G_SI = 6.6743e-11; // m^3 kg^-1 s^-2
const KB_EV_K = 8.617333262145179e-5; // eV/K
const MPC_M = 3.085677581491367e22; // m
export const C_KM_S = 299792.458;

// astropy fitting constants (Komatsu et al. 2011, eq. 26)
const NEUTRINO_FERMI_DIRAC_CORRECTION = 0.22710731766; // 7/8 (4/11)^(4/3)
const KOMATSU_P = 1.83;
const KOMATSU_INVP = 0.54644808743;
const KOMATSU_K = 0.3173;
const TNU0_TCMB0 = Math.cbrt(4 / 11);

export interface FlatLCDMParams {
  H0: number; // km/s/Mpc
  Om0: number;
  Tcmb0: number; // K
  Neff: number;
  mNu: number[]; // eV, one entry per species
}

export const PLANCK15: FlatLCDMParams = {
  H0: 67.74,
  Om0: 0.3075,
  Tcmb0: 2.7255,
  Neff: 3.046,
  mNu: [0.0, 0.0, 0.06],
};

export class FlatLambdaCDM {
  readonly hubbleDistance: number; // Mpc
  readonly Ogamma0: number;
  readonly Onu0: number;
  readonly Ode0: number;
  private readonly nuY: number[];
  private readonly nMassless: number;
  private readonly neffPerNu: number;

  constructor(readonly p: FlatLCDMParams = PLANCK15) {
    this.hubbleDistance = C_KM_S / p.H0;
    // critical density today in g/cm^3: 3 H0^2 / (8 pi G)
    const H0_s = (p.H0 * 1000) / MPC_M;
    const G_cgs = G_SI * 1e3; // cm^3 g^-1 s^-2
    const rhoCrit0 = (3 * H0_s * H0_s) / (8 * Math.PI * G_cgs);
    // radiation constant over c^2 in cgs: 4 sigma / c^3
    const aBc2 = (4 * (SIGMA_SB * 1e3)) / Math.pow(C_M_S * 100, 3);
    this.Ogamma0 = (aBc2 * p.Tcmb0 ** 4) / rhoCrit0;

    const Tnu0 = TNU0_TCMB0 * p.Tcmb0;
    const massive = p.mNu.filter((m) => m > 0);
    this.nuY = massive.map((m) => m / (KB_EV_K * Tnu0));
    this.nMassless = p.mNu.length - massive.length;
    this.neffPerNu = p.Neff / p.mNu.length;
    this.Onu0 = this.Ogamma0 * this.nuRelativeDensity(0);
    this.Ode0 = 1.0 - p.Om0 - this.Ogamma0 - this.Onu0;
  }

  nuRelativeDensity(z: number): number {
    let relMass = this.nMassless;
    for (const y of this.nuY) {
      relMass += Math.pow(1.0 + Math.pow((KOMATSU_K * y) / (1.0 + z), KOMATSU_P), KOMATSU_INVP);
    }
    return NEUTRINO_FERMI_DIRAC_CORRECTION * this.neffPerNu * relMass;
  }

  /** 1 / E(z) */
  invEfunc(z: number): number {
    const Or = this.Ogamma0 * (1.0 + this.nuRelativeDensity(z));
    const zp1 = z + 1.0;
    return Math.pow(zp1 ** 3 * (Or * zp1 + this.p.Om0) + this.Ode0, -0.5);
  }

  /** Line-of-sight comoving distance between z1 < z2, in Mpc. */
  comovingDistanceZ1Z2(z1: number, z2: number): number {
    return this.hubbleDistance * integrate((z) => this.invEfunc(z), z1, z2);
  }

  comovingDistance(z: number): number {
    return this.comovingDistanceZ1Z2(0, z);
  }

  angularDiameterDistance(z: number): number {
    return this.comovingDistance(z) / (1.0 + z);
  }

  /** Flat universe: D_A(z1, z2) = D_C(z1, z2) / (1 + z2). */
  angularDiameterDistanceZ1Z2(z1: number, z2: number): number {
    return this.comovingDistanceZ1Z2(z1, z2) / (1.0 + z2);
  }
}

export const planck15 = new FlatLambdaCDM(PLANCK15);

/**
 * simgenv2.sigma_to_thetaE_arcsec: SIS Einstein radius for velocity dispersion
 * sigma (km/s), lens at z_lens, source at z_source.
 */
export function sigmaToThetaEArcsec(
  sigmaKms: number,
  zLens: number,
  zSource: number,
  cosmo: FlatLambdaCDM = planck15,
): number {
  const Ds = cosmo.angularDiameterDistance(zSource);
  const Dls = cosmo.angularDiameterDistanceZ1Z2(zLens, zSource);
  const thetaRad = ((4.0 * Math.PI * sigmaKms ** 2) / C_KM_S ** 2) * (Dls / Ds);
  return thetaRad * (180.0 / Math.PI) * 3600.0;
}

// ---------------------------------------------------------------------------
// Composite 20-point Gauss-Legendre quadrature. The integrand 1/E(z) is smooth,
// so panels of width <= 0.25 converge to ~1e-15 relative.

const GL = gaussLegendre(20);

function integrate(f: (x: number) => number, a: number, b: number): number {
  if (a === b) return 0;
  const panels = Math.max(1, Math.ceil(Math.abs(b - a) / 0.25));
  const h = (b - a) / panels;
  let total = 0;
  for (let p = 0; p < panels; p++) {
    const lo = a + p * h;
    const mid = lo + h / 2;
    let s = 0;
    for (let i = 0; i < GL.x.length; i++) s += GL.w[i] * f(mid + (h / 2) * GL.x[i]);
    total += (h / 2) * s;
  }
  return total;
}

function gaussLegendre(n: number): { x: number[]; w: number[] } {
  const x: number[] = new Array(n);
  const w: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5));
    let pp = 0;
    for (let iter = 0; iter < 100; iter++) {
      let p1 = 1.0;
      let p2 = 0.0;
      for (let j = 1; j <= n; j++) {
        const p3 = p2;
        p2 = p1;
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
      }
      pp = (n * (z * p1 - p2)) / (z * z - 1);
      const z1 = z;
      z = z1 - p1 / pp;
      if (Math.abs(z - z1) < 1e-15) break;
    }
    x[i] = z;
    w[i] = 2 / ((1 - z * z) * pp * pp);
  }
  return { x, w };
}
