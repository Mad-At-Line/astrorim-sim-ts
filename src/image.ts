/**
 * Array helpers mirroring the numpy/scipy calls in simgenv2. Images are
 * row-major Float64Arrays: index = row * width + col, row = y, col = x.
 */

/** np.linspace(start, stop, num) with endpoint=True (numpy's formula). */
export function linspace(start: number, stop: number, num: number): Float64Array {
  const out = new Float64Array(num);
  if (num === 1) {
    out[0] = start;
    return out;
  }
  const step = (stop - start) / (num - 1);
  for (let i = 0; i < num; i++) out[i] = i * step + start;
  out[num - 1] = stop;
  return out;
}

/** np.meshgrid(g, g) flattened: X[r, c] = g[c], Y[r, c] = g[r]. */
export function meshgrid(g: Float64Array): { X: Float64Array; Y: Float64Array } {
  const n = g.length;
  const X = new Float64Array(n * n);
  const Y = new Float64Array(n * n);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      X[r * n + c] = g[c];
      Y[r * n + c] = g[r];
    }
  }
  return { X, Y };
}

/** simgenv2.downsample: mean over factor x factor blocks. */
export function downsample(img: Float64Array, size: number, factor: number): Float64Array {
  const n = Math.floor(size / factor);
  const out = new Float64Array(n * n);
  const inv = 1 / (factor * factor);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      let s = 0;
      for (let dr = 0; dr < factor; dr++) {
        const row = (r * factor + dr) * size + c * factor;
        for (let dc = 0; dc < factor; dc++) s += img[row + dc];
      }
      out[r * n + c] = s * inv;
    }
  }
  return out;
}

/** np.percentile(arr, q) with the default 'linear' method. */
export function percentile(arr: ArrayLike<number>, q: number): number {
  const sorted = Float64Array.from(arr).sort();
  const pos = ((sorted.length - 1) * q) / 100;
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, sorted.length - 1);
  const t = pos - lo;
  const a = sorted[lo];
  const b = sorted[hi];
  // numpy's _lerp: a + t*(b-a), evaluated from the upper end when t >= 0.5
  return t >= 0.5 ? b - (b - a) * (1 - t) : a + (b - a) * t;
}

export function maxOf(arr: ArrayLike<number>): number {
  let m = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
  return m;
}

export function sumOf(arr: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s;
}

/** Round every element to float32 precision (mirrors numpy .astype(np.float32)). */
export function toF32Precision(arr: Float64Array): Float64Array {
  for (let i = 0; i < arr.length; i++) arr[i] = Math.fround(arr[i]);
  return arr;
}

/**
 * scipy RegularGridInterpolator((gy, gx), values, method='linear',
 * bounds_error=False, fill_value=0) evaluated at points (py[i], px[i]).
 * Assumes both axes are the same uniform grid g (true for simgenv2).
 */
export function bilinearOnGrid(
  g: Float64Array,
  values: Float64Array,
  px: ArrayLike<number>,
  py: ArrayLike<number>,
  out: Float64Array = new Float64Array(px.length),
): Float64Array {
  const n = g.length;
  const g0 = g[0];
  const gN = g[n - 1];
  const step = (gN - g0) / (n - 1);
  const locate = (v: number): number => {
    let i = Math.floor((v - g0) / step);
    if (i < 0) i = 0;
    if (i > n - 2) i = n - 2;
    // correct for rounding so that g[i] <= v <= g[i+1]
    while (i > 0 && v < g[i]) i--;
    while (i < n - 2 && v > g[i + 1]) i++;
    return i;
  };
  for (let k = 0; k < px.length; k++) {
    const x = px[k];
    const y = py[k];
    if (!(x >= g0 && x <= gN && y >= g0 && y <= gN)) {
      out[k] = 0.0;
      continue;
    }
    const ix = locate(x);
    const iy = locate(y);
    const tx = (x - g[ix]) / (g[ix + 1] - g[ix]);
    const ty = (y - g[iy]) / (g[iy + 1] - g[iy]);
    const v00 = values[iy * n + ix];
    const v01 = values[iy * n + ix + 1];
    const v10 = values[(iy + 1) * n + ix];
    const v11 = values[(iy + 1) * n + ix + 1];
    out[k] =
      v00 * (1 - ty) * (1 - tx) + v01 * (1 - ty) * tx + v10 * ty * (1 - tx) + v11 * ty * tx;
  }
  return out;
}
