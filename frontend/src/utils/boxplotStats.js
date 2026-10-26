/**
 * Boxplot stats aligned with matplotlib ``boxplot(..., whis=(5, 95))``.
 */

function percentile(sorted, p) {
  const n = sorted.length;
  if (n === 0) return NaN;
  if (n === 1) return sorted[0];
  const idx = (p / 100) * (n - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

/**
 * @param {string[]} algorithms
 * @param {number[][]} valuesByAlgorithm Parallel arrays of raw sample values.
 * @param {{ whis?: [number, number], logY?: boolean }} options
 * @returns {{ categories: string[], boxData: number[][], outliers: number[][] }}
 */
export function buildBoxplotSeries(algorithms, valuesByAlgorithm, options = {}) {
  const whis = options.whis ?? [5, 95];
  const logY = Boolean(options.logY);
  const categories = [];
  const rows = [];
  const outliers = [];

  (algorithms || []).forEach((label, xi) => {
    const raw = valuesByAlgorithm?.[xi];
    let pts = (Array.isArray(raw) ? raw : [])
      .map((v) => Number(v))
      .filter((v) => Number.isFinite(v));
    if (logY) {
      pts = pts.filter((v) => v > 0);
    }
    if (!pts.length) {
      return;
    }
    pts.sort((a, b) => a - b);
    const q1 = percentile(pts, 25);
    const median = percentile(pts, 50);
    const q3 = percentile(pts, 75);
    const low = percentile(pts, whis[0]);
    const high = percentile(pts, whis[1]);
    const row = {
      label: String(label),
      median,
      box: [low, q1, median, q3, high],
      outlierValues: [],
    };
    pts.forEach((value) => {
      if (value < low || value > high) {
        row.outlierValues.push(value);
      }
    });
    rows.push(row);
  });

  rows.sort((a, b) => a.median - b.median || a.label.localeCompare(b.label));
  const boxData = [];
  rows.forEach((row, catIndex) => {
    categories.push(row.label);
    boxData.push(row.box);
    row.outlierValues.forEach((value) => {
      outliers.push([catIndex, value]);
    });
  });

  return { categories, boxData, outliers };
}

export function formatBoxplotAxisValue(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "";
  if (num === 0) return "0";
  const abs = Math.abs(num);
  if (abs >= 1000 || abs < 0.001) {
    return num.toExponential(1);
  }
  if (abs >= 10) return num.toFixed(1);
  if (abs >= 1) return num.toFixed(2);
  return num.toFixed(3);
}
