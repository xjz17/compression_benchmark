/**
 * Browser cache for per-dataset statistics JSON (detail panel only).
 * Dataset overview uses GET /api/datasets/?part=summary; the table uses ?part=items.
 */

const STATS_CACHE_VERSION = 5;
const STATS_PREFIX = `wc_dataset_stats_v${STATS_CACHE_VERSION}_`;

function storageKey(prefix, suffix) {
  return `${prefix}${suffix}`;
}

function readJson(key) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function writeJson(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (error) {
    console.warn("dataset statistics cache write failed", error);
    return false;
  }
}

/** @param {string} datasetKey @param {string} [floatPrecision] @param {string} [operator] */
function statisticsCacheStorageKey(datasetKey, floatPrecision = "all", operator = "") {
  const key = String(datasetKey || "").trim();
  const fp = String(floatPrecision || "all").trim() || "all";
  const op = String(operator || "").trim();
  return storageKey(
    STATS_PREFIX,
    `${encodeURIComponent(key)}_${encodeURIComponent(fp)}_${encodeURIComponent(op || "raw")}`,
  );
}

/** @param {string} datasetKey @param {string} [floatPrecision] @param {string} [operator] */
export function readDatasetStatisticsCache(datasetKey, floatPrecision = "all", operator = "") {
  const key = String(datasetKey || "").trim();
  if (!key) return null;
  const entry = readJson(statisticsCacheStorageKey(key, floatPrecision, operator));
  if (!entry || entry.version !== STATS_CACHE_VERSION) {
    return null;
  }
  if (!Array.isArray(entry.rows)) {
    return null;
  }
  return entry.rows;
}

/** @param {string} datasetKey @param {Array<object>} rows @param {string} [floatPrecision] @param {string} [operator] */
export function writeDatasetStatisticsCache(
  datasetKey,
  rows,
  floatPrecision = "all",
  operator = "",
) {
  const key = String(datasetKey || "").trim();
  if (!key || !Array.isArray(rows)) {
    return false;
  }
  return writeJson(statisticsCacheStorageKey(key, floatPrecision, operator), {
    version: STATS_CACHE_VERSION,
    cachedAt: Date.now(),
    rows,
  });
}
