import { isActiveMethodName } from "./activeMethodPolicy.js";

const normalizeKey = (value) => String(value ?? "").trim().replace(/^\ufeff/, "").toLowerCase();

const normalizeLabel = (value) => String(value ?? "").trim().replace(/\s+/g, " ").toLowerCase();

export const HOME_LEADERBOARD_DISPLAY_ALGORITHM_KEYS = Object.freeze([
  "DELTA_BINARY_PACKED",
  "DBP",
]);

const HOME_LEADERBOARD_DISPLAY_KEY_ALIASES = Object.freeze([
  ...HOME_LEADERBOARD_DISPLAY_ALGORITHM_KEYS,
  "2DIFF",
  "SVB",
  "spdp_float_level_9",
  "ts_2diff_bos_int",
  "ts_2diff_bos_float",
]);

export const HOME_LEADERBOARD_DISPLAY_ALGORITHM_LABELS = Object.freeze([
  "2DIFF",
  "ALP",
  "ANS",
  "Arithmetic",
  "Blosc",
  "Blosc2",
  "BOS",
  "BriefLZ",
  "Brotli",
  "BSC",
  "BUFF",
  "Bzip2",
  "Bzip3",
  "Camel",
  "CHIMP",
  "Chimp128",
  "Dictionary",
  "CORAD",
  "Crush",
  "CSC",
  "DBP",
  "DELTA_BINARY_PACKED",
  "Density",
  "Elf",
  "Elf+",
  "FastLZ",
  "FastLZMA2",
  "FastPFor",
  "FPC",
  "FPCompress DPratio",
  "FPCompress DPspeed",
  "FPZIP",
  "FSE",
  "Gipfeli",
  "GLZA",
  "GORILLA",
  "GZIP",
  "Huff0",
  "Huffman",
  "Kanzi",
  "LECO",
  "Libdeflate",
  "Lizard",
  "LZ4",
  "LZ4 Fast",
  "LZ4HC",
  "LZAV",
  "LZF",
  "LZFSE",
  "LZG",
  "LZHAM",
  "LZJB",
  "Lzlib",
  "LZMAT",
  "LZO1",
  "LZO2A",
  "LZRW",
  "LZSSE",
  "LZVN",
  "PFPC",
  "Pithy",
  "Pongo",
  "PPMD8",
  "QuickLZ",
  "REGER",
  "RLBE",
  "RLE",
  "Shrinker",
  "Simple8b",
  "SLZ",
  "SNAPPY",
  "SPDP",
  "SPRINTZ",
  "Stream VByte",
  "SVB",
  "TAMP",
  "Tornado",
  "TS_2DIFF",
  "UCL NRV",
  "WFLZ",
  "XPACK",
  "XZ",
  "YALZ77",
  "Yappy",
  "Zlib",
  "Zlib-ng",
  "Zling",
  "ZSTD",
  "ZXC",
].filter(isActiveMethodName));

const ALLOWED_KEY_SET = new Set(HOME_LEADERBOARD_DISPLAY_KEY_ALIASES.map(normalizeKey));
const ALLOWED_LABEL_SET = new Set(HOME_LEADERBOARD_DISPLAY_ALGORITHM_LABELS.map(normalizeLabel));

function normalizeBaseLabel(value) {
  return normalizeLabel(String(value ?? "").replace(/\s*\([^)]*\)/g, ""));
}

export function isHomeLeaderboardDisplayAlgorithmKey(value) {
  return ALLOWED_KEY_SET.has(normalizeKey(value));
}

export function isHomeLeaderboardDisplayAlgorithmLabel(value) {
  const label = normalizeLabel(value);
  return ALLOWED_LABEL_SET.has(label) || ALLOWED_LABEL_SET.has(normalizeBaseLabel(value));
}

export function isHomeLeaderboardDisplayRow(row) {
  if (!row || typeof row !== "object") return false;
  return (
    isHomeLeaderboardDisplayAlgorithmKey(row.algorithm_key) ||
    isHomeLeaderboardDisplayAlgorithmKey(row.algorithm) ||
    isHomeLeaderboardDisplayAlgorithmLabel(row.algorithm)
  );
}

export function filterHomeLeaderboardRows(rows) {
  return (Array.isArray(rows) ? rows : []).filter(isHomeLeaderboardDisplayRow);
}

function filterRankGroups(rankGroups) {
  return (Array.isArray(rankGroups) ? rankGroups : [])
    .map((group) => {
      const algorithms = (Array.isArray(group?.algorithms) ? group.algorithms : [])
        .filter(isHomeLeaderboardDisplayAlgorithmLabel);
      return { ...group, algorithms };
    })
    .filter((group) => group.algorithms.length >= 2);
}

function filterChartByAlgorithmLabel(chart) {
  if (!chart || typeof chart !== "object") return chart;
  const algorithms = Array.isArray(chart.algorithms) ? chart.algorithms : [];
  const values = Array.isArray(chart.values) ? chart.values : [];
  const keepIndexes = algorithms
    .map((algorithm, index) => ({ algorithm, index }))
    .filter(({ algorithm }) => isHomeLeaderboardDisplayAlgorithmLabel(algorithm));
  const out = {
    ...chart,
    algorithms: keepIndexes.map(({ algorithm }) => algorithm),
    values: keepIndexes.map(({ index }) => (index < values.length ? values[index] : [])),
  };
  if (Array.isArray(chart.rank_groups)) {
    out.rank_groups = filterRankGroups(chart.rank_groups);
  }
  return out;
}

export function filterHomeLeaderboardBoxplotGroups(groups) {
  if (!groups || typeof groups !== "object") return groups;
  return Object.fromEntries(
    Object.entries(groups).map(([metricKey, group]) => [
      metricKey,
      {
        ...group,
        charts: (Array.isArray(group?.charts) ? group.charts : [])
          .map(filterChartByAlgorithmLabel),
      },
    ]),
  );
}

function isLeveledParetoPoint(point) {
  return point?.level_order !== null && point?.level_order !== undefined && point?.level_order !== "";
}

function isHomeLeaderboardParetoPoint(point) {
  if (!point || typeof point !== "object") return false;
  if (isHomeLeaderboardDisplayAlgorithmKey(point.algorithm_key)) return true;
  if (isLeveledParetoPoint(point)) {
    return isHomeLeaderboardDisplayAlgorithmLabel(point.algorithm);
  }
  return (
    isHomeLeaderboardDisplayAlgorithmLabel(point.algorithm) ||
    isHomeLeaderboardDisplayAlgorithmLabel(point.family)
  );
}

function withRecomputedPareto(points) {
  return points.map((point, index) => {
    const rate = Number(point?.rate);
    const compressionTime = Number(point?.compression_time);
    const pareto = points.every((candidate, candidateIndex) => {
      if (candidateIndex === index) return true;
      const otherRate = Number(candidate?.rate);
      const otherTime = Number(candidate?.compression_time);
      if (!Number.isFinite(otherRate) || !Number.isFinite(otherTime)) return true;
      return !(
        otherRate <= rate &&
        otherTime <= compressionTime &&
        (otherRate < rate || otherTime < compressionTime)
      );
    });
    return { ...point, pareto };
  });
}

function frontierFromPoints(points) {
  const seen = new Set();
  return points
    .filter((point) => point?.pareto)
    .map((point) => [Number(point.rate), Number(point.compression_time)])
    .filter(([rate, time]) => Number.isFinite(rate) && Number.isFinite(time))
    .sort((a, b) => a[0] - b[0] || a[1] - b[1])
    .filter(([rate, time]) => {
      const key = `${rate}:${time}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

export function filterHomeLeaderboardParetoChart(chart) {
  if (!chart || typeof chart !== "object") return chart;
  const points = withRecomputedPareto(
    (Array.isArray(chart.points) ? chart.points : []).filter(isHomeLeaderboardParetoPoint),
  );
  const levelSeries = (Array.isArray(chart.level_series) ? chart.level_series : [])
    .map((series) => {
      const seriesPoints = (Array.isArray(series?.points) ? series.points : [])
        .filter(isHomeLeaderboardParetoPoint);
      return { ...series, points: seriesPoints };
    })
    .filter((series) => series.points.length >= 2);
  return {
    ...chart,
    points,
    level_series: levelSeries,
    frontier: frontierFromPoints(points),
  };
}
