/**
 * lzbench codec catalog (mirrors backend/myapp/lzbench_codec_catalog.py).
 */

export const MAX_SAMPLED_LEVELS = 10;
export const ENABLE_LZBENCH_RUNTIME_CATALOG = true;

/** @typedef {{ codecId: string, displayName: string, firstLevel?: number, lastLevel?: number, presets?: string[], presetLabels?: Record<string,string>, selector?: string, sampledLevels?: number[], defaultPreset?: string, controlLabel?: string }} LzbenchCodecEntry */

/** @type {LzbenchCodecEntry[]} */
export const LZBENCH_CODEC_ENTRIES = [
  { codecId: "brieflz", displayName: "BriefLZ", firstLevel: 1, lastLevel: 9 },
  { codecId: "bsc", displayName: "BSC", presets: ["bsc0", "bsc1", "bsc2", "bsc3", "bsc4", "bsc5", "bsc6"] },
  { codecId: "bzip2", displayName: "Bzip2", firstLevel: 1, lastLevel: 9 },
  { codecId: "bzip3", displayName: "Bzip3", firstLevel: 1, lastLevel: 10 },
  { codecId: "crush", displayName: "Crush", firstLevel: 0, lastLevel: 2 },
  { codecId: "csc", displayName: "CSC", firstLevel: 1, lastLevel: 5 },
  { codecId: "fastlz", displayName: "FastLZ", firstLevel: 1, lastLevel: 2 },
  { codecId: "fastlzma2", displayName: "FastLZMA2", firstLevel: 1, lastLevel: 10 },
  { codecId: "kanzi", displayName: "Kanzi", firstLevel: 1, lastLevel: 9 },
  { codecId: "libdeflate", displayName: "Libdeflate", firstLevel: 1, lastLevel: 12 },
  { codecId: "lizard", displayName: "Lizard", firstLevel: 10, lastLevel: 49 },
  { codecId: "lz4fast", displayName: "LZ4 Fast", sampledLevels: [1, 3, 9, 17] },
  { codecId: "lz4hc", displayName: "LZ4HC", firstLevel: 1, lastLevel: 12 },
  { codecId: "lzav", displayName: "LZAV", firstLevel: 1, lastLevel: 2 },
  { codecId: "lzf", displayName: "LZF", firstLevel: 0, lastLevel: 1 },
  { codecId: "lzfse", displayName: "LZFSE", presets: ["lzfse"] },
  { codecId: "lzg", displayName: "LZG", firstLevel: 1, lastLevel: 9 },
  { codecId: "lzham", displayName: "LZHAM", firstLevel: 0, lastLevel: 4 },
  { codecId: "lzjb", displayName: "LZJB", presets: ["lzjb"] },
  { codecId: "lzlib", displayName: "Lzlib", firstLevel: 0, lastLevel: 9 },
  { codecId: "lzo1x", displayName: "LZO1X", firstLevel: 1, lastLevel: 999 },
  { codecId: "ppmd8", displayName: "PPMD8", firstLevel: 1, lastLevel: 9 },
  { codecId: "quicklz", displayName: "QuickLZ", firstLevel: 1, lastLevel: 3 },
  { codecId: "slz_deflate", displayName: "Slz_deflate", firstLevel: 1, lastLevel: 3 },
  { codecId: "slz_gzip", displayName: "Slz_gzip", firstLevel: 1, lastLevel: 3 },
  { codecId: "slz_zlib", displayName: "Slz_zlib", firstLevel: 1, lastLevel: 3 },
  { codecId: "tornado", displayName: "Tornado", firstLevel: 1, lastLevel: 16 },
  { codecId: "ucl_nrv2b", displayName: "UCL NRV2b", firstLevel: 1, lastLevel: 9 },
  { codecId: "zlib", displayName: "Zlib", firstLevel: 1, lastLevel: 9 },
  { codecId: "zlib-ng", displayName: "Zlib-ng", firstLevel: 1, lastLevel: 9, selector: "zlib-ng" },
  { codecId: "zling", displayName: "Zling", firstLevel: 0, lastLevel: 4 },
  { codecId: "zxc", displayName: "ZXC", firstLevel: 1, lastLevel: 6 },
  { codecId: "tamp", displayName: "TAMP", firstLevel: 8, lastLevel: 15 },
];

export const PARAMETER_VARIANT_ENTRIES = [
  {
    codecId: "lzo1_variant",
    displayName: "LZO1",
    presets: ["lzo1", "lzo1a", "lzo1b", "lzo1c", "lzo1f", "lzo1x_999", "lzo1y", "lzo1z"],
    presetLabels: {
      lzo1: "LZO1",
      lzo1a: "LZO1A",
      lzo1b: "LZO1B",
      lzo1c: "LZO1C",
      lzo1f: "LZO1F",
      lzo1x_999: "LZO1X",
      lzo1y: "LZO1Y",
      lzo1z: "LZO1Z",
    },
    defaultPreset: "lzo1z",
    controlLabel: "variant",
  },
  {
    codecId: "slz_variant",
    displayName: "SLZ",
    presets: ["slz_deflate_3", "slz_gzip_3", "slz_zlib_3"],
    presetLabels: {
      slz_deflate_3: "deflate",
      slz_gzip_3: "gzip",
      slz_zlib_3: "zlib",
    },
    defaultPreset: "slz_zlib_3",
    controlLabel: "variant",
  },
  {
    codecId: "ucl_nrv_variant",
    displayName: "UCL NRV",
    presets: ["ucl_nrv2b_9", "ucl_nrv2d", "ucl_nrv2e"],
    presetLabels: {
      ucl_nrv2b_9: "NRV2b",
      ucl_nrv2d: "NRV2d",
      ucl_nrv2e: "NRV2e",
    },
    defaultPreset: "ucl_nrv2e",
    controlLabel: "variant",
  },
];

const PARAMETER_VARIANT_BACKING_CODEC_IDS = new Set([
  "lzo1x",
  "slz_deflate",
  "slz_gzip",
  "slz_zlib",
  "ucl_nrv2b",
]);

export const PARAMETER_CODEC_ENTRIES = [
  ...PARAMETER_VARIANT_ENTRIES,
  ...LZBENCH_CODEC_ENTRIES.filter((entry) => !PARAMETER_VARIANT_BACKING_CODEC_IDS.has(entry.codecId)),
];

/** @param {number} first @param {number} last @param {number} [maxCount] */
export function sampleLevels(first, last, maxCount = MAX_SAMPLED_LEVELS) {
  if (first > last) return [];
  const total = last - first + 1;
  if (total <= maxCount) {
    return Array.from({ length: total }, (_, i) => first + i);
  }
  if (maxCount <= 1) return [first];
  const out = [];
  const seen = new Set();
  for (let i = 0; i < maxCount; i++) {
    const lv =
      i === maxCount - 1
        ? last
        : Math.max(first, Math.min(last, Math.round(first + ((last - first) * i) / (maxCount - 1))));
    if (!seen.has(lv)) {
      out.push(lv);
      seen.add(lv);
    }
  }
  if (out.length < maxCount) {
    for (let lv = first; lv <= last && out.length < maxCount; lv++) {
      if (!seen.has(lv)) {
        out.push(lv);
        seen.add(lv);
      }
    }
  }
  return out.sort((a, b) => a - b);
}

/** @param {string} codecId @param {number} level */
export function algorithmKeyForLevel(codecId, level) {
  return `${codecId.replace(/-/g, "_")}_${level}`;
}

/** @param {LzbenchCodecEntry} entry */
export function entrySampledLevels(entry) {
  if (entry.sampledLevels) return [...entry.sampledLevels];
  if (entry.presets) {
    if (entry.codecId === "bsc") {
      return entry.presets.map((_, i) => i);
    }
    return [0];
  }
  return sampleLevels(entry.firstLevel, entry.lastLevel);
}

/** @param {LzbenchCodecEntry} entry */
export function entryAlgorithmKeys(entry) {
  if (entry.presets) {
    if (entry.codecId === "bsc") return [...entry.presets];
    return [...entry.presets];
  }
  return entrySampledLevels(entry).map((lv) => algorithmKeyForLevel(entry.codecId, lv));
}

export function allLzbenchCatalogAlgorithmKeys() {
  return LZBENCH_CODEC_ENTRIES.flatMap((entry) => entryAlgorithmKeys(entry));
}

export function allParameterControlledAlgorithmKeys() {
  return PARAMETER_CODEC_ENTRIES.flatMap((entry) => entryAlgorithmKeys(entry));
}

export function runnableLzbenchCatalogAlgorithmKeys() {
  return ENABLE_LZBENCH_RUNTIME_CATALOG ? allLzbenchCatalogAlgorithmKeys() : [];
}

/** @param {LzbenchCodecEntry} entry */
export function defaultLevelForEntry(entry) {
  if (entry.defaultPreset) {
    return entry.defaultPreset;
  }
  if (entry.codecId === "bsc") {
    const presets = entry.presets || [];
    return presets[presets.length - 1] ?? presets[0];
  }
  if (entry.codecId === "lz4fast") {
    const levels = entrySampledLevels(entry);
    return levels[0];
  }
  if (entry.presets) {
    return entry.presets[0];
  }
  const levels = entrySampledLevels(entry);
  return levels[levels.length - 1] ?? levels[0];
}

export function defaultLzbenchLevelSelection() {
  /** @type {Record<string, number|string>} */
  const out = {};
  for (const entry of PARAMETER_CODEC_ENTRIES) {
    out[entry.codecId] = defaultLevelForEntry(entry);
  }
  return out;
}

/** Codecs that expose more than one selectable variant in the UI. */
export function lzbenchCodecLevelControls() {
  if (!ENABLE_LZBENCH_RUNTIME_CATALOG) return [];
  return PARAMETER_CODEC_ENTRIES.filter((entry) => {
    if (entry.presets) return (entry.presets?.length ?? 0) > 1;
    return entrySampledLevels(entry).length > 1;
  });
}

/** @param {string} key */
export function isLzbenchCatalogAlgorithm(key) {
  const s = String(key || "").trim();
  if (!s) return false;
  if (s.startsWith("lzbench_")) return true;
  return allLzbenchCatalogAlgorithmKeys().includes(s) || allParameterControlledAlgorithmKeys().includes(s);
}

/** @param {string} key */
export function parseLzbenchAlgorithmKey(key) {
  const s = String(key || "").trim();
  if (!s) return null;
  if (s.startsWith("lzbench_")) {
    return parseLzbenchAlgorithmKey(s.slice("lzbench_".length));
  }
  for (const entry of PARAMETER_CODEC_ENTRIES) {
    if (entry.presets?.includes(s)) {
      return { entry, level: s, preset: s };
    }
  }
  for (const entry of PARAMETER_CODEC_ENTRIES) {
    if (entry.presets) continue;
    const prefix = `${entry.codecId.replace(/-/g, "_")}_`;
    if (s.startsWith(prefix)) {
      const tail = s.slice(prefix.length);
      const level = Number(tail);
      if (Number.isFinite(level)) {
        return { entry, level };
      }
    }
  }
  return null;
}

/** @param {LzbenchCodecEntry} entry @param {number|string|null|undefined} selected */
export function resolveSelectedAlgorithmKey(entry, selected) {
  if (entry.codecId === "bsc") {
    const presets = entry.presets || [];
    if (typeof selected === "string" && presets.includes(selected)) return selected;
    if (typeof selected === "number" && selected >= 0 && selected < presets.length) {
      return presets[selected];
    }
    return presets[presets.length - 1] ?? presets[0];
  }
  if (entry.presets) {
    const presets = entry.presets || [];
    if (typeof selected === "string" && presets.includes(selected)) return selected;
    if (typeof selected === "number" && selected >= 0 && selected < presets.length) {
      return presets[selected];
    }
    if (entry.defaultPreset && presets.includes(entry.defaultPreset)) return entry.defaultPreset;
    return presets[0];
  }
  const levels = entrySampledLevels(entry);
  if (!levels.length) return null;
  let want = selected;
  if (want == null || want === "") {
    want = levels[levels.length - 1];
  }
  const num = Number(want);
  if (levels.includes(num)) {
    return algorithmKeyForLevel(entry.codecId, num);
  }
  const nearest = levels.reduce((best, lv) =>
    Math.abs(lv - num) < Math.abs(best - num) ? lv : best
  );
  return algorithmKeyForLevel(entry.codecId, nearest);
}

/** @param {Record<string, number|string>|null|undefined} lzbenchLevels */
export function selectedLzbenchAlgorithmKeys(lzbenchLevels) {
  const levels = { ...defaultLzbenchLevelSelection(), ...(lzbenchLevels || {}) };
  const keys = new Set();
  for (const entry of PARAMETER_CODEC_ENTRIES) {
    const chosen = resolveSelectedAlgorithmKey(entry, levels[entry.codecId]);
    if (chosen) keys.add(chosen);
  }
  return keys;
}

/** @param {LzbenchCodecEntry} entry */
export function sliderLabelsForEntry(entry) {
  if (entry.presets) {
    return entry.presets || [];
  }
  return entrySampledLevels(entry).map(String);
}

/** @param {LzbenchCodecEntry} entry @param {number|string} selected */
export function displayLevelLabel(entry, selected) {
  if (entry.codecId === "bsc") {
    const presets = entry.presets || [];
    const key = typeof selected === "string" ? selected : presets[Number(selected)];
    return entry.presetLabels?.[key] || key || String(selected);
  }
  if (entry.presets) {
    const presets = entry.presets || [];
    const key = typeof selected === "string" ? selected : presets[Number(selected)];
    return entry.presetLabels?.[key] || key || String(selected);
  }
  return String(selected);
}
