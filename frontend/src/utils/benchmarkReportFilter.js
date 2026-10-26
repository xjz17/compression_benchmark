/**
 * Client-side mirror of backend/myapp/report_filters.py for Upload re-filtering.
 */

import { isLegacyColumnRleAlgorithm } from "./algorithmCompressionMode.js";
import {
  isLzbenchCatalogAlgorithm,
  selectedLzbenchAlgorithmKeys,
} from "./lzbenchCodecCatalog.js";

const RESEARCH_FAMILY_VARIANTS = {
  simpiece: [
    ["simpiece_relRange_0p005", "simpiece_float_relRange_0p005"],
    ["simpiece_relRange_0p01", "simpiece_float_relRange_0p01"],
  ],
  serf: [
    ["serf_float_ABS_1e-1"],
    ["serf_float_ABS_1e-2"],
    ["serf_float_ABS_1e-3"],
    ["serf_float_ABS_1e-4"],
  ],
};

const SZ_ABS_ORDER = [
  "sz_ABS_1e-1",
  "sz_ABS_1e-2",
  "sz_ABS_1e-3",
  "sz_ABS_1e-4",
];
const SZ_REL_ORDER = ["sz_REL_1e-2", "sz_REL_1e-3"];

function isSzAlgorithm(algorithm) {
  const a = String(algorithm ?? "");
  return a.startsWith("sz_") && !a.startsWith("sz3_");
}
const SZ3_ABS_GROUPS = [
  ["sz3_ft_ABS_1e-1", "sz3_ABS_1e-1"],
  ["sz3_ABS_1e-2"],
  ["sz3_ABS_1e-3"],
  ["sz3_ABS_1e-4"],
];
const SZ3_REL_ORDER = ["sz3_REL_1e-2", "sz3_REL_1e-3", "sz3_REL_1e-4"];
const MACHETE_REL_ORDER = [
  "machete_rel_1e-1",
  "machete_rel_1e-2",
  "machete_rel_1e-3",
  "machete_rel_1e-4",
];
const ZFP_TOL_ORDER = [
  "ext_zfp_tol_1e-1",
  "ext_zfp_tol_1e-2",
  "ext_zfp_tol_1e-3",
];
const SPDP_LEVEL_ORDER = Array.from({ length: 10 }, (_, i) => `spdp_level_${i}`);
const CHIMP_HISTORY_1_KEYS = ["CHIMP"];
const CHIMP_HISTORY_128_KEYS = ["CHIMP128"];
const CHIMP_LEGACY_KEYS = ["CHIMP"];

function canonicalAlgorithmKey(algorithm) {
  const raw = String(algorithm ?? "").trim();
  if (raw === "lzma") return "xz";
  if (raw.startsWith("lzma_level_")) return `xz_level_${raw.slice("lzma_level_".length)}`;
  return raw;
}

function pickSliderKey(algorithms, orderedKeys, index) {
  if (!orderedKeys.length) return null;
  const i = Math.max(0, Math.min(index, orderedKeys.length - 1));
  for (let j = i; j >= 0; j--) {
    if (algorithms.includes(orderedKeys[j])) return orderedKeys[j];
  }
  for (let j = i + 1; j < orderedKeys.length; j++) {
    if (algorithms.includes(orderedKeys[j])) return orderedKeys[j];
  }
  return null;
}

function filterSzRows(rows, absrel, epsIdx) {
  const algos = rows.map((r) => r.algorithm);
  if (!algos.some(isSzAlgorithm)) return rows;
  const rel = String(absrel).toUpperCase() === "REL";
  const order = rel ? SZ_REL_ORDER : SZ_ABS_ORDER;
  const chosen = pickSliderKey(algos, order, epsIdx);
  if (!chosen) return rows;
  return rows.filter((r) => !isSzAlgorithm(r.algorithm) || r.algorithm === chosen);
}

function filterSz3Rows(rows, absrel, epsIdx) {
  const algos = rows.map((r) => r.algorithm);
  if (!algos.some((a) => a.startsWith("sz3_"))) return rows;
  const rel = String(absrel).toUpperCase() === "REL";
  let chosen = null;
  if (rel) {
    chosen = pickSliderKey(algos, SZ3_REL_ORDER, epsIdx);
  } else {
    const gi = Math.max(0, Math.min(epsIdx, SZ3_ABS_GROUPS.length - 1));
    for (let gix = gi; gix >= 0 && !chosen; gix--) {
      for (const k of SZ3_ABS_GROUPS[gix]) {
        if (algos.includes(k)) {
          chosen = k;
          break;
        }
      }
    }
    if (!chosen) {
      for (let gix = gi + 1; gix < SZ3_ABS_GROUPS.length && !chosen; gix++) {
        for (const k of SZ3_ABS_GROUPS[gix]) {
          if (algos.includes(k)) {
            chosen = k;
            break;
          }
        }
      }
    }
  }
  if (!chosen) return rows;
  return rows.filter((r) => !r.algorithm.startsWith("sz3_") || r.algorithm === chosen);
}

function filterMacheteRows(rows, epsIdx) {
  const algos = rows.map((r) => r.algorithm);
  const pred = (a) =>
    a.startsWith("machete_rel_") || a.startsWith("machete_float_rel_");
  if (!algos.some(pred)) return rows;
  let chosen = pickSliderKey(algos, MACHETE_REL_ORDER, epsIdx);
  if (!chosen) {
    const legacy = MACHETE_REL_ORDER.map((k) =>
      k.replace("machete_rel_", "machete_float_rel_")
    );
    chosen = pickSliderKey(algos, legacy, epsIdx);
  }
  if (!chosen) return rows;
  return rows.filter((r) => !pred(r.algorithm) || r.algorithm === chosen);
}

function filterSpdpRows(rows, levelIdx) {
  const algos = rows.map((r) => r.algorithm);
  const pred = (a) => a.startsWith("spdp_level_") || a.startsWith("spdp_float_level_");
  if (!algos.some(pred)) return rows;
  const chosen = pickSliderKey(algos, SPDP_LEVEL_ORDER, levelIdx);
  if (!chosen) return rows;
  return rows.filter((r) => !pred(r.algorithm) || r.algorithm === chosen);
}

function filterZfpTolRows(rows, epsIdx) {
  const algos = rows.map((r) => r.algorithm);
  const pred = (a) => a.startsWith("ext_zfp_tol_");
  if (!algos.some(pred)) return rows;
  const chosen = pickSliderKey(algos, ZFP_TOL_ORDER, epsIdx);
  if (!chosen) return rows;
  return rows.filter((r) => !pred(r.algorithm) || r.algorithm === chosen);
}

function filterResearchFamily(rows, familyKey, paramIndex) {
  const groups = RESEARCH_FAMILY_VARIANTS[familyKey];
  if (!groups) return rows;
  const pred =
    familyKey === "simpiece"
      ? (a) => a.startsWith("simpiece_")
      : familyKey === "serf"
        ? (a) => a.startsWith("serf_float_")
        : () => false;
  const algos = rows.map((r) => r.algorithm);
  if (!algos.some(pred)) return rows;
  const i = Math.max(0, Math.min(paramIndex, groups.length - 1));
  let chosen = null;
  for (const k of groups[i]) {
    if (algos.includes(k)) {
      chosen = k;
      break;
    }
  }
  if (!chosen) {
    for (let gi = i - 1; gi >= 0 && !chosen; gi--) {
      for (const k of groups[gi]) {
        if (algos.includes(k)) {
          chosen = k;
          break;
        }
      }
    }
  }
  if (!chosen) {
    for (let gi = i + 1; gi < groups.length && !chosen; gi++) {
      for (const k of groups[gi]) {
        if (algos.includes(k)) {
          chosen = k;
          break;
        }
      }
    }
  }
  if (!chosen) return rows;
  return rows.filter((r) => !pred(r.algorithm) || r.algorithm === chosen);
}

function filterLzbenchRows(rows, lzbenchLevels) {
  if (!rows.length) return rows;
  const hasLzbench = rows.some((r) => isLzbenchCatalogAlgorithm(r.algorithm));
  if (!hasLzbench) return rows;
  const keep = selectedLzbenchAlgorithmKeys(lzbenchLevels);
  return rows.filter((r) => !isLzbenchCatalogAlgorithm(r.algorithm) || keep.has(r.algorithm));
}

function canonicalChimpAlgorithmKey(algorithm) {
  const raw = canonicalAlgorithmKey(algorithm);
  if (raw === "CHIMP" || raw === "Chimp" || raw === "chimp") return "CHIMP";
  if (raw === "CHIMP128" || raw === "Chimp128" || raw === "chimp128" || raw === "chimp_128") {
    return "CHIMP128";
  }
  return raw;
}

function filterChimpRows(rows, chimpHistory) {
  if (!rows.length) return rows;
  const normalized = rows.map((row) => canonicalChimpAlgorithmKey(row.algorithm));
  const hasChimp = normalized.some(
    (a) => CHIMP_HISTORY_1_KEYS.includes(a) || CHIMP_HISTORY_128_KEYS.includes(a)
  );
  if (!hasChimp) return rows;
  let keep = Number(chimpHistory) === 1 ? CHIMP_HISTORY_1_KEYS : CHIMP_HISTORY_128_KEYS;
  if (Number(chimpHistory) !== 1 && !normalized.some((a) => CHIMP_HISTORY_128_KEYS.includes(a))) {
    keep = [...keep, ...CHIMP_LEGACY_KEYS];
  }
  return rows
    .map((row, index) => ({ ...row, algorithm: normalized[index] }))
    .filter((row) => {
      const a = row.algorithm;
      const isChimp = CHIMP_HISTORY_1_KEYS.includes(a) || CHIMP_HISTORY_128_KEYS.includes(a);
      return !isChimp || keep.includes(a);
    });
}

/**
 * @param {Array<{algorithm: string}>} items
 * @param {object} config
 */
export function filterBenchmarkReportItems(items, config) {
  if (!Array.isArray(items) || !items.length) return [];
  const lang = config.implement_language;
  const xz = Number(config.xz_level ?? config.lzma_level ?? 9);
  const zstd = Number(config.zstd_level ?? 3);

  let rows = items.map((row) => ({ ...row, algorithm: canonicalAlgorithmKey(row.algorithm) }));
  rows = rows.filter((row) => {
    const a = String(row.algorithm || "");
    if (a.startsWith("xz_level_")) return a === `xz_level_${xz}`;
    if (a.startsWith("zstd_level_")) return a === `zstd_level_${zstd}`;
    return true;
  });
  rows = filterChimpRows(rows, config.chimp_history ?? 128);
  rows = filterLzbenchRows(rows, config.lzbench_levels);

  if (lang === "java" || lang === "python" || lang === "cpp") {
    rows = filterMacheteRows(rows, Number(config.machete_eps ?? 2));
    rows = filterSpdpRows(rows, Number(config.spdp_level ?? 9));
    rows = filterZfpTolRows(rows, Number(config.zfp_tol_eps ?? 2));
    rows = filterResearchFamily(rows, "simpiece", Number(config.simpiece_param ?? 0));
    rows = filterSzRows(rows, config.sz_absrel ?? "ABS", Number(config.sz_eps ?? 0));
    rows = filterSz3Rows(rows, config.sz3_absrel ?? "ABS", Number(config.sz3_eps ?? 0));
    rows = filterResearchFamily(rows, "serf", Number(config.serf_param ?? 0));
  }
  return rows.filter((row) => !isLegacyColumnRleAlgorithm(row.algorithm));
}

export function buildUploadFormFilterFields(config) {
  const lang = config.implement_language;
  const base = {
    xz_level: config.xz_level ?? config.lzma_level ?? 9,
    zstd_level: config.zstd_level,
    chimp_history: config.chimp_history ?? 128,
    lzbench_levels: JSON.stringify(config.lzbench_levels || {}),
  };
  if (lang === "java" || lang === "python" || lang === "cpp") {
    return {
      ...base,
      simpiece_param: config.simpiece_param,
      sz_absrel: config.sz_absrel,
      sz_eps: config.sz_eps,
      sz3_absrel: config.sz3_absrel,
      sz3_eps: config.sz3_eps,
      machete_eps: config.machete_eps,
      spdp_level: config.spdp_level,
      zfp_tol_eps: config.zfp_tol_eps,
      serf_param: config.serf_param,
    };
  }
  return base;
}
