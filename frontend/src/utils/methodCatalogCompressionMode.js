/**
 * Map Methods page entries to benchmark catalog algorithm keys and lossy/lossless mode
 * (same rules as Home "Available Algorithms" via algorithmCompressionMode.js).
 */

import { getBenchmarkCatalogAlgorithms } from "./benchmarkAlgorithmCatalog.js";
import { isLossyAlgorithm } from "./algorithmCompressionMode.js";
import { displayAlgorithmCatalogName } from "./algorithmDisplayName.js";
import { LZBENCH_CODEC_ENTRIES, entryAlgorithmKeys } from "./lzbenchCodecCatalog.js";

/** @param {string} text */
function normalizeMethodMatchToken(text) {
  return String(text ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");
}

function buildLzbenchMethodCatalogMappings() {
  /** @type {Record<string, string[]>} */
  const explicit = {};
  /** @type {Record<string, string>} */
  const prefixes = {};
  for (const entry of LZBENCH_CODEC_ENTRIES) {
    const keys = entryAlgorithmKeys(entry);
    const name = entry.displayName;
    if (entry.presets) {
      explicit[name] = keys;
    } else {
      prefixes[name] = `${entry.codecId.replace(/-/g, "_")}_`;
    }
  }
  return { explicit, prefixes };
}

const { explicit: LZBENCH_METHOD_EXPLICIT, prefixes: LZBENCH_METHOD_PREFIXES } =
  buildLzbenchMethodCatalogMappings();

const LZO1_FAMILY_ALGORITHM_KEYS = [
  "lzo1",
  "lzo1a",
  "lzo1b",
  "lzo1c",
  "lzo1f",
  "lzo1x_1",
  "lzo1x_112",
  "lzo1x_223",
  "lzo1x_334",
  "lzo1x_445",
  "lzo1x_555",
  "lzo1x_666",
  "lzo1x_777",
  "lzo1x_888",
  "lzo1x_999",
  "lzo1y",
  "lzo1z",
];
const SLZ_FAMILY_ALGORITHM_KEYS = ["slz_deflate", "slz_gzip", "slz_zlib"].flatMap((codec) =>
  [1, 2, 3].map((level) => `${codec}_${level}`)
);
const UCL_NRV_FAMILY_ALGORITHM_KEYS = [
  ...Array.from({ length: 9 }, (_, i) => `ucl_nrv2b_${i + 1}`),
  "ucl_nrv2d",
  "ucl_nrv2e",
];

/** Explicit method metadata name -> catalog report keys. */
const METHOD_NAME_ALGORITHM_KEYS = {
  gzip: ["gz"],
  buff: ["buff_kernel_native"],
  Chimp128: ["CHIMP128"],
  elf: ["elf_on_chimp"],
  fpc: ["fpc"],
  pfpc: ["pfpc"],
  MOST: ["most_minimal"],
  LZ4: ["lz4-framed"],
  snappy: ["snappy-framed"],
  "Stream VByte": ["streamvbyte_delta1"],
  BOS: ["ts_2diff_bos"],
  REGER: ["reger"],
  DELTA_BINARY_PACKED: ["DELTA_BINARY_PACKED"],
  FPZIP: ["fpzip"],
  Blosc: ["blosc_blosclz"],
  Blosc2: ["blosc2_blosclz"],
  "Blosc BloscLZ": ["blosc_blosclz"],
  "Blosc2 BloscLZ": ["blosc2_blosclz"],
  BLOSC: ["blosc_blosclz"],
  BLOSC2: ["blosc2_blosclz"],
  ZFP: ["ext_zfp_zfpy"],
  "FPCompress DPratio": ["fpcompress_dpratio"],
  "FPCompress DPspeed": ["fpcompress_dpspeed"],
  LZO1: LZO1_FAMILY_ALGORITHM_KEYS,
  SLZ: SLZ_FAMILY_ALGORITHM_KEYS,
  "UCL NRV": UCL_NRV_FAMILY_ALGORITHM_KEYS,
  ...LZBENCH_METHOD_EXPLICIT,
};

/** Method metadata name -> prefix of catalog keys (level/preset families). */
const METHOD_NAME_KEY_PREFIXES = {
  zstd: "zstd_level_",
  XZ: "xz_level_",
  SZ: "sz_",
  SZ3: "sz3_",
  Machete: "machete_rel_",
  SimPiece: "simpiece_",
  spdp: "spdp_level_",
  ...LZBENCH_METHOD_PREFIXES,
};

/**
 * @param {string} methodName
 * @returns {string[]}
 */
export function algorithmKeysForMethodName(methodName) {
  const catalog = getBenchmarkCatalogAlgorithms();
  const catalogSet = new Set(catalog);
  const name = String(methodName ?? "").trim();
  if (!name) {
    return [];
  }

  const explicit = METHOD_NAME_ALGORITHM_KEYS[name];
  if (explicit?.length) {
    return explicit.filter((k) => catalogSet.has(k));
  }

  const prefix = METHOD_NAME_KEY_PREFIXES[name];
  if (prefix) {
    return catalog.filter((k) => k.startsWith(prefix));
  }

  const normMethod = normalizeMethodMatchToken(name);
  return catalog.filter((key) => {
    if (normalizeMethodMatchToken(key) === normMethod) {
      return true;
    }
    return normalizeMethodMatchToken(displayAlgorithmCatalogName(key)) === normMethod;
  });
}

/**
 * @param {string} methodName
 * @returns {"lossless"|"lossy"}
 */
export function compressionModeForMethodName(methodName) {
  const keys = algorithmKeysForMethodName(methodName);
  if (!keys.length) {
    return "lossless";
  }
  return keys.some((k) => isLossyAlgorithm(k)) ? "lossy" : "lossless";
}

/** @param {"lossless"|"lossy"|string} mode */
export function methodCompressionModeLabel(mode) {
  return mode === "lossy" ? "Lossy" : "Lossless";
}

/** @param {"lossless"|"lossy"|string} mode */
export function methodCompressionModeTagType(mode) {
  return mode === "lossy" ? "warning" : "success";
}
