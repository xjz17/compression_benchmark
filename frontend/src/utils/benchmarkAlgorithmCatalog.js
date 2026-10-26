/**
 * Canonical algorithm key lists for Available Algorithms / UI catalog.
 * Aligned with tri-language int/float benchmark CSVs (e.g. AI4I-Maintenance, 62 rows).
 * Java keys have no _int/_float suffix; int runs cast integer columns to float for shared codecs.
 */

import { runnableLzbenchCatalogAlgorithmKeys } from "./lzbenchCodecCatalog.js";
import { filterActiveAlgorithmKeys } from "./activeMethodPolicy.js";

export const PY_XZ_LEVELS = Array.from(
  { length: 10 },
  (_, i) => `xz_level_${i}`,
);
export const PY_ZSTD_LEVELS = Array.from(
  { length: 9 },
  (_, i) => `zstd_level_${i + 1}`,
);

const LEGACY_NATIVE_LZ_FAMILY_ALGORITHM_KEYS = [
  "density",
  "gipfeli",
  "glza",
  "lzmat",
  "lzo1",
  "lzo1a",
  "lzo1b",
  "lzo1c",
  "lzo1f",
  "lzo1y",
  "lzo1z",
  "lzo2a",
  "lzrw",
  "lzsse2",
  "lzsse4",
  "lzsse4fast",
  "lzsse8",
  "lzsse8fast",
  "lzvn",
  "pithy",
  "shrinker",
  "ucl_nrv2d",
  "ucl_nrv2e",
  "wflz",
  "xpack",
  "yalz77",
  "yappy",
];

export const NATIVE_LZ_FAMILY_ALGORITHM_KEYS = Object.freeze(
  filterActiveAlgorithmKeys(LEGACY_NATIVE_LZ_FAMILY_ALGORITHM_KEYS),
);

/** Generic byte compressors for string columns (Home staticdata). */
export const STRING_GENERIC_ALGORITHM_BASE = [
  "brotli",
  "gz",
  "lz4-framed",
  "xz",
  "snappy-framed",
  "fpzip",
  "zstd",
  "blosc_blosclz",
  "blosc2_blosclz",
  "ext_zfp_zfpy",
  "fse",
  "huff0",
  ...NATIVE_LZ_FAMILY_ALGORITHM_KEYS,
];

/** @returns {string[]} Expanded keys (23) for generic string byte codecs only. */
export function getStringGenericCatalogAlgorithms() {
  return expandCatalogLevels([...STRING_GENERIC_ALGORITHM_BASE]);
}

/** @returns {string[]} Expanded keys for string benchmark CSVs (generic byte codecs + lzbench). */
export function getStringBenchmarkCatalogAlgorithms() {
  return [...getStringGenericCatalogAlgorithms(), ...LZBENCH_ALGORITHM_KEYS];
}

/** Placeholders expanded via {@link expandCatalogLevels}. */
const BENCHMARK_ALGORITHM_BASE = [
  "brotli",
  "gz",
  "lz4-framed",
  "xz",
  "snappy-framed",
  "zstd",
  ...NATIVE_LZ_FAMILY_ALGORITHM_KEYS,
  "ans",
  "fse",
  "huffman",
  "huff0",
  "arithmetic",
  "CHIMP",
  "GORILLA",
  "RLE",
  "RLBE",
  "SPRINTZ",
  "TS_2DIFF",
  "ts_2diff_bos",
  "reger",
  "col_dictionary",
  "col_bss_lz4",
  "DELTA_BINARY_PACKED",
  "streamvbyte_delta1",
  "fastpfor",
  "fpc",
  "pfpc",
  "simple8b",
  "spdp",
  "most_minimal",
  "corad_tricklet",
  "machete_rel_1e-1",
  "machete_rel_1e-2",
  "machete_rel_1e-3",
  "machete_rel_1e-4",
  "leco",
  "alp",
  "elf_on_chimp",
  "elf_plus",
  "camel",
  "simpiece_relRange_0p005",
  "simpiece_relRange_0p01",
  "sz_ABS_1e-1",
  "sz_ABS_1e-2",
  "sz_ABS_1e-3",
  "sz_ABS_1e-4",
  "sz_REL_1e-2",
  "sz_REL_1e-3",
  "sz3_ABS_1e-1",
  "sz3_ABS_1e-2",
  "sz3_ABS_1e-3",
  "sz3_ABS_1e-4",
  "sz3_REL_1e-2",
  "sz3_REL_1e-3",
  "buff_kernel_native",
];

/** lzbench-origin report rows are backed by the tri-language native byte reference codec. */
export const LZBENCH_ALGORITHM_KEYS = runnableLzbenchCatalogAlgorithmKeys();

/** @deprecated Same as {@link BENCHMARK_ALGORITHM_BASE}; kept for add-benchmark-codec docs. */
export const JAVA_INT_ALGORITHMS = BENCHMARK_ALGORITHM_BASE;

/** @deprecated Same as {@link BENCHMARK_ALGORITHM_BASE}. */
export const JAVA_FLOAT_ALGORITHMS = BENCHMARK_ALGORITHM_BASE;

export const PY_SPDP_LEVELS = Array.from({ length: 10 }, (_, i) => `spdp_level_${i}`);

/** Expand catalog placeholders to report CSV keys (xz_level_*, zstd_level_*, spdp_level_*). */
export function expandCatalogLevels(javaKeys) {
  return javaKeys.flatMap((k) =>
    k === "xz"
      ? [...PY_XZ_LEVELS]
      : k === "zstd"
        ? [...PY_ZSTD_LEVELS]
        : k === "spdp"
          ? [...PY_SPDP_LEVELS]
          : [k],
  );
}

/** @deprecated Use {@link expandCatalogLevels}. */
export const expandPythonLevels = expandCatalogLevels;

/** FPCompress LC-framework CPU codecs (report CSV keys). */
export const FPCOMPRESS_ALGORITHM_KEYS = [
  "fpcompress_dpratio",
  "fpcompress_dpspeed",
];

/** ZFP rows (float32 wire; lossless zfpy + fixed-accuracy tol variants). */
export const ZFP_ALGORITHM_KEYS = [
  "ext_zfp_zfpy",
  "ext_zfp_tol_1e-1",
  "ext_zfp_tol_1e-2",
  "ext_zfp_tol_1e-3",
];

/** Blosc1 blosclz + shuffle (int64 / float32 wire). */
export const BLOSC1_ALGORITHM_KEYS = ["blosc_blosclz"];

/** Blosc2 blosclz + shuffle (int64 / float32 wire). */
export const BLOSC2_ALGORITHM_KEYS = ["blosc2_blosclz"];

/** @param {string[]} keys */
export function insertBloscAfterFpzip(keys) {
  const block = [...BLOSC1_ALGORITHM_KEYS, ...BLOSC2_ALGORITHM_KEYS];
  const i = keys.findIndex((k) => k === "fpzip");
  if (i === -1) {
    return [...keys, ...block];
  }
  return [...keys.slice(0, i + 1), ...block, ...keys.slice(i + 1)];
}

/** @deprecated use insertBloscAfterFpzip */
export function insertBlosc2AfterFpzip(keys) {
  return insertBloscAfterFpzip(keys);
}

/** @param {string[]} keys */
export function insertZfpAfterFpzip(keys) {
  const i = keys.findIndex((k) => k === "blosc2_blosclz");
  const j = i === -1 ? keys.findIndex((k) => k === "fpzip") : i;
  if (j === -1) {
    return [...keys, ...ZFP_ALGORITHM_KEYS];
  }
  return [...keys.slice(0, j + 1), ...ZFP_ALGORITHM_KEYS, ...keys.slice(j + 1)];
}

/** @param {string[]} keys */
export function insertFpzipAfterSnappy(keys) {
  const i = keys.findIndex((k) => k === "snappy-framed" || k === "snappy");
  if (i === -1) {
    return ["fpzip", ...keys];
  }
  return [...keys.slice(0, i + 1), "fpzip", ...keys.slice(i + 1)];
}

/** @param {string[]} keys */
export function insertFpcompressAfterFpzip(keys) {
  const i = keys.findIndex((k) => k === "fpzip");
  if (i === -1) {
    return [...keys, ...FPCOMPRESS_ALGORITHM_KEYS];
  }
  return [...keys.slice(0, i + 1), ...FPCOMPRESS_ALGORITHM_KEYS, ...keys.slice(i + 1)];
}

/** Full catalog keys after level expansion, fpzip, Blosc2, ZFP, and FPCompress inserts. */
export function getBenchmarkCatalogAlgorithms() {
  return filterActiveAlgorithmKeys([
    ...insertZfpAfterFpzip(
      insertFpcompressAfterFpzip(
        insertBloscAfterFpzip(
          insertFpzipAfterSnappy(expandCatalogLevels([...BENCHMARK_ALGORITHM_BASE])),
        ),
      ),
    ),
    ...LZBENCH_ALGORITHM_KEYS,
  ]);
}

/**
 * Same algorithm list for java/python/cpp and int/float (matches static benchmark CSVs).
 */
export function buildAlgorithmsByLanguageAndType() {
  const list = getBenchmarkCatalogAlgorithms();
  const bucket = { int: list, float: list };
  return {
    java: bucket,
    python: bucket,
    cpp: bucket,
  };
}
