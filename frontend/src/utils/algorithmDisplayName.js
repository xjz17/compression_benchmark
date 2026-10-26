/**
 * Human-facing names so users see "SZ" / "ZFP" in the UI while API keys stay unchanged.
 */

import { normalizeAlgorithmKey } from "./algorithmCompressionMode.js";
import { parseLzbenchAlgorithmKey } from "./lzbenchCodecCatalog.js";

/** Short labels for ranking tables and general UI (no parenthetical suffixes). */
const ALGORITHM_KEY_LABEL_OVERRIDES = {
  brotli: "Brotli",
  col_dictionary: "Dictionary",
  col_bss_lz4: "BSS+LZ4",
  DELTA_BINARY_PACKED: "DBP",
  fastpfor: "FastPFor",
  fpc: "FPC",
  pfpc: "PFPC",
  pFPC: "PFPC",
  fastpfor_float: "FastPFor",
  fastpfor_int: "FastPFor",
  simple8b: "Simple8b",
  simple8b_float: "Simple8b",
  simple8b_int: "Simple8b",
  leco: "LECO",
  leco_float: "LECO",
  alp: "ALP",
  alp_float: "ALP",
  xz: "XZ",
  lzma: "XZ",
  zstd: "ZSTD",
  "lz4-framed": "LZ4",
  "snappy-framed": "SNAPPY",
  fpzip: "FPZIP",
  blosc_blosclz: "Blosc",
  blosc2_blosclz: "Blosc2",
  density: "Density",
  gipfeli: "Gipfeli",
  glza: "GLZA",
  lzmat: "LZMAT",
  lzo1: "LZO1",
  lzo1a: "LZO1",
  lzo1b: "LZO1",
  lzo1c: "LZO1",
  lzo1f: "LZO1",
  lzo1y: "LZO1",
  lzo1z: "LZO1",
  lzo2a: "LZO2A",
  lzrw: "LZRW",
  lzsse2: "LZSSE",
  lzsse4: "LZSSE",
  lzsse4fast: "LZSSE",
  lzsse8: "LZSSE",
  lzsse8fast: "LZSSE",
  lzvn: "LZVN",
  pithy: "Pithy",
  shrinker: "Shrinker",
  slz_deflate: "SLZ",
  slz_gzip: "SLZ",
  slz_zlib: "SLZ",
  ucl_nrv2d: "UCL NRV",
  ucl_nrv2e: "UCL NRV",
  wflz: "WFLZ",
  xpack: "XPACK",
  yalz77: "YALZ77",
  yappy: "Yappy",
  fpcompress_dpratio: "FPCompress DPratio",
  fpcompress_dpspeed: "FPCompress DPspeed",
  gz: "GZIP",
  buff_scalar: "BUFF",
  buff_simd: "BUFF",
  buff_kernel_native: "BUFF",
  buff_float_scalar: "BUFF",
  buff_float_simd: "BUFF",
  buff_float_kernel_native: "BUFF",
  ext_cmix: "Cmix",
  ext_msdzip_www25: "MSDZip",
  ext_msdlc_icassp25: "MSDLC",
  streamvbyte_delta1: "Stream VByte",
  streamvbyte_delta1_int: "Stream VByte",
  most_minimal: "MOST",
  most_float_minimal: "MOST",
  corad_tricklet: "CORAD",
  corad_tricklet_float: "CORAD",
  elf_on_chimp: "Elf",
  elf_on_chimp_float: "Elf",
  elf_plus: "Elf+",
  elf_plus_float: "Elf+",
  camel: "Camel",
  camel_float: "Camel",
  ts_2diff_bos: "BOS",
  reger: "REGER",
  ts_2diff_bos_int: "BOS",
  ts_2diff_bos_float: "BOS",
  gzip: "GZIP",
  lz4: "LZ4",
  LZMA: "XZ",
  XZ: "XZ",
  snappy: "SNAPPY",
  ans: "ANS",
  fse: "FSE",
  huffman: "Huffman",
  huff0: "Huff0",
  arithmetic: "Arithmetic",
  "Arithmetic coding": "Arithmetic",
  zstd: "ZSTD",
  CHIMP: "CHIMP",
  Chimp: "CHIMP",
  chimp: "CHIMP",
  CHIMP128: "Chimp128",
  Chimp128: "Chimp128",
  chimp128: "Chimp128",
  chimp_128: "Chimp128",
};

function withLeadingCapital(label) {
  return String(label ?? "").replace(/^([a-z])/, (first) => first.toUpperCase());
}

/**
 * Join catalog labels for a list of algorithm keys, deduping by label (first occurrence wins).
 * Used for Lossy / lossless lines where multiple presets could map to the same short name.
 */
export function joinUniqueCatalogLabels(algorithmKeys) {
  const seen = new Set();
  const labels = [];
  for (const k of algorithmKeys) {
    const lab = displayAlgorithmCatalogName(k).trim().replace(/\s+/g, " ");
    if (!lab) {
      continue;
    }
    const dedupeKey = lab.toLowerCase();
    if (seen.has(dedupeKey)) {
      continue;
    }
    seen.add(dedupeKey);
    labels.push(lab);
  }
  return labels.join(", ");
}

/**
 * After {@link formatAvailableAlgorithmsCatalogLine}, parentheticals may collapse distinct keys
 * (e.g. BUFF variants) to duplicate tokens — dedupe comma-separated head, keep first occurrence.
 * Preserves optional `"; "` tail (literature suffix) unchanged.
 */
export function dedupeCommaSeparatedCatalogDisplayLine(line) {
  const s = String(line ?? "").trim();
  if (!s) {
    return "";
  }
  const semi = s.indexOf("; ");
  const head = semi === -1 ? s : s.slice(0, semi).trim();
  const tail = semi === -1 ? "" : s.slice(semi + 2).trim();
  const parts = head.split(/\s*,\s*/).map((x) => x.trim()).filter(Boolean);
  const seen = new Set();
  const out = [];
  for (const p of parts) {
    const k = p.toLowerCase().replace(/\s+/g, " ");
    if (seen.has(k)) {
      continue;
    }
    seen.add(k);
    out.push(p);
  }
  let result = out.join(", ");
  if (tail) {
    result = result ? `${result}; ${tail}` : tail;
  }
  return result;
}

/** @param {string|undefined|null} algorithm */
export function displayAlgorithmName(algorithm) {
  const s = normalizeAlgorithmKey(algorithm);
  if (s === "") {
    return "";
  }
  if (s.startsWith("user_lossless:") || s.startsWith("user_lossy:")) {
    const label = s.includes(":") ? s.split(":", 2)[1] : s;
    return withLeadingCapital(label || s);
  }
  if (ALGORITHM_KEY_LABEL_OVERRIDES[s] != null) {
    return withLeadingCapital(ALGORITHM_KEY_LABEL_OVERRIDES[s]);
  }
  if (
    s.startsWith("sz_float_ABS_") ||
    s.startsWith("sz_float_REL_") ||
    s.startsWith("sz_")
  ) {
    return "SZ";
  }
  if (
    s.startsWith("sz3_float_ABS_") ||
    s.startsWith("sz3_float_REL_") ||
    s.startsWith("sz3_")
  ) {
    return "SZ3";
  }
  if (s.startsWith("serf_float_") || s.startsWith("serf_ABS_")) {
    return "Serf";
  }
  if (s.startsWith("simpiece_")) {
    return "Sim-Piece";
  }
  if (s === "elf_on_chimp_float") {
    return "Elf";
  }
  if (s === "elf_plus_float" || s === "elf_float") {
    return "Elf+";
  }
  if (s === "camel_float") {
    return "Camel";
  }
  if (s === "chimp32" || s === "chimp_int" || s === "chimp_float") {
    return "Chimp32";
  }
  if (s.startsWith("ext_zfp_tol_")) {
    return "ZFP";
  }
  if (s === "ext_zfp_zfpy") {
    return "ZFP";
  }
  if (s.startsWith("spdp_level_") || s.startsWith("spdp_float_level_")) {
    return "SPDP";
  }
  if (s.startsWith("machete_rel_") || s.startsWith("machete_float_rel_")) {
    return "Machete";
  }
  if (s.startsWith("xz_level_") || s.startsWith("lzma_level_")) {
    return "XZ";
  }
  if (s.startsWith("zstd_level_")) {
    return "ZSTD";
  }
  const lzParsed = parseLzbenchAlgorithmKey(s);
  if (lzParsed) {
    return withLeadingCapital(lzParsed.entry.displayName);
  }
  return withLeadingCapital(s);
}

/**
 * Parameter configuration sliders / tooltips: preset detail in parentheses.
 * @param {string|undefined|null} algorithm
 */
export function displayAlgorithmParamName(algorithm) {
  const s = normalizeAlgorithmKey(algorithm);
  if (s === "") {
    return "";
  }
  if (s.startsWith("user_lossless:") || s.startsWith("user_lossy:")) {
    const label = s.includes(":") ? s.split(":", 2)[1] : s;
    return label ? `${label} (my algorithm)` : s;
  }
  const lzsseLevelLabels = {
    lzsse2: "2",
    lzsse4: "4",
    lzsse4fast: "4 Fast",
    lzsse8: "8",
    lzsse8fast: "8 Fast",
  };
  if (lzsseLevelLabels[s] != null) {
    return `LZSSE (level ${lzsseLevelLabels[s]})`;
  }
  if (ALGORITHM_KEY_LABEL_OVERRIDES[s] != null) {
    return displayAlgorithmName(algorithm);
  }
  if (s.startsWith("sz_float_ABS_")) {
    const rest = s.slice("sz_float_ABS_".length).replace(/_/g, "");
    return `SZ (ABS, ${rest})`;
  }
  if (s.startsWith("sz_float_REL_")) {
    const rest = s.slice("sz_float_REL_".length).replace(/_/g, "");
    return `SZ (REL, ${rest})`;
  }
  if (s.startsWith("sz_ABS_")) {
    const rest = s.slice("sz_ABS_".length).replace(/_/g, "");
    return `SZ (ABS, ${rest})`;
  }
  if (s.startsWith("sz_REL_")) {
    const rest = s.slice("sz_REL_".length).replace(/_/g, "");
    return `SZ (REL, ${rest})`;
  }
  if (s.startsWith("sz3_float_ABS_")) {
    const rest = s.slice("sz3_float_ABS_".length).replace(/_/g, "");
    return `SZ3 (ABS, ${rest})`;
  }
  if (s.startsWith("sz3_float_REL_")) {
    const rest = s.slice("sz3_float_REL_".length).replace(/_/g, "");
    return `SZ3 (REL, ${rest})`;
  }
  if (s.startsWith("sz3_")) {
    return `SZ3 (${s.replace(/^sz3_/, "")})`;
  }
  if (s.startsWith("sz_")) {
    return `SZ (${s.replace(/^sz_/, "")})`;
  }
  if (s.startsWith("serf_float_ABS_")) {
    const rest = s.slice("serf_float_ABS_".length).replace(/_/g, "");
    return `Serf (ABS, ${rest})`;
  }
  if (s.startsWith("serf_float_")) {
    return `Serf (${s.replace(/^serf_float_/, "")})`;
  }
  if (s.startsWith("simpiece_float_relRange_")) {
    const r = s.replace("simpiece_float_relRange_", "");
    return `Sim-Piece (ε vs range ${r})`;
  }
  if (s.startsWith("simpiece_")) {
    return `Sim-Piece (${s.replace(/^simpiece_/, "")})`;
  }
  if (s.startsWith("ext_zfp_tol_")) {
    const t = s.slice("ext_zfp_tol_".length);
    return `ZFP lossy (tol ${t})`;
  }
  if (s === "ext_zfp_zfpy") {
    return "ZFP lossless (zfpy)";
  }
  if (s.startsWith("spdp_level_") || s.startsWith("spdp_float_level_")) {
    const rest = s.startsWith("spdp_level_")
      ? s.slice("spdp_level_".length)
      : s.slice("spdp_float_level_".length);
    return `SPDP (level ${rest})`;
  }
  if (s.startsWith("machete_rel_") || s.startsWith("machete_float_rel_")) {
    const rest = (s.startsWith("machete_rel_")
      ? s.slice("machete_rel_".length)
      : s.slice("machete_float_rel_".length)
    ).replace(/_/g, "");
    return `Machete (rel ${rest})`;
  }
  if (s.startsWith("serf_ABS_")) {
    const rest = s.slice("serf_ABS_".length).replace(/_/g, "");
    return `Serf (ABS, ${rest})`;
  }
  const lzParsed = parseLzbenchAlgorithmKey(s);
  if (lzParsed) {
    const { entry, level } = lzParsed;
    if (entry.codecId === "bsc" || entry.presets) {
      return `${entry.displayName} (${entry.presetLabels?.[level] || level})`;
    }
    return `${entry.displayName} (level ${level})`;
  }
  return displayAlgorithmName(algorithm);
}

/**
 * Short label for “Available Algorithms” lists: drop parenthetical segments
 * from merged catalog lines (ASCII () and full-width （）).
 */
export function displayAlgorithmCatalogName(algorithm) {
  const key = normalizeAlgorithmKey(algorithm);
  if (key === "") {
    return "";
  }
  if (key.startsWith("xz_level_") || key.startsWith("lzma_level_")) {
    return "XZ";
  }
  if (key.startsWith("zstd_level_")) {
    return "ZSTD";
  }
  const lzParsed = parseLzbenchAlgorithmKey(key);
  if (lzParsed) {
    return withLeadingCapital(lzParsed.entry.displayName);
  }
  if (ALGORITHM_KEY_LABEL_OVERRIDES[key] != null) {
    return withLeadingCapital(ALGORITHM_KEY_LABEL_OVERRIDES[key]);
  }
  return withLeadingCapital(displayAlgorithmName(algorithm));
}

/**
 * Remove every parenthetical (ASCII () and full-width （）), repeatedly until stable.
 * For Java Available Algorithms: strip technical keys like (elf_on_chimp_float) from suffix text too.
 */
export function stripCatalogLineParentheticals(text) {
  let t = String(text ?? "");
  let prev;
  let guard = 0;
  do {
    prev = t;
    t = t.replace(/\s*（[^）]*）/g, "");
    t = t.replace(/\s*\([^)]*\)/g, "");
    guard++;
  } while (t !== prev && guard < 48);
  return t
    .replace(/[ \t]+/g, " ")
    .replace(/\s*,\s*/g, ", ")
    .replace(/\s*;\s*/g, "; ")
    .trim();
}

/**
 * Strip parentheticals from merged Available Algorithms lines (Java + Python catalogs).
 * Preserves empty-state messages that start with "(None" so hints stay readable.
 * @param {string} line
 */
export function formatAvailableAlgorithmsCatalogLine(line) {
  const s = String(line ?? "");
  if (/^\s*\(None/i.test(s)) {
    return s.trim();
  }
  return stripCatalogLineParentheticals(s);
}

/** @deprecated Use {@link formatAvailableAlgorithmsCatalogLine}. */
export const formatJavaAvailableAlgorithmsLine = formatAvailableAlgorithmsCatalogLine;

/**
 * Methods page catalog row label — aligned with homepage via {@link displayAlgorithmCatalogName}.
 * @param {string|undefined|null} methodName
 */
export function displayMethodCatalogName(methodName) {
  const raw = String(methodName ?? "").trim();
  if (!raw) {
    return "";
  }
  if (ALGORITHM_KEY_LABEL_OVERRIDES[raw] != null) {
    return withLeadingCapital(ALGORITHM_KEY_LABEL_OVERRIDES[raw]);
  }
  const catalogLabel = displayAlgorithmCatalogName(raw);
  if (catalogLabel !== raw) {
    return withLeadingCapital(catalogLabel);
  }
  const METHOD_PAGE_OVERRIDES = {
    elf_plus: "Elf+",
    Elf_PLUS: "Elf+",
    ELF_PLUS: "Elf+",
    Blosc: "Blosc",
    Blosc2: "Blosc2",
    "Blosc BloscLZ": "Blosc",
    "Blosc2 BloscLZ": "Blosc2",
    BLOSC: "Blosc",
    BLOSC2: "Blosc2",
    libdeflate: "Libdeflate",
    "zlib-ng": "Zlib-ng",
    slz_gzip: "Slz_gzip",
  };
  if (METHOD_PAGE_OVERRIDES[raw] != null) {
    return withLeadingCapital(METHOD_PAGE_OVERRIDES[raw]);
  }
  return withLeadingCapital(raw
    .split(/(\s+|_)/)
    .map((part) => {
      if (part === "" || /^\s+$/.test(part) || part === "_") {
        return part;
      }
      if (/^[a-z]/.test(part)) {
        return part.charAt(0).toUpperCase() + part.slice(1);
      }
      return part;
    })
    .join(""));
}
