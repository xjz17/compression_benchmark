/**
 * Lossy vs lossless for benchmark `algorithm` CSV keys.
 * Extend explicit rules as new error-bounded or approximate codecs are added.
 */

/** @param {string|undefined|null} algorithm */
export function normalizeAlgorithmKey(algorithm) {
  if (algorithm == null || algorithm === "") {
    return "";
  }
  return String(algorithm).trim().replace(/^\ufeff/, "");
}

/** Legacy columnar RLE keys in old CSV reports; TsFile ``RLE`` is canonical. */
const LEGACY_COLUMN_RLE_ALGORITHMS = new Set(["col_rle", "col_rle_int"]);

/** @param {string|undefined|null} algorithm */
export function isLegacyColumnRleAlgorithm(algorithm) {
  return LEGACY_COLUMN_RLE_ALGORITHMS.has(normalizeAlgorithmKey(algorithm));
}

/** Collapse legacy report keys to their canonical catalog name. */
export function canonicalAlgorithmKey(algorithm) {
  return normalizeAlgorithmKey(algorithm);
}

/** Error-bounded or approximate codecs (unified keys; int runs use cast-to-float columns). */
const LOSSY_PREFIXES = [
  "sz_",
  "sz3_",
  "serf_",
  "simpiece_",
  "ext_zfp_tol_",
  "machete_rel_",
  "machete_float_rel_",
  "most_minimal",
  "most_float_minimal",
];

/** @param {string|undefined|null} algorithm */
export function getAlgorithmCompressionMode(algorithm) {
  const key = normalizeAlgorithmKey(algorithm);
  if (key === "") {
    return "lossless";
  }
  if (key.startsWith("user_lossy:")) {
    return "lossy";
  }
  if (key.startsWith("user_lossless:")) {
    return "lossless";
  }
  if (LOSSY_PREFIXES.some((p) => key.startsWith(p))) {
    return "lossy";
  }
  return "lossless";
}

/** @param {string|undefined|null} algorithm */
export function isLossyAlgorithm(algorithm) {
  return getAlgorithmCompressionMode(algorithm) === "lossy";
}

/** Table / filter label (English UI). */
export function compressionModeLabel(algorithm) {
  return isLossyAlgorithm(algorithm) ? "Lossy" : "Lossless";
}

/** Element Plus tag type */
export function compressionModeTagType(algorithm) {
  return isLossyAlgorithm(algorithm) ? "warning" : "success";
}
