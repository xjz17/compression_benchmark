/**
 * Optional suffix lines for Available Algorithms (see algorithmDisplayName.js).
 */

const LITERATURE_PYTHON_FLOAT_MSD = "";

/**
 * @param {"java"|"python"|"cpp"} language
 * @param {"int"|"float"} dataType
 */
export function literatureLosslessSuffix(language, dataType) {
  if (language === "python" && dataType === "float") return LITERATURE_PYTHON_FLOAT_MSD;
  if (language === "python") return "";
  if (language === "cpp") return "";
  if (language === "java" && dataType === "float") return "";
  return "";
}

/**
 * @param {"java"|"python"|"cpp"} _language
 * @param {"int"|"float"} dataType
 * @returns {string} empty when no literature lossy entries apply
 */
export function literatureLossySuffix(_language, dataType) {
  void _language;
  void dataType;
  return "";
}

/**
 * @param {string} baseText from live benchmark keys only
 * @param {string} suffix from literatureLosslessSuffix / literatureLossySuffix
 */
export function mergeAlgorithmLine(baseText, suffix) {
  const b = (baseText || "").trim();
  const s = (suffix || "").trim();
  if (!s) return b;
  if (!b) return s;
  return `${b}; ${s}`;
}
