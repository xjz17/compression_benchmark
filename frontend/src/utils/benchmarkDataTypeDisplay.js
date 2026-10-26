/** User-facing benchmark data type labels. */

export function displayBenchmarkDataType(dtype) {
  const key = String(dtype || "").trim().toLowerCase();
  return key;
}

export function displayBenchmarkDataTypeLabel(dtype) {
  const displayed = displayBenchmarkDataType(dtype);
  if (!displayed) {
    return "";
  }
  return displayed.charAt(0).toUpperCase() + displayed.slice(1);
}
