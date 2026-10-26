/**
 * Labels for dataset dropdowns: built-in catalog keys ``(public)``,
 * user uploads follow visibility ``(public|private)``.
 */

export function isUserUploadDatasetKey(key) {
  return String(key || "").startsWith("user-upload-");
}

/** Label for one catalog API row (`/api/datasets/` items). */
export function datasetSelectLabelFromRow(row) {
  if (!row) return "";
  if (row.is_user_upload) {
    const name = String(row.display_name || row.key || "").trim();
    const visibility = row.is_public ? "public" : "private";
    return `${name || row.key} (${visibility})`;
  }
  return `${String(row.key || "").trim()} (public)`;
}

/** Label when only key/name is known; optional visibility for user uploads. */
export function datasetSelectLabelFromKey(key, displayName = "", isPublic = null) {
  const k = String(key || "").trim();
  if (!k) return "";
  if (isUserUploadDatasetKey(k)) {
    const name = String(displayName || "").trim();
    const visibility = isPublic === true ? "public" : "private";
    return `${name || k} (${visibility})`;
  }
  return `${k} (public)`;
}
