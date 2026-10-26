import axios from "@/utils/httpClient.js";
import { sortOperatorsByTaxonomy } from "@/utils/operatorTaxonomy.js";

/**
 * @returns {Promise<Array<{ id: string, name: string, description: string }>>}
 */
export async function fetchEncodingOperatorCatalog() {
  const { data } = await axios.get("/api/encoding-operators/");
  return Array.isArray(data?.operators) ? data.operators : [];
}

/**
 * Build Dataset page operator select options: ``original`` plus full encoding catalog.
 * @param {Array<{ id: string, name: string, description?: string }>} catalog
 */
export function buildDatasetOperatorOptions(catalog) {
  const base = [{ value: "", label: "original" }];
  const fromCatalog = sortOperatorsByTaxonomy(catalog)
    .map((row) => {
      const id = String(row?.id || "").trim();
      if (!id) {
        return null;
      }
      const name = String(row?.name || id).trim() || id;
      return {
        value: id,
        label: id === "grouping" ? `${name} (multi-column layout)` : name,
        classId: row.classId,
        classLabel: row.classLabel,
        disabled: id === "grouping",
      };
    })
    .filter(Boolean);
  return [...base, ...fromCatalog];
}
