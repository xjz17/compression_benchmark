import { ref, computed, watch } from "vue";
import axios from "@/utils/httpClient.js";
import { CATALOG_API_TIMEOUT_MS } from "@/constants/apiTimeouts.js";
import { HIDE_LOSSY_ALGORITHMS } from "@/constants/hideLossyAlgorithms.js";
import { ElMessage } from "element-plus";
import { useAuth } from "@/composables/useAuth.js";
import {
  dedupeCommaSeparatedCatalogDisplayLine,
  displayAlgorithmName,
  displayAlgorithmParamName,
  formatAvailableAlgorithmsCatalogLine,
  joinUniqueCatalogLabels,
} from "@/utils/algorithmDisplayName.js";
import { getBenchmarkCatalogAlgorithms } from "@/utils/benchmarkAlgorithmCatalog.js";
import { CURRENT_CPP_ACTIVE_METHOD_NAMES } from "@/utils/currentCppMethodCatalog.js";
import {
  defaultLzbenchLevelSelection,
  displayLevelLabel,
  lzbenchCodecLevelControls,
  sliderLabelsForEntry,
} from "@/utils/lzbenchCodecCatalog.js";
import {
  compressionModeLabel,
  compressionModeTagType,
  isLossyAlgorithm,
} from "@/utils/algorithmCompressionMode.js";
import {
  literatureLosslessSuffix,
  literatureLossySuffix,
  mergeAlgorithmLine,
} from "@/utils/literatureBenchmarkAlgorithms.js";
import {
  buildUploadFormFilterFields,
  filterBenchmarkReportItems,
} from "@/utils/benchmarkReportFilter.js";
import { datasetSelectLabelFromRow } from "@/utils/datasetSelectLabel.js";
import { withBalancedScores } from "@/utils/leaderboardScoring.js";

export const BENCHMARK_CONFIG_KEY = Symbol("benchmarkConfig");

const SIMPIECE_PARAM_KEYS = [
  "simpiece_relRange_0p005",
  "simpiece_relRange_0p01",
];
const SZ_ABS_EPS_LABELS = ["1e-1", "1e-2", "1e-3", "1e-4"];
const SZ_REL_EPS_LABELS = ["1e-2", "1e-3"];
const SZ3_ABS_EPS_LABELS = ["1e-1", "1e-2", "1e-3", "1e-4"];
const SZ3_REL_EPS_LABELS = ["1e-2", "1e-3", "1e-4"];
const MACHETE_REL_EPS_LABELS = ["1e-1", "1e-2", "1e-3", "1e-4"];
const ZFP_TOL_EPS_LABELS = ["1e-1", "1e-2", "1e-3"];
const SPDP_LEVEL_LABELS = Array.from({ length: 10 }, (_, i) => String(i));
const CHIMP_HISTORY_VALUES = [1, 128];
const SERF_PARAM_KEYS = [
  "serf_float_ABS_1e-1",
  "serf_float_ABS_1e-2",
  "serf_float_ABS_1e-3",
  "serf_float_ABS_1e-4",
];
const SIMPIECE_MARK_LABELS = ["0.005", "0.01"];
const SERF_MARK_LABELS = ["1e-1", "1e-2", "1e-3", "1e-4"];
const HIDDEN_AVAILABLE_ALGORITHM_KEYS = new Set();

function labelsToSliderMarks(labels) {
  const o = {};
  labels.forEach((label, i) => {
    o[i] = label;
  });
  return o;
}

function isSingleDatasetLikeDimension(dim) {
  return dim === "single_dataset" || dim === "single_column";
}

/** Dataset-level Home dimensions (not single column). */
function isDatasetLevelDataDimension(dim) {
  return dim === "single_dataset" || dim === "single_field" || dim === "all_datasets";
}

export const FLOAT_PRECISION_FILTER_ALL = "all";
export const FLOAT_PRECISION_FILTER_FIXED = "fixed";
export const FLOAT_PRECISION_FILTER_NON_FIXED = "non_fixed";
function benchmarkDataTypeFromLogicalType(logicalType) {
  const t = String(logicalType || "").trim().toLowerCase();
  if (t === "integer" || t === "int" || t === "long") return "int";
  if (t === "float" || t === "double") return "float";
  return "";
}

function benchmarkDataTypeToSelectValue(benchmarkType) {
  if (benchmarkType === "int") return "1";
  if (benchmarkType === "float") return "2";
  if (benchmarkType === "overall") return "4";
  return "";
}

const languageMap = { 1: "java", 2: "python", 3: "cpp" };
const dataTypeMap = { 1: "int", 2: "float", 4: "overall" };

export const mapImplementLanguage = (v) => {
  if (v === null || v === undefined || v === "") return undefined;
  return languageMap[v] ?? languageMap[Number(v)];
};
export const mapDataType = (v) => {
  if (v === null || v === undefined || v === "") return undefined;
  return dataTypeMap[v] ?? dataTypeMap[Number(v)];
};

const INTEGER_LOGICAL_TYPES = new Set(["integer", "int", "long"]);
const FLOAT_LOGICAL_TYPES = new Set(["float", "double"]);
export function datasetLogicalTypes(row) {
  if (!row) return [];
  if (!Array.isArray(row.logical_types)) return [];
  return row.logical_types.map((t) => String(t).trim().toLowerCase()).filter(Boolean);
}

export function datasetHasIntegerColumns(row) {
  return datasetLogicalTypes(row).some((t) => INTEGER_LOGICAL_TYPES.has(t));
}

export function datasetHasFloatColumns(row) {
  const types = datasetLogicalTypes(row);
  if (!types.length) return true;
  return types.some((t) => FLOAT_LOGICAL_TYPES.has(t));
}

/** Prefer ``benchmark_dtypes`` from catalog API when present. */
export function datasetHasBenchmarkDtype(row, dtype) {
  if (!row) {
    if (dtype === "int" || dtype === "float") return true;
    return false;
  }
  const dtypes = row.benchmark_dtypes;
  if (Array.isArray(dtypes) && dtypes.length > 0) {
    return dtypes.includes(dtype);
  }
  if (dtype === "int") return datasetHasIntegerColumns(row);
  if (dtype === "float") return datasetHasFloatColumns(row);
  return false;
}

function datasetHasVisibleBenchmarkDtype(row) {
  return datasetHasBenchmarkDtype(row, "int") || datasetHasBenchmarkDtype(row, "float");
}

function datasetMajorCategory(category) {
  const raw = String(category || "").trim();
  if (!raw) return "Uncategorized";
  return raw.split("/")[0].trim() || raw;
}

/**
 * @param {{ rawItemsRef?: import('vue').Ref, includeDatasetDimension?: boolean }} options
 */
export function useBenchmarkRunConfig(options = {}) {
  const { rawItemsRef = null, includeDatasetDimension = false } = options;

  const implement_language = ref("3");
  const ranking_criteria = ref("4");
  const compression_rate_mode = ref("average");
  const compression_time_mode = ref("average");
  const decompression_time_mode = ref("average");
  const compression_scope = ref(HIDE_LOSSY_ALGORITHMS ? "lossless" : "all");
  const data_type = ref("4");
  const float_precision = ref(FLOAT_PRECISION_FILTER_ALL);
  const xz_level = ref(9);
  const zstd_level = ref(9);
  const chimp_history = ref(128);
  const simpiece_param = ref(0);
  const sz_absrel = ref("ABS");
  const sz_eps = ref(0);
  const sz3_absrel = ref("ABS");
  const sz3_eps = ref(0);
  const machete_eps = ref(2);
  const spdp_level = ref(9);
  const zfp_tol_eps = ref(2);
  const serf_param = ref(0);
  const lzbench_levels = ref(defaultLzbenchLevelSelection());
  const lzbench_slider_idx = ref({});

  function syncLzbenchSliderIndexFromLevels() {
    const idx = {};
    for (const entry of lzbenchCodecLevelControls()) {
      const labels = sliderLabelsForEntry(entry);
      const current = lzbench_levels.value[entry.codecId];
      let i = labels.indexOf(String(current));
      if (i < 0) i = Math.max(0, labels.length - 1);
      idx[entry.codecId] = i;
    }
    lzbench_slider_idx.value = idx;
  }

  function syncLzbenchLevelsFromSliderIndex() {
    const next = { ...lzbench_levels.value };
    for (const entry of lzbenchCodecLevelControls()) {
      const labels = sliderLabelsForEntry(entry);
      const raw = lzbench_slider_idx.value[entry.codecId] ?? 0;
      const i = Math.max(0, Math.min(raw, labels.length - 1));
      const label = labels[i];
      next[entry.codecId] = entry.codecId === "bsc" ? label : Number(label);
    }
    lzbench_levels.value = next;
  }

  syncLzbenchSliderIndexFromLevels();

  watch(lzbench_slider_idx, () => syncLzbenchLevelsFromSliderIndex(), { deep: true });

  const lzbenchLevelControls = computed(() => lzbenchCodecLevelControls());

  const apiItems = ref([]);
  const data_dimension = ref("all_datasets");
  const dataset = ref("AI4I-Maintenance");
  const datasets = ref([]);
  const datasetCatalogByKey = ref({});
  const datasetsLoading = ref(false);
  const { isAuthenticated } = useAuth();
  const selected_category = ref("");
  const categoryOptions = ref([]);
  const selected_column = ref("");
  const columnOptions = ref([]);
  const columnOptionsLoading = ref(false);

  const catalogAlgorithms = getBenchmarkCatalogAlgorithms();

  const reportFilterParams = computed(() => {
    const lang = mapImplementLanguage(implement_language.value);
    return {
      implement_language: lang,
      xz_level: xz_level.value,
      zstd_level: zstd_level.value,
      chimp_history: chimp_history.value,
      simpiece_param: simpiece_param.value,
      sz_absrel: sz_absrel.value,
      sz_eps: sz_eps.value,
      sz3_absrel: sz3_absrel.value,
      sz3_eps: sz3_eps.value,
      machete_eps: machete_eps.value,
      spdp_level: spdp_level.value,
      zfp_tol_eps: zfp_tol_eps.value,
      serf_param: serf_param.value,
      lzbench_levels: lzbench_levels.value,
    };
  });

  const uploadFilterFormFields = computed(() =>
    buildUploadFormFilterFields(reportFilterParams.value)
  );

  function normalizeSelectValue(v) {
    if (v != null && typeof v === "object" && "value" in v) {
      return v.value;
    }
    return v;
  }

  const showResearchSliders = computed(() => {
    if (HIDE_LOSSY_ALGORITHMS) return false;
    const lang = mapImplementLanguage(implement_language.value);
    return lang === "java" || lang === "python" || lang === "cpp";
  });

  /** SPDP is lossless; keep visible when lossy research sliders are hidden. */
  const showSpdpLevelSlider = computed(() => {
    const lang = mapImplementLanguage(implement_language.value);
    return lang === "java" || lang === "python" || lang === "cpp";
  });

  /** ZFP fixed-accuracy tol rows (float only; lossless ext_zfp_zfpy is always kept). */
  const showZfpTolSlider = computed(() => {
    if (HIDE_LOSSY_ALGORITHMS) return false;
    const lang = mapImplementLanguage(implement_language.value);
    const type = mapDataType(data_type.value);
    if (type !== "float") return false;
    return lang === "java" || lang === "python" || lang === "cpp";
  });

  const activeCatalogDatasets = computed(() => {
    if (!includeDatasetDimension) return [];
    const byKey = datasetCatalogByKey.value;
    const dim = data_dimension.value;
    if (isSingleDatasetLikeDimension(dim)) {
      const row = byKey[normalizeSelectValue(dataset.value)];
      return row ? [row] : [];
    }
    if (dim === "single_field") {
      const cat = String(normalizeSelectValue(selected_category.value) || "").trim();
      if (!cat) return [];
      return Object.values(byKey).filter(
        (row) => datasetMajorCategory(row.category) === cat
      );
    }
    if (dim === "all_datasets") return Object.values(byKey);
    return [];
  });

  const selectedColumnOption = computed(() => {
    if (data_dimension.value !== "single_column") return null;
    const selected = String(selected_column.value || "");
    if (!selected) return null;
    return columnOptions.value.find((o) => String(o.value) === selected) || null;
  });

  function columnOptionsHaveBenchmarkType(benchmarkType) {
    return columnOptions.value.some((o) => o.benchmark_data_type === benchmarkType);
  }

  const canSelectIntDataType = computed(() => {
    if (!includeDatasetDimension) return true;
    if (data_dimension.value === "single_column") {
      const colType = selectedColumnOption.value?.benchmark_data_type || "";
      if (colType) return colType === "int";
      return columnOptionsHaveBenchmarkType("int");
    }
    return activeCatalogDatasets.value.some((row) => datasetHasIntegerColumns(row));
  });

  const canSelectFloatDataType = computed(() => {
    if (!includeDatasetDimension) return true;
    if (data_dimension.value === "single_column") {
      const colType = selectedColumnOption.value?.benchmark_data_type || "";
      if (colType) return colType === "float";
      return columnOptionsHaveBenchmarkType("float");
    }
    const active = activeCatalogDatasets.value;
    if (!active.length) return true;
    return active.some((row) => datasetHasFloatColumns(row));
  });

  const canSelectStringDataType = computed(() => {
    return false;
  });

  const canSelectOverallDataType = computed(() => {
    if (!includeDatasetDimension) return true;
    if (data_dimension.value === "single_column") return false;
    const active = activeCatalogDatasets.value;
    if (!active.length) return true;
    return active.some((row) => datasetHasVisibleBenchmarkDtype(row));
  });

  const isStringDataType = computed(() => false);
  const isFloatDataType = computed(() => mapDataType(data_type.value) === "float");
  const showFloatPrecisionSelector = computed(() => {
    if (!includeDatasetDimension) return false;
    if (!isFloatDataType.value) return false;
    return isDatasetLevelDataDimension(data_dimension.value);
  });

  const dataTypeSelectionHint = computed(() => {
    if (!includeDatasetDimension || !activeCatalogDatasets.value.length) return "";
    // In single-column mode, dtype is auto-aligned to selected column and the other
    // option is disabled, so hint text would be redundant/noisy.
    if (data_dimension.value === "single_column") return "";
    if (!canSelectIntDataType.value) {
      if (isSingleDatasetLikeDimension(data_dimension.value)) {
        return "This dataset has no integer columns.";
      }
      if (data_dimension.value === "single_field") {
        return "No dataset in this category has integer columns.";
      }
      return "None of the registered datasets include integer columns.";
    }
    if (!canSelectFloatDataType.value) {
      return "No floating-point columns are available for the current selection.";
    }
    return "";
  });

  const simpieceParamMax = computed(() => Math.max(0, SIMPIECE_PARAM_KEYS.length - 1));
  const serfParamMax = computed(() => Math.max(0, SERF_PARAM_KEYS.length - 1));
  const szEpsMax = computed(() =>
    String(sz_absrel.value).toUpperCase() === "REL"
      ? SZ_REL_EPS_LABELS.length - 1
      : SZ_ABS_EPS_LABELS.length - 1
  );
  const sz3EpsMax = computed(() =>
    String(sz3_absrel.value).toUpperCase() === "REL"
      ? SZ3_REL_EPS_LABELS.length - 1
      : SZ3_ABS_EPS_LABELS.length - 1
  );
  const macheteEpsMax = computed(() => Math.max(0, MACHETE_REL_EPS_LABELS.length - 1));
  const zfpTolEpsMax = computed(() => Math.max(0, ZFP_TOL_EPS_LABELS.length - 1));
  const spdpLevelMax = computed(() => Math.max(0, SPDP_LEVEL_LABELS.length - 1));

  const simpieceSliderMarks = computed(() => labelsToSliderMarks(SIMPIECE_MARK_LABELS));
  const szSliderMarks = computed(() =>
    labelsToSliderMarks(
      String(sz_absrel.value).toUpperCase() === "REL" ? SZ_REL_EPS_LABELS : SZ_ABS_EPS_LABELS
    )
  );
  const sz3SliderMarks = computed(() =>
    labelsToSliderMarks(
      String(sz3_absrel.value).toUpperCase() === "REL" ? SZ3_REL_EPS_LABELS : SZ3_ABS_EPS_LABELS
    )
  );
  const macheteSliderMarks = computed(() => labelsToSliderMarks(MACHETE_REL_EPS_LABELS));
  const zfpTolSliderMarks = computed(() => labelsToSliderMarks(ZFP_TOL_EPS_LABELS));
  const spdpSliderMarks = computed(() => labelsToSliderMarks(SPDP_LEVEL_LABELS));
  const serfSliderMarks = computed(() => labelsToSliderMarks(SERF_MARK_LABELS));

  const simpieceCurrentLabel = computed(() =>
    displayAlgorithmParamName(
      SIMPIECE_PARAM_KEYS[Math.min(simpiece_param.value, SIMPIECE_PARAM_KEYS.length - 1)] ?? ""
    )
  );
  const serfCurrentLabel = computed(() =>
    displayAlgorithmParamName(SERF_PARAM_KEYS[Math.min(serf_param.value, SERF_PARAM_KEYS.length - 1)] ?? "")
  );
  const szEpsCurrentLabel = computed(() => {
    const rel = String(sz_absrel.value).toUpperCase() === "REL";
    const labels = rel ? SZ_REL_EPS_LABELS : SZ_ABS_EPS_LABELS;
    return labels[Math.max(0, Math.min(sz_eps.value, labels.length - 1))];
  });
  const sz3EpsCurrentLabel = computed(() => {
    const rel = String(sz3_absrel.value).toUpperCase() === "REL";
    const labels = rel ? SZ3_REL_EPS_LABELS : SZ3_ABS_EPS_LABELS;
    return labels[Math.max(0, Math.min(sz3_eps.value, labels.length - 1))];
  });
  const macheteEpsCurrentLabel = computed(() => {
    const labels = MACHETE_REL_EPS_LABELS;
    return labels[Math.max(0, Math.min(machete_eps.value, labels.length - 1))];
  });
  const zfpTolEpsCurrentLabel = computed(() => {
    const labels = ZFP_TOL_EPS_LABELS;
    return labels[Math.max(0, Math.min(zfp_tol_eps.value, labels.length - 1))];
  });
  const spdpLevelCurrentLabel = computed(() => {
    const labels = SPDP_LEVEL_LABELS;
    return labels[Math.max(0, Math.min(spdp_level.value, labels.length - 1))];
  });

  const simpieceTooltip = (val) =>
    displayAlgorithmParamName(SIMPIECE_PARAM_KEYS[Math.min(val, SIMPIECE_PARAM_KEYS.length - 1)] ?? "");
  const szEpsTooltip = (val) => {
    const rel = String(sz_absrel.value).toUpperCase() === "REL";
    const labels = rel ? SZ_REL_EPS_LABELS : SZ_ABS_EPS_LABELS;
    return labels[Math.max(0, Math.min(val, labels.length - 1))];
  };
  const sz3EpsTooltip = (val) => {
    const rel = String(sz3_absrel.value).toUpperCase() === "REL";
    const labels = rel ? SZ3_REL_EPS_LABELS : SZ3_ABS_EPS_LABELS;
    return labels[Math.max(0, Math.min(val, labels.length - 1))];
  };
  const macheteEpsTooltip = (val) => {
    const labels = MACHETE_REL_EPS_LABELS;
    return labels[Math.max(0, Math.min(val, labels.length - 1))];
  };
  const zfpTolEpsTooltip = (val) => {
    const labels = ZFP_TOL_EPS_LABELS;
    return labels[Math.max(0, Math.min(val, labels.length - 1))];
  };
  const spdpLevelTooltip = (val) => {
    const labels = SPDP_LEVEL_LABELS;
    return labels[Math.max(0, Math.min(val, labels.length - 1))];
  };
  const serfTooltip = (val) =>
    displayAlgorithmParamName(SERF_PARAM_KEYS[Math.min(val, SERF_PARAM_KEYS.length - 1)] ?? "");

  watch(sz_absrel, () => {
    if (sz_eps.value > szEpsMax.value) sz_eps.value = szEpsMax.value;
  });
  watch(sz3_absrel, () => {
    if (sz3_eps.value > sz3EpsMax.value) sz3_eps.value = sz3EpsMax.value;
  });
  watch(machete_eps, () => {
    if (machete_eps.value > macheteEpsMax.value) machete_eps.value = macheteEpsMax.value;
  });
  watch(zfp_tol_eps, () => {
    if (zfp_tol_eps.value > zfpTolEpsMax.value) zfp_tol_eps.value = zfpTolEpsMax.value;
  });
  watch(spdp_level, () => {
    if (spdp_level.value > spdpLevelMax.value) spdp_level.value = spdpLevelMax.value;
  });

  const availableAlgorithmsLossless = computed(() => {
    const language = mapImplementLanguage(implement_language.value);
    const type = mapDataType(data_type.value);
    const displayCatalog = language === "cpp"
      ? CURRENT_CPP_ACTIVE_METHOD_NAMES
      : catalogAlgorithms;
    const losslessKeys = displayCatalog.filter((a) => !isLossyAlgorithm(a));
    const visibleLosslessKeys = losslessKeys.filter(
      (key) => !HIDDEN_AVAILABLE_ALGORITHM_KEYS.has(String(key)),
    );
    let line = mergeAlgorithmLine(
      joinUniqueCatalogLabels(visibleLosslessKeys),
      type === "overall" ? "" : literatureLosslessSuffix(language, type)
    );
    line = formatAvailableAlgorithmsCatalogLine(line);
    return dedupeCommaSeparatedCatalogDisplayLine(line);
  });

  const availableAlgorithmsLossy = computed(() => {
    if (HIDE_LOSSY_ALGORITHMS) {
      return "";
    }
    const type = mapDataType(data_type.value);
    const language = mapImplementLanguage(implement_language.value);
    const displayCatalog = language === "cpp"
      ? CURRENT_CPP_ACTIVE_METHOD_NAMES
      : catalogAlgorithms;
    const lossy = displayCatalog.filter((a) => isLossyAlgorithm(a));
    const base = lossy.length
      ? joinUniqueCatalogLabels(lossy)
      : "(None for this combination)";
    let line = mergeAlgorithmLine(base, literatureLossySuffix(language, type));
    return formatAvailableAlgorithmsCatalogLine(line);
  });

  const filteredSourceItems = computed(() => {
    const raw = rawItemsRef ? rawItemsRef.value : apiItems.value;
    return filterBenchmarkReportItems(raw, reportFilterParams.value);
  });

  const sortedBenchmarkItems = computed(() => {
    const source = filteredSourceItems.value || [];
    const scoreRateMode = compression_rate_mode.value;
    const scoreTimeMode = compression_time_mode.value;
    const arr = withBalancedScores(source).map((row) => ({
      ...row,
      balanced_score_rate_mode: scoreRateMode,
      balanced_score_time_mode: scoreTimeMode,
    }));
    const rk = String(ranking_criteria.value ?? "1");
    if (rk === "1") {
      arr.sort((a, b) => parseFloat(a.compression_rate) - parseFloat(b.compression_rate));
    } else if (rk === "2") {
      arr.sort((a, b) => parseFloat(a.compression_time) - parseFloat(b.compression_time));
    } else if (rk === "3") {
      arr.sort((a, b) => parseFloat(a.decompression_time) - parseFloat(b.decompression_time));
    } else if (rk === "4") {
      arr.sort((a, b) => a.balanced_score - b.balanced_score);
    }
    return arr;
  });

  const leaderboardItems = computed(() => {
    if (HIDE_LOSSY_ALGORITHMS) {
      return sortedBenchmarkItems.value.filter((row) => !isLossyAlgorithm(row.algorithm));
    }
    const scope = String(compression_scope.value ?? "all");
    if (scope === "all") return sortedBenchmarkItems.value;
    if (scope === "lossy") {
      return sortedBenchmarkItems.value.filter((row) => isLossyAlgorithm(row.algorithm));
    }
    return sortedBenchmarkItems.value.filter((row) => !isLossyAlgorithm(row.algorithm));
  });

  async function fetchHomeDatasetOptions() {
    if (!includeDatasetDimension) return;
    datasetsLoading.value = true;
    try {
      const params = { lite: "1" };
      if (isAuthenticated.value) {
        params.include_user_uploads = "true";
      }
      const { data } = await axios.get("/api/datasets/", {
        params,
        timeout: CATALOG_API_TIMEOUT_MS,
      });
      const items = (data.items || []).filter(datasetHasVisibleBenchmarkDtype);
      datasetCatalogByKey.value = Object.fromEntries(items.map((row) => [row.key, row]));
      datasets.value = items.map((row) => ({
        value: row.key,
        label: datasetSelectLabelFromRow(row),
      }));
      const keys = new Set(datasets.value.map((d) => d.value));
      const currentKey = normalizeSelectValue(dataset.value);
      if (keys.size && !keys.has(currentKey)) {
        dataset.value = datasets.value[0].value;
      } else if (currentKey && currentKey !== dataset.value) {
        dataset.value = currentKey;
      }
      const dist = data.summary?.category_distribution || [];
      categoryOptions.value = dist.map((row) => ({
        value: row.category,
        label: row.category,
      }));
      const catValues = new Set(categoryOptions.value.map((c) => c.value));
      if (catValues.size && !catValues.has(selected_category.value)) {
        selected_category.value = categoryOptions.value[0].value;
      }
    } catch (error) {
      console.error(error);
      datasets.value = [];
      datasetCatalogByKey.value = {};
      categoryOptions.value = [];
      ElMessage.error("Failed to load dataset list.");
    } finally {
      datasetsLoading.value = false;
    }
  }

  async function fetchDatasetColumnOptions() {
    if (!includeDatasetDimension || data_dimension.value !== "single_column") {
      columnOptions.value = [];
      selected_column.value = "";
      return;
    }
    const key = normalizeSelectValue(dataset.value);
    if (!key) {
      columnOptions.value = [];
      selected_column.value = "";
      return;
    }
    columnOptionsLoading.value = true;
    try {
      const { data } = await axios.get(`/api/datasets/${encodeURIComponent(String(key))}/columns/`);
      const rows = Array.isArray(data?.columns) ? data.columns : [];
      const opts = rows
        .filter((r) => {
          if (!r || r.column_name == null) return false;
          const benchmarkType =
            String(r.benchmark_data_type || "").trim() ||
            benchmarkDataTypeFromLogicalType(r.logical_type);
          return benchmarkType === "int" || benchmarkType === "float";
        })
        .map((r) => {
          const idx = Number.isInteger(r.column_index) ? r.column_index : null;
          const name = String(r.column_name);
          const benchmarkType =
            String(r.benchmark_data_type || "").trim() ||
            benchmarkDataTypeFromLogicalType(r.logical_type);
          return {
            value: idx != null ? String(idx) : name,
            label: idx != null ? `${idx}: ${name}` : name,
            logical_type: String(r.logical_type || "").trim().toLowerCase(),
            benchmark_data_type: benchmarkType,
          };
        });
      columnOptions.value = opts;
      const selected = String(selected_column.value || "");
      const valid = opts.some((o) => o.value === selected);
      selected_column.value = valid ? selected : opts[0]?.value || "";
    } catch (error) {
      console.error(error);
      columnOptions.value = [];
      selected_column.value = "";
      ElMessage.error("Failed to load dataset columns.");
    } finally {
      columnOptionsLoading.value = false;
    }
  }

  function buildStaticDataParams(options = {}) {
    const leaderboardMode = String(options.leaderboardMode ?? "").trim().toLowerCase();
    const implLang = mapImplementLanguage(implement_language.value);
    const dtype = mapDataType(data_type.value);
    const paperTable3Mode =
      leaderboardMode === "paper_table3";
    const params = {
      data_type: dtype,
      implement_language: paperTable3Mode ? "cpp" : implLang,
      xz_level: xz_level.value,
      zstd_level: zstd_level.value,
      data_dimension: data_dimension.value,
      compression_rate_mode: paperTable3Mode ? "average" : compression_rate_mode.value,
      compression_time_mode: paperTable3Mode ? "average" : compression_time_mode.value,
      decompression_time_mode: paperTable3Mode ? "average" : decompression_time_mode.value,
    };
    if (paperTable3Mode) {
      params.leaderboard_mode = "paper_table3";
    }
    if (data_dimension.value === "single_field") {
      params.category = normalizeSelectValue(selected_category.value);
    } else if (data_dimension.value === "single_column") {
      params.dataset = normalizeSelectValue(dataset.value);
      params.column_index = selected_column.value;
    } else if (isSingleDatasetLikeDimension(data_dimension.value)) {
      params.dataset = normalizeSelectValue(dataset.value);
    }
    if (implLang === "java" || implLang === "python" || implLang === "cpp") {
      params.simpiece_param = simpiece_param.value;
      params.chimp_history = chimp_history.value;
      params.sz_absrel = sz_absrel.value;
      params.sz_eps = sz_eps.value;
      params.sz3_absrel = sz3_absrel.value;
      params.sz3_eps = sz3_eps.value;
      params.machete_eps = machete_eps.value;
      params.spdp_level = spdp_level.value;
      params.zfp_tol_eps = zfp_tol_eps.value;
      params.serf_param = serf_param.value;
    }
    if (showFloatPrecisionSelector.value) {
      params.float_precision = float_precision.value;
    }
    params.lzbench_levels = JSON.stringify(lzbench_levels.value);
    return params;
  }

  watch(isAuthenticated, () => {
    if (includeDatasetDimension) {
      fetchHomeDatasetOptions();
      fetchDatasetColumnOptions();
    }
  });

  function pickFirstAvailableDataType() {
    if (canSelectIntDataType.value) {
      data_type.value = "1";
    } else if (canSelectFloatDataType.value) {
      data_type.value = "2";
    } else if (canSelectOverallDataType.value) {
      data_type.value = "4";
    }
  }

  watch(selectedColumnOption, (col) => {
    if (!includeDatasetDimension || data_dimension.value !== "single_column") return;
    const next = benchmarkDataTypeToSelectValue(col?.benchmark_data_type || "");
    if (next && next !== "3") {
      data_type.value = next;
    } else if (String(data_type.value) === "3") {
      pickFirstAvailableDataType();
    }
  });

  watch([dataset, data_dimension, selected_category, canSelectIntDataType, canSelectFloatDataType, canSelectStringDataType, canSelectOverallDataType], () => {
    if (!includeDatasetDimension) return;
    if (data_dimension.value === "single_column") return;
    if (!canSelectIntDataType.value && String(data_type.value) === "1") {
      pickFirstAvailableDataType();
    }
    if (!canSelectFloatDataType.value && String(data_type.value) === "2") {
      pickFirstAvailableDataType();
    }
    if (!canSelectStringDataType.value && String(data_type.value) === "3") {
      pickFirstAvailableDataType();
    }
    if (!canSelectOverallDataType.value && String(data_type.value) === "4") {
      pickFirstAvailableDataType();
    }
  });
  watch([data_dimension, selectedColumnOption], () => {
    if (!includeDatasetDimension) return;
    if (data_dimension.value === "single_column" && String(data_type.value) === "4") {
      pickFirstAvailableDataType();
    }
  });
  watch([dataset, data_dimension], () => {
    if (!includeDatasetDimension) return;
    fetchDatasetColumnOptions();
  });

  watch(data_type, () => {
    if (!showFloatPrecisionSelector.value) {
      float_precision.value = FLOAT_PRECISION_FILTER_ALL;
    }
  });

  watch(data_dimension, () => {
    if (!showFloatPrecisionSelector.value) {
      float_precision.value = FLOAT_PRECISION_FILTER_ALL;
    }
  });

  function lzbenchSliderMax(entry) {
    return Math.max(0, sliderLabelsForEntry(entry).length - 1);
  }

  function lzbenchSliderMarks(entry) {
    const labels = sliderLabelsForEntry(entry);
    const o = {};
    labels.forEach((label, i) => {
      o[i] = displayLevelLabel(entry, label);
    });
    return o;
  }

  function lzbenchSliderHint(entry) {
    return displayLevelLabel(entry, lzbench_levels.value[entry.codecId]);
  }

  function lzbenchSliderTooltip(entry, val) {
    const labels = sliderLabelsForEntry(entry);
    const label = labels[Math.max(0, Math.min(val, labels.length - 1))] ?? "";
    return displayLevelLabel(entry, label);
  }

  return {
    includeDatasetDimension,
    implement_language,
    ranking_criteria,
    compression_rate_mode,
    compression_time_mode,
    decompression_time_mode,
    compression_scope,
    data_type,
    float_precision,
    xz_level,
    zstd_level,
    chimp_history,
    chimpHistoryValues: CHIMP_HISTORY_VALUES,
    simpiece_param,
    sz_absrel,
    sz_eps,
    sz3_absrel,
    sz3_eps,
    machete_eps,
    spdp_level,
    zfp_tol_eps,
    serf_param,
    lzbench_levels,
    lzbench_slider_idx,
    lzbenchLevelControls,
    lzbenchSliderMax,
    lzbenchSliderMarks,
    lzbenchSliderHint,
    lzbenchSliderTooltip,
    apiItems,
    data_dimension,
    dataset,
    datasets,
    datasetsLoading,
    selected_category,
    categoryOptions,
    selected_column,
    columnOptions,
    columnOptionsLoading,
    showResearchSliders,
    showSpdpLevelSlider,
    showZfpTolSlider,
    canSelectIntDataType,
    canSelectFloatDataType,
    canSelectStringDataType,
    canSelectOverallDataType,
    isStringDataType,
    isFloatDataType,
    showFloatPrecisionSelector,
    dataTypeSelectionHint,
    simpieceParamMax,
    serfParamMax,
    szEpsMax,
    sz3EpsMax,
    macheteEpsMax,
    zfpTolEpsMax,
    spdpLevelMax,
    simpieceSliderMarks,
    szSliderMarks,
    sz3SliderMarks,
    macheteSliderMarks,
    zfpTolSliderMarks,
    spdpSliderMarks,
    serfSliderMarks,
    simpieceCurrentLabel,
    serfCurrentLabel,
    szEpsCurrentLabel,
    sz3EpsCurrentLabel,
    macheteEpsCurrentLabel,
    zfpTolEpsCurrentLabel,
    spdpLevelCurrentLabel,
    simpieceTooltip,
    szEpsTooltip,
    sz3EpsTooltip,
    macheteEpsTooltip,
    zfpTolEpsTooltip,
    spdpLevelTooltip,
    serfTooltip,
    availableAlgorithmsLossless,
    availableAlgorithmsLossy,
    leaderboardItems,
    reportFilterParams,
    uploadFilterFormFields,
    fetchHomeDatasetOptions,
    fetchDatasetColumnOptions,
    buildStaticDataParams,
    displayAlgorithmName,
    compressionModeLabel,
    compressionModeTagType,
    formatFileSize: (bytes) => {
      if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(2) + " MB";
      return (bytes / 1024).toFixed(2) + " KB";
    },
  };
}
