<template>
  <AppLayout title="Time Series Compression Benchmark for Apache TsFile">
    <!-- Introduction Section -->
    <div class="introduction-section">
      <div class="introduction-content">
        <div class="text-content">
          <h2>File Format for Internet of Things</h2>
          <p>
            TsFile is a columnar storage file format designed for time series data, which supports efficient
            compression, high throughput of read and write, and compatibility with various frameworks, such as Spark and
            Flink. It is easy to integrate TsFile into IoT big data processing frameworks.
          </p>

          <h3>Efficient Storage and Compression</h3>
          <p>
            TsFile employs advanced compression techniques to minimize storage requirements, resulting in reduced disk
            space consumption and improved system efficiency.
          </p>
        </div>

        <div class="image-content">
          <img src="@/assets/tsfile-logo.svg" alt="TsFile Logo" class="header-image" />
        </div>
      </div>
    </div>

    <el-row class="option-row option-single">
      <el-col :span="8">
        <span class="option-label">Curated dataset</span>
      </el-col>
      <el-col :span="12">
        <el-select
          v-model="dataset"
          class="home-dataset-select"
          placeholder="Select a dataset"
          filterable
          size="large"
          :loading="isStaticLoading"
          :disabled="datasetOptions.length === 0"
          popper-class="dataset-bundle-select-dropdown"
          @change="scheduleLoadStaticData"
        >
          <el-option
            v-for="item in datasetOptions"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
        <div v-if="dataset" class="tsfile-bundle-link-wrap">
          <button
            type="button"
            class="tsfile-bundle-link"
            :disabled="bundleDownloading"
            @click="downloadDatasetTsfileBundle"
          >
            {{ bundleDownloading ? 'Downloading…' : 'Download TsFile bundle' }}
          </button>
        </div>
      </el-col>
    </el-row>

    <el-row class="option-row">
      <el-col :span="8">
        <span class="option-label">Data Type</span>
      </el-col>
      <el-col :span="16">
        <el-radio-group v-model="data_type" @change="scheduleLoadStaticData">
          <el-radio value="1" size="large" :disabled="!canSelectIntDataType">Int</el-radio>
          <el-radio value="2" size="large" :disabled="!canSelectFloatDataType">Float</el-radio>
          <!-- Byte/string inputs are hidden while the benchmark focuses on int and float data. -->
          <!--
          <el-radio value="3" size="large" :disabled="!canSelectStringDataType">Byte</el-radio>
          -->
        </el-radio-group>
        <p v-if="dataTypeSelectionHint" class="data-type-hint">
          {{ dataTypeSelectionHint }}
        </p>
      </el-col>
    </el-row>

    <el-row class="option-row">
      <el-col :span="8">
        <span class="option-label">Available Algorithms</span>
      </el-col>
      <el-col :span="16">
        <div class="algorithms-display algorithms-by-mode">
          <div class="algo-line">
            <span class="algo-mode-label success">Lossless</span>
            <span>{{ availableAlgorithmsLossless }}</span>
          </div>
          <div class="algo-line">
            <span class="algo-mode-label warning">Lossy</span>
            <span>{{ availableAlgorithmsLossy }}</span>
          </div>
        </div>
      </el-col>
    </el-row>

    <el-row class="option-row option-single">
      <el-col :span="8">
        <span class="option-label">Compression mode</span>
      </el-col>
      <el-col :span="16">
        <el-radio-group v-model="compression_scope">
          <el-radio value="all" size="large">All</el-radio>
          <el-radio value="lossless" size="large">Lossless only</el-radio>
          <el-radio value="lossy" size="large">Lossy only</el-radio>
        </el-radio-group>
      </el-col>
    </el-row>

    <el-row class="option-row option-single">
      <el-col :span="8">
        <span class="option-label">Ranking Criteria</span>
      </el-col>
      <el-col :span="16">
        <el-radio-group v-model="ranking_criteria">
          <el-radio value="1" size="large">Compression Rate</el-radio>
          <el-radio value="2" size="large">Compression Time</el-radio>
          <el-radio value="3" size="large">Decompression Time</el-radio>
        </el-radio-group>
      </el-col>
    </el-row>

    <el-table :data="items" :height="items.length > 0 ? 400 : undefined" :empty-text="tableEmptyText">
      <el-table-column label="Rank" width="100" type="index"> </el-table-column>
      <el-table-column
        label="Compression Mode"
        width="150"
        column-key="compression_type"
        :show-overflow-tooltip="false"
      >
        <template #default="scope">
          <CompressionTypeTag :algorithm="scope.row.algorithm" />
        </template>
      </el-table-column>
      <el-table-column prop="algorithm" label="Algorithm" min-width="140">
        <template #default="scope">
          {{ displayAlgorithmName(scope.row.algorithm) }}
        </template>
      </el-table-column>
      <el-table-column prop="compression_rate" label="Compression Rate">
        <template #default="scope">
          {{ parseFloat(scope.row.compression_rate).toFixed(3) }}
        </template>
      </el-table-column>
      <el-table-column prop="compression_time" label="Write time (TsFile) (s)">
        <template #default="scope">
          {{ parseFloat(scope.row.compression_time).toFixed(3) }}
        </template>
      </el-table-column>
      <el-table-column prop="decompression_time" label="Read time (TsFile) (s)">
        <template #default="scope">
          {{ parseFloat(scope.row.decompression_time).toFixed(3) }}
        </template>
      </el-table-column>
      <el-table-column label="Space Cost">
        <template #default="scope">
          {{ formatFileSize(scope.row.space_cost) }}
        </template>
      </el-table-column>
      <el-table-column
        v-if="SHOW_TSFILE_RESULTS_ACTIONS_COLUMN"
        label="Actions"
        width="140"
      >
        <template #default="scope">
          <el-button
            type="primary"
            size="small"
            @click="downloadFile(scope.row)"
            :loading="scope.row.downloading"
            :disabled="!dataset"
          >
            Download
          </el-button>
        </template>
      </el-table-column>
    </el-table>
  </AppLayout>
</template>

<script setup>
import { ref, computed, onMounted, watch, nextTick } from "vue";
import { ElMessage } from "element-plus";
import { useAuth } from "@/composables/useAuth.js";
import { datasetHasBenchmarkDtype } from "@/composables/useBenchmarkRunConfig.js";
import _ from "lodash";
import axios from "@/utils/httpClient.js";
import { CATALOG_API_TIMEOUT_MS } from "@/constants/apiTimeouts.js";
import AppLayout from "@/components/AppLayout.vue";
import CompressionTypeTag from "@/components/CompressionTypeTag.vue";
import {
  dedupeCommaSeparatedCatalogDisplayLine,
  displayAlgorithmName,
  formatAvailableAlgorithmsCatalogLine,
  joinUniqueCatalogLabels,
} from "@/utils/algorithmDisplayName.js";
import { isLossyAlgorithm } from "@/utils/algorithmCompressionMode.js";
import {
  literatureLosslessSuffix,
  literatureLossySuffix,
  mergeAlgorithmLine,
} from "@/utils/literatureBenchmarkAlgorithms.js";
import { datasetSelectLabelFromKey } from "@/utils/datasetSelectLabel.js";

/** Set false to hide the Download button column in the results table. */
const SHOW_TSFILE_RESULTS_ACTIONS_COLUMN = false;

/** Same keys as Home when manifest is unavailable. */
const FALLBACK_STATIC_DATASETS = [
  "AI4I-Maintenance",
  "Air-Quality",
  "Metro-Traffic",
];

const dataset = ref("AI4I-Maintenance");
const datasetOptions = ref([]);
const datasetCatalogByKey = ref({});

const data_type = ref("2");
const ranking_criteria = ref("1");
const compression_scope = ref("all");
const apiItems = ref([]);
const isStaticLoading = ref(false);
const staticDataError = ref("");
const { isAuthenticated } = useAuth();

const bundleDownloading = ref(false);

const dataTypeMap = {
  1: "int",
  2: "float",
};

const mapDataType = (v) => dataTypeMap[v] ?? dataTypeMap[Number(v)];

async function loadDatasetCatalogByKey() {
  const params = {};
  if (isAuthenticated.value) {
    params.include_user_uploads = true;
  }
  try {
    const { data } = await axios.get("/api/datasets/", {
      params,
      timeout: CATALOG_API_TIMEOUT_MS,
    });
    datasetCatalogByKey.value = Object.fromEntries(
      (data.items || []).map((row) => [row.key, row]),
    );
  } catch {
    datasetCatalogByKey.value = {};
  }
}

async function loadDatasetOptions() {
  const params = {};
  if (isAuthenticated.value) {
    params.include_user_uploads = true;
  }
  const displayNameByKey = new Map();
  const visibilityByKey = new Map();
  await loadDatasetCatalogByKey();
  try {
    const { data } = await axios.get("/api/staticdata/manifest/", { params });
    const keys = Array.isArray(data?.tsfile_datasets) ? data.tsfile_datasets : [];
    if (isAuthenticated.value) {
      try {
        const { data: uploads } = await axios.get("/api/my-uploads/");
        for (const row of uploads?.items || []) {
          if (row.processing_status === "completed" && row.id != null) {
            const key = `user-upload-${row.id}`;
            displayNameByKey.set(
              key,
              row.display_name || "",
            );
            visibilityByKey.set(key, row.is_public === true);
          }
        }
      } catch {
        /* ignore */
      }
    }
    datasetOptions.value = keys.length
      ? keys.map((k) => ({
          value: k,
          label: datasetSelectLabelFromKey(
            k,
            displayNameByKey.get(k),
            visibilityByKey.get(k) ?? null,
          ),
        }))
      : FALLBACK_STATIC_DATASETS.map((k) => ({
          value: k,
          label: datasetSelectLabelFromKey(k),
        }));
    if (!datasetOptions.value.some((d) => d.value === dataset.value)) {
      dataset.value =
        datasetOptions.value[0]?.value ?? FALLBACK_STATIC_DATASETS[0];
    }
  } catch {
    datasetOptions.value = FALLBACK_STATIC_DATASETS.map((k) => ({
      value: k,
      label: datasetSelectLabelFromKey(k),
    }));
  }
}

async function loadStaticData() {
  const dtype = mapDataType(data_type.value);
  if (!dtype) {
    return;
  }
  isStaticLoading.value = true;
  staticDataError.value = "";
  try {
    const params = {
      dataset: dataset.value,
      data_type: dtype,
    };
    const { data } = await axios.get("/api/tsfile-staticdata/", { params });
    const rows = data?.items;
    apiItems.value = Array.isArray(rows)
      ? rows.map((r) => ({ ...r, downloading: false }))
      : [];
  } catch (e) {
    const detail = e?.response?.data?.error;
    staticDataError.value =
      typeof detail === "string"
        ? detail
        : "Failed to load TsFile benchmark static data.";
    apiItems.value = [];
    ElMessage.error(staticDataError.value);
  } finally {
    isStaticLoading.value = false;
  }
}

const activeDatasetRow = computed(
  () => datasetCatalogByKey.value[dataset.value] ?? null,
);

const canSelectIntDataType = computed(() =>
  datasetHasBenchmarkDtype(activeDatasetRow.value, "int"),
);

const canSelectFloatDataType = computed(() =>
  datasetHasBenchmarkDtype(activeDatasetRow.value, "float"),
);

const canSelectStringDataType = computed(() => false);

const dataTypeSelectionHint = computed(() => {
  if (!activeDatasetRow.value) {
    return "";
  }
  if (!canSelectIntDataType.value) {
    return "This dataset has no integer columns.";
  }
  if (!canSelectFloatDataType.value) {
    return "This dataset has no floating-point columns.";
  }
  return "";
});

function pickFirstAvailableTsfileDataType() {
  if (canSelectIntDataType.value) {
    data_type.value = "1";
  } else if (canSelectFloatDataType.value) {
    data_type.value = "2";
  }
}

const scheduleLoadStaticData = _.debounce(() => {
  nextTick(() => {
    loadStaticData();
  });
}, 350);

/** TsFile page pipeline (chunk codecs + TSEncodings + TS_2DIFF+BOS); matches Java processCsvSimple(tsfile). */
const algorithmsByType = {
  int: [
    "gzip",
    "lz4",
    "XZ",
    "snappy",
    "zstd",
    "CHIMP",
    "GORILLA",
    "RLE",
    "RLBE",
    "SPRINTZ",
    "TS_2DIFF",
    "ts_2diff_bos",
  ],
  float: [
    "gzip",
    "lz4",
    "XZ",
    "snappy",
    "zstd",
    "CHIMP",
    "GORILLA",
    "RLE",
    "RLBE",
    "SPRINTZ",
    "TS_2DIFF",
    "ts_2diff_bos",
  ],
};

const availableAlgorithmsLossless = computed(() => {
  const type = mapDataType(data_type.value);
  const algorithms = algorithmsByType[type] || [];
  const losslessKeys = algorithms.filter((a) => !isLossyAlgorithm(a));
  const base = joinUniqueCatalogLabels(losslessKeys);
  return dedupeCommaSeparatedCatalogDisplayLine(
    formatAvailableAlgorithmsCatalogLine(
      mergeAlgorithmLine(base, literatureLosslessSuffix("java", type)),
    ),
  );
});

const availableAlgorithmsLossy = computed(() => {
  const type = mapDataType(data_type.value);
  const algorithms = algorithmsByType[type] || [];
  const lossy = algorithms.filter((a) => isLossyAlgorithm(a));
  const base = lossy.length
    ? joinUniqueCatalogLabels(lossy)
    : "(No dedicated lossy codecs on this TsFile page)";
  return formatAvailableAlgorithmsCatalogLine(
    mergeAlgorithmLine(base, literatureLossySuffix("java", type)),
  );
});

const sortedBenchmarkItems = computed(() => {
  const arr = [...apiItems.value];
  if (ranking_criteria.value === "1") {
    arr.sort(
      (a, b) =>
        parseFloat(a.compression_rate) - parseFloat(b.compression_rate)
    );
  } else if (ranking_criteria.value === "2") {
    arr.sort(
      (a, b) =>
        parseFloat(a.compression_time) - parseFloat(b.compression_time)
    );
  } else if (ranking_criteria.value === "3") {
    arr.sort(
      (a, b) =>
        parseFloat(a.decompression_time) - parseFloat(b.decompression_time)
    );
  }
  return arr;
});

const items = computed(() => {
  const scope = compression_scope.value;
  if (scope === "all") {
    return sortedBenchmarkItems.value;
  }
  if (scope === "lossy") {
    return sortedBenchmarkItems.value.filter((row) =>
      isLossyAlgorithm(row.algorithm)
    );
  }
  return sortedBenchmarkItems.value.filter(
    (row) => !isLossyAlgorithm(row.algorithm)
  );
});

const tableEmptyText = computed(() => {
  if (isStaticLoading.value) {
    return "Loading TsFile benchmark results...";
  }
  if (staticDataError.value) {
    return staticDataError.value;
  }
  if (apiItems.value.length > 0 && items.value.length === 0) {
    return "No rows match the current Compression mode filter. Switch to All or Lossless only.";
  }
  return "No Data";
});

const formatFileSize = (bytes) => {
  if (bytes >= 1024 * 1024) {
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  } else {
    return (bytes / 1024).toFixed(2) + ' KB';
  }
};

async function downloadDatasetTsfileBundle() {
  if (!dataset.value) {
    return;
  }
  bundleDownloading.value = true;
  try {
    const response = await axios.get(
      `/api/datasets/download/${dataset.value}/tsfile/`,
      { responseType: "blob" },
    );
    let filename = `${dataset.value}_tsfile_bundle.zip`;
    const cd = response.headers["content-disposition"];
    if (cd) {
      const m = cd.match(/filename="([^"]+)"/);
      if (m?.[1]) {
        filename = m[1];
      }
    }
    const contentType = response.headers["content-type"] || "application/zip";
    const url = window.URL.createObjectURL(new Blob([response.data], { type: contentType }));
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
    ElMessage.success("Download complete.");
  } catch (error) {
    let detail = "Download failed.";
    const data = error?.response?.data;
    if (data instanceof Blob) {
      try {
        const text = await data.text();
        const parsed = JSON.parse(text);
        if (parsed?.error) {
          detail = String(parsed.error);
        }
      } catch {
        /* ignore */
      }
    } else if (error?.response?.data?.error) {
      detail = String(error.response.data.error);
    }
    ElMessage.error(detail);
  } finally {
    bundleDownloading.value = false;
  }
}

const downloadFile = async (fileInfo) => {
  if (!dataset.value || !fileInfo?.algorithm) {
    ElMessage.warning("Select a dataset and algorithm before downloading.");
    return;
  }
  const dt = mapDataType(data_type.value);
  if (!dt) {
    ElMessage.warning("Select Int or Float before downloading.");
    return;
  }

  fileInfo.downloading = true;
  try {
    const response = await axios.get("/api/tsfile-staticdata/download/", {
      params: {
        dataset: dataset.value,
        data_type: dt,
        algorithm: String(fileInfo.algorithm),
      },
      responseType: "blob",
    });

    let filename = "downloaded.tsfile";
    const contentDisposition = response.headers["content-disposition"];
    if (contentDisposition) {
      const matches = contentDisposition.match(/filename="([^"]+)"/);
      if (matches && matches.length > 1) {
        filename = matches[1];
      }
    }

    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);

    ElMessage.success("File downloaded successfully!");
  } catch (error) {
    console.error(error);
    ElMessage.error(
      error?.message
        ? `Download failed: ${error.message}`
        : "Error downloading file.",
    );
  } finally {
    fileInfo.downloading = false;
  }
};

watch(isAuthenticated, async () => {
  await loadDatasetOptions();
  await loadStaticData();
});

watch(
  [dataset, canSelectIntDataType, canSelectFloatDataType],
  () => {
    const current = String(data_type.value);
    const invalid =
      (current === "1" && !canSelectIntDataType.value) ||
      (current === "2" && !canSelectFloatDataType.value) ||
      current === "3";
    if (invalid) {
      pickFirstAvailableTsfileDataType();
      scheduleLoadStaticData();
    }
  },
);

onMounted(async () => {
  await loadDatasetOptions();
  await loadStaticData();
});

</script>

<style scoped>
/* styles/theme.css */

.tsfile-bundle-link-wrap {
  margin-top: 0.5rem;
}

.tsfile-bundle-link {
  font-size: 0.875rem;
  color: var(--bc-accent, #4da3ff);
  background: none;
  border: none;
  padding: 0;
  cursor: pointer;
  text-align: left;
  text-decoration: underline;
  text-underline-offset: 2px;
}

.tsfile-bundle-link:hover:not(:disabled) {
  color: #7eb8ff;
}

.tsfile-bundle-link:disabled {
  opacity: 0.6;
  cursor: wait;
}

.introduction-section {
  margin-bottom: 40px;
  line-height: 1.8;
}

.introduction-content {
  display: flex;
  gap: 20px;
  align-items: center;
}

.text-content {
  flex: 1;
}

.image-content {
  flex-shrink: 0;
  width: 400px;
  display: flex;
  justify-content: center;
  align-items: center;
}

.header-image {
  width: 90%;
  height: auto;
}


.introduction-section h2 {
  color: var(--bc-accent, #409eff);
  font-size: 28px;
  font-weight: 600;
  margin-bottom: 20px;
}

.introduction-section h3 {
  color: var(--bc-accent, #409eff);
  font-size: 24px;
  font-weight: 600;
  margin: 30px 0 15px 0;
}

.introduction-section p {
  margin-bottom: 20px;
  color: var(--bc-text, #e0e0e0);
  font-size: 18px;
}

.algorithms-by-mode .algo-line {
  margin-bottom: 8px;
  line-height: 1.5;
  color: var(--bc-text, #e0e0e0);
}

.algorithms-by-mode .algo-mode-label {
  display: inline-block;
  min-width: 3em;
  margin-right: 8px;
  font-weight: 600;
  font-size: 14px;
}

.algorithms-by-mode .algo-mode-label.success {
  color: #67c23a;
}

.algorithms-by-mode .algo-mode-label.warning {
  color: #e6a23c;
}

.ranking-criteria-footnote {
  margin: 10px 0 0 0;
  font-size: 14px;
  line-height: 1.5;
  color: var(--bc-text-muted, #b0b0b0);
  max-width: 52em;
}

.data-type-hint {
  margin: 8px 0 0;
  font-size: 14px;
  color: var(--bc-text-subtle, #a8abb2);
  line-height: 1.45;
  max-width: 36rem;
}

/* Responsive layout */
@media (max-width: 768px) {
  .introduction-content {
    flex-direction: column;
    gap: 15px;
  }

  .image-content {
    width: 100%;
    justify-content: center;
  }

  .header-image {
    max-width: 400px;
  }
}
</style>
