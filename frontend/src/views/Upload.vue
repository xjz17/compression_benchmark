<template>
  <AppLayout
    title="Upload"
    subtitle="Upload datasets and custom compression algorithms. Results appear on Home after background processing."
  >
    <el-tabs v-model="activeTab" class="upload-tabs">
      <el-tab-pane label="Datasets" name="datasets">
        <section class="my-uploads-section">
          <h2 class="section-heading">My datasets</h2>
          <p class="section-desc">
            Manage datasets you uploaded. View rankings on
            <router-link to="/">Home</router-link> when processing completes.
          </p>
          <div class="my-upload-table-wrap">
            <el-table
              v-loading="myUploadsLoading"
              :data="myUploads"
              class="my-datasets-table"
              empty-text="No saved datasets yet. Upload a file below."
              style="width: 100%"
              :max-height="MY_UPLOAD_TABLE_MAX_HEIGHT_PX"
            >
            <el-table-column prop="display_name" label="Name" min-width="200">
              <template #default="scope">
                {{ scope.row.display_name }}
                <el-tag
                  v-if="
                    !scope.row.is_catalog &&
                    scope.row.processing_status &&
                    scope.row.processing_status !== 'completed'
                  "
                  size="small"
                  :type="processingTagType(scope.row.processing_status)"
                  class="status-tag"
                >
                  {{ processingLabel(scope.row.processing_status) }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="major_category" label="Category" min-width="160" />
            <el-table-column prop="file_format" label="Format" width="120" />
            <el-table-column label="Visibility" width="120">
              <template #default="scope">
                <span v-if="scope.row.is_catalog" class="catalog-fixed-field">Public</span>
                <el-switch
                  v-else
                  class="upload-visibility-switch"
                  :model-value="scope.row.is_public"
                  active-text="Public"
                  inactive-text="Private"
                  inline-prompt
                  :loading="visibilitySavingId === scope.row.id"
                  :disabled="scope.row.processing_status === 'running'"
                  @change="(val) => updateUploadVisibility(scope.row, val)"
                />
              </template>
            </el-table-column>
            <el-table-column label="Uploaded" width="170">
              <template #default="scope">
                <span v-if="!scope.row.is_catalog">{{
                  formatUploadedAt(scope.row.created_at)
                }}</span>
              </template>
            </el-table-column>
            <el-table-column label="Actions" width="120" fixed="right">
              <template #default="scope">
                <el-button
                  v-if="!scope.row.is_catalog"
                  type="danger"
                  size="small"
                  :loading="deleteSavingId === scope.row.id"
                  @click="confirmDeleteUpload(scope.row)"
                >
                  Delete
                </el-button>
              </template>
            </el-table-column>
            </el-table>
          </div>
        </section>

        <h2 class="section-heading upload-heading">Upload a new dataset</h2>

        <el-row class="option-row">
          <el-col :span="8">
            <span class="option-label">Category</span>
          </el-col>
          <el-col :span="16">
            <el-select
              v-model="uploadMajorCategory"
              class="home-dataset-select"
              size="large"
              popper-class="dataset-bundle-select-dropdown"
              :disabled="!isAuthenticated"
            >
              <el-option
                v-for="cat in DATASET_MAJOR_CATEGORIES"
                :key="cat"
                :label="cat"
                :value="cat"
              />
            </el-select>
          </el-col>
        </el-row>

        <el-row class="option-row option-single">
          <el-col :span="8">
            <span class="option-label">Dataset visibility</span>
          </el-col>
          <el-col :span="16">
            <div class="upload-visibility-row">
              <el-switch
                class="upload-visibility-switch"
                v-model="uploadIsPublic"
                active-text="Public"
                inactive-text="Private"
                inline-prompt
                :disabled="!isAuthenticated"
              />
            </div>
          </el-col>
        </el-row>

        <el-row class="option-row">
          <el-col :span="8">
            <span class="option-label">File</span>
          </el-col>
          <el-col :span="16">
            <el-upload
              ref="datasetUpload"
              accept=".csv,.arrow"
              action="/api/upload/"
              :data="uploadFormData"
              :headers="uploadHeaders"
              :limit="1"
              :disabled="!isAuthenticated"
              :before-upload="beforeDatasetUpload"
              :on-exceed="handleDatasetExceed"
              :on-success="handleDatasetSuccess"
              :on-error="handleUploadError"
              :on-remove="handleRemove"
            >
              <el-button type="primary" :disabled="!isAuthenticated">Select dataset file</el-button>
              <template #tip>
                <div class="el-upload__tip">CSV or Arrow, max 1GB.</div>
              </template>
            </el-upload>
          </el-col>
        </el-row>
      </el-tab-pane>

      <el-tab-pane label="Algorithms" name="algorithms">
        <section class="my-uploads-section">
          <h2 class="section-heading">My algorithms</h2>
          <p class="section-desc">
            Custom codecs are benchmarked on your completed datasets. Rankings appear on
            <router-link to="/">Home</router-link> alongside built-in algorithms.
          </p>
          <div class="my-upload-table-wrap">
            <el-table
              v-loading="myAlgorithmsLoading"
              :data="myAlgorithms"
              class="my-algorithms-table"
              empty-text="No saved algorithms yet. Upload one below."
              style="width: 100%"
              :max-height="MY_UPLOAD_TABLE_MAX_HEIGHT_PX"
            >
            <el-table-column prop="display_name" label="Name" min-width="160" />
            <el-table-column prop="implement_language" label="Language" width="100" />
            <el-table-column label="Data type" width="120">
              <template #default="scope">
                {{ algorithmTargetLabel(scope.row.target_column_type) }}
              </template>
            </el-table-column>
            <el-table-column label="Mode" width="110">
              <template #default="scope">
                <el-tag
                  size="small"
                  :type="scope.row.lossy_lossless === 'lossy' ? 'warning' : 'success'"
                >
                  {{ scope.row.lossy_lossless === 'lossy' ? 'Lossy' : 'Lossless' }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="entry_script" label="Entry" min-width="140" show-overflow-tooltip />
            <el-table-column label="Status" width="120">
              <template #default="scope">
                <el-tag
                  v-if="scope.row.processing_status !== 'completed'"
                  size="small"
                  :type="processingTagType(scope.row.processing_status)"
                >
                  {{ processingLabel(scope.row.processing_status) }}
                </el-tag>
                <span v-else class="status-ready">Ready</span>
              </template>
            </el-table-column>
            <el-table-column label="Uploaded" width="170">
              <template #default="scope">
                {{ formatUploadedAt(scope.row.created_at) }}
              </template>
            </el-table-column>
            <el-table-column label="Actions" width="120" fixed="right">
              <template #default="scope">
                <el-button
                  type="danger"
                  size="small"
                  :loading="algoDeleteSavingId === scope.row.id"
                  @click="confirmDeleteAlgorithm(scope.row)"
                >
                  Delete
                </el-button>
              </template>
            </el-table-column>
            </el-table>
          </div>
        </section>

        <h2 class="section-heading upload-heading">Upload a new algorithm</h2>

        <el-row class="option-row">
          <el-col :span="8">
            <span class="option-label">Algorithm name</span>
          </el-col>
          <el-col :span="16">
            <el-input
              v-model="algoDisplayName"
              placeholder="e.g. XorDelta"
              maxlength="255"
              :disabled="!isAuthenticated"
            />
          </el-col>
        </el-row>

        <el-row class="option-row">
          <el-col :span="8">
            <span class="option-label">Language</span>
          </el-col>
          <el-col :span="16">
            <el-radio-group v-model="algoLanguage" :disabled="!isAuthenticated">
              <el-radio value="python" size="large">Python</el-radio>
              <el-radio value="java" size="large">Java</el-radio>
              <el-radio value="cpp" size="large">C++</el-radio>
            </el-radio-group>
          </el-col>
        </el-row>

        <el-row class="option-row">
          <el-col :span="8">
            <span class="option-label">Compression mode</span>
          </el-col>
          <el-col :span="16">
            <el-radio-group v-model="algoLossyMode" :disabled="!isAuthenticated">
              <el-radio value="lossless" size="large">Lossless</el-radio>
              <el-radio value="lossy" size="large">Lossy</el-radio>
            </el-radio-group>
          </el-col>
        </el-row>

        <el-row class="option-row">
          <el-col :span="8">
            <span class="option-label">Data type</span>
          </el-col>
          <el-col :span="16">
            <el-radio-group v-model="algoTargetColumnType" :disabled="!isAuthenticated">
              <el-radio value="int" size="large">Integer</el-radio>
              <el-radio value="float" size="large">Float</el-radio>
            </el-radio-group>
          </el-col>
        </el-row>

        <el-row class="option-row">
          <el-col :span="8">
            <span class="option-label">Algorithm source</span>
          </el-col>
          <el-col :span="16">
            <el-upload
              ref="algorithmUpload"
              accept=".zip,.py,.java,.cpp"
              action="/api/my-algorithms/"
              :data="algorithmFormData"
              :headers="uploadHeaders"
              :limit="1"
              :disabled="!isAuthenticated"
              :before-upload="beforeAlgorithmUpload"
              :on-exceed="handleAlgorithmExceed"
              :on-success="handleAlgorithmSuccess"
              :on-error="handleAlgorithmError"
              :on-remove="handleRemove"
            >
              <el-button type="primary" :disabled="!isAuthenticated">
                Select algorithm source
              </el-button>
              <template #tip>
                <div class="el-upload__tip">
                  ZIP or single source file; entry must match the language.
                </div>
              </template>
            </el-upload>
          </el-col>
        </el-row>

        <aside class="algorithm-requirements" aria-label="Algorithm code requirements">
          <h3 class="algorithm-requirements-title">Code requirements</h3>
          <ul class="algorithm-requirements-list">
            <li>
              <strong>Package:</strong> a <code>.zip</code> archive, or a single
              <code>.py</code> / <code>.java</code> / <code>.cpp</code> file.
            </li>
            <li>
              <strong>Entry file</strong>:
              Python <code>main.py</code>; Java <code>Main.java</code> with
              <code>public class Main</code>; C++ <code>main.cpp</code> (compiled here with
              <code>c++ -O2 -std=c++17</code>).
            </li>
            <li>
              <strong>Command-line interface</strong> — your program must accept the same flags
              in every language:
              <code>--mode</code> (<code>compress</code> or <code>decompress</code>),
              <code>--input</code> &lt;path&gt;, <code>--output</code> &lt;path&gt;,
              <code>--data-type</code> (<code>int</code> or <code>float</code>).
            </li>
            <li>
              <strong>compress:</strong> read the numeric CSV at <code>--input</code>, write your
              compressed payload to <code>--output</code>.
              <strong>decompress:</strong> read <code>--input</code>, write a CSV to
              <code>--output</code> (optional but used for decompression timing when it
              succeeds).
            </li>
            <li>
              <strong>Input CSV:</strong> one logical type per file (header row, numeric columns
              only). Parse values
              according to <code>--data-type</code>.
            </li>
            <li>
              <strong>Data type</strong>: <em>Integer</em> runs on <code>int</code>
              columns only; <em>Float</em> runs on <code>int</code> and <code>float</code> columns
              when the dataset has them. Each run passes the matching <code>--data-type</code>.
            </li>
          </ul>
        </aside>
      </el-tab-pane>
    </el-tabs>
  </AppLayout>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch } from "vue";
import { genFileId, ElMessage, ElMessageBox } from "element-plus";

import AppLayout from "@/components/AppLayout.vue";
import axios from "@/utils/httpClient.js";
import { CATALOG_API_TIMEOUT_MS } from "@/constants/apiTimeouts.js";
import { getAuthToken, useAuth } from "@/composables/useAuth.js";

const { isAuthenticated } = useAuth();

const activeTab = ref("datasets");
const myUploads = ref([]);
const myUploadsLoading = ref(false);
const myAlgorithms = ref([]);
const myAlgorithmsLoading = ref(false);
const visibilitySavingId = ref(null);
const deleteSavingId = ref(null);
const algoDeleteSavingId = ref(null);
/** Element Plus pins header and scrolls the body when the list is long (see Dataset.vue). */
const MY_UPLOAD_TABLE_MAX_HEIGHT_PX = 400;

const DATASET_MAJOR_CATEGORIES = [
  "Climate & Weather",
  "Cloud & Infrastructure",
  "Energy & Power",
  "Finance & Economics",
  "Geoscience & Space",
  "Transport & Mobility",
  "Others",
];

const uploadIsPublic = ref(false);
const uploadMajorCategory = ref("Others");
const datasetUpload = ref(null);
const algorithmUpload = ref(null);
const algoDisplayName = ref("");
const algoLanguage = ref("python");
const algoLossyMode = ref("lossless");
const algoTargetColumnType = ref("float");
let pollTimer = null;

function catalogMajorCategory(row) {
  const raw = String(row.category || "").trim();
  if (!raw) return "";
  if (raw.includes("/")) return raw.split("/")[0].trim() || raw;
  return raw;
}

function mapCatalogDatasetRow(row) {
  return {
    is_catalog: true,
    display_name: row.key,
    major_category: catalogMajorCategory(row),
    file_format: row.size_bytes > 0 || row.row_count > 0 ? "csv" : "",
    is_public: true,
    created_at: null,
    id: null,
    processing_status: "completed",
  };
}

async function fetchMyUploads() {
  myUploadsLoading.value = true;
  try {
    const catalogReq = axios.get("/api/datasets/", { timeout: CATALOG_API_TIMEOUT_MS });
    const uploadsReq = isAuthenticated.value
      ? axios.get("/api/my-uploads/")
      : Promise.resolve({ data: { items: [] } });
    const [{ data: catalogData }, { data: uploadsData }] = await Promise.all([
      catalogReq,
      uploadsReq,
    ]);
    const catalogRows = (catalogData?.items || [])
      .filter((row) => !row.is_user_upload)
      .map(mapCatalogDatasetRow);
    const uploadRows = (uploadsData?.items || [])
      .map((row) => ({
        ...row,
        is_catalog: false,
      }))
      .sort((a, b) => {
        const ta = a.created_at ? new Date(a.created_at).getTime() : 0;
        const tb = b.created_at ? new Date(b.created_at).getTime() : 0;
        return tb - ta;
      });
    myUploads.value = [...uploadRows, ...catalogRows];
  } catch {
    ElMessage.error("Could not load datasets.");
    myUploads.value = [];
  } finally {
    myUploadsLoading.value = false;
  }
}

async function fetchMyAlgorithms() {
  if (!isAuthenticated.value) {
    myAlgorithms.value = [];
    return;
  }
  myAlgorithmsLoading.value = true;
  try {
    const { data } = await axios.get("/api/my-algorithms/");
    myAlgorithms.value = data.items || [];
  } catch {
    ElMessage.error("Could not load your algorithms.");
  } finally {
    myAlgorithmsLoading.value = false;
  }
}

async function refreshAll() {
  await Promise.all([fetchMyUploads(), fetchMyAlgorithms()]);
  schedulePollIfNeeded();
}

function schedulePollIfNeeded() {
  const pending =
    myUploads.value.some(
      (r) =>
        !r.is_catalog && ["pending", "running"].includes(r.processing_status),
    ) ||
    myAlgorithms.value.some((r) => ["pending", "running"].includes(r.processing_status));
  if (pending && !pollTimer) {
    pollTimer = setInterval(refreshAll, 8000);
  } else if (!pending && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function updateUploadVisibility(row, isPublic) {
  if (row.is_catalog) return;
  visibilitySavingId.value = row.id;
  try {
    const { data } = await axios.patch(`/api/my-uploads/${row.id}/`, {
      is_public: isPublic,
    });
    const idx = myUploads.value.findIndex((r) => r.id === row.id);
    if (idx >= 0 && data.item) {
      myUploads.value[idx] = data.item;
    } else {
      row.is_public = isPublic;
    }
    ElMessage.success(isPublic ? "Dataset is now public." : "Dataset is now private.");
  } catch {
    ElMessage.error("Could not update visibility.");
  } finally {
    visibilitySavingId.value = null;
  }
}

async function confirmDeleteUpload(row) {
  if (row.is_catalog) return;
  try {
    await ElMessageBox.confirm(
      `Delete “${row.display_name}”? This removes saved files and cannot be undone.`,
      "Delete dataset",
      { type: "warning", confirmButtonText: "Delete", cancelButtonText: "Cancel" }
    );
  } catch {
    return;
  }
  deleteSavingId.value = row.id;
  try {
    await axios.delete(`/api/my-uploads/${row.id}/`);
    myUploads.value = myUploads.value.filter((r) => r.id !== row.id);
    ElMessage.success("Dataset deleted.");
    schedulePollIfNeeded();
  } catch {
    ElMessage.error("Could not delete dataset.");
  } finally {
    deleteSavingId.value = null;
  }
}

async function confirmDeleteAlgorithm(row) {
  try {
    await ElMessageBox.confirm(
      `Delete “${row.display_name}”? This removes the algorithm files and benchmark results.`,
      "Delete algorithm",
      { type: "warning", confirmButtonText: "Delete", cancelButtonText: "Cancel" }
    );
  } catch {
    return;
  }
  algoDeleteSavingId.value = row.id;
  try {
    await axios.delete(`/api/my-algorithms/${row.id}/`);
    myAlgorithms.value = myAlgorithms.value.filter((r) => r.id !== row.id);
    ElMessage.success("Algorithm deleted.");
    schedulePollIfNeeded();
  } catch {
    ElMessage.error("Could not delete algorithm.");
  } finally {
    algoDeleteSavingId.value = null;
  }
}

function formatUploadedAt(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function processingLabel(status) {
  const map = {
    pending: "Queued",
    running: "Processing",
    failed: "Failed",
    completed: "Ready",
  };
  return map[status] || status;
}

function processingTagType(status) {
  if (status === "failed") return "danger";
  if (status === "running") return "warning";
  return "info";
}

const uploadFormData = computed(() => ({
  is_public: uploadIsPublic.value ? "true" : "false",
  major_category: uploadMajorCategory.value,
}));

const algorithmFormData = computed(() => ({
  display_name: algoDisplayName.value.trim(),
  implement_language: algoLanguage.value,
  lossy_lossless: algoLossyMode.value,
  target_column_type: algoTargetColumnType.value,
}));

function algorithmTargetLabel(value) {
  return value === "int" ? "Integer" : "Float";
}

const uploadHeaders = computed(() => {
  const token = getAuthToken();
  if (!token) return {};
  return { Authorization: `Token ${token}` };
});

function requireAuth() {
  if (!isAuthenticated.value) {
    ElMessage.error("Sign in required to upload.");
    return false;
  }
  return true;
}

const beforeDatasetUpload = (file) => {
  if (!requireAuth()) return false;
  if (file.size > 1024 * 1024 * 1024) {
    ElMessage.error("File size cannot exceed 1GB!");
    return false;
  }
  ElMessage.info({ message: "Uploading dataset…", duration: 0, showClose: false });
  return true;
};

const beforeAlgorithmUpload = (file) => {
  if (!requireAuth()) return false;
  if (!algoDisplayName.value.trim()) {
    ElMessage.error("Enter an algorithm name first.");
    return false;
  }
  if (file.size > 50 * 1024 * 1024) {
    ElMessage.error("Algorithm package cannot exceed 50MB.");
    return false;
  }
  ElMessage.info({ message: "Uploading algorithm…", duration: 0, showClose: false });
  return true;
};

const handleDatasetSuccess = async (response) => {
  ElMessage.closeAll();
  ElMessage.success(response?.message || "Dataset upload saved.");
  datasetUpload.value?.clearFiles();
  await fetchMyUploads();
  schedulePollIfNeeded();
};

const handleAlgorithmSuccess = async (response) => {
  ElMessage.closeAll();
  if (response?.error) {
    ElMessage.warning(response.error);
    algorithmUpload.value?.clearFiles();
    return;
  }
  ElMessage.success(response?.message || "Algorithm saved.");
  algorithmUpload.value?.clearFiles();
  await fetchMyAlgorithms();
  schedulePollIfNeeded();
};

const parseUploadErrorBody = (err) => {
  try {
    const body =
      typeof err?.message === "string" && err.message.startsWith("{")
        ? JSON.parse(err.message)
        : err?.response?.data;
    if (body?.error) return body.error;
    if (body?.detail) return body.detail;
  } catch {
    /* ignore */
  }
  return null;
};

const handleUploadError = (err) => {
  ElMessage.closeAll();
  let detail = parseUploadErrorBody(err) || "Upload failed. Please try again.";
  const status = err?.response?.status;
  if (status === 401 || status === 403) {
    detail = "Sign in required. Please sign in and try again.";
  }
  ElMessage.error(detail);
};

const handleAlgorithmError = (err) => {
  handleUploadError(err);
  algorithmUpload.value?.clearFiles();
};

const handleDatasetExceed = (files) => {
  datasetUpload.value.clearFiles();
  const file = files[0];
  file.uid = genFileId();
  datasetUpload.value.handleStart(file);
  datasetUpload.value.submit();
};

const handleAlgorithmExceed = (files) => {
  algorithmUpload.value.clearFiles();
  const file = files[0];
  file.uid = genFileId();
  algorithmUpload.value.handleStart(file);
  algorithmUpload.value.submit();
};

const handleRemove = () => {};

onMounted(refreshAll);
onBeforeUnmount(() => {
  if (pollTimer) clearInterval(pollTimer);
});
watch(isAuthenticated, refreshAll);
</script>

<style scoped>
.upload-tabs {
  margin-top: 0.25rem;
}

.upload-tabs :deep(.el-tabs__header) {
  margin-bottom: 1.25rem;
  border-bottom: 1px solid var(--bc-border, rgba(120, 170, 255, 0.14));
}

.upload-tabs :deep(.el-tabs__nav-wrap::after) {
  height: 1px;
  background-color: transparent;
}

.upload-tabs :deep(.el-tabs__item) {
  color: var(--bc-text, #e8ecf4);
  font-size: 0.95rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  padding: 0 1.15rem 0.85rem;
  opacity: 0.88;
  transition:
    color var(--bc-transition, 0.2s ease),
    opacity var(--bc-transition, 0.2s ease);
}

.upload-tabs :deep(.el-tabs__item:hover) {
  color: var(--bc-cyan, #5ae9ff);
  opacity: 1;
}

.upload-tabs :deep(.el-tabs__item.is-active) {
  color: var(--bc-accent, #4da3ff);
  opacity: 1;
}

.upload-tabs :deep(.el-tabs__active-bar) {
  height: 2px;
  border-radius: 2px 2px 0 0;
  background-color: var(--bc-accent, #4da3ff);
}

.my-uploads-section {
  margin-bottom: 2rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid rgba(120, 170, 255, 0.14);
}

.my-upload-table-wrap :deep(.el-table__body-wrapper) {
  scrollbar-color: var(--bc-scroll-thumb, rgba(120, 170, 255, 0.45))
    var(--bc-scroll-track, rgba(20, 26, 40, 0.5));
}

.my-upload-table-wrap :deep(.el-table__body-wrapper)::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

.my-upload-table-wrap :deep(.el-table__body-wrapper)::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: var(--bc-scroll-thumb, rgba(120, 170, 255, 0.35));
}

.my-upload-table-wrap :deep(.el-table__body-wrapper)::-webkit-scrollbar-track {
  background: var(--bc-scroll-track, rgba(20, 26, 40, 0.45));
  border-radius: 999px;
}

.section-heading {
  margin: 0 0 0.5rem;
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--bc-text, #e8ecf4);
}

.upload-heading {
  margin-bottom: 1.25rem;
}

.algorithm-requirements {
  margin: 0.75rem 0 1.5rem;
  padding: 1rem 1.15rem;
  max-width: 52rem;
  border-radius: 8px;
  border: 1px solid rgba(120, 170, 255, 0.2);
  background: var(--bc-soft-panel-bg, rgba(18, 28, 48, 0.45));
}

.algorithm-requirements-title {
  margin: 0 0 0.65rem;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--bc-text, #d8e0f0);
}

.algorithm-requirements-list {
  margin: 0;
  padding-left: 1.2rem;
  font-size: 0.88rem;
  line-height: 1.55;
  color: var(--bc-text-subtle, #a8abb2);
}

.algorithm-requirements-list li {
  margin-bottom: 0.45rem;
}

.algorithm-requirements-list li:last-child {
  margin-bottom: 0;
}

.algorithm-requirements-list strong {
  color: var(--bc-text, #c8d4e8);
  font-weight: 600;
}

.algorithm-requirements-list code {
  font-size: 0.84em;
  color: var(--bc-code-text, #9ec8ff);
  background: var(--bc-code-bg, rgba(40, 60, 100, 0.35));
  padding: 0.1em 0.35em;
  border-radius: 3px;
}

.section-desc {
  margin: 0 0 1rem;
  font-size: 0.92rem;
  line-height: 1.5;
  color: var(--bc-text-subtle, #a8abb2);
  max-width: 52rem;
}

.section-desc a {
  color: var(--bc-accent, #4da3ff);
}

.status-tag {
  margin-left: 0.5rem;
  vertical-align: middle;
}

.status-ready {
  color: #67c23a;
  font-size: 13px;
}

.catalog-fixed-field {
  color: var(--bc-text-subtle, #a8abb2);
  font-size: 13px;
}

.upload-visibility-row {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.option-row {
  margin-bottom: 1.25rem;
}

.option-label {
  font-weight: 500;
  color: var(--bc-text, #c8d0e0);
}

:deep(.upload-visibility-switch .el-switch__core) {
  min-width: 72px;
}

</style>
