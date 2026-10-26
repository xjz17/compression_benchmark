<template>
  <AppLayout
    title="Time Series Compression Benchmark"
    subtitle="Compare codecs on curated datasets — rate, latency, and storage footprint."
  >
    <div class="benchmark-page">
    <section class="introduction-section intro-card">
      <div class="introduction-content">
        <div class="text-content">
          <p>
            Welcome to the Time Series Compression Benchmark website! Our platform
            is dedicated to providing a comprehensive evaluation of various
            compression algorithms specifically tailored for time series data. We
            aim to offer clear, comparative insights into how different datasets
            perform under various compression techniques.
          </p>

          <p>
            Here, you'll find detailed rankings based on crucial metrics such as:
          </p>

          <ul>
            <li>
              <strong>Compression Rate:</strong> Compressed size divided by original
              size (lower is better).
            </li>
            <li>
              <strong>Compression Time:</strong> The speed at which data is
              compressed.
            </li>
            <li>
              <strong>Decompression Time:</strong> The efficiency of reconstructing
              the original data.
            </li>
          </ul>

          <p>
            Balanced Score = 0.40 * compression rate + 0.30 * normalized
            compression time + 0.30 * normalized decompression time. Lower scores are better.
          </p>

        </div>

        <div class="image-content">
          <img src="@/assets/header-image.png" alt="Time Series Compression Illustration" class="header-image" />
        </div>
      </div>

      <p>
        Beyond just numbers, we delve into the unique characteristics of time
        series data to analyze and explain why certain algorithms excel or
        falter. We believe that understanding the underlying features of your
        data, like its variability, periodicity, or presence of outliers, is key
        to selecting the most effective compression strategy.
      </p>

      <p>
        <strong>Collaborate With Us!</strong>
      </p>
      <p>
        We are always looking to expand our benchmark and welcome contributions from the community. If you've developed
        a novel time series compression algorithm or have a dataset you'd like to see evaluated, we invite you to
        collaborate with us. Join our efforts to advance the field of time series data management and contribute to a
        valuable resource for researchers and practitioners alike.
      </p>

      <p>
        <a
          href="https://github.com/xjz17/web_compression"
          target="_blank"
          rel="noopener noreferrer"
          class="intro-github-link"
        >
          <strong>Get Involved</strong>
        </a>
      </p>
      <p>
        You can find our project and contribute to its development on GitHub.
      </p>
      <p>
        We look forward to your contributions and insights!
      </p>
    </section>

    <section class="surface-panel">
      <header class="panel-head">
        <h2 class="panel-title">Run configuration</h2>
        <p class="panel-desc">Choose dataset, engine, and parameters. The leaderboard updates as you adjust options.</p>
      </header>
      <div class="panel-inner">
        <BenchmarkRunConfigPanel />
      </div>
    </section>

    <section class="surface-panel ranking-panel">
      <header class="panel-head">
        <h2 class="panel-title">Leaderboard</h2>
        <p class="panel-desc">
          Average rate uses the arithmetic mean of per-dataset compression rates.
          Overall rate uses total compressed size divided by total original size.
          Compression time and decompression time are shown in ns/point.
          Sort by compression rate, compression time, decompression time, or the balanced score.
        </p>
      </header>
      <div class="panel-inner leaderboard-table-inner">
        <BenchmarkLeaderboardPanel :table-height="400" :processing="staticDataLoading">
          <template #empty>
            <span class="ranking-table-empty">
              {{ staticDataLoading ? "Loading current results..." : "No data available." }}
            </span>
          </template>
        </BenchmarkLeaderboardPanel>
      </div>
    </section>

    <section class="surface-panel leaderboard-charts-panel">
      <header class="panel-head">
        <h2 class="panel-title">Leaderboard Charts</h2>
        <p class="panel-desc">
          Pareto trade-offs for the selected leaderboard metric.
        </p>
      </header>
      <div class="panel-inner leaderboard-charts-inner">
        <div class="leaderboard-charts-grid">
          <section class="leaderboard-chart-panel leaderboard-pareto-section">
            <h3 class="leaderboard-boxplot-title">{{ leaderboardParetoTitle }}</h3>
            <p class="leaderboard-boxplot-desc">
              Thin paths connect configurations from the same algorithm family across levels;
              dots are individual algorithm settings, and the bold path marks the Pareto frontier.
            </p>
            <div v-if="levelSeriesSummary" class="leaderboard-pareto-note">
              {{ levelSeriesSummary }}
            </div>
            <LeaderboardParetoChart
              v-if="hasParetoChartData"
              :chart="leaderboardParetoChart"
              :aria-label="leaderboardParetoTitle"
            />
            <p v-else class="leaderboard-boxplot-empty">No Pareto chart data available.</p>
          </section>
        </div>
      </div>
    </section>
    </div>
  </AppLayout>
</template>

<script setup>
import { ref, onMounted, watch, computed, nextTick, provide } from "vue";
import axios from "@/utils/httpClient.js";
import _ from "lodash";
import { ElMessage } from "element-plus";
import { useRoute } from "vue-router";
import AppLayout from "@/components/AppLayout.vue";
import BenchmarkRunConfigPanel from "@/components/BenchmarkRunConfigPanel.vue";
import BenchmarkLeaderboardPanel from "@/components/BenchmarkLeaderboardPanel.vue";
import LeaderboardParetoChart from "@/components/LeaderboardParetoChart.vue";
import {
  useBenchmarkRunConfig,
  BENCHMARK_CONFIG_KEY,
  mapImplementLanguage,
  mapDataType,
} from "@/composables/useBenchmarkRunConfig.js";
import {
  filterHomeLeaderboardParetoChart,
  filterHomeLeaderboardRows,
} from "@/utils/homeLeaderboardDisplayFilter.js";

const bench = useBenchmarkRunConfig({ includeDatasetDimension: true });
provide(BENCHMARK_CONFIG_KEY, bench);
const route = useRoute();

const {
  apiItems,
  data_dimension,
  dataset,
  selected_category,
  selected_column,
  categoryOptions,
  data_type,
  float_precision,
  implement_language,
  compression_rate_mode,
  compression_time_mode,
  decompression_time_mode,
  xz_level,
  zstd_level,
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
  fetchHomeDatasetOptions,
  fetchDatasetColumnOptions,
  buildStaticDataParams,
} = bench;

const EMPTY_LEADERBOARD_PARETO_CHART = {
  title: "Pareto Frontier (Compression Rate vs Time)",
  x_label: "Compression rate",
  x_min: 0,
  x_max: 0.9,
  y_label: "Compression time (ns/point)",
  points: [],
  level_series: [],
  frontier_style: "actual_polyline",
  frontier: [],
};

const leaderboardParetoChart = ref({ ...EMPTY_LEADERBOARD_PARETO_CHART });
const staticDataLoading = ref(false);
const suppressStaticDataWatch = ref(true);
let staticDataRequestGeneration = 0;

const CURRENT_CPP_DEFAULT_SNAPSHOT_URL =
  `${import.meta.env.BASE_URL}data/current-cpp-overall-average.json`;

function canUseCurrentCppDefaultSnapshot(params) {
  return (
    params?.data_dimension === "all_datasets" &&
    params?.data_type === "overall" &&
    params?.implement_language === "cpp" &&
    params?.compression_rate_mode === "average" &&
    params?.compression_time_mode === "average" &&
    params?.decompression_time_mode === "average" &&
    !params?.leaderboard_mode
  );
}

async function fetchStaticData(params) {
  if (canUseCurrentCppDefaultSnapshot(params)) {
    try {
      const { data } = await axios.get(CURRENT_CPP_DEFAULT_SNAPSHOT_URL);
      if (
        data?.schema === "benchmark208-current-cpp-default-snapshot-v1" &&
        Array.isArray(data?.items)
      ) {
        return data;
      }
    } catch (_error) {
      // The dynamic endpoint remains the fallback if the frozen asset is absent.
    }
  }
  const { data } = await axios.get("/api/staticdata/", { params });
  return data;
}

const leaderboardParetoTitle = computed(
  () => leaderboardParetoChart.value?.title || EMPTY_LEADERBOARD_PARETO_CHART.title,
);

const hasParetoChartData = computed(() => {
  const points = leaderboardParetoChart.value?.points;
  return Array.isArray(points) && points.length > 0;
});

function loadStaticData() {
  const implLang = mapImplementLanguage(implement_language.value);
  const dtype = mapDataType(data_type.value);
  if (!implLang || !dtype) {
    ElMessage.error("Invalid implement language or data type.");
    return;
  }

  if (data_dimension.value === "single_field" && !selected_category.value) {
    ElMessage.warning("Select a dataset category first.");
    return;
  }
  if (data_dimension.value === "single_column" && !selected_column.value) {
    ElMessage.warning("Select a column first.");
    return;
  }

  const params = buildStaticDataParams();

  if (
    (data_dimension.value === "single_dataset" || data_dimension.value === "single_column") &&
    !params.dataset
  ) {
    ElMessage.warning("Select a dataset first.");
    return;
  }

  const requestGeneration = ++staticDataRequestGeneration;
  staticDataLoading.value = true;
  fetchStaticData(params)
    .then((data) => {
      if (requestGeneration !== staticDataRequestGeneration) return;
      const rows = data?.items;
      apiItems.value = filterHomeLeaderboardRows(rows);
    })
    .catch((error) => {
      if (requestGeneration !== staticDataRequestGeneration) return;
      console.log(error);
      ElMessage.error("Failed to load data.");
      apiItems.value = [];
    })
    .finally(() => {
      if (requestGeneration !== staticDataRequestGeneration) return;
      staticDataLoading.value = false;
    });
}

async function loadLeaderboardBoxplotManifest() {
  try {
    const { data } = await axios.get("/api/benchmark-boxplots-manifest/", {
      params: buildStaticDataParams({ leaderboardMode: "legacy" }),
    });
    const pareto = data?.pareto_chart;
    if (pareto && typeof pareto === "object") {
      leaderboardParetoChart.value = {
        ...EMPTY_LEADERBOARD_PARETO_CHART,
        ...filterHomeLeaderboardParetoChart(pareto),
      };
    }
  } catch (_error) {
    // Keep empty placeholder charts when manifest is unavailable.
  }
}

const scheduleLoadStaticData = _.debounce(loadStaticData, 400);
const scheduleLoadLeaderboardBoxplotManifest = _.debounce(loadLeaderboardBoxplotManifest, 500);

async function applyHomeRoutePreset() {
  const dim = String(route.query?.data_dimension || "").trim();
  const ds = String(route.query?.dataset || "").trim();
  const colIdx = String(route.query?.column_index || "").trim();

  if (
    dim === "single_dataset" ||
    dim === "single_field" ||
    dim === "all_datasets" ||
    dim === "single_column"
  ) {
    data_dimension.value = dim;
  }
  if (ds) {
    const allowedDatasets = new Set(datasets.value.map((row) => String(row.value)));
    if (!allowedDatasets.size || allowedDatasets.has(ds)) {
      dataset.value = ds;
    }
  }
  if (data_dimension.value === "single_column") {
    if (colIdx) {
      selected_column.value = colIdx;
    }
    await fetchDatasetColumnOptions();
  }
}

watch(
  [
    data_dimension,
    dataset,
    selected_category,
    selected_column,
    data_type,
    implement_language,
    compression_rate_mode,
    compression_time_mode,
    decompression_time_mode,
    xz_level,
    zstd_level,
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
    float_precision,
  ],
  () => {
    if (suppressStaticDataWatch.value) return;
    nextTick(() => {
      scheduleLoadStaticData();
      scheduleLoadLeaderboardBoxplotManifest();
    });
  }
);

watch(data_dimension, (dim) => {
  if (dim === "single_field" && categoryOptions.value.length && !selected_category.value) {
    selected_category.value = categoryOptions.value[0].value;
  }
});

watch(
  () => [route.query.data_dimension, route.query.dataset, route.query.column_index],
  async () => {
    if (route.name !== "Home") {
      return;
    }
    await applyHomeRoutePreset();
    await nextTick();
    loadStaticData();
    loadLeaderboardBoxplotManifest();
  },
);

onMounted(async () => {
  ElMessage.closeAll();
  data_dimension.value = "all_datasets";
  dataset.value = "";

  await fetchHomeDatasetOptions();
  await applyHomeRoutePreset();
  suppressStaticDataWatch.value = false;
  loadStaticData();
  loadLeaderboardBoxplotManifest();
});
</script>

<style scoped>
.benchmark-page {
  display: flex;
  flex-direction: column;
  gap: 1.35rem;
}

.intro-card {
  padding: 1.5rem 1.35rem 1.6rem;
  border-radius: var(--bc-radius-md, 12px);
  border: 1px solid var(--bc-border, rgba(120, 170, 255, 0.14));
  background: var(--bc-card-bg);
  box-shadow: var(--bc-shadow);
}

.surface-panel {
  border-radius: var(--bc-radius-lg, 18px);
  border: 1px solid var(--bc-border, rgba(120, 170, 255, 0.14));
  background: var(--bc-panel-bg, rgba(10, 12, 20, 0.45));
  overflow: hidden;
  transition: border-color var(--bc-transition, 0.2s ease), box-shadow var(--bc-transition, 0.2s ease);
}

.surface-panel:hover {
  border-color: rgba(120, 170, 255, 0.22);
  box-shadow: 0 0 0 1px rgba(77, 163, 255, 0.06);
}

.panel-head {
  padding: 1.15rem 1.35rem 0.85rem;
  border-bottom: 1px solid rgba(120, 170, 255, 0.1);
  background: linear-gradient(180deg, rgba(77, 163, 255, 0.06) 0%, transparent 100%);
}

.panel-title {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--bc-text, #e8ecf4);
}

.panel-desc {
  margin: 0.35rem 0 0;
  font-size: 0.85rem;
  line-height: 1.45;
  color: var(--bc-text-muted, #9aa3b2);
  max-width: 42rem;
}

.panel-inner {
  padding: 0.35rem 1.25rem 1.35rem;
}

.data-dimension-hint {
  margin: 0.5rem 0 0;
}

.ranking-table-empty {
  display: inline-block;
  line-height: 1.4;
  color: var(--bc-text, #cfd6e4);
  font-size: 14px;
  text-align: left;
  padding: 8px 10px;
}

.leaderboard-table-inner {
  padding-top: 1rem;
}

.leaderboard-table-inner :deep(.el-table) {
  width: 100%;
}

.leaderboard-charts-inner {
  padding-top: 1rem;
}

.leaderboard-charts-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 0.9rem;
  align-items: start;
  justify-items: center;
}

.leaderboard-chart-panel,
.leaderboard-pareto-section {
  min-width: 0;
  display: flex;
  flex-direction: column;
  overflow: visible;
  width: min(100%, 840px);
}

.leaderboard-boxplot-title {
  margin: 0;
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--bc-text, #e8ecf4);
}

.leaderboard-boxplot-desc {
  margin: 0.35rem 0 0.8rem;
  font-size: 0.83rem;
  color: var(--bc-text-muted, #9aa3b2);
}

.leaderboard-pareto-note {
  display: inline-flex;
  align-items: center;
  max-width: 100%;
  min-height: 28px;
  margin: 0 0 0.7rem;
  padding: 0.28rem 0.55rem;
  border: 1px solid rgba(120, 170, 255, 0.16);
  border-radius: 8px;
  background: rgba(77, 163, 255, 0.06);
  color: var(--bc-text, #d9e2f2);
  font-size: 0.8rem;
  line-height: 1.35;
}

.slider-param-hint {
  margin-top: 8px;
  font-size: 14px;
  color: var(--bc-text-subtle, #a8abb2);
  line-height: 1.4;
}

.data-type-hint {
  margin: 8px 0 0;
  font-size: 14px;
  color: var(--bc-text-subtle, #a8abb2);
  line-height: 1.45;
  max-width: 36rem;
}

.research-absrel-radio {
  display: block;
  margin-bottom: 8px;
}

.research-eps-slider {
  margin-top: 4px;
}

.research-slider-marks {
  padding-bottom: 18px;
}

.introduction-section {
  margin-bottom: 0;
  line-height: 1.65;
}

.introduction-content {
  display: flex;
  gap: 1.5rem;
  align-items: flex-start;
}

.text-content {
  flex: 1;
  min-width: 0;
}

.image-content {
  flex-shrink: 0;
  width: min(100%, 320px);
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding-top: 0.25rem;
}

.header-image {
  width: 100%;
  max-width: 280px;
  height: auto;
  border-radius: 12px;
  box-shadow: var(--bc-shadow);
}

.introduction-section p {
  margin-bottom: 1rem;
  color: var(--bc-text-muted);
  font-size: 0.95rem;
}

.introduction-section ul {
  margin: 1rem 0 1.25rem;
  padding-left: 1.15rem;
}

.introduction-section li {
  margin-bottom: 0.5rem;
  color: var(--bc-text);
  font-size: 0.92rem;
}

.introduction-section strong {
  color: var(--bc-accent, #4da3ff);
  font-weight: 600;
}

.intro-github-link {
  color: var(--bc-accent, #4da3ff);
  text-decoration: none;
  border-bottom: 1px solid transparent;
  transition: color 0.15s ease, border-color 0.15s ease;
}

.intro-github-link:hover {
  color: var(--bc-cyan, #5ae9ff);
  border-bottom-color: rgba(90, 233, 255, 0.45);
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

.charts-section {
  margin: 0.5rem 0 1rem;
  padding: 0;
  background: transparent;
  border: none;
}

.charts-title {
  text-align: left;
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--bc-text, #e8ecf4);
  margin-bottom: 0.85rem;
}

.statistics-section {
  margin-bottom: 0;
}

.statistics-title {
  text-align: left;
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--bc-text, #e8ecf4);
  margin-bottom: 0.75rem;
}

.ranking-criteria-footnote {
  margin: 10px 0 0 0;
  font-size: 14px;
  line-height: 1.5;
  color: var(--bc-text-muted, #b0b0b0);
  max-width: 52em;
}

/* Responsive layout */
@media (max-width: 768px) {
  .introduction-content {
    flex-direction: column;
    gap: 20px;
  }

  .image-content {
    width: 100%;
    justify-content: center;
  }

  .header-image {
    max-width: 280px;
  }

  .leaderboard-boxplot-grid {
    grid-template-columns: 1fr;
  }

  .leaderboard-charts-grid {
    grid-template-columns: 1fr;
  }
}

</style>
