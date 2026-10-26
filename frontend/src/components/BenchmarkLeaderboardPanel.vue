<template>
  <div class="benchmark-leaderboard">
    <div class="leaderboard-option-row">
      <div class="leaderboard-option-label">
        <span class="option-label">Compression rate</span>
      </div>
      <div class="leaderboard-option-control">
        <el-radio-group v-model="compression_rate_mode" class="leaderboard-radio-group">
          <el-radio value="overall" size="large">Overall rate</el-radio>
          <el-radio value="average" size="large">Average rate</el-radio>
        </el-radio-group>
      </div>
    </div>

    <div class="leaderboard-option-row">
      <div class="leaderboard-option-label">
        <span class="option-label">Ranking Criteria</span>
      </div>
      <div class="leaderboard-option-control">
        <el-radio-group v-model="ranking_criteria" class="leaderboard-radio-group ranking-radio-group">
          <el-radio value="1" size="large">Compression Rate</el-radio>
          <el-radio value="2" size="large">Compression Time</el-radio>
          <el-radio value="3" size="large">Decompression Time</el-radio>
          <el-radio value="4" size="large">Balanced Score</el-radio>
          <el-radio value="5" size="large">Average Rank</el-radio>
        </el-radio-group>
      </div>
    </div>

    <p class="timing-contract-note">
      Timing is total measured nanoseconds divided by valid numeric values within each dataset;
      multi-dataset views apply the selected aggregation across datasets. Rate and timing use
      their own available measurements, so unavailable timing never removes a valid rate.
    </p>

    <el-table
      class="leaderboard-table"
      :data="tableRows"
      :height="tableHeight"
      :empty-text="emptyText"
      table-layout="fixed"
    >
      <template v-if="$slots.empty" #empty>
        <slot name="empty" />
      </template>
      <el-table-column
        label="Rank"
        width="66"
        type="index"
        align="center"
        class-name="rank-col"
        label-class-name="rank-col"
      />
      <el-table-column
        prop="algorithm"
        label="Algorithm"
        min-width="205"
        class-name="algorithm-col"
        label-class-name="algorithm-col"
      >
        <template #default="scope">
          <span class="algorithm-cell">
            {{ displayAlgorithmName(scope.row.algorithm) }}
          </span>
        </template>
      </el-table-column>
      <el-table-column prop="compression_rate" label="Compression Rate" min-width="145" align="right">
        <template #default="scope">
          {{ formatMetric(scope.row.compression_rate, 4) }}
        </template>
      </el-table-column>
      <el-table-column prop="compression_time" label="Compression Time (ns/point)" min-width="180" align="right">
        <template #default="scope">
          {{ formatMetric(scope.row.compression_time, 1) }}
        </template>
      </el-table-column>
      <el-table-column prop="decompression_time" label="Decompression Time (ns/point)" min-width="190" align="right">
        <template #default="scope">
          {{ formatMetric(scope.row.decompression_time, 1) }}
        </template>
      </el-table-column>
      <el-table-column prop="balanced_score" label="Balance Score (↓)" min-width="160" align="right">
        <template #default="scope">
          {{ formatMetric(scope.row.balanced_score, 4) }}
        </template>
      </el-table-column>
      <el-table-column prop="average_rank" label="Average Rank" min-width="145" align="right">
        <template #default="scope">
          {{ formatAverageRank(scope.row.average_rank, scope.row.average_rank_digits) }}
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<script setup>
import { inject, computed } from "vue";
import { BENCHMARK_CONFIG_KEY } from "@/composables/useBenchmarkRunConfig.js";

const props = defineProps({
  tableHeight: { type: [Number, String], default: 400 },
  processing: { type: Boolean, default: false },
  emptyTextOverride: { type: String, default: "" },
});

const bench = inject(BENCHMARK_CONFIG_KEY);
if (!bench) {
  throw new Error("BenchmarkLeaderboardPanel requires BENCHMARK_CONFIG_KEY provider");
}

const {
  compression_rate_mode,
  ranking_criteria,
  leaderboardItems,
  displayAlgorithmName,
} = bench;

const emptyText = computed(() => {
  if (props.emptyTextOverride) return props.emptyTextOverride;
  if (props.processing) return "Processing file, please wait...";
  return "No data available.";
});

const tableRows = computed(() => {
  const rows = withAverageRanks(leaderboardItems.value);
  if (String(ranking_criteria.value ?? "1") !== "5") {
    return rows;
  }
  return [...rows].sort(
    (a, b) =>
      sortValue(a.average_rank) - sortValue(b.average_rank) ||
      String(a.algorithm || "").localeCompare(String(b.algorithm || "")),
  );
});

function sortValue(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : Number.POSITIVE_INFINITY;
}

function numericMetric(row, key) {
  const value = Number(row?.[key]);
  return Number.isFinite(value) ? value : null;
}

function metricRanks(rows, key) {
  const ranked = rows
    .map((row, index) => ({
      index,
      value: numericMetric(row, key),
    }))
    .filter((entry) => entry.value != null)
    .sort((a, b) => a.value - b.value || a.index - b.index);
  const ranks = new Map();
  for (let i = 0; i < ranked.length; ) {
    let end = i;
    while (end + 1 < ranked.length && ranked[end + 1].value === ranked[i].value) {
      end += 1;
    }
    const averageRank = (i + 1 + end + 1) / 2;
    for (let j = i; j <= end; j += 1) {
      ranks.set(ranked[j].index, averageRank);
    }
    i = end + 1;
  }
  return ranks;
}

function withAverageRanks(rows) {
  const source = Array.isArray(rows) ? rows : [];
  const metricKeys = ["compression_rate", "compression_time", "decompression_time"];
  const rankMaps = metricKeys.map((key) => metricRanks(source, key));
  return source.map((row, index) => {
    const providedRank = Number(row?.average_rank);
    if (Number.isFinite(providedRank)) {
      return {
        ...row,
        average_rank: providedRank,
        average_rank_digits: 1,
      };
    }
    const ranks = rankMaps
      .map((ranksByMetric) => ranksByMetric.get(index))
      .filter((rank) => Number.isFinite(rank));
    const averageRank = ranks.length
      ? ranks.reduce((sum, rank) => sum + rank, 0) / ranks.length
      : Number.POSITIVE_INFINITY;
    return {
      ...row,
      average_rank: averageRank,
    };
  });
}

function formatAverageRank(value, digits = 2) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(Number(digits)) : "N/A";
}

function formatMetric(value, digits) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "N/A";
}
</script>

<style scoped>
.benchmark-leaderboard {
  min-width: 0;
}

.timing-contract-note {
  margin: 0.25rem 0 0.75rem;
  color: #606266;
  font-size: 0.86rem;
  line-height: 1.4;
}

.leaderboard-option-row {
  display: grid;
  grid-template-columns: 220px minmax(0, 1fr);
  align-items: center;
  column-gap: 1rem;
  margin: 0.75rem 0;
  min-width: 0;
}

.leaderboard-option-label,
.leaderboard-option-control {
  min-width: 0;
}

.leaderboard-option-label .option-label {
  font-size: 1.1rem;
  line-height: 1.25;
  font-weight: 650;
  white-space: nowrap;
}

.leaderboard-radio-group {
  display: flex;
  flex-wrap: nowrap;
  align-items: center;
  column-gap: 1.05rem;
  row-gap: 0.28rem;
  min-width: 0;
}

.leaderboard-radio-group :deep(.el-radio) {
  margin-right: 0;
  white-space: nowrap;
}

.leaderboard-radio-group :deep(.el-radio__label) {
  padding-left: 0.32rem;
  font-size: 1.05rem;
  line-height: 1.2;
}

.ranking-radio-group {
  justify-content: flex-start;
  width: 100%;
}

.ranking-radio-group :deep(.el-radio__label) {
  font-size: 1rem;
}

.leaderboard-table {
  width: 100%;
  min-width: 0;
}

.leaderboard-table :deep(.el-table__inner-wrapper),
.leaderboard-table :deep(.el-table__header-wrapper),
.leaderboard-table :deep(.el-table__body-wrapper) {
  overflow-x: hidden;
}

.leaderboard-table :deep(th.el-table__cell),
.leaderboard-table :deep(td.el-table__cell) {
  padding: 8px 6px !important;
  font-size: 1rem;
}

.leaderboard-table :deep(.el-table__header-wrapper th) {
  height: 46px !important;
  padding: 7px 6px !important;
  font-size: 0.95rem !important;
  line-height: 1.15 !important;
}

.leaderboard-table :deep(.el-table__header-wrapper .cell) {
  padding: 0 6px;
  white-space: normal;
  overflow: visible;
  text-overflow: clip;
}

.leaderboard-table :deep(.rank-col .cell) {
  white-space: nowrap;
}

.leaderboard-table :deep(.el-table__body-wrapper .cell) {
  padding: 0 6px;
  white-space: nowrap;
  overflow: visible;
  text-overflow: clip;
}

.leaderboard-table :deep(.algorithm-col .cell) {
  white-space: normal;
  overflow: visible;
  text-overflow: clip;
  line-height: 1.2;
}

.algorithm-cell {
  display: inline-block;
  max-width: 100%;
  overflow-wrap: anywhere;
  word-break: break-word;
  font-size: 1rem;
}

@media (max-width: 1080px) {
  .leaderboard-radio-group {
    flex-wrap: wrap;
  }
}

@media (max-width: 760px) {
  .leaderboard-option-row {
    grid-template-columns: 1fr;
    row-gap: 0.35rem;
  }
}
</style>
