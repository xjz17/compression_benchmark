<template>
  <div
    ref="chartRoot"
    class="entropy-before-after-chart"
    :style="{ height: chartHeight }"
    role="img"
    :aria-label="ariaLabel"
  />
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef, watch } from "vue";
import * as echarts from "echarts/core";
import { BarChart, LinesChart, ScatterChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import { formatBoxplotAxisValue } from "@/utils/boxplotStats.js";
import {
  getChartTheme,
  chartAxisLabelStyle,
  chartAxisNameStyle,
  chartTooltipStyle,
} from "@/utils/chartTheme.js";
import { useTheme } from "@/composables/useTheme.js";

echarts.use([
  BarChart,
  LinesChart,
  ScatterChart,
  GridComponent,
  TooltipComponent,
  CanvasRenderer,
]);

const props = defineProps({
  chart: {
    type: Object,
    default: () => ({}),
  },
  ariaLabel: {
    type: String,
    default: "Entropy before-after chart",
  },
});

const chartRoot = ref(null);
const chartInstance = shallowRef(null);
let resizeObserver = null;
const { themeMode } = useTheme();

const isBarChart = computed(() =>
  String(props.chart?.chart_type || "").toLowerCase() === "bar" &&
  (Array.isArray(props.chart?.mean_changes) || Array.isArray(props.chart?.mean_reductions)),
);

const arrowRows = computed(() => {
  const algorithms = Array.isArray(props.chart?.algorithms) ? props.chart.algorithms : [];
  const before = Array.isArray(props.chart?.before_medians) ? props.chart.before_medians : [];
  const after = Array.isArray(props.chart?.after_medians) ? props.chart.after_medians : [];
  return algorithms
    .map((label, index) => {
      const b = Number(before[index]);
      const a = Number(after[index]);
      const hasData = Number.isFinite(b) && Number.isFinite(a);
      return {
        label: String(label),
        before: hasData ? b : null,
        after: hasData ? a : null,
        delta: hasData ? a - b : null,
        hasData,
      };
    })
    .filter((row) => row.label);
});

const barRows = computed(() => {
  const algorithms = Array.isArray(props.chart?.algorithms) ? props.chart.algorithms : [];
  const changes = Array.isArray(props.chart?.mean_changes)
    ? props.chart.mean_changes
    : (Array.isArray(props.chart?.mean_reductions) ? props.chart.mean_reductions : []);
  const sampleCounts = Array.isArray(props.chart?.sample_counts) ? props.chart.sample_counts : [];
  return algorithms
    .map((label, index) => {
      const change = Number(changes[index]);
      const hasData = Number.isFinite(change);
      return {
        label: String(label),
        change: hasData ? change : null,
        sampleCount: Number(sampleCounts[index]) || 0,
        hasData,
      };
    })
    .filter((row) => row.label);
});

const allRows = computed(() => (isBarChart.value ? barRows.value : arrowRows.value));
const rows = computed(() => allRows.value.filter((row) => row.hasData));

const chartHeight = computed(() => {
  const compact = props.chart?.compact_height === true;
  const minHeight = compact ? 360 : 400;
  const rowHeight = compact ? 13 : 18;
  const baseHeight = compact ? 84 : 96;
  return `${Math.max(minHeight, allRows.value.length * rowHeight + baseHeight)}px`;
});

function tooltipFor(row) {
  if (isBarChart.value) {
    const valueLabel = String(props.chart?.y_label || "Mean entropy reduction (bits/byte)");
    const direction = row.change > 0
      ? "entropy reduction"
      : (row.change < 0 ? "entropy increase" : "no change");
    const sampleCountLabel = String(props.chart?.sample_count_label || "Samples");
    const lines = [
      row.label,
      `${valueLabel}: ${formatBoxplotAxisValue(row.change)}`,
      `Direction: ${direction}`,
    ];
    if (row.sampleCount) {
      lines.push(`${sampleCountLabel}: ${row.sampleCount}`);
    }
    return lines.join("<br/>");
  }
  const deltaLabel = String(props.chart?.delta_label || "Δ median");
  return [
    row.label,
    `Before: ${formatBoxplotAxisValue(row.before)}`,
    `After: ${formatBoxplotAxisValue(row.after)}`,
    `${deltaLabel}: ${formatBoxplotAxisValue(row.delta)}`,
  ].join("<br/>");
}

function buildBarOption() {
  if (!allRows.value.length || !rows.value.length) return null;
  const theme = getChartTheme();
  const labels = allRows.value.map((row) => row.label);
  const yLabel = String(props.chart?.y_label || "Mean entropy reduction (bits/byte)");
  const xValues = rows.value.map((row) => row.change);
  const minX = Math.min(0, ...xValues);
  const maxX = Math.max(0, ...xValues);
  const pad = Math.max(0.05, (maxX - minX) * 0.08);

  return {
    animation: false,
    grid: {
      left: 14,
      right: 26,
      top: 24,
      bottom: 8,
      containLabel: true,
    },
    tooltip: {
      trigger: "item",
      ...chartTooltipStyle(theme),
      formatter(params) {
        const row = params?.data?.row;
        return row ? tooltipFor(row) : "";
      },
    },
    xAxis: {
      type: "value",
      name: yLabel,
      min: minX < 0 ? minX - pad : 0,
      max: maxX + pad,
      nameLocation: "end",
      nameGap: 10,
      nameTextStyle: chartAxisNameStyle(theme, 11, { align: "left" }),
      axisLine: { show: true, lineStyle: { color: theme.axisLine } },
      axisTick: { show: true, lineStyle: { color: theme.axisTick } },
      axisLabel: chartAxisLabelStyle(theme, 10, {
        formatter: (value) => formatBoxplotAxisValue(value),
      }),
      splitLine: { lineStyle: { color: theme.splitLine } },
    },
    yAxis: {
      type: "category",
      data: labels,
      inverse: true,
      axisLine: { lineStyle: { color: theme.axisLine } },
      axisTick: { alignWithLabel: true, lineStyle: { color: theme.axisTick } },
      axisLabel: chartAxisLabelStyle(theme, labels.length > 18 ? 9 : 10, {
        interval: 0,
        hideOverlap: false,
      }),
    },
    series: [
      {
        name: yLabel,
        type: "bar",
        barMaxWidth: 12,
        data: allRows.value.map((row) => ({
          value: row.hasData ? row.change : null,
          row,
          itemStyle: {
            color: row.hasData && row.change > 0 ? "#2ca02c" : "#d95f02",
          },
        })),
        emphasis: {
          itemStyle: {
            color: theme.accentStrong,
          },
        },
      },
    ],
  };
}

function buildArrowOption() {
  if (!allRows.value.length || !rows.value.length) return null;
  const theme = getChartTheme();
  const labels = allRows.value.map((row) => row.label);
  const yLabel = String(props.chart?.entropy_label || props.chart?.y_label || "Entropy");
  const xValues = rows.value.flatMap((row) => [row.before, row.after]);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const pad = Math.max(0.05, (maxX - minX) * 0.08);

  return {
    animation: false,
    grid: {
      left: 14,
      right: 22,
      top: 24,
      bottom: 8,
      containLabel: true,
    },
    tooltip: {
      trigger: "item",
      ...chartTooltipStyle(theme),
      formatter(params) {
        const row = params?.data?.row;
        return row ? tooltipFor(row) : "";
      },
    },
    xAxis: {
      type: "value",
      name: yLabel,
      min: Math.max(0, minX - pad),
      max: maxX + pad,
      nameLocation: "end",
      nameGap: 10,
      nameTextStyle: chartAxisNameStyle(theme, 11, { align: "left" }),
      axisLine: { show: true, lineStyle: { color: theme.axisLine } },
      axisTick: { show: true, lineStyle: { color: theme.axisTick } },
      axisLabel: chartAxisLabelStyle(theme, 10, {
        formatter: (value) => formatBoxplotAxisValue(value),
      }),
      splitLine: { lineStyle: { color: theme.splitLine } },
    },
    yAxis: {
      type: "category",
      data: labels,
      inverse: true,
      axisLine: { lineStyle: { color: theme.axisLine } },
      axisTick: { alignWithLabel: true, lineStyle: { color: theme.axisTick } },
      axisLabel: chartAxisLabelStyle(theme, labels.length > 18 ? 9 : 10, {
        interval: 0,
        hideOverlap: false,
      }),
    },
    series: [
      {
        name: "Before to after",
        type: "lines",
        coordinateSystem: "cartesian2d",
        symbol: ["circle", "arrow"],
        symbolSize: [5, 9],
        lineStyle: {
          width: 2,
          opacity: 0.86,
        },
        data: rows.value.map((row) => ({
          coords: [
            [row.before, row.label],
            [row.after, row.label],
          ],
          row,
          lineStyle: {
            color: row.delta <= 0 ? theme.accent : "#f59e0b",
          },
        })),
      },
      {
        name: "Before",
        type: "scatter",
        symbolSize: 7,
        data: rows.value.map((row) => ({
          value: [row.before, row.label],
          row,
        })),
        itemStyle: { color: theme.scatterMuted },
      },
      {
        name: "After",
        type: "scatter",
        symbolSize: 8,
        data: rows.value.map((row) => ({
          value: [row.after, row.label],
          row,
        })),
        itemStyle: { color: theme.accentStrong },
      },
    ],
  };
}

function buildOption() {
  return isBarChart.value ? buildBarOption() : buildArrowOption();
}

function renderChart() {
  if (!chartInstance.value) return;
  const option = buildOption();
  if (!option) {
    chartInstance.value.clear();
    return;
  }
  chartInstance.value.setOption(option, true);
  chartInstance.value.resize();
}

async function initChart() {
  if (!chartRoot.value || chartInstance.value) return;
  await nextTick();
  if (!chartRoot.value) return;
  chartInstance.value = echarts.init(chartRoot.value, undefined, { renderer: "canvas" });
  renderChart();
}

function disposeChart() {
  resizeObserver?.disconnect();
  resizeObserver = null;
  chartInstance.value?.dispose();
  chartInstance.value = null;
}

onMounted(() => {
  void initChart();
  if (chartRoot.value && typeof ResizeObserver !== "undefined") {
    resizeObserver = new ResizeObserver(() => chartInstance.value?.resize());
    resizeObserver.observe(chartRoot.value);
  }
});

onBeforeUnmount(disposeChart);

watch(
  () => props.chart,
  () => {
    void (async () => {
      if (!chartInstance.value && chartRoot.value) await initChart();
      renderChart();
    })();
  },
  { deep: true },
);

watch(themeMode, renderChart);
</script>

<style scoped>
.entropy-before-after-chart {
  display: block;
  width: 100%;
  min-height: 360px;
  height: 400px;
  padding: 4px 2px 0 4px;
  box-sizing: border-box;
  overflow: visible;
}
</style>
