<template>
  <div
    ref="chartRoot"
    class="leaderboard-critical-difference-chart"
    role="img"
    :aria-label="ariaLabel"
  />
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef, watch } from "vue";
import * as echarts from "echarts/core";
import { LineChart, ScatterChart } from "echarts/charts";
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
import {
  BOS_HIGHLIGHT_COLOR,
  bosRichText,
  echartsDataValue,
  isBosAlgorithmLabel,
  withBosItemStyle,
} from "@/utils/chartAlgorithmHighlight.js";
import { useTheme } from "@/composables/useTheme.js";

echarts.use([
  LineChart,
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
    default: "Critical difference diagram",
  },
});

const chartRoot = ref(null);
const chartInstance = shallowRef(null);
let resizeObserver = null;
const { themeMode } = useTheme();

const rankRows = computed(() => {
  const algorithms = Array.isArray(props.chart?.algorithms) ? props.chart.algorithms : [];
  const values = Array.isArray(props.chart?.values) ? props.chart.values : [];
  return algorithms
    .map((algorithm, index) => {
      const pts = Array.isArray(values[index])
        ? values[index].map((value) => Number(value)).filter((value) => Number.isFinite(value))
        : [];
      if (!pts.length) return null;
      return {
        algorithm: String(algorithm),
        rank: pts.reduce((sum, value) => sum + value, 0) / pts.length,
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.rank - b.rank || a.algorithm.localeCompare(b.algorithm));
});

const rankGroups = computed(() =>
  (Array.isArray(props.chart?.rank_groups) ? props.chart.rank_groups : [])
    .map((group, index) => ({
      startRank: Number(group?.start_rank),
      endRank: Number(group?.end_rank),
      lane: Number.isFinite(Number(group?.lane)) ? Number(group.lane) : index % 4,
      algorithms: Array.isArray(group?.algorithms)
        ? group.algorithms.map((name) => String(name)).filter(Boolean)
        : [],
    }))
    .filter(
      (group) =>
        Number.isFinite(group.startRank) &&
        Number.isFinite(group.endRank) &&
        group.endRank >= group.startRank,
    )
    .slice(0, 8),
);

function buildOption() {
  const rows = rankRows.value;
  if (!rows.length) {
    return null;
  }
  const theme = getChartTheme();
  const maxRank = Math.max(...rows.map((row) => row.rank), 1);
  const criticalDifference = Number(props.chart?.critical_difference);
  const cd = Number.isFinite(criticalDifference) && criticalDifference > 0 ? criticalDifference : 0;
  const cdEnd = Math.min(maxRank, 1 + cd);
  const sampleCount = Number(props.chart?.sample_count);
  const xMax = Math.max(maxRank, cdEnd, 2);
  const scatterY = 1.15;
  const groupBaseY = 2.95;
  const groupGap = 0.18;

  const series = [];
  if (cd > 0) {
    series.push({
      name: "Critical difference",
      type: "line",
      data: [
        [1, 3.75, "CD"],
        [cdEnd, 3.75, "CD"],
      ],
      symbol: "none",
      lineStyle: {
        color: theme.accentStrong,
        width: 2,
      },
      label: {
        show: true,
        position: "top",
        formatter: () => `CD ${formatBoxplotAxisValue(cd)}`,
        color: theme.axisText,
        fontSize: 10,
      },
      z: 4,
    });
  }

  for (const group of rankGroups.value) {
    const y = groupBaseY - (group.lane % 4) * groupGap;
    series.push({
      name: "Rank clique",
      type: "line",
      data: [
        [group.startRank, y, group.algorithms.join(", ")],
        [group.endRank, y, group.algorithms.join(", ")],
      ],
      symbol: "none",
      lineStyle: {
        color: group.algorithms.some((name) => isBosAlgorithmLabel(name))
          ? BOS_HIGHLIGHT_COLOR
          : theme.line,
        width: 2,
      },
      z: 3,
    });
  }

  series.push({
    name: "Average rank",
    type: "scatter",
    data: rows.map((row) =>
      withBosItemStyle(
        [row.rank, scatterY, row.algorithm],
        row.algorithm,
        {
          symbolSize: 10,
        },
      ),
    ),
    symbolSize: 5,
    itemStyle: {
      color: theme.barFill,
      borderColor: theme.pieSliceBorder,
      borderWidth: 0.8,
    },
    label: {
      show: true,
      position: "bottom",
      rotate: 90,
      align: "right",
      verticalAlign: "middle",
      distance: 10,
      color: theme.axisText,
      fontSize: 14,
      formatter(params) {
        const data = echartsDataValue(params?.data);
        const name = Array.isArray(data) ? data[2] : "";
        return isBosAlgorithmLabel(name) ? "{bos|BOS}" : "";
      },
      rich: bosRichText(14),
    },
    labelLayout: {
      hideOverlap: true,
    },
    z: 5,
  });

  return {
    animation: false,
    grid: {
      left: 24,
      right: 24,
      top: 44,
      bottom: rows.length > 45 ? 210 : 180,
      containLabel: true,
    },
    tooltip: {
      trigger: "item",
      ...chartTooltipStyle(theme),
      formatter(params) {
        const data = echartsDataValue(params?.data);
        if (!Array.isArray(data)) return "";
        if (params?.seriesName === "Average rank") {
          return [
            data[2] || "",
            `Average rank: ${formatBoxplotAxisValue(data[0])}`,
            Number.isFinite(sampleCount) && sampleCount > 0
              ? `Datasets: ${formatBoxplotAxisValue(sampleCount)}`
              : "",
          ].filter(Boolean).join("<br/>");
        }
        if (params?.seriesName === "Rank clique") {
          return [
            "No critical difference within group",
            `Rank span: ${formatBoxplotAxisValue(data[0])}`,
            data[2] || "",
          ].filter(Boolean).join("<br/>");
        }
        return `Critical difference: ${formatBoxplotAxisValue(cd)}`;
      },
    },
    xAxis: {
      type: "value",
      min: 1,
      max: xMax,
      name: "Average rank (lower is better)",
      nameLocation: "middle",
      nameGap: 34,
      nameTextStyle: chartAxisNameStyle(theme, 11),
      axisLine: {
        lineStyle: { color: theme.axisLine },
      },
      axisTick: {
        lineStyle: { color: theme.axisTick },
      },
      axisLabel: chartAxisLabelStyle(theme, 10, {
        formatter: (value) => formatBoxplotAxisValue(value),
      }),
      splitLine: {
        lineStyle: { color: theme.splitLine },
      },
    },
    yAxis: {
      type: "value",
      min: 0.75,
      max: 4.05,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { show: false },
      splitLine: { show: false },
    },
    series,
  };
}

function renderChart() {
  if (!chartInstance.value) {
    return;
  }
  const option = buildOption();
  if (!option) {
    chartInstance.value.clear();
    return;
  }
  chartInstance.value.setOption(option, true);
  chartInstance.value.resize();
}

async function initChart() {
  if (!chartRoot.value || chartInstance.value) {
    return;
  }
  await nextTick();
  if (!chartRoot.value) {
    return;
  }
  chartInstance.value = echarts.init(chartRoot.value, undefined, {
    renderer: "canvas",
  });
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
    resizeObserver = new ResizeObserver(() => {
      chartInstance.value?.resize();
    });
    resizeObserver.observe(chartRoot.value);
  }
});

onBeforeUnmount(disposeChart);

watch(
  () => props.chart,
  () => {
    void (async () => {
      if (!chartInstance.value && chartRoot.value) {
        await initChart();
      }
      renderChart();
    })();
  },
  { deep: true },
);

watch(themeMode, () => {
  renderChart();
});
</script>

<style scoped>
.leaderboard-critical-difference-chart {
  display: block;
  width: 100%;
  min-height: 440px;
  height: 520px;
  padding: 4px 2px 0 4px;
  box-sizing: border-box;
  overflow: visible;
}
</style>
