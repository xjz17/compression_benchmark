<template>
  <div
    ref="chartRoot"
    class="leaderboard-pareto-chart"
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
  BOS_HIGHLIGHT_BORDER,
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
    default: "Pareto frontier chart",
  },
});

const chartRoot = ref(null);
const chartInstance = shallowRef(null);
let resizeObserver = null;
const { themeMode } = useTheme();

const normalizedPoints = computed(() =>
  (props.chart?.points || [])
    .map((row) => ({
      algorithm: String(row?.algorithm || "").trim(),
      rate: Number(row?.rate),
      compressionTime: Number(row?.compression_time),
      pareto: Boolean(row?.pareto),
      family: String(row?.family || "").trim(),
      levelOrder:
        row?.level_order == null || row?.level_order === ""
          ? null
          : Number(row.level_order),
    }))
    .filter(
      (row) =>
        row.algorithm &&
        Number.isFinite(row.rate) &&
        Number.isFinite(row.compressionTime) &&
        row.compressionTime > 0,
    ),
);

const normalizedLevelSeries = computed(() =>
  (props.chart?.level_series || [])
    .map((row) => ({
      family: String(row?.family || "").trim(),
      points: (row?.points || [])
        .map((pt) => ({
          algorithm: String(pt?.algorithm || "").trim(),
          rate: Number(pt?.rate),
          compressionTime: Number(pt?.compression_time),
          levelOrder: Number(pt?.level_order),
        }))
        .filter(
          (pt) =>
            pt.algorithm &&
            Number.isFinite(pt.rate) &&
            Number.isFinite(pt.compressionTime) &&
            pt.compressionTime > 0 &&
            Number.isFinite(pt.levelOrder),
        )
        .sort((a, b) => a.levelOrder - b.levelOrder),
    }))
    .filter((row) => row.family && row.points.length >= 2),
);

function levelSeriesColor(family) {
  let hash = 0;
  for (let i = 0; i < family.length; i += 1) {
    hash = (hash * 31 + family.charCodeAt(i)) >>> 0;
  }
  const hue = hash % 360;
  return `hsla(${hue}, 55%, 58%, 0.72)`;
}

const frontierLine = computed(() => {
  const raw = props.chart?.frontier;
  if (Array.isArray(raw) && raw.length) {
    return raw
      .map((pair) => [Number(pair?.[0]), Number(pair?.[1])])
      .filter((pair) => Number.isFinite(pair[0]) && Number.isFinite(pair[1]) && pair[1] > 0);
  }
  return normalizedPoints.value
    .filter((row) => row.pareto)
    .sort((a, b) => a.rate - b.rate)
    .map((row) => [row.rate, row.compressionTime]);
});

function finiteAxisBound(value, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function buildOption() {
  const points = normalizedPoints.value;
  if (!points.length) {
    return null;
  }

  const xLabel = String(props.chart?.x_label || "Compression rate");
  const yLabel = String(props.chart?.y_label || "Compression time (ns/point)");
  const allScatter = points.map((row) =>
    withBosItemStyle(
      [row.rate, row.compressionTime, row.algorithm],
      row.algorithm,
      {
        symbolSize: 12,
      },
    ),
  );
  const paretoScatter = points
    .filter((row) => row.pareto)
    .map((row) =>
      withBosItemStyle(
        [row.rate, row.compressionTime, row.algorithm],
        row.algorithm,
        {
          symbolSize: 13,
        },
      ),
    );
  const lineData = frontierLine.value;
  const theme = getChartTheme();
  const levelLineSeries = normalizedLevelSeries.value.map((row) => ({
    name: `${row.family} levels`,
    type: "line",
    data: row.points.map((pt) => [pt.rate, pt.compressionTime, pt.algorithm, pt.levelOrder]),
    showSymbol: true,
    symbolSize: 5,
    lineStyle: {
      color: levelSeriesColor(row.family),
      width: 1.5,
    },
    itemStyle: {
      color: levelSeriesColor(row.family),
      borderColor: theme.pieSliceBorder,
      borderWidth: 0.6,
    },
    z: 1,
  }));

  return {
    animation: false,
    grid: {
      left: 14,
      right: 18,
      top: 34,
      bottom: 40,
      containLabel: true,
    },
    tooltip: {
      trigger: "item",
      ...chartTooltipStyle(theme),
      formatter(params) {
        const data = echartsDataValue(params?.data);
        if (!Array.isArray(data)) {
          return "";
        }
        const name = data[2] || "";
        const rows = [
          name,
          `${xLabel}: ${formatBoxplotAxisValue(data[0])}`,
          `${yLabel}: ${formatBoxplotAxisValue(data[1])}`,
        ];
        if (params?.seriesName && String(params.seriesName).endsWith(" levels")) {
          rows.push(`Level order: ${formatBoxplotAxisValue(data[3])}`);
          rows.push("Path: same algorithm family across parameter levels");
        }
        return rows.join("<br/>");
      },
    },
    xAxis: {
      type: "value",
      name: xLabel,
      min: finiteAxisBound(props.chart?.x_min, 0),
      max: finiteAxisBound(props.chart?.x_max, 0.9),
      nameLocation: "middle",
      nameGap: 28,
      nameTextStyle: chartAxisNameStyle(theme, 11),
      axisLine: {
        lineStyle: { color: theme.axisLine },
      },
      axisLabel: chartAxisLabelStyle(theme, 10, {
        formatter: (value) => formatBoxplotAxisValue(value),
      }),
      splitLine: {
        lineStyle: { color: theme.splitLine },
      },
    },
    yAxis: {
      type: "log",
      name: yLabel,
      nameLocation: "end",
      nameRotate: 0,
      nameGap: 10,
      nameTextStyle: chartAxisNameStyle(theme, 11, { align: "left" }),
      axisLine: {
        lineStyle: { color: theme.axisLine },
      },
      axisLabel: chartAxisLabelStyle(theme, 10, {
        formatter: (value) => formatBoxplotAxisValue(value),
      }),
      splitLine: {
        lineStyle: { color: theme.splitLine },
      },
    },
    series: [
      ...levelLineSeries,
      {
        name: "Algorithms",
        type: "scatter",
        data: allScatter,
        symbolSize: 7,
        itemStyle: {
          color: theme.scatterMuted,
          borderColor: theme.scatterBorder,
          borderWidth: 0.8,
        },
        label: {
          show: true,
          position: "right",
          distance: 5,
          formatter(params) {
            const data = echartsDataValue(params?.data);
            const name = Array.isArray(data) ? data[2] : "";
            return isBosAlgorithmLabel(name) ? "{bos|BOS}" : "";
          },
          rich: bosRichText(14),
        },
        z: 2,
      },
      {
        name: "Pareto optimal",
        type: "scatter",
        data: paretoScatter,
        symbolSize: 9,
        itemStyle: {
          color: theme.barFill,
          borderColor: theme.pieSliceBorder,
          borderWidth: 1,
        },
        label: {
          show: true,
          position: "right",
          distance: 5,
          formatter(params) {
            const data = echartsDataValue(params?.data);
            const name = Array.isArray(data) ? data[2] : "";
            return isBosAlgorithmLabel(name) ? "{bos|BOS}" : "";
          },
          rich: bosRichText(14),
        },
        z: 4,
      },
      {
        name: "Pareto frontier",
        type: "line",
        data: lineData,
        smooth: false,
        showSymbol: true,
        symbolSize: 6,
        lineStyle: {
          color: theme.line,
          width: 2,
        },
        itemStyle: {
          color(params) {
            const data = echartsDataValue(params?.data);
            const name = Array.isArray(data) ? data[2] : "";
            return isBosAlgorithmLabel(name) ? BOS_HIGHLIGHT_COLOR : theme.barFill;
          },
          borderColor(params) {
            const data = echartsDataValue(params?.data);
            const name = Array.isArray(data) ? data[2] : "";
            return isBosAlgorithmLabel(name) ? BOS_HIGHLIGHT_BORDER : theme.pieSliceBorder;
          },
        },
        z: 3,
      },
    ],
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
.leaderboard-pareto-chart {
  display: block;
  width: 100%;
  aspect-ratio: 1 / 1;
  min-height: 0;
  height: auto;
  padding: 4px 2px 0 4px;
  box-sizing: border-box;
  overflow: visible;
}
</style>
