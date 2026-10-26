<template>
  <div
    ref="chartRoot"
    class="catalog-compression-rate-bar-chart"
    :class="{ 'catalog-compression-rate-bar-chart--clickable': clickable }"
    role="img"
    :aria-label="title"
  />
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref, shallowRef, watch } from "vue";
import * as echarts from "echarts/core";
import { BarChart } from "echarts/charts";
import { GridComponent, TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import {
  getChartTheme,
  chartAxisLabelStyle,
  chartTooltipStyle,
} from "@/utils/chartTheme.js";
import { displayBenchmarkDataType } from "@/utils/benchmarkDataTypeDisplay.js";
import { useTheme } from "@/composables/useTheme.js";

echarts.use([BarChart, GridComponent, TooltipComponent, CanvasRenderer]);

const props = defineProps({
  rows: {
    type: Array,
    default: () => [],
  },
  title: {
    type: String,
    default: "Compression rate bar chart",
  },
  chartKey: {
    type: String,
    default: "",
  },
  clickable: {
    type: Boolean,
    default: false,
  },
  legendHighlight: {
    type: String,
    default: "",
  },
  colorForName: {
    type: Function,
    required: true,
  },
  formatValue: {
    type: Function,
    required: true,
  },
});

const emit = defineEmits(["slice-click"]);

const chartRoot = ref(null);
const chartInstance = shallowRef(null);
let resizeObserver = null;
const { themeMode } = useTheme();

function normalizedRows() {
  return (props.rows || [])
    .map((row) => ({
      category: String(row?.category || "").trim(),
      row,
      rate: Number(row?.compression_rate ?? row?.value ?? 0),
      percentage: Number(row?.percentage ?? 0),
    }))
    .filter((entry) => entry.category && Number.isFinite(entry.rate) && entry.rate > 0)
    .sort((a, b) => b.rate - a.rate || a.category.localeCompare(b.category));
}

function buildSeriesData() {
  const highlight = String(props.legendHighlight || "").trim();
  const hasHighlight = Boolean(
    highlight && normalizedRows().some((row) => row.category === highlight),
  );

  return normalizedRows().map((entry) => ({
    name: entry.category,
    value: entry.rate,
    rowMeta: entry.row,
    itemStyle: {
      color: props.colorForName(entry.category),
      opacity:
        hasHighlight && entry.category !== highlight ? 0.32 : 1,
      borderRadius: [2, 2, 0, 0],
    },
  }));
}

function buildOption() {
  const rows = normalizedRows();
  const data = buildSeriesData();
  const theme = getChartTheme();
  const categories = rows.map((row) => displayBenchmarkDataType(row.category));
  const maxRate = rows.reduce((max, row) => Math.max(max, row.rate), 0);
  const yMax = maxRate > 0 ? Math.min(1, maxRate * 1.12) : 1;
  const rotateLabels = categories.length > 3 ? 40 : 0;

  return {
    animation: false,
    grid: {
      left: 4,
      right: 4,
      top: 8,
      bottom: rotateLabels ? 2 : 4,
      containLabel: true,
    },
    tooltip: {
      trigger: "axis",
      ...chartTooltipStyle(theme),
      axisPointer: { type: "shadow" },
      formatter(params) {
        const item = Array.isArray(params) ? params[0] : params;
        if (!item) {
          return "";
        }
        const rowMeta = item?.data?.rowMeta;
        if (!rowMeta) {
          return "";
        }
        const valueText = props.formatValue(rowMeta);
        const pct = Number(rowMeta?.percentage ?? 0).toFixed(1);
        return [
          `<span style="font-weight:700;color:${theme.accentStrong}">${props.title}</span>`,
          displayBenchmarkDataType(item.name),
          `Rate: ${valueText}`,
          `Share: ${pct}%`,
        ].join("<br/>");
      },
    },
    xAxis: {
      type: "category",
      data: categories,
      axisLine: {
        lineStyle: { color: theme.axisLine },
      },
      axisTick: {
        alignWithLabel: true,
        lineStyle: { color: theme.axisTick },
      },
      axisLabel: chartAxisLabelStyle(theme, 8, {
        interval: 0,
        rotate: rotateLabels,
        hideOverlap: true,
        margin: rotateLabels ? 10 : 4,
        formatter: (value) => {
          const text = String(value || "");
          return text.length > 10 ? `${text.slice(0, 9)}…` : text;
        },
      }),
    },
    yAxis: {
      type: "value",
      min: 0,
      max: yMax,
      splitNumber: 3,
      axisLine: {
        show: true,
        lineStyle: { color: theme.axisLine },
      },
      axisTick: {
        show: true,
        lineStyle: { color: theme.axisTick },
      },
      axisLabel: chartAxisLabelStyle(theme, 8, {
        formatter: (value) => Number(value).toFixed(2),
      }),
      splitLine: {
        lineStyle: { color: theme.splitLine },
      },
    },
    series: [
      {
        type: "bar",
        data,
        barMaxWidth: 18,
        emphasis: {
          itemStyle: {
            shadowBlur: 8,
            shadowColor: theme.emphasisShadow,
          },
        },
      },
    ],
  };
}

function applyHighlightState() {
  if (!chartInstance.value) {
    return;
  }
  const rows = normalizedRows();
  if (!rows.length) {
    return;
  }
  chartInstance.value.dispatchAction({ type: "downplay", seriesIndex: 0 });

  const highlight = String(props.legendHighlight || "").trim();
  if (highlight && rows.some((row) => row.category === highlight)) {
    chartInstance.value.dispatchAction({
      type: "highlight",
      seriesIndex: 0,
      name: highlight,
    });
  }
}

function renderChart() {
  if (!chartInstance.value) {
    return;
  }
  if (!normalizedRows().length) {
    chartInstance.value.clear();
    return;
  }
  chartInstance.value.setOption(buildOption(), true);
  applyHighlightState();
}

function onChartClick(params) {
  if (!props.clickable) {
    return;
  }
  const category = String(params?.data?.rowMeta?.category || params?.name || "").trim();
  if (category) {
    emit("slice-click", category);
  }
  chartInstance.value?.dispatchAction({ type: "downplay", seriesIndex: 0 });
}

async function initChart() {
  if (!chartRoot.value || chartInstance.value) {
    return;
  }
  chartInstance.value = echarts.init(chartRoot.value, undefined, {
    renderer: "canvas",
  });
  chartInstance.value.on("click", "series", onChartClick);
  renderChart();
}

function disposeChart() {
  resizeObserver?.disconnect();
  resizeObserver = null;
  if (chartInstance.value) {
    chartInstance.value.off("click", onChartClick);
    chartInstance.value.dispose();
    chartInstance.value = null;
  }
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
  () => [
    props.rows,
    props.title,
    props.chartKey,
    props.clickable,
    props.legendHighlight,
  ],
  () => {
    if (!chartInstance.value && chartRoot.value) {
      void initChart();
    }
    renderChart();
  },
  { deep: true },
);

watch(themeMode, () => {
  renderChart();
});
</script>

<style scoped>
.catalog-compression-rate-bar-chart {
  display: block;
  width: 100%;
  height: 100%;
  min-height: 160px;
}

.catalog-compression-rate-bar-chart--clickable {
  cursor: pointer;
}
</style>
