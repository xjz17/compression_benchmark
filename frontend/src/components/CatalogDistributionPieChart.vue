<template>
  <div
    ref="chartRoot"
    class="catalog-distribution-pie-chart"
    :class="{ 'catalog-distribution-pie-chart--clickable': clickable }"
    role="img"
    :aria-label="title"
  />
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref, shallowRef, watch } from "vue";
import * as echarts from "echarts/core";
import { PieChart } from "echarts/charts";
import { TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import { getChartTheme, chartTooltipStyle } from "@/utils/chartTheme.js";
import { displayBenchmarkDataType } from "@/utils/benchmarkDataTypeDisplay.js";
import { useTheme } from "@/composables/useTheme.js";

echarts.use([PieChart, TooltipComponent, CanvasRenderer]);

const props = defineProps({
  rows: {
    type: Array,
    default: () => [],
  },
  title: {
    type: String,
    default: "Distribution pie chart",
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
      percentage: Number(row?.percentage ?? 0),
      value: Number(row?.value ?? row?.count ?? row?.percentage ?? 0),
    }))
    .filter((entry) => entry.category && Number.isFinite(entry.percentage));
}

function buildSeriesData() {
  const theme = getChartTheme();
  return normalizedRows().map((entry) => ({
    name: entry.category,
    value: Math.max(0, entry.percentage),
    rowMeta: entry.row,
    itemStyle: {
      color: props.colorForName(entry.category),
      borderColor: theme.pieSliceBorder,
      borderWidth: 1.2,
    },
  }));
}

function buildOption() {
  const data = buildSeriesData();
  const theme = getChartTheme();

  return {
    animation: false,
    tooltip: {
      trigger: "item",
      ...chartTooltipStyle(theme),
      formatter(params) {
        const rowMeta = params?.data?.rowMeta;
        if (!rowMeta) {
          return "";
        }
        const valueText = props.formatValue(rowMeta);
        const pct = Number(rowMeta?.percentage ?? params.percent ?? 0).toFixed(1);
        return [
          `<span style="font-weight:700;color:${theme.accentStrong}">${props.title}</span>`,
          displayBenchmarkDataType(params.name),
          valueText,
          `${pct}%`,
        ].join(" · ");
      },
    },
    series: [
      {
        type: "pie",
        radius: "86%",
        center: ["50%", "50%"],
        selectedMode: false,
        avoidLabelOverlap: true,
        label: { show: false },
        labelLine: { show: false },
        emphasis: {
          scale: true,
          scaleSize: 6,
          itemStyle: {
            shadowBlur: 8,
            shadowColor: theme.emphasisShadow,
          },
        },
        data,
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
  chartInstance.value.dispatchAction({ type: "unselect", seriesIndex: 0 });

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
  chartInstance.value?.dispatchAction({ type: "unselect", seriesIndex: 0 });
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
.catalog-distribution-pie-chart {
  display: block;
  width: 100%;
  height: 100%;
  min-height: 160px;
}

.catalog-distribution-pie-chart--clickable {
  cursor: pointer;
}
</style>
