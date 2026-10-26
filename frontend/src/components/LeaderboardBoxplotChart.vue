<template>
  <div
    ref="chartRoot"
    class="leaderboard-boxplot-chart"
    role="img"
    :aria-label="ariaLabel"
  />
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef, watch } from "vue";
import * as echarts from "echarts/core";
import { BarChart, BoxplotChart, ScatterChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import {
  buildBoxplotSeries,
  formatBoxplotAxisValue,
} from "@/utils/boxplotStats.js";
import {
  getChartTheme,
  chartAxisLabelStyle,
  chartAxisNameStyle,
  chartTooltipStyle,
} from "@/utils/chartTheme.js";
import {
  bosRichText,
  echartsDataValue,
  isBosAlgorithmLabel,
  withBosItemStyle,
} from "@/utils/chartAlgorithmHighlight.js";
import { useTheme } from "@/composables/useTheme.js";

echarts.use([
  BarChart,
  BoxplotChart,
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
    default: "Leaderboard boxplot",
  },
});

const chartRoot = ref(null);
const chartInstance = shallowRef(null);
let resizeObserver = null;
const { themeMode } = useTheme();

const chartType = computed(() => String(props.chart?.chart_type || "boxplot").toLowerCase());

const seriesInput = computed(() => {
  const algorithms = props.chart?.algorithms || [];
  const values = props.chart?.values || [];
  return buildBoxplotSeries(algorithms, values, {
    logY: Boolean(props.chart?.log_y),
  });
});

const barInput = computed(() => {
  const algorithms = Array.isArray(props.chart?.algorithms) ? props.chart.algorithms : [];
  const values = Array.isArray(props.chart?.values) ? props.chart.values : [];
  const categories = [];
  const data = [];
  algorithms.forEach((name, index) => {
    const pts = Array.isArray(values[index])
      ? values[index].map((value) => Number(value)).filter((value) => Number.isFinite(value))
      : [];
    if (!pts.length) return;
    categories.push(String(name));
    data.push(withBosItemStyle(
      pts.reduce((sum, value) => sum + value, 0) / pts.length,
      name,
    ));
  });
  return { categories, data };
});

function buildBarOption() {
  const { categories, data } = barInput.value;
  if (!categories.length) {
    return null;
  }

  const yLabel = String(props.chart?.y_label || "Value");
  const logY = Boolean(props.chart?.log_y);
  const categoryCount = categories.length;
  const xLabelFontSize = categoryCount > 40 ? 8 : 9;
  const theme = getChartTheme();

  return {
    animation: false,
    grid: {
      left: 14,
      right: 16,
      top: 34,
      bottom: 26,
      containLabel: true,
    },
    tooltip: {
      trigger: "item",
      ...chartTooltipStyle(theme),
      formatter(params) {
        const name = categories[params?.dataIndex] || "";
        return `${name}<br/>${yLabel}: ${formatBoxplotAxisValue(params?.value)}`;
      },
    },
    xAxis: {
      type: "category",
      data: categories,
      boundaryGap: true,
      axisLine: {
        lineStyle: { color: theme.axisLine },
      },
      axisTick: {
        alignWithLabel: true,
        lineStyle: { color: theme.axisTick },
      },
      axisLabel: chartAxisLabelStyle(theme, xLabelFontSize, {
        rotate: 0,
        interval: 0,
        hideOverlap: false,
        margin: 6,
        formatter(value) {
          return isBosAlgorithmLabel(value) ? "{bos|BOS}" : "";
        },
        rich: bosRichText(14),
      }),
    },
    yAxis: {
      type: logY ? "log" : "value",
      name: yLabel,
      nameLocation: "end",
      nameRotate: 0,
      nameGap: 10,
      nameTextStyle: chartAxisNameStyle(theme, 11, { align: "left" }),
      axisLine: {
        show: true,
        lineStyle: { color: theme.axisLine },
      },
      axisTick: {
        show: true,
        lineStyle: { color: theme.axisTick },
      },
      axisLabel: chartAxisLabelStyle(theme, 10, {
        formatter: (value) => formatBoxplotAxisValue(value),
      }),
      splitLine: {
        lineStyle: { color: theme.splitLine },
      },
    },
    series: [
      {
        name: yLabel,
        type: "bar",
        data,
        barMaxWidth: 24,
        itemStyle: {
          color: theme.barFill,
          borderColor: theme.pieSliceBorder,
          borderWidth: 0.8,
        },
        emphasis: {
          itemStyle: {
            color: theme.accentStrong,
          },
        },
        label: {
          show: true,
          position: "top",
          color: theme.axisText,
          fontSize: 10,
          formatter(params) {
            const name = categories[params?.dataIndex] || "";
            return isBosAlgorithmLabel(name) ? "{bos|BOS}" : "";
          },
          rich: bosRichText(14),
        },
      },
    ],
  };
}

function buildOption() {
  if (chartType.value === "bar") {
    return buildBarOption();
  }

  const { categories, boxData, outliers } = seriesInput.value;
  if (!categories.length) {
    return null;
  }

  const yLabel = String(props.chart?.y_label || "Value");
  const logY = Boolean(props.chart?.log_y);
  const categoryCount = categories.length;
  const xLabelFontSize = categoryCount > 40 ? 8 : 9;
  const theme = getChartTheme();
  const styledBoxData = boxData.map((row, index) =>
    withBosItemStyle(row, categories[index], {
      itemStyle: {
        color: "rgba(239, 68, 68, 0.28)",
      },
    }),
  );
  const styledOutliers = outliers.map((row) =>
    withBosItemStyle(row, categories[Number(row?.[0])], {
      itemStyle: {
        color: "#ef4444",
      },
    }),
  );

  return {
    animation: false,
    grid: {
      left: 14,
      right: 16,
      top: 34,
      bottom: 26,
      containLabel: true,
    },
    tooltip: {
      trigger: "item",
      ...chartTooltipStyle(theme),
      formatter(params) {
        if (!params) return "";
        if (params.seriesType === "scatter") {
          const value = echartsDataValue(params.data ?? params.value);
          return `${categories[value[0]]}<br/>${yLabel}: ${formatBoxplotAxisValue(value[1])}`;
        }
        const data = echartsDataValue(params.data ?? params.value);
        if (!Array.isArray(data) || data.length < 5) {
          return "";
        }
        const name = categories[params.dataIndex] || "";
        return [
          name,
          `Min: ${formatBoxplotAxisValue(data[0])}`,
          `Q1: ${formatBoxplotAxisValue(data[1])}`,
          `Median: ${formatBoxplotAxisValue(data[2])}`,
          `Q3: ${formatBoxplotAxisValue(data[3])}`,
          `Max: ${formatBoxplotAxisValue(data[4])}`,
        ].join("<br/>");
      },
    },
    xAxis: {
      type: "category",
      data: categories,
      boundaryGap: true,
      axisLine: {
        lineStyle: { color: theme.axisLine },
      },
      axisTick: {
        alignWithLabel: true,
        lineStyle: { color: theme.axisTick },
      },
      axisLabel: chartAxisLabelStyle(theme, xLabelFontSize, {
        rotate: 0,
        interval: 0,
        hideOverlap: false,
        margin: 6,
        formatter(value) {
          return isBosAlgorithmLabel(value) ? "{bos|BOS}" : "";
        },
        rich: bosRichText(14),
      }),
    },
    yAxis: {
      type: logY ? "log" : "value",
      name: yLabel,
      nameLocation: "end",
      nameRotate: 0,
      nameGap: 10,
      nameTextStyle: chartAxisNameStyle(theme, 11, { align: "left" }),
      axisLine: {
        show: true,
        lineStyle: { color: theme.axisLine },
      },
      axisTick: {
        show: true,
        lineStyle: { color: theme.axisTick },
      },
      axisLabel: chartAxisLabelStyle(theme, 10, {
        formatter: (value) => formatBoxplotAxisValue(value),
      }),
      splitLine: {
        lineStyle: { color: theme.splitLine },
      },
    },
    series: [
      {
        name: yLabel,
        type: "boxplot",
        data: styledBoxData,
        itemStyle: {
          color: theme.boxFill,
          borderColor: theme.boxBorder,
          borderWidth: 1.4,
        },
        emphasis: {
          itemStyle: {
            color: theme.boxFill,
            borderColor: theme.accentStrong,
            borderWidth: 1.8,
          },
        },
      },
      {
        name: "Outlier",
        type: "scatter",
        data: styledOutliers,
        symbolSize: 4,
        itemStyle: {
          color: theme.accentStrong,
        },
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
.leaderboard-boxplot-chart {
  display: block;
  width: 100%;
  min-height: 360px;
  height: 400px;
  padding: 4px 2px 0 4px;
  box-sizing: border-box;
  overflow: visible;
}
</style>
