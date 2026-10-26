<template>
  <div
    ref="chartRoot"
    class="distribution-bar-chart"
    role="img"
    :aria-label="ariaLabel"
  />
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef, watch } from "vue";
import * as echarts from "echarts/core";
import { BarChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import {
  getChartTheme,
  chartAxisLabelStyle,
  chartAxisNameStyle,
  chartTooltipStyle,
} from "@/utils/chartTheme.js";
import { useTheme } from "@/composables/useTheme.js";

echarts.use([BarChart, GridComponent, TooltipComponent, CanvasRenderer]);

const props = defineProps({
  distribution: {
    type: Array,
    default: () => [],
  },
  xAxisName: {
    type: String,
    default: "Category",
  },
  yAxisName: {
    type: String,
    default: "Freq (%)",
  },
  ariaLabel: {
    type: String,
    default: "Distribution bar chart",
  },
});

const chartRoot = ref(null);
const chartInstance = shallowRef(null);
let resizeObserver = null;
const { themeMode } = useTheme();

const normalizedRows = computed(() =>
  (props.distribution || [])
    .map((row) => ({
      category: Number(row?.category),
      count: Number(row?.count),
      percentage: Number(row?.percentage),
    }))
    .filter(
      (row) =>
        Number.isFinite(row.category) &&
        Number.isFinite(row.percentage) &&
        row.percentage >= 0,
    ),
);

function buildOption() {
  const rows = normalizedRows.value;
  const categories = rows.map((row) => String(row.category));
  const values = rows.map((row) => Number(row.percentage.toFixed(2)));
  const labelInterval = rows.length > 16 ? Math.ceil(rows.length / 8) - 1 : 0;
  const theme = getChartTheme();

  return {
    animation: false,
    grid: {
      left: 6,
      right: 8,
      top: 22,
      bottom: 6,
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
        const row = rows[item.dataIndex];
        if (!row) {
          return "";
        }
        const countText = Number.isFinite(row.count) ? `<br/>Count: ${row.count}` : "";
        return `${props.xAxisName}: ${row.category}${countText}<br/>${props.yAxisName}: ${row.percentage.toFixed(2)}%`;
      },
    },
    xAxis: {
      type: "category",
      name: props.xAxisName,
      nameLocation: "middle",
      nameGap: 12,
      nameTextStyle: chartAxisNameStyle(theme, 10),
      data: categories,
      axisLine: {
        lineStyle: { color: theme.axisLine },
      },
      axisTick: {
        alignWithLabel: true,
        lineStyle: { color: theme.axisTick },
      },
      axisLabel: chartAxisLabelStyle(theme, 9, {
        interval: labelInterval,
        hideOverlap: true,
        margin: 4,
      }),
    },
    yAxis: {
      type: "value",
      min: 0,
      name: props.yAxisName,
      nameLocation: "end",
      nameRotate: 0,
      nameGap: 6,
      nameTextStyle: chartAxisNameStyle(theme, 9, { align: "left" }),
      splitNumber: 3,
      axisLine: {
        show: true,
        lineStyle: { color: theme.axisLine },
      },
      axisTick: {
        show: true,
        lineStyle: { color: theme.axisTick },
      },
      axisLabel: chartAxisLabelStyle(theme, 9, {
        formatter: (value) => `${value}%`,
      }),
      splitLine: {
        lineStyle: { color: theme.splitLine },
      },
    },
    series: [
      {
        type: "bar",
        data: values,
        barMaxWidth: 14,
        itemStyle: {
          color: theme.barFill,
          borderRadius: [1, 1, 0, 0],
        },
      },
    ],
  };
}

function renderChart() {
  if (!chartInstance.value) {
    return;
  }
  if (!normalizedRows.value.length) {
    chartInstance.value.clear();
    return;
  }
  chartInstance.value.setOption(buildOption(), true);
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
  () => [props.distribution, props.xAxisName, props.yAxisName],
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
.distribution-bar-chart {
  display: block;
  width: 100%;
  min-width: 148px;
  height: 112px;
  padding: 2px 0 0;
  box-sizing: border-box;
}
</style>
