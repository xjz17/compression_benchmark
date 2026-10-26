<template>
  <div class="catalog-correlation-heatmap">
    <header class="catalog-correlation-heatmap__head">
      <h4 class="catalog-correlation-heatmap__title">{{ heatmap.title }}</h4>
      <p v-if="heatmap.description" class="catalog-correlation-heatmap__desc">
        {{ heatmap.description }}
      </p>
    </header>
    <div
      ref="chartRoot"
      class="catalog-correlation-heatmap__chart"
      :class="{ 'catalog-correlation-heatmap__chart--fill': fillHeight }"
      :style="chartAreaStyle"
      role="img"
      :aria-label="heatmap.title"
    />
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef, watch } from "vue";
import * as echarts from "echarts/core";
import { HeatmapChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import {
  getChartTheme,
  chartAxisLabelStyle,
  chartTooltipStyle,
} from "@/utils/chartTheme.js";
import { useTheme } from "@/composables/useTheme.js";

echarts.use([
  HeatmapChart,
  DataZoomComponent,
  GridComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

const props = defineProps({
  heatmap: {
    type: Object,
    required: true,
  },
  fillHeight: {
    type: Boolean,
    default: false,
  },
});

const chartRoot = ref(null);
const chartInstance = shallowRef(null);
let resizeObserver = null;
const { themeMode } = useTheme();

const yLabelCount = computed(
  () => (Array.isArray(props.heatmap?.y_labels) ? props.heatmap.y_labels.length : 0),
);

const xLabelCount = computed(
  () => (Array.isArray(props.heatmap?.x_labels) ? props.heatmap.x_labels.length : 0),
);

const chartHeightPx = computed(() => {
  const rows = yLabelCount.value;
  const visibleRows = rows > 24 ? 24 : rows;
  return Math.max(240, Math.min(620, 88 + visibleRows * 22));
});

const resolvedChartHeightPx = computed(() =>
  props.fillHeight ? null : chartHeightPx.value,
);

const chartAreaStyle = computed(() => {
  if (props.fillHeight) {
    return null;
  }
  return {
    height: `${chartHeightPx.value}px`,
    minHeight: "120px",
  };
});

const matrixCells = computed(() => {
  const xLabels = props.heatmap?.x_labels || [];
  const yLabels = props.heatmap?.y_labels || [];
  const values = props.heatmap?.values || [];
  const cells = [];
  for (let y = 0; y < yLabels.length; y += 1) {
    const row = Array.isArray(values[y]) ? values[y] : [];
    for (let x = 0; x < xLabels.length; x += 1) {
      const raw = row[x];
      const num = raw == null || raw === "" ? null : Number(raw);
      cells.push([x, y, Number.isFinite(num) ? num : null]);
    }
  }
  return cells;
});

function yZoomEndPercent() {
  const rows = yLabelCount.value;
  if (rows <= 18) {
    return 100;
  }
  return Math.min(100, Math.round((18 / rows) * 100));
}

function buildOption() {
  const theme = getChartTheme();
  const xLabels = props.heatmap?.x_labels || [];
  const yLabels = props.heatmap?.y_labels || [];
  const cells = matrixCells.value;
  const finite = cells
    .map((cell) => cell[2])
    .filter((value) => value != null && Number.isFinite(value));
  const maxAbs = finite.length
    ? Math.max(0.35, ...finite.map((value) => Math.abs(value)))
    : 1;
  const useYZoom = yLabelCount.value > 18;
  const yZoomEnd = yZoomEndPercent();

  return {
    animation: true,
    animationDurationUpdate: 280,
    grid: {
      left: 8,
      right: useYZoom ? 98 : 84,
      top: 12,
      bottom: 8,
      containLabel: true,
    },
    tooltip: {
      position: "top",
      ...chartTooltipStyle(theme),
      formatter(params) {
        const x = xLabels[params?.data?.[0]] || "";
        const y = yLabels[params?.data?.[1]] || "";
        const value = params?.data?.[2];
        const valueText =
          value == null || !Number.isFinite(value) ? "—" : Number(value).toFixed(3);
        return `${y}<br/>${x}<br/>r = ${valueText}`;
      },
    },
    dataZoom: useYZoom
      ? [
          {
            type: "slider",
            yAxisIndex: 0,
            orient: "vertical",
            right: 78,
            width: 12,
            start: 0,
            end: yZoomEnd,
            brushSelect: false,
            showDetail: false,
          },
          {
            type: "inside",
            yAxisIndex: 0,
            start: 0,
            end: yZoomEnd,
            zoomOnMouseWheel: true,
            moveOnMouseMove: true,
            moveOnMouseWheel: true,
          },
        ]
      : [],
    xAxis: {
      type: "category",
      data: xLabels,
      splitArea: { show: true },
      axisLine: { lineStyle: { color: theme.axisLine } },
      axisTick: { lineStyle: { color: theme.axisTick } },
      axisLabel: chartAxisLabelStyle(theme, 9, {
        interval: 0,
        rotate: xLabels.length > 4 ? 32 : 0,
        hideOverlap: true,
        formatter: (value) => {
          const text = String(value || "");
          return text.length > 16 ? `${text.slice(0, 15)}…` : text;
        },
      }),
    },
    yAxis: {
      type: "category",
      data: yLabels,
      splitArea: { show: true },
      axisLine: { lineStyle: { color: theme.axisLine } },
      axisTick: { lineStyle: { color: theme.axisTick } },
      axisLabel: chartAxisLabelStyle(theme, 9, {
        width: 108,
        overflow: "truncate",
      }),
    },
    visualMap: {
      min: -maxAbs,
      max: maxAbs,
      calculable: true,
      orient: "vertical",
      right: 4,
      top: "middle",
      itemHeight: Math.min(180, (resolvedChartHeightPx.value ?? 280) - 40),
      text: ["+", "−"],
      textStyle: chartAxisLabelStyle(theme, 9),
      inRange: {
        color: ["#3b4cc0", "#f7f7f7", "#b40426"],
      },
    },
    series: [
      {
        type: "heatmap",
        data: cells,
        progressive: 800,
        animation: true,
        label: {
          show: yLabels.length <= 14 && xLabels.length <= 9,
          formatter(params) {
            const value = params?.data?.[2];
            if (value == null || !Number.isFinite(value)) {
              return "";
            }
            return Number(value).toFixed(2);
          },
          fontSize: 9,
          color: theme.axisText,
        },
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

async function renderChart() {
  if (!chartInstance.value) {
    return;
  }
  if (!matrixCells.value.length) {
    chartInstance.value.clear();
    return;
  }
  chartInstance.value.setOption(buildOption(), true);
  await nextTick();
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
  await renderChart();
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
  () => [props.heatmap, props.fillHeight, resolvedChartHeightPx.value, yLabelCount.value, xLabelCount.value],
  async () => {
    if (!chartInstance.value && chartRoot.value) {
      await initChart();
      return;
    }
    await renderChart();
  },
  { deep: true },
);

watch(themeMode, () => {
  void renderChart();
});
</script>

<style scoped>
.catalog-correlation-heatmap {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
  min-width: 0;
  height: 100%;
  min-height: 0;
}

.catalog-correlation-heatmap__head {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.catalog-correlation-heatmap__title {
  margin: 0;
  font-size: 0.9rem;
  font-weight: 700;
  color: var(--bc-text, #e8ecf4);
}

.catalog-correlation-heatmap__desc {
  margin: 0;
  font-size: 0.78rem;
  line-height: 1.35;
  color: var(--bc-text-muted, rgba(220, 230, 255, 0.72));
}

.catalog-correlation-heatmap__chart {
  width: 100%;
}

.catalog-correlation-heatmap__chart--fill {
  flex: 1 1 auto;
  min-height: 0;
  height: 100%;
}
</style>
