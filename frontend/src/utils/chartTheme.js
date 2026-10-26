/** Theme-aware ECharts colors sourced from CSS variables in styles/theme.css */

function readChartCssVar(name, fallback) {
  if (typeof document === "undefined") {
    return fallback;
  }
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

export function getChartTheme() {
  return {
    axisText: readChartCssVar("--bc-chart-axis-text", "#c8d0e0"),
    axisLine: readChartCssVar("--bc-chart-axis-line", "rgba(140, 155, 180, 0.55)"),
    axisTick: readChartCssVar("--bc-chart-axis-tick", "rgba(140, 155, 180, 0.45)"),
    splitLine: readChartCssVar("--bc-chart-split-line", "rgba(140, 155, 180, 0.18)"),
    tooltipBg: readChartCssVar("--bc-chart-tooltip-bg", "rgba(16, 22, 36, 0.94)"),
    tooltipBorder: readChartCssVar("--bc-chart-tooltip-border", "rgba(120, 170, 255, 0.28)"),
    tooltipText: readChartCssVar("--bc-chart-tooltip-text", "#e8ecf4"),
    accent: readChartCssVar("--bc-chart-accent", "#5ae9ff"),
    accentStrong: readChartCssVar("--bc-accent", "#4da3ff"),
    boxFill: readChartCssVar("--bc-chart-box-fill", "rgba(200, 214, 229, 0.35)"),
    boxBorder: readChartCssVar("--bc-chart-box-border", "rgba(140, 170, 210, 0.95)"),
    scatterMuted: readChartCssVar("--bc-chart-scatter-muted", "rgba(120, 150, 190, 0.55)"),
    scatterBorder: readChartCssVar("--bc-chart-scatter-border", "rgba(180, 200, 230, 0.8)"),
    barFill: readChartCssVar("--bc-chart-bar-fill", "rgba(90, 233, 255, 0.85)"),
    line: readChartCssVar("--bc-chart-line", "rgba(90, 233, 255, 0.75)"),
    pieSliceBorder: readChartCssVar("--bc-chart-pie-slice-border", "rgba(12, 16, 26, 0.8)"),
    emphasisShadow: readChartCssVar("--bc-chart-emphasis-shadow", "rgba(0, 0, 0, 0.35)"),
  };
}

export function chartAxisLabelStyle(theme, fontSize = 10, extra = {}) {
  return {
    fontSize,
    color: theme.axisText,
    ...extra,
  };
}

export function chartAxisNameStyle(theme, fontSize = 11, extra = {}) {
  return {
    fontSize,
    color: theme.axisText,
    ...extra,
  };
}

export function chartTooltipStyle(theme, extra = {}) {
  return {
    confine: true,
    appendToBody: true,
    backgroundColor: theme.tooltipBg,
    borderColor: theme.tooltipBorder,
    textStyle: {
      color: theme.tooltipText,
      fontSize: 12,
    },
    ...extra,
  };
}
