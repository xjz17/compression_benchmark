export const BOS_HIGHLIGHT_COLOR = "#ef4444";
export const BOS_HIGHLIGHT_BORDER = "#991b1b";

export function isBosAlgorithmLabel(value) {
  const label = String(value ?? "").trim().toLowerCase();
  if (!label) {
    return false;
  }
  return (
    label === "bos" ||
    label.startsWith("bos ") ||
    label.startsWith("bos(") ||
    label.startsWith("bos l") ||
    label.startsWith("ts_2diff_bos")
  );
}

export function truncateAlgorithmLabel(value, maxLength = 13) {
  const label = String(value ?? "").trim();
  if (label.length <= maxLength) {
    return label;
  }
  return `${label.slice(0, Math.max(1, maxLength - 3))}...`;
}

export function formatBosRichLabel(value, maxLength = 13) {
  const label = truncateAlgorithmLabel(value, maxLength);
  return isBosAlgorithmLabel(value) ? `{bos|${label}}` : label;
}

export function bosRichText(fontSize = 10) {
  return {
    bos: {
      color: BOS_HIGHLIGHT_COLOR,
      fontSize,
      fontWeight: 800,
    },
  };
}

export function shouldShowCompactCategoryLabel(label, index, total, maxVisible = 14) {
  if (isBosAlgorithmLabel(label)) {
    return true;
  }
  const count = Number(total);
  if (!Number.isFinite(count) || count <= maxVisible) {
    return true;
  }
  const step = Math.max(1, Math.ceil(count / maxVisible));
  return Number(index) % step === 0;
}

export function withBosItemStyle(value, label, extra = {}) {
  if (!isBosAlgorithmLabel(label)) {
    return value;
  }
  return {
    value,
    itemStyle: {
      color: BOS_HIGHLIGHT_COLOR,
      borderColor: BOS_HIGHLIGHT_BORDER,
      borderWidth: 1.4,
      ...(extra.itemStyle || {}),
    },
    ...extra,
  };
}

export function echartsDataValue(data) {
  if (Array.isArray(data)) {
    return data;
  }
  if (Array.isArray(data?.value)) {
    return data.value;
  }
  return data?.value ?? data;
}
