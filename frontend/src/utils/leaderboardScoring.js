function finitePositive(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? numeric : null;
}

function finiteNonNegative(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric >= 0 ? numeric : null;
}

function nonNegativeValueRange(values) {
  const sorted = values
    .map(finiteNonNegative)
    .filter((value) => value != null)
    .sort((a, b) => a - b);
  if (!sorted.length) return null;
  return { min: sorted[0], max: sorted[sorted.length - 1] };
}

function normalizedByRange(value, range) {
  const numeric = finiteNonNegative(value);
  if (numeric == null || !range) return null;
  if (range.max <= range.min) return 0;
  return (numeric - range.min) / (range.max - range.min);
}

const BALANCED_SCORE_COMPRESSION_TIME_WEIGHT = 0.30;
const BALANCED_SCORE_DECOMPRESSION_TIME_WEIGHT = 0.30;
const BALANCED_SCORE_COMPRESSION_RATE_WEIGHT = 0.40;

export function balancedScore(row, compressionTimeRange, decompressionTimeRange) {
  const rate = finitePositive(row?.compression_rate);
  const compressionTime = finiteNonNegative(row?.compression_time);
  const decompressionTime = finiteNonNegative(row?.decompression_time);
  if (rate == null || compressionTime == null || decompressionTime == null) {
    return Number.POSITIVE_INFINITY;
  }
  const normalizedCompressionTime = normalizedByRange(compressionTime, compressionTimeRange);
  const normalizedDecompressionTime = normalizedByRange(decompressionTime, decompressionTimeRange);
  if (normalizedCompressionTime == null || normalizedDecompressionTime == null) {
    return Number.POSITIVE_INFINITY;
  }
  return (
    BALANCED_SCORE_COMPRESSION_TIME_WEIGHT * normalizedCompressionTime +
    BALANCED_SCORE_DECOMPRESSION_TIME_WEIGHT * normalizedDecompressionTime +
    BALANCED_SCORE_COMPRESSION_RATE_WEIGHT * rate
  );
}

export function withBalancedScores(rows) {
  const source = Array.isArray(rows) ? rows : [];
  if (source.some((row) => Number.isFinite(Number(row?.balanced_score)))) {
    return source.map((row) => {
      const score = Number(row?.balanced_score);
      return {
        ...row,
        balanced_score: Number.isFinite(score) ? score : Number.POSITIVE_INFINITY,
      };
    });
  }
  const compressionTimeRange = nonNegativeValueRange(source.map((row) => row?.compression_time));
  const decompressionTimeRange = nonNegativeValueRange(source.map((row) => row?.decompression_time));
  return source.map((row) => ({
    ...row,
    balanced_score: balancedScore(row, compressionTimeRange, decompressionTimeRange),
  }));
}
