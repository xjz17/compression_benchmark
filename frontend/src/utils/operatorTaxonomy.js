export const OPERATOR_CLASSES = [
  { id: "residual_formation", label: "Residual formation", operatorIds: ["delta", "regression_predictor"] },
  { id: "reordering_symbol_recoding", label: "Reordering and symbol recoding", operatorIds: ["value_reorder", "block_transform", "symbol_recode"] },
  { id: "match_repetition", label: "Match and repetition", operatorIds: ["dictionary", "run_length", "lz_match"] },
  { id: "lz_decisions_format", label: "LZ decisions and format", operatorIds: ["match_parse_policy", "match_window_model", "match_token_format", "literal_entropy"] },
  { id: "probability_coding", label: "Probability coding", operatorIds: ["entropy_huffman", "entropy_range", "entropy_fse", "entropy_special"] },
  { id: "numeric_representation", label: "Numeric representation", operatorIds: ["scaling", "frame_of_reference", "outlier_separation", "zigzag", "xor"] },
  { id: "bit_processing", label: "Bit processing", operatorIds: ["eraser", "bitpack_int", "trailing_zero_pack", "integer_code", "bitstream_pack"] },
  { id: "layout_state", label: "Layout and state", operatorIds: ["block_state_control", "container_frame"] },
];

export const PAPER_OPERATOR_ORDER = [
  "eraser",
  "scaling",
  "frame_of_reference",
  "value_reorder",
  "regression_predictor",
  "outlier_separation",
  "bitpack_int",
  "delta",
  "xor",
  "trailing_zero_pack",
  "dictionary",
  "zigzag",
  "block_transform",
  "symbol_recode",
  "run_length",
  "lz_match",
  "match_parse_policy",
  "match_window_model",
  "match_token_format",
  "integer_code",
  "bitstream_pack",
  "entropy_huffman",
  "entropy_range",
  "entropy_fse",
  "entropy_special",
  "literal_entropy",
  "block_state_control",
  "container_frame",
];

const OPERATOR_ORDER = new Map(
  PAPER_OPERATOR_ORDER.map((operatorId, orderIndex) => [operatorId, orderIndex]),
);

const OPERATOR_POSITION = new Map();
for (const [classIndex, group] of OPERATOR_CLASSES.entries()) {
  for (const [operatorIndex, operatorId] of group.operatorIds.entries()) {
    OPERATOR_POSITION.set(operatorId, {
      classId: group.id,
      classLabel: group.label,
      classIndex,
      operatorIndex,
      orderIndex: OPERATOR_ORDER.get(operatorId) ?? Number.MAX_SAFE_INTEGER,
    });
  }
}

export const OPERATOR_COUNT = OPERATOR_POSITION.size;

export function operatorTaxonomy(operatorId) {
  return OPERATOR_POSITION.get(String(operatorId || "").trim()) || {
    classId: "uncategorized",
    classLabel: "Uncategorized",
    classIndex: OPERATOR_CLASSES.length,
    operatorIndex: Number.MAX_SAFE_INTEGER,
    orderIndex: Number.MAX_SAFE_INTEGER,
  };
}

export function sortOperatorsByTaxonomy(operators) {
  return [...(operators || [])]
    .map((row) => ({ ...row, ...operatorTaxonomy(row?.id) }))
    .sort((left, right) =>
      left.orderIndex - right.orderIndex ||
      left.classIndex - right.classIndex ||
      String(left.name || left.id).localeCompare(String(right.name || right.id)),
    );
}
