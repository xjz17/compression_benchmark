<template>
  <AppLayout
    title="Method"
    subtitle="Overview of benchmark methods."
  >
    <section class="surface-panel">
      <header class="panel-head">
        <h2 class="panel-title">Methods</h2>
        <p class="panel-desc">
          Method metadata with filters, plus a combined operator step matrix and automaton.
        </p>
      </header>
      <div class="panel-inner">
        <el-row class="option-row option-single">
          <el-col :span="8">
            <span class="option-label">Data Type</span>
          </el-col>
          <el-col :span="16">
            <el-radio-group v-model="data_type">
              <el-radio value="all" size="large">All</el-radio>
              <el-radio value="int" size="large">Int</el-radio>
              <el-radio value="float" size="large">Float</el-radio>
            </el-radio-group>
          </el-col>
        </el-row>

        <div class="methods-tables-stack">
          <div class="methods-info-table-wrap">
            <el-table
              :data="filteredMethods"
              style="width: 100%"
              :loading="loading"
              :max-height="METHODS_TABLE_MAX_HEIGHT_PX"
              :empty-text="tableEmptyText"
            >
              <el-table-column prop="name" label="Method" min-width="170">
                <template #default="scope">
                  {{ displayMethodCatalogName(scope.row.name) }}
                </template>
              </el-table-column>
              <el-table-column label="Paper" min-width="260" show-overflow-tooltip>
                <template #default="scope">
                  <div class="method-paper-cell">
                    <a
                      v-if="scope.row.paper_url"
                      :href="scope.row.paper_url"
                      target="_blank"
                      rel="noopener noreferrer"
                      class="paper-link"
                    >
                      {{ scope.row.paper || "Reference" }}
                    </a>
                    <span v-else>{{ scope.row.paper || "—" }}</span>
                  </div>
                </template>
              </el-table-column>
              <el-table-column label="Description" min-width="300" class-name="method-desc-col">
                <template #default="scope">
                  <div class="method-desc-cell">
                    {{ scope.row.description || "—" }}
                  </div>
                </template>
              </el-table-column>
              <el-table-column
                label="Compression Mode"
                width="150"
                column-key="compression_type"
                :show-overflow-tooltip="false"
              >
                <template #default="scope">
                  <CompressionTypeTag :mode="scope.row.compression_mode" />
                </template>
              </el-table-column>
              <el-table-column label="Data Types" min-width="170">
                <template #default="scope">
                  <div class="dtype-tags">
                    <el-tag
                      v-for="dtype in visibleMethodDataTypes(scope.row)"
                      :key="`${scope.row.name}-${dtype}`"
                      size="small"
                      type="info"
                      effect="plain"
                    >
                      {{ displayBenchmarkDataType(dtype) }}
                    </el-tag>
                    <span v-if="!visibleMethodDataTypes(scope.row).length">—</span>
                  </div>
                </template>
              </el-table-column>
            </el-table>
          </div>

          <div
            v-show="showPipelineMatrix"
            class="methods-operator-table-wrap"
          >
            <div class="combined-pipeline-layout">
              <div class="combined-pipeline-matrix" @mouseleave="onCombinedPipelineLeave">
                <h3 class="operators-matrix-title">Compression pipelines</h3>
                <p class="operators-matrix-desc">
                  All 28 operators follow the same order and names as Table 1 in the paper.
                  Operators are grouped by class, and the class label is shown once for each group.
                </p>
                <el-table
                  class="operator-pipeline-matrix-table"
                  :data="combinedPipelineMatrixMethods"
                  :style="combinedPipelineTableStyle"
                  :loading="loading"
                  :max-height="METHODS_TABLE_MAX_HEIGHT_PX"
                  :empty-text="combinedPipelineMatrixEmptyText"
                  @cell-mouse-enter="onCombinedPipelineMethodEnter"
                >
                  <el-table-column
                    label="Method"
                    :width="PIPELINE_METHOD_COL_WIDTH_PX"
                    fixed
                    align="center"
                    label-class-name="method-name-col"
                    class-name="method-name-col"
                    show-overflow-tooltip
                  >
                    <template #default="scope">
                      <div class="pipeline-method-cell">
                        <span>{{ displayMethodCatalogName(scope.row.name) }}</span>
                      </div>
                    </template>
                  </el-table-column>
                  <el-table-column
                    label="Operators"
                    align="center"
                  >
                  <el-table-column
                    v-for="group in pipelineMatrixOperatorGroups"
                    :key="`pipeline-op-group-${group.id}`"
                    :label="group.label"
                    :min-width="group.width"
                    :width="group.width"
                    align="center"
                    label-class-name="operator-category-col"
                  >
                    <el-table-column
                      v-for="op in group.operators"
                      :key="`pipeline-op-${op.id}`"
                      :column-key="op.id"
                      :min-width="OPERATOR_COL_MIN_WIDTH_PX"
                      :width="OPERATOR_COL_WIDTH_PX"
                      align="center"
                      label-class-name="operator-header-col"
                      class-name="operator-step-col"
                    >
                      <template #header>
                        <span
                          class="operator-header-vertical"
                          :title="operatorPipelineHeaderTitle(op)"
                        >
                          {{ op.name }}
                        </span>
                      </template>
                      <template #default="scope">
                        <span
                          v-if="pipelineOperatorStep(scope.row, op.id)"
                          class="operator-step-num"
                          @mouseenter="onCombinedPipelineMethodEnter(scope.row)"
                        >
                          {{ pipelineOperatorStep(scope.row, op.id) }}
                        </span>
                      </template>
                    </el-table-column>
                  </el-table-column>
                </el-table-column>
              </el-table>
              </div>
              <div class="automata-row">
                <div
                  v-for="view in automatonStartViews"
                  :key="`operator-automaton-${view.id}`"
                  class="operator-automaton combined-operator-automaton"
                  :aria-label="`Finite-state automaton for operators from ${view.state} input`"
                >
                <h3 class="operators-matrix-title">{{ view.title }}</h3>
                <svg
                  class="automaton-svg"
                  viewBox="0 0 540 360"
                  role="img"
                  :aria-labelledby="`operatorAutomatonTitle-${view.id} operatorAutomatonDesc-${view.id}`"
                >
                  <title :id="`operatorAutomatonTitle-${view.id}`">Finite-state automaton for operators</title>
                  <desc :id="`operatorAutomatonDesc-${view.id}`">
                    States are data representations; labeled arrows are operator transitions.
                  </desc>
                  <defs>
                    <marker
                      id="automaton-arrow"
                      viewBox="0 0 10 10"
                      refX="9"
                      refY="5"
                      markerWidth="7"
                      markerHeight="7"
                      orient="auto-start-reverse"
                    >
                      <path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke" />
                    </marker>
                  </defs>

                  <g
                    v-for="edge in automatonEdges"
                    :key="edge.id"
                    v-show="shouldShowAutomatonEdgeForStart(view.state, edge.id)"
                    class="automaton-edge"
                    :class="{ 'is-active': isActiveAutomatonEdgeForStart(view.state, edge.id) }"
                  >
                    <path
                      :d="edge.path"
                      :class="{ 'automaton-loop': edge.loop }"
                    />
                    <foreignObject
                      v-show="!isAutomatonFocused || !fullAutomatonRunForStart(view.state).length"
                      :x="edge.label.x"
                      :y="edge.label.y"
                      :width="edge.label.width"
                      :height="edge.label.height"
                    >
                      <div
                        xmlns="http://www.w3.org/1999/xhtml"
                        class="automaton-edge-label"
                        :class="{ 'is-active': isActiveAutomatonEdgeForStart(view.state, edge.id) }"
                      >
                        {{ automatonEdgeLabelTextForStart(view.state, edge) }}
                      </div>
                    </foreignObject>
                  </g>

                  <g
                    v-for="(run, index) in activeAutomatonRunForStart(view.state)"
                    :key="`run-${view.id}-${index}-${run.edgeId}-${run.operatorId}`"
                    class="automaton-run-edge"
                    :style="{ '--run-index': index }"
                  >
                    <path :d="automatonEdgePath(run.edgeId)" />
                    <foreignObject
                      :x="run.badge.x"
                      :y="run.badge.y"
                      width="34"
                      height="22"
                    >
                      <div xmlns="http://www.w3.org/1999/xhtml" class="automaton-run-badge">
                        {{ run.stepNumber || index + 1 }}
                      </div>
                    </foreignObject>
                    <foreignObject
                      v-if="isAutomatonFocused"
                      :x="run.label.x"
                      :y="run.label.y"
                      :width="run.label.width"
                      :height="run.label.height"
                    >
                      <div xmlns="http://www.w3.org/1999/xhtml" class="automaton-run-label">
                        {{ automatonRunLabel(run, index) }}
                      </div>
                    </foreignObject>
                  </g>

                  <g
                    v-for="state in automatonStates"
                    :key="state.id"
                    v-show="shouldShowAutomatonStateForStart(view.state, state.id)"
                    class="automaton-state"
                    :class="{
                      'is-active': isActiveAutomatonStateForStart(view.state, state.id),
                      'is-start': isActiveAutomatonStartStateForStart(view.state, state.id),
                      'is-final': isActiveAutomatonFinalStateForStart(view.state, state.id),
                    }"
                    :transform="`translate(${state.x} ${state.y})`"
                  >
                    <circle :r="state.r" />
                    <text text-anchor="middle" dominant-baseline="middle">
                      {{ state.label }}
                    </text>
                  </g>
                </svg>
                <div class="automaton-caption">
                  <span class="automaton-caption-label">Hover:</span>
                  {{ automatonCaptionForStart(view.state) }}
                </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="surface-panel">
      <header class="panel-head">
        <h2 class="panel-title">Compression Operators</h2>
        <p class="panel-desc">
          {{ compressionOperatorsPanelDesc }}
        </p>
        <div class="operator-downloads" aria-label="Operator result downloads">
          <a
            href="/benchmark/downloads/operator-byte-effect-current135-summary.json"
            download
          >
            Download current-135 operator summary (JSON)
          </a>
          <a
            href="/benchmark/downloads/operator135_gap400.current_value_columns.public.csv.gz"
            download
          >
            Download the 4,000-row lossless byte-effect patch (CSV.gz)
          </a>
          <span>
            Byte effects only; timing fields are intentionally empty. The summary discloses retained historical input qualification.
          </span>
        </div>
      </header>
      <div class="panel-inner operators-reference-stack">
        <div>
          <h3 class="operators-reference-title">Operators</h3>
          <el-table
            :data="operatorReferenceRows"
            style="width: 100%"
            :loading="loading"
            :max-height="METHODS_TABLE_MAX_HEIGHT_PX"
            empty-text="No operators in catalog."
          >
            <el-table-column prop="name" label="Operator" min-width="160" />
            <el-table-column prop="classLabel" label="Class" min-width="190" />
            <el-table-column label="Description" min-width="280" class-name="method-desc-col">
              <template #default="scope">
                <div class="method-desc-cell">
                  {{ scope.row.description || "—" }}
                </div>
              </template>
            </el-table-column>
            <el-table-column label="Good ratio pattern" min-width="360" class-name="method-desc-col">
              <template #default="scope">
                <div class="method-desc-cell">
                  {{ operatorGoodRatioExample(scope.row) }}
                </div>
              </template>
            </el-table-column>
            <el-table-column label="Poor ratio pattern" min-width="360" class-name="method-desc-col">
              <template #default="scope">
                <div class="method-desc-cell">
                  {{ operatorPoorRatioExample(scope.row) }}
                </div>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
    </section>
  </AppLayout>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { ElMessage } from "element-plus";
import axios from "@/utils/httpClient.js";
import AppLayout from "@/components/AppLayout.vue";
import CompressionTypeTag from "@/components/CompressionTypeTag.vue";
import { compressionModeForMethodName } from "@/utils/methodCatalogCompressionMode.js";
import { displayMethodCatalogName } from "@/utils/algorithmDisplayName.js";
import { displayBenchmarkDataType } from "@/utils/benchmarkDataTypeDisplay.js";
import { HIDE_LOSSY_ALGORITHMS } from "@/constants/hideLossyAlgorithms.js";
import {
  methodHasOperatorsInFamily,
  pipelineOperatorGroups,
  operatorsUsedByMethods,
  pipelineOperatorIds,
  stepsForOperatorFamily,
} from "@/utils/methodOperatorMatrix.js";
import {
  OPERATOR_CLASSES,
  OPERATOR_COUNT,
  sortOperatorsByTaxonomy,
} from "@/utils/operatorTaxonomy.js";

const loading = ref(false);
const allMethods = ref([]);
const encodingOperators = ref([]);
const byteCodecOperators = ref([]);
const hoveredValuePipelineMethod = ref(null);
const hoveredBytePipelineMethod = ref(null);
const valueAutomatonStepIndex = ref(0);
const valueAutomatonComplete = ref(false);
const byteAutomatonStepIndex = ref(0);
const byteAutomatonComplete = ref(false);
const compression_scope = ref(HIDE_LOSSY_ALGORITHMS ? "lossless" : "all");
const data_type = ref("all");
const METHODS_TABLE_MAX_HEIGHT_PX = 450;
const PIPELINE_METHOD_COL_WIDTH_PX = 220;
const OPERATOR_COL_WIDTH_PX = 32;
const OPERATOR_COL_MIN_WIDTH_PX = 18;
const OPERATOR_CATEGORY_HEADER_HEIGHT_PX = 116;
const OPERATOR_HEADER_HEIGHT_PX = 124;
/** Half of global table body row height (theme.css sets td to 48px). */
const PIPELINE_BODY_ROW_HEIGHT_PX = 24;
const METHODS_REFRESH_MIN_INTERVAL_MS = 750;
const AUTOMATON_STEP_INTERVAL_MS = 1000;
const HIDDEN_METHOD_TABLE_NAMES = new Set(["col_bss_lz4"]);
const HIDDEN_VALUE_PIPELINE_OPERATOR_IDS = new Set(["rd"]);
const VALUE_PIPELINE_OPERATOR_DISPLAY_NAMES = {
  eraser: "eraser",
  ease: "eraser",
  pongo_eraser: "eraser",
  rle: "run length",
  run_length: "run length",
};
let lastMethodsLoadAt = 0;
let valueAutomatonTimer = null;
let byteAutomatonTimer = null;

const automatonStates = [
  { id: "float", label: "float", x: 270, y: 74, r: 36 },
  { id: "int", label: "int", x: 130, y: 250, r: 36 },
  { id: "byte", label: "byte", x: 410, y: 250, r: 36 },
];

const automatonStartViews = [
  { id: "int", state: "int", title: "Operators automaton: int start" },
  { id: "float", state: "float", title: "Operators automaton: float start" },
];

const automatonEdges = [
  {
    id: "float-to-int",
    path: "M 248 103 C 206 139 170 184 148 219",
    label: { x: 142, y: 137, width: 126, height: 38, text: "scaling / dictionary / run length" },
  },
  {
    id: "int-to-byte",
    path: "M 166 250 C 240 294 322 294 374 250",
    label: { x: 220, y: 285, width: 120, height: 44, text: "bitpacking / dictionary / output" },
  },
  {
    id: "float-to-byte",
    path: "M 292 103 C 335 139 374 184 392 219",
    label: { x: 320, y: 137, width: 120, height: 40, text: "eraser / bitpacking / dictionary / output" },
  },
  {
    id: "byte-to-int",
    path: "M 374 236 C 304 202 226 202 166 236",
    label: { x: 228, y: 201, width: 94, height: 26, text: "dictionary / run length" },
  },
  {
    id: "int-self",
    loop: true,
    path: "M 109 220 C 36 151 36 334 127 286",
    label: { x: 30, y: 248, width: 126, height: 36, text: "for / delta / xor / dictionary / zigzag / outlier / run length" },
  },
  {
    id: "float-self",
    loop: true,
    path: "M 247 47 C 178 -20 362 -20 293 47",
    label: { x: 207, y: 4, width: 118, height: 34, text: "for / delta / xor / outlier / run length" },
  },
  {
    id: "byte-self",
    loop: true,
    path: "M 431 220 C 504 151 504 334 413 286",
    label: { x: 378, y: 248, width: 132, height: 36, text: "for / delta / xor / bitpacking / trailing-zero / outlier / run length" },
  },
];

const automatonEdgeById = new Map(automatonEdges.map((edge) => [edge.id, edge]));

const automatonSelfEdges = {
  int: "int-self",
  float: "float-self",
  byte: "byte-self",
};

const byteAutomatonStates = [
  { id: "float", label: "float", x: 112, y: 74, r: 34 },
  { id: "int", label: "int", x: 112, y: 210, r: 34 },
  { id: "byte", label: "byte", x: 390, y: 142, r: 42 },
];

const byteAutomatonEdges = [
  {
    id: "byte-float-to-byte",
    path: "M 145 82 C 222 78 284 94 350 121",
    label: { x: 206, y: 58, width: 132, height: 34, text: "first byte-codec step" },
  },
  {
    id: "byte-int-to-byte",
    path: "M 145 202 C 222 206 284 190 350 163",
    label: { x: 206, y: 198, width: 132, height: 34, text: "first byte-codec step" },
  },
  {
    id: "byte-input-to-byte",
    loop: true,
    path: "M 350 163 C 246 254 534 254 430 163",
    label: { x: 300, y: 230, width: 174, height: 34, text: "first byte-codec step" },
  },
  {
    id: "byte-self",
    loop: true,
    path: "M 357 116 C 278 28 502 28 423 116",
    label: { x: 296, y: 26, width: 190, height: 34, text: "byte-codec operators" },
  },
];

const byteAutomatonEdgeById = new Map(byteAutomatonEdges.map((edge) => [edge.id, edge]));

function normalizeValuePipelineOperator(row) {
  const displayName = VALUE_PIPELINE_OPERATOR_DISPLAY_NAMES[row.id];
  return displayName ? { ...row, name: displayName } : row;
}

function methodSupportsDataType(row, dtype) {
  const types = (row.data_types || [])
    .map((t) => String(t).trim().toLowerCase())
    .filter(Boolean);
  return types.includes(String(dtype).trim().toLowerCase());
}

function isVisibleMethodDataType(dtype) {
  const normalized = String(dtype || "").trim().toLowerCase();
  return normalized === "int" || normalized === "float";
}

function visibleMethodDataTypes(row) {
  return (row?.data_types || []).filter(isVisibleMethodDataType);
}

function methodHasValueBasedOperators(row) {
  return methodHasOperatorsInFamily(
    row,
    columnOperatorIds.value,
    dtypeFilter.value,
  );
}

function methodHasByteBasedOperators(row) {
  return methodHasOperatorsInFamily(
    row,
    byteOperatorIds.value,
    dtypeFilter.value,
  );
}

const dtypeFilter = computed(() =>
  data_type.value === "all" ? "" : data_type.value,
);

const fullAutomatonRunsByStart = computed(() => {
  const row = hoveredValuePipelineMethod.value;
  const out = {};
  for (const view of automatonStartViews) {
    out[view.state] = row ? buildAutomatonRun(row, view.state) : [];
  }
  return out;
});

const fullAutomatonRun = computed(() => {
  const runsByStart = fullAutomatonRunsByStart.value;
  return automatonStartViews.flatMap((view) => runsByStart[view.state] || []);
});

const isAutomatonFocused = computed(() => hoveredValuePipelineMethod.value != null);

const activeAutomatonRun = computed(() =>
  activeRunSlice(
    fullAutomatonRun.value,
    valueAutomatonStepIndex.value,
    valueAutomatonComplete.value,
  ),
);

const maxAutomatonStepCount = computed(() =>
  Math.max(
    0,
    ...automatonStartViews.map((view) =>
      automatonStepCount(fullAutomatonRunForStart(view.state)),
    ),
  ),
);

const fullByteAutomatonRun = computed(() => {
  const row = hoveredBytePipelineMethod.value;
  if (!row) {
    return [];
  }
  return buildByteAutomatonRun(row);
});

const isByteAutomatonFocused = computed(() => hoveredBytePipelineMethod.value != null);

function filteredRunForView(run, viewId) {
  return run.filter((step) => step.route === viewId);
}

function activeRunSliceForView(run, viewId, stepIndex, complete) {
  const viewRun = filteredRunForView(run, viewId);
  if (!viewRun.length) {
    return [];
  }
  if (complete) {
    return viewRun;
  }
  const stepNumber = Math.min(stepIndex + 1, automatonStepCount(viewRun));
  return viewRun.filter((step) => step.stepNumber === stepNumber);
}

function activeRunSlice(run, stepIndex, complete) {
  if (!run.length) {
    return [];
  }
  if (complete) {
    return run;
  }
  const stepNumber = Math.min(stepIndex + 1, automatonStepCount(run));
  return run.filter((step) => step.stepNumber === stepNumber);
}

function fullAutomatonRunForStart(startState) {
  return fullAutomatonRunsByStart.value[startState] || [];
}

function activeAutomatonRunForStart(startState) {
  return activeRunSlice(
    fullAutomatonRunForStart(startState),
    valueAutomatonStepIndex.value,
    valueAutomatonComplete.value,
  );
}

function activeAutomatonRunForView(viewId) {
  return activeRunSliceForView(
    fullAutomatonRun.value,
    viewId,
    valueAutomatonStepIndex.value,
    valueAutomatonComplete.value,
  );
}

function activeByteAutomatonRunForView(viewId) {
  return activeRunSliceForView(
    fullByteAutomatonRun.value,
    viewId,
    byteAutomatonStepIndex.value,
    byteAutomatonComplete.value,
  );
}

function activeStartStateIdsForRun(run) {
  const ids = new Set();
  const firstStep = Math.min(...run.map((step) => Number(step.stepNumber) || 0));
  for (const step of run) {
    if (step.stepNumber === firstStep) {
      ids.add(step.from);
    }
  }
  return ids;
}

function activeFinalStateIdsForRun(run) {
  const ids = new Set();
  const finalStepNumber = automatonStepCount(run);
  for (const step of run) {
    if (step.stepNumber === finalStepNumber) {
      ids.add(step.to);
    }
  }
  return ids;
}

function activeEdgeIdsForRun(run) {
  return new Set(run.map((step) => step.edgeId));
}

function activeStateIdsForRun(run) {
  const ids = new Set();
  for (const step of run) {
    ids.add(step.from);
    ids.add(step.to);
  }
  return ids;
}

const automatonCaption = computed(() => {
  const row = hoveredValuePipelineMethod.value;
  if (!row) {
    return "move over a method to inspect its operator pipeline.";
  }
  const run = fullAutomatonRun.value;
  if (!run.length) {
    return `${displayMethodCatalogName(row.name)} has no operator path for the current filter.`;
  }
  if (!valueAutomatonComplete.value) {
    const stepNumber = Math.min(
      valueAutomatonStepIndex.value + 1,
      automatonStepCount(run),
    );
    const steps = run.filter((step) => step.stepNumber === stepNumber);
    const ops = steps.map(displayAutomatonStepLabel).join(" | ");
    const paths = steps
      .map((step) => `${displayAutomatonState(step.from)} -> ${displayAutomatonState(step.to)}`)
      .join(" | ");
    return `${displayMethodCatalogName(row.name)}: step ${stepNumber}/${automatonStepCount(run)}, ${paths} via ${ops}.`;
  }
  const ops = [...new Set(run.map(displayAutomatonStepLabel))].join(" -> ");
  return `${displayMethodCatalogName(row.name)}: ${automatonStepCount(run)} steps via ${ops}.`;
});

function automatonCaptionForView() {
  return automatonCaption.value;
}

function automatonCaptionForStart(startState) {
  const row = hoveredValuePipelineMethod.value;
  if (!row) {
    return `move over a method to inspect the ${startState} start pipeline.`;
  }
  const run = fullAutomatonRunForStart(startState);
  if (!run.length) {
    return `${displayMethodCatalogName(row.name)} has no operator path from ${startState} start for the current filter.`;
  }
  if (!valueAutomatonComplete.value) {
    const stepNumber = Math.min(
      valueAutomatonStepIndex.value + 1,
      automatonStepCount(run),
    );
    const steps = run.filter((step) => step.stepNumber === stepNumber);
    const ops = steps.map(displayAutomatonStepLabel).join(" | ");
    const paths = steps
      .map((step) => `${displayAutomatonState(step.from)} -> ${displayAutomatonState(step.to)}`)
      .join(" | ");
    return `${displayMethodCatalogName(row.name)}: step ${stepNumber}/${automatonStepCount(run)}, ${paths} via ${ops}.`;
  }
  const ops = [...new Set(run.map(displayAutomatonStepLabel))].join(" -> ");
  return `${displayMethodCatalogName(row.name)}: ${automatonStepCount(run)} steps from ${startState} start via ${ops}.`;
}

function byteAutomatonCaptionForView(viewId) {
  const row = hoveredBytePipelineMethod.value;
  if (!row) {
    return "move over a byte-based method to inspect this input pipeline.";
  }
  const run = filteredRunForView(fullByteAutomatonRun.value, viewId);
  if (!run.length) {
    return `${displayMethodCatalogName(row.name)} has no byte-operator path for the current filter.`;
  }
  if (!byteAutomatonComplete.value) {
    const stepNumber = Math.min(
      byteAutomatonStepIndex.value + 1,
      automatonStepCount(run),
    );
    const steps = run.filter((step) => step.stepNumber === stepNumber);
    const ops = [...new Set(steps.map((step) => displayByteAutomatonOperator(step.operatorId)))].join(" | ");
    const paths = steps
      .map((step) => `${displayAutomatonState(step.from)} -> ${displayAutomatonState(step.to)}`)
      .join(" | ");
    return `${displayMethodCatalogName(row.name)}: step ${stepNumber}/${automatonStepCount(run)}, ${paths} via ${ops}.`;
  }
  const ops = run.map((step) => displayByteAutomatonOperator(step.operatorId)).join(" -> ");
  return `${displayMethodCatalogName(row.name)}: ${viewId} -> byte via ${ops}.`;
}

const allColumnEncodingOperators = computed(() =>
  encodingOperators.value
    .filter((row) => row.category !== "byte_codec")
    .map(normalizeValuePipelineOperator),
);

const columnEncodingOperators = computed(() =>
  allColumnEncodingOperators.value.filter(
    (row) => !HIDDEN_VALUE_PIPELINE_OPERATOR_IDS.has(row.id),
  ),
);

const columnOperatorIds = computed(
  () => new Set(columnEncodingOperators.value.map((row) => row.id)),
);

const byteCodecOperatorRows = computed(() => {
  if (byteCodecOperators.value.length) {
    return byteCodecOperators.value;
  }
  return encodingOperators.value.filter((row) => row.category === "byte_codec");
});

const byteOperatorIds = computed(
  () => new Set(byteCodecOperatorRows.value.map((row) => row.id)),
);

const showPipelineMatrix = computed(() => true);

const compressionOperatorsPanelDesc = computed(() => {
  return `${OPERATOR_COUNT} operators in ${OPERATOR_CLASSES.length} functional classes, using the same visible names as the paper.`;
});

const valueBasedMatrixMethods = computed(() =>
  filteredMethods.value.filter((row) => methodHasValueBasedOperators(row)),
);

const byteBasedMatrixMethods = computed(() =>
  [],
);

const combinedPipelineMatrixMethods = computed(() => filteredMethods.value);

const valueBasedMatrixOperators = computed(() => columnEncodingOperators.value);

const compressionMatrixOperators = computed(() =>
  byteCodecOperatorRows.value,
);

const pipelineMatrixOperators = computed(() =>
  sortOperatorsByTaxonomy([
    ...valueBasedMatrixOperators.value,
    ...compressionMatrixOperators.value,
  ]),
);

const pipelineMatrixOperatorGroups = computed(() => {
  const operatorsById = new Map(
    pipelineMatrixOperators.value.map((row) => [row.id, row]),
  );
  const seen = new Set();
  const groups = [];
  for (const group of OPERATOR_CLASSES) {
    const operators = group.operatorIds
      .map((operatorId) => operatorsById.get(operatorId))
      .filter(Boolean)
      .sort((left, right) =>
        left.orderIndex - right.orderIndex ||
        left.operatorIndex - right.operatorIndex ||
        String(left.name || left.id).localeCompare(String(right.name || right.id)),
      );
    if (!operators.length) {
      continue;
    }
    operators.forEach((row) => seen.add(row.id));
    groups.push({
      ...group,
      operators,
      width: operators.length * OPERATOR_COL_WIDTH_PX,
    });
  }
  const uncategorized = pipelineMatrixOperators.value.filter((row) => !seen.has(row.id));
  if (uncategorized.length) {
    groups.push({
      id: "uncategorized",
      label: "Uncategorized",
      operators: uncategorized,
      width: uncategorized.length * OPERATOR_COL_WIDTH_PX,
    });
  }
  return groups;
});

const pipelineMatrixOperatorIds = computed(
  () => new Set(pipelineMatrixOperatorGroups.value.flatMap((group) => group.operators.map((row) => row.id))),
);

const operatorReferenceRows = computed(() =>
  sortOperatorsByTaxonomy([
    ...columnEncodingOperators.value,
    ...byteCodecOperatorRows.value,
  ]),
);

function operatorGoodRatioExample(row) {
  return row?.good_ratio_example || row?.numeric_example || "—";
}

function operatorPoorRatioExample(row) {
  return row?.poor_ratio_example || "—";
}

function operatorPipelineHeaderTitle(row) {
  const detail = row?.example || row?.description || row?.name || "";
  return [row?.classLabel, detail].filter(Boolean).join(": ");
}

const combinedPipelineTableStyle = computed(() => {
  const width =
    PIPELINE_METHOD_COL_WIDTH_PX +
    pipelineMatrixOperators.value.length * OPERATOR_COL_WIDTH_PX +
    4;
  return {
    width: `${width}px`,
    maxWidth: "100%",
  };
});

function pipelineOperatorStep(row, operatorId) {
  const steps = stepsForOperatorFamily(
    pipelineOperatorGroups(row, dtypeFilter.value),
    pipelineMatrixOperatorIds.value,
  );
  const step = steps[operatorId];
  return step != null && step !== "" ? step : "";
}

function valueBasedOperatorStep(row, operatorId) {
  const steps = stepsForOperatorFamily(
    pipelineOperatorGroups(row, dtypeFilter.value),
    columnOperatorIds.value,
  );
  const step = steps[operatorId];
  return step != null && step !== "" ? step : "";
}

function byteBasedOperatorStep(row, operatorId) {
  const steps = stepsForOperatorFamily(
    pipelineOperatorGroups(row, dtypeFilter.value),
    byteOperatorIds.value,
  );
  const step = steps[operatorId];
  return step != null && step !== "" ? step : "";
}

function onValuePipelineMethodEnter(row) {
  if (hoveredValuePipelineMethod.value === row) {
    return;
  }
  hoveredValuePipelineMethod.value = row;
  startValueAutomatonAnimation();
}

function onValuePipelineCellLeave() {
  hoveredValuePipelineMethod.value = null;
  stopValueAutomatonAnimation();
}

function onBytePipelineMethodEnter(row) {
  if (hoveredBytePipelineMethod.value === row) {
    return;
  }
  hoveredBytePipelineMethod.value = row;
  startByteAutomatonAnimation();
}

function onBytePipelineLeave() {
  hoveredBytePipelineMethod.value = null;
  stopByteAutomatonAnimation();
}

function onCombinedPipelineMethodEnter(row) {
  onValuePipelineMethodEnter(row);
}

function onCombinedPipelineLeave() {
  onValuePipelineCellLeave();
  onBytePipelineLeave();
}

function stopValueAutomatonAnimation() {
  if (valueAutomatonTimer != null) {
    window.clearInterval(valueAutomatonTimer);
    valueAutomatonTimer = null;
  }
  valueAutomatonStepIndex.value = 0;
  valueAutomatonComplete.value = false;
}

function stopByteAutomatonAnimation() {
  if (byteAutomatonTimer != null) {
    window.clearInterval(byteAutomatonTimer);
    byteAutomatonTimer = null;
  }
  byteAutomatonStepIndex.value = 0;
  byteAutomatonComplete.value = false;
}

function startValueAutomatonAnimation() {
  stopValueAutomatonAnimation();
  const stepCount = maxAutomatonStepCount.value;
  if (!stepCount) {
    return;
  }
  valueAutomatonTimer = window.setInterval(() => {
    if (valueAutomatonStepIndex.value < stepCount - 1) {
      valueAutomatonStepIndex.value += 1;
      return;
    }
    valueAutomatonComplete.value = true;
    window.clearInterval(valueAutomatonTimer);
    valueAutomatonTimer = null;
  }, AUTOMATON_STEP_INTERVAL_MS);
}

function startByteAutomatonAnimation() {
  stopByteAutomatonAnimation();
  const stepCount = automatonStepCount(fullByteAutomatonRun.value);
  if (!stepCount) {
    return;
  }
  byteAutomatonTimer = window.setInterval(() => {
    if (byteAutomatonStepIndex.value < stepCount - 1) {
      byteAutomatonStepIndex.value += 1;
      return;
    }
    byteAutomatonComplete.value = true;
    window.clearInterval(byteAutomatonTimer);
    byteAutomatonTimer = null;
  }, AUTOMATON_STEP_INTERVAL_MS);
}

function resetAutomatonHoverState() {
  hoveredValuePipelineMethod.value = null;
  hoveredBytePipelineMethod.value = null;
  stopValueAutomatonAnimation();
  stopByteAutomatonAnimation();
}

function displayAutomatonState(stateId) {
  return String(stateId || "").replace("_", "+");
}

function displayAutomatonOperator(operatorId) {
  if (operatorId === "output_bytes") {
    return "output";
  }
  if (operatorId === "outlier_separation") {
    return "outlier";
  }
  if (operatorId === "trailing_zero_pack") {
    return "trailing-zero";
  }
  const found = pipelineMatrixOperators.value.find((row) => row.id === operatorId)
    || operatorReferenceRows.value.find((row) => row.id === operatorId);
  return found?.name || operatorId;
}

function displayAutomatonStepLabel(step) {
  if (["rle", "run_length"].includes(step?.operatorId)) {
    if (step.rleStream === "count") {
      return "int run-length";
    }
    if (step.rleStream === "value") {
      return `${displayAutomatonState(step.rleValueType || step.to)} value`;
    }
  }
  return displayAutomatonOperator(step?.operatorId);
}

function displayByteAutomatonOperator(operatorId) {
  const found = byteCodecOperatorRows.value.find((row) => row.id === operatorId);
  return found?.name || operatorId;
}

function automatonEdgePath(edgeId) {
  return automatonEdgeById.get(edgeId)?.path || "";
}

function byteAutomatonEdgePath(edgeId) {
  return byteAutomatonEdgeById.get(edgeId)?.path || "";
}

function isActiveAutomatonEdge(edgeId) {
  return activeEdgeIdsForRun(activeAutomatonRun.value).has(edgeId);
}

function isActiveAutomatonEdgeForStart(startState, edgeId) {
  return activeEdgeIdsForRun(activeAutomatonRunForStart(startState)).has(edgeId);
}

function isActiveByteAutomatonEdge(viewId, edgeId) {
  return activeEdgeIdsForRun(activeByteAutomatonRunForView(viewId)).has(edgeId);
}

function isActiveAutomatonState(stateId) {
  return activeStateIdsForRun(activeAutomatonRun.value).has(stateId);
}

function isActiveAutomatonStateForStart(startState, stateId) {
  return activeStateIdsForRun(activeAutomatonRunForStart(startState)).has(stateId);
}

function isActiveByteAutomatonState(viewId, stateId) {
  return activeStateIdsForRun(activeByteAutomatonRunForView(viewId)).has(stateId);
}

function isActiveAutomatonStartState(stateId) {
  return activeStartStateIdsForRun(activeAutomatonRun.value).has(stateId);
}

function isActiveAutomatonStartStateForStart(startState, stateId) {
  if (!isAutomatonFocused.value || !fullAutomatonRunForStart(startState).length) {
    return stateId === startState;
  }
  return activeStartStateIdsForRun(activeAutomatonRunForStart(startState)).has(stateId);
}

function isActiveByteAutomatonStartState(viewId, stateId) {
  return activeStartStateIdsForRun(activeByteAutomatonRunForView(viewId)).has(stateId);
}

function isActiveAutomatonFinalState(stateId) {
  return activeFinalStateIdsForRun(activeAutomatonRun.value).has(stateId);
}

function isActiveAutomatonFinalStateForStart(startState, stateId) {
  return activeFinalStateIdsForRun(activeAutomatonRunForStart(startState)).has(stateId);
}

function isActiveByteAutomatonFinalState(viewId, stateId) {
  return activeFinalStateIdsForRun(activeByteAutomatonRunForView(viewId)).has(stateId);
}

function defaultAutomatonEdgeIdsForView(viewId) {
  if (viewId === "float") {
    return new Set(["float-to-int", "int-self", "int-to-byte"]);
  }
  return new Set(["int-self", "int-to-byte"]);
}

function defaultAutomatonStateIdsForView(viewId) {
  if (viewId === "float") {
    return new Set(["float", "int", "byte"]);
  }
  return new Set(["int", "byte"]);
}

function shouldShowAutomatonEdge(edgeId) {
  return isAutomatonFocused.value
    ? isActiveAutomatonEdge(edgeId)
    : true;
}

function shouldShowAutomatonEdgeForStart(startState, edgeId) {
  if (!isAutomatonFocused.value || !fullAutomatonRunForStart(startState).length) {
    return true;
  }
  return isActiveAutomatonEdgeForStart(startState, edgeId);
}

function shouldShowByteAutomatonEdge(viewId, edgeId) {
  if (isByteAutomatonFocused.value) {
    return isActiveByteAutomatonEdge(viewId, edgeId);
  }
  if (viewId === "float") {
    return edgeId === "byte-float-to-byte" || edgeId === "byte-self";
  }
  if (viewId === "int") {
    return edgeId === "byte-int-to-byte" || edgeId === "byte-self";
  }
  return edgeId === "byte-input-to-byte" || edgeId === "byte-self";
}

function shouldShowAutomatonState(stateId) {
  return isAutomatonFocused.value
    ? isActiveAutomatonState(stateId)
    : true;
}

function shouldShowAutomatonStateForStart(startState, stateId) {
  if (!isAutomatonFocused.value || !fullAutomatonRunForStart(startState).length) {
    return true;
  }
  return isActiveAutomatonStateForStart(startState, stateId);
}

function shouldShowByteAutomatonState(viewId, stateId) {
  if (isByteAutomatonFocused.value) {
    return isActiveByteAutomatonState(viewId, stateId);
  }
  return viewId === stateId || stateId === "byte";
}

function automatonEdgeLabelText(edge) {
  if (!isAutomatonFocused.value || !isActiveAutomatonEdge(edge.id)) {
    return edge.label.text;
  }
  const activeOperators = activeAutomatonRun.value
    .filter((step) => step.edgeId === edge.id)
    .map(displayAutomatonStepLabel);
  return activeOperators.join(" / ") || edge.label.text;
}

function automatonEdgeLabelTextForStart(startState, edge) {
  if (
    !isAutomatonFocused.value ||
    !fullAutomatonRunForStart(startState).length ||
    !isActiveAutomatonEdgeForStart(startState, edge.id)
  ) {
    return edge.label.text;
  }
  const activeOperators = activeAutomatonRunForStart(startState)
    .filter((step) => step.edgeId === edge.id)
    .map(displayAutomatonStepLabel);
  return activeOperators.join(" / ") || edge.label.text;
}

function automatonRunLabel(run, index) {
  return `${run.stepNumber || index + 1} ${displayAutomatonStepLabel(run)}`;
}

function automatonStateForDataType(dtype) {
  const normalized = String(dtype || "").trim().toLowerCase();
  if (normalized === "float") {
    return "float";
  }
  if (normalized === "string" || normalized === "byte") {
    return "byte";
  }
  return "int";
}

function dtypeForAutomatonStartState(startState) {
  return startState === "byte" ? "string" : startState;
}

function methodSupportsAutomatonStartState(row, startState) {
  if (startState === "int") {
    return true;
  }
  return methodSupportsDataType(row, dtypeForAutomatonStartState(startState));
}

function operatorAutomatonStartStates(row) {
  if (dtypeFilter.value === "int") {
    return ["int"];
  }
  if (dtypeFilter.value === "float") {
    return ["float"];
  }
  if (dtypeFilter.value === "string") {
    return ["byte"];
  }
  const states = new Set(
    visibleMethodDataTypes(row)
      .map(automatonStateForDataType)
      .filter(Boolean),
  );
  return states.size ? [...states] : ["float", "int"];
}

function rleTransitionsForState(state) {
  const valueState = state === "float" || state === "byte" ? state : "int";
  const countEdgeId =
    valueState === "float"
      ? "float-to-int"
      : valueState === "byte"
        ? "byte-to-int"
        : "int-self";
  const valueEdgeId = automatonSelfEdges[valueState] || "int-self";
  return [
    {
      to: "int",
      edgeId: countEdgeId,
      rleStream: "count",
      rleValueType: "int",
    },
    {
      to: valueState,
      edgeId: valueEdgeId,
      rleStream: "value",
      rleValueType: valueState,
    },
  ];
}

function transitionForOperator(operatorId, state, options = {}) {
  const terminal = Boolean(options.terminal);
  const selfEdge = automatonSelfEdges[state] || "int-self";
  if (operatorId === "scaling") {
    return state === "float" ? { to: "int", edgeId: "float-to-int" } : null;
  }
  if (operatorId === "eraser" || operatorId === "ease" || operatorId === "pongo_eraser") {
    return state === "float"
      ? { to: "byte", edgeId: "float-to-byte" }
      : { to: state, edgeId: selfEdge };
  }
  if (
    operatorId === "frame_of_reference" ||
    operatorId === "delta" ||
    operatorId === "rd" ||
    operatorId === "outlier_separation"
  ) {
    return { to: state, edgeId: selfEdge };
  }
  if (operatorId === "xor") {
    return { to: state, edgeId: selfEdge };
  }
  if (
    operatorId === "bitpacking" ||
    operatorId === "bitpack_int" ||
    operatorId === "trailing_zero_pack"
  ) {
    if (state === "float") {
      return { to: "byte", edgeId: "float-to-byte" };
    }
    if (state === "byte") {
      return { to: "byte", edgeId: "byte-self" };
    }
    return { to: "byte", edgeId: "int-to-byte" };
  }
  if (operatorId === "dictionary") {
    if (terminal) {
      if (state === "float") {
        return { to: "byte", edgeId: "float-to-byte" };
      }
      return { to: "byte", edgeId: state === "byte" ? "byte-self" : "int-to-byte" };
    }
    if (state === "byte") {
      return { to: "int", edgeId: "byte-to-int" };
    }
    if (state === "float") {
      return { to: "int", edgeId: "float-to-int" };
    }
    return { to: "int", edgeId: "int-self" };
  }
  if (operatorId === "zigzag") {
    return { to: "int", edgeId: "int-self" };
  }
  if (byteOperatorIds.value.has(operatorId)) {
    if (state === "float") {
      return { to: "byte", edgeId: "float-to-byte" };
    }
    if (state === "byte") {
      return { to: "byte", edgeId: "byte-self" };
    }
    return { to: "byte", edgeId: "int-to-byte" };
  }
  return { to: state, edgeId: selfEdge };
}

function transitionsForOperator(operatorId, state, options = {}) {
  if (["rle", "run_length"].includes(operatorId)) {
    return rleTransitionsForState(state);
  }
  const transition = transitionForOperator(operatorId, state, options);
  return transition ? [transition] : [];
}

function automatonBadgeForEdge(edgeId) {
  const edge = automatonEdgeById.get(edgeId);
  const label = edge?.label;
  if (!label) {
    return { x: 0, y: 0 };
  }
  return {
    x: label.x + Math.max(0, label.width - 34) / 2,
    y: label.y + label.height + 2,
  };
}

function automatonRunLayoutForStep(edgeId, occurrenceIndex) {
  const gridLayouts = {
    "int-self": [
      { x: 28, y: 136, width: 102 },
      { x: 28, y: 164, width: 102 },
      { x: 28, y: 192, width: 102 },
      { x: 28, y: 304, width: 102 },
      { x: 28, y: 332, width: 102 },
    ],
    "float-self": [
      { x: 126, y: 8, width: 96 },
      { x: 222, y: 8, width: 96 },
      { x: 318, y: 8, width: 96 },
    ],
    "byte-self": [
      { x: 410, y: 150, width: 108 },
      { x: 410, y: 178, width: 108 },
      { x: 410, y: 206, width: 108 },
      { x: 410, y: 326, width: 108 },
      { x: 410, y: 298, width: 108 },
    ],
    "float-to-byte": [
      { x: 286, y: 118, width: 104 },
      { x: 304, y: 150, width: 104 },
      { x: 322, y: 182, width: 104 },
    ],
    "int-to-byte": [
      { x: 190, y: 292, width: 112 },
      { x: 314, y: 292, width: 112 },
      { x: 252, y: 320, width: 112 },
    ],
    "float-to-int": [
      { x: 154, y: 118, width: 104 },
      { x: 136, y: 150, width: 104 },
    ],
    "byte-to-int": [
      { x: 204, y: 188, width: 112 },
      { x: 320, y: 188, width: 112 },
    ],
  };
  const positions = gridLayouts[edgeId];
  const fallback = automatonBadgeForEdge(edgeId);
  const base = positions?.[occurrenceIndex] || positions?.[positions.length - 1] || {
    x: fallback.x + 26,
    y: fallback.y,
  };
  const extraOffset = positions && occurrenceIndex >= positions.length
    ? (occurrenceIndex - positions.length + 1) * 26
    : 0;
  const label = {
    x: base.x,
    y: base.y + extraOffset,
    width: base.width || 104,
    height: 22,
  };
  return {
    label,
    badge: {
      x: label.x - 26,
      y: label.y,
    },
  };
}

function layoutAutomatonRun(run) {
  const edgeCounts = {};
  return run.map((step) => {
    const occurrenceIndex = edgeCounts[step.edgeId] || 0;
    edgeCounts[step.edgeId] = occurrenceIndex + 1;
    return {
      ...step,
      ...automatonRunLayoutForStep(step.edgeId, occurrenceIndex),
    };
  });
}

function normalizeRouteStepNumbers(run) {
  const routeMaps = new Map();
  return run.map((step) => {
    const route = step.route || "";
    if (!routeMaps.has(route)) {
      routeMaps.set(route, new Map());
    }
    const map = routeMaps.get(route);
    if (!map.has(step.stepNumber)) {
      map.set(step.stepNumber, map.size + 1);
    }
    return {
      ...step,
      stepNumber: map.get(step.stepNumber),
    };
  });
}

function automatonStepCount(run) {
  return Math.max(0, ...run.map((step) => Number(step.stepNumber) || 0));
}

function outputTransitionForState(state) {
  if (state === "float") {
    return { to: "byte", edgeId: "float-to-byte" };
  }
  if (state === "byte") {
    return { to: "byte", edgeId: "byte-self" };
  }
  return { to: "byte", edgeId: "int-to-byte" };
}

function byteAutomatonRunLayoutForStep(index) {
  const positions = {
    "byte-float-to-byte": [
      { x: 174, y: 106 },
    ],
    "byte-int-to-byte": [
      { x: 174, y: 160 },
    ],
    "byte-input-to-byte": [
      { x: 296, y: 222 },
    ],
    "byte-self": [
      { x: 270, y: 72 },
      { x: 384, y: 72 },
      { x: 270, y: 98 },
      { x: 384, y: 98 },
      { x: 270, y: 230 },
      { x: 384, y: 230 },
    ],
  };
  const edgeId = typeof index === "string" ? index : "byte-self";
  const occurrenceIndex = typeof index === "string" ? 0 : index;
  const candidates = positions[edgeId] || positions["byte-self"];
  const base = candidates[occurrenceIndex % candidates.length];
  const rowOffset = Math.floor(occurrenceIndex / candidates.length) * 26;
  const label = {
    x: base.x,
    y: Math.min(252, base.y + rowOffset),
    width: 122,
    height: 22,
  };
  return {
    label,
    badge: {
      x: label.x - 26,
      y: label.y,
    },
  };
}

function byteAutomatonRunLayout(edgeId, occurrenceIndex) {
  return byteAutomatonRunLayoutForStep(edgeId, occurrenceIndex);
}

function buildByteAutomatonRun(row) {
  const pipeline = pipelineOperatorIds(row, dtypeFilter.value).filter((operatorId) =>
    byteOperatorIds.value.has(operatorId),
  );
  if (!pipeline.length) {
    return [];
  }
  const run = [
    {
      from: "float",
      to: "byte",
      operatorId: pipeline[0],
      edgeId: "byte-float-to-byte",
      route: "float",
      stepNumber: 1,
      ...byteAutomatonRunLayout("byte-float-to-byte", 0),
    },
    {
      from: "int",
      to: "byte",
      operatorId: pipeline[0],
      edgeId: "byte-int-to-byte",
      route: "int",
      stepNumber: 1,
      ...byteAutomatonRunLayout("byte-int-to-byte", 0),
    },
    {
      from: "byte",
      to: "byte",
      operatorId: pipeline[0],
      edgeId: "byte-input-to-byte",
      route: "byte",
      stepNumber: 1,
      ...byteAutomatonRunLayout("byte-input-to-byte", 0),
    },
  ];
  pipeline.slice(1).forEach((operatorId, index) => {
    ["float", "int", "byte"].forEach((route, routeIndex) => {
      run.push({
        from: "byte",
        to: "byte",
        operatorId,
        edgeId: "byte-self",
        route,
        stepNumber: index + 2,
        ...byteAutomatonRunLayout("byte-self", index * 3 + routeIndex + 1),
      });
    });
  });
  return run;
}

function buildRleAutomatonRun() {
  const rleTransitions = ["int", "float", "byte"].flatMap((state) =>
    rleTransitionsForState(state).map((transition) => ({
      from: state,
      to: transition.to,
      edgeId: transition.edgeId,
      route: state,
      stepNumber: 1,
      operatorId: "run_length",
      rleStream: transition.rleStream,
      rleValueType: transition.rleValueType,
    })),
  );
  return layoutAutomatonRun(rleTransitions.map((transition) => ({
    ...transition,
    badge: automatonBadgeForEdge(transition.edgeId),
  })));
}

function buildAutomatonRun(row, startState = "") {
  const fixedStartState = String(startState || "").trim().toLowerCase();
  if (fixedStartState && !methodSupportsAutomatonStartState(row, fixedStartState)) {
    return [];
  }
  const pipelineDtype = fixedStartState
    ? dtypeForAutomatonStartState(fixedStartState)
    : dtypeFilter.value;
  const groups = pipelineOperatorGroups(row, pipelineDtype)
    .map((group) => ({
      ...group,
      operatorIds: group.operatorIds.filter((operatorId) =>
        pipelineMatrixOperatorIds.value.has(operatorId),
      ),
    }))
    .filter((group) => group.operatorIds.length);
  const run = [];
  let states = (fixedStartState ? [fixedStartState] : operatorAutomatonStartStates(row))
    .map((state) => ({ state, route: state }));
  let stepNumber = 0;
  groups.forEach((group, groupIndex) => {
    stepNumber += 1;
    const isTerminalGroup = groupIndex === groups.length - 1;
    const nextStates = [];
    for (const current of states) {
      for (const operatorId of group.operatorIds) {
        const transitions = transitionsForOperator(operatorId, current.state, {
          terminal: isTerminalGroup && group.anyOf,
        });
        if (!transitions.length) {
          nextStates.push(current);
          continue;
        }
        for (const transition of transitions) {
          run.push({
            from: current.state,
            to: transition.to,
            operatorId,
            edgeId: transition.edgeId,
            stepNumber,
            route: current.route,
            anyOf: group.anyOf,
            rleStream: transition.rleStream,
            rleValueType: transition.rleValueType,
            badge: automatonBadgeForEdge(transition.edgeId),
          });
          nextStates.push({ state: transition.to, route: current.route });
        }
      }
    }
    states = nextStates;
  });
  const unfinishedStates = states.filter((current) => current.state !== "byte");
  if (groups.length && unfinishedStates.length) {
    stepNumber += 1;
    for (const current of unfinishedStates) {
      const transition = outputTransitionForState(current.state);
      run.push({
        from: current.state,
        to: transition.to,
        operatorId: "output_bytes",
        edgeId: transition.edgeId,
        implicit: true,
        stepNumber,
        route: current.route,
        badge: automatonBadgeForEdge(transition.edgeId),
      });
    }
  }
  return layoutAutomatonRun(normalizeRouteStepNumbers(run));
}

const combinedPipelineMatrixEmptyText = computed(() => {
  if (loading.value) {
    return "Loading methods...";
  }
  if (!filteredMethods.value.length) {
    return "No methods match the current filters.";
  }
  return "No compression methods with operator mappings match the current filters.";
});

const filteredMethods = computed(() => {
  let list = allMethods.value;
  list = list.filter((row) => !HIDDEN_METHOD_TABLE_NAMES.has(row.name));
  if (HIDE_LOSSY_ALGORITHMS) {
    list = list.filter((row) => row.compression_mode === "lossless");
  } else {
    const scope = compression_scope.value;
    if (scope === "lossless") {
      list = list.filter((row) => row.compression_mode === "lossless");
    } else if (scope === "lossy") {
      list = list.filter((row) => row.compression_mode === "lossy");
    }
  }
  const dtype = data_type.value;
  if (dtype !== "all") {
    list = list.filter((row) => methodSupportsDataType(row, dtype));
  } else {
    list = list.filter((row) => visibleMethodDataTypes(row).length);
  }
  return list;
});

const tableEmptyText = computed(() => {
  if (loading.value) {
    return "Loading methods...";
  }
  if (!allMethods.value.length) {
    return "No methods in catalog.";
  }
  return "No methods match the current filters. Try All for data type.";
});

async function loadMethods() {
  lastMethodsLoadAt = Date.now();
  loading.value = true;
  try {
    const { data } = await axios.get("/api/methods/");
    const items = Array.isArray(data?.items) ? data.items : [];
    allMethods.value = items.map((row) => ({
      ...row,
      compression_mode:
        row.compression_mode === "lossy" || row.compression_mode === "lossless"
          ? row.compression_mode
          : compressionModeForMethodName(row.name),
    }));
    encodingOperators.value = Array.isArray(data?.encoding_operators)
      ? data.encoding_operators
      : [];
    byteCodecOperators.value = Array.isArray(data?.byte_codec_operators)
      ? data.byte_codec_operators
      : encodingOperators.value.filter((row) => row.category === "byte_codec");
  } catch (error) {
    console.error(error);
    ElMessage.error("Failed to load method catalog.");
    allMethods.value = [];
    encodingOperators.value = [];
    byteCodecOperators.value = [];
  } finally {
    loading.value = false;
  }
}

function refreshMethodsWhenVisible() {
  if (document.visibilityState === "hidden") {
    return;
  }
  if (Date.now() - lastMethodsLoadAt < METHODS_REFRESH_MIN_INTERVAL_MS) {
    return;
  }
  loadMethods();
}

onMounted(() => {
  loadMethods();
  window.addEventListener("focus", refreshMethodsWhenVisible);
  document.addEventListener("visibilitychange", refreshMethodsWhenVisible);
});

watch(data_type, () => {
  if (data_type.value === "string") {
    data_type.value = "all";
    return;
  }
  resetAutomatonHoverState();
});

onBeforeUnmount(() => {
  resetAutomatonHoverState();
  window.removeEventListener("focus", refreshMethodsWhenVisible);
  document.removeEventListener("visibilitychange", refreshMethodsWhenVisible);
});
</script>

<style scoped>
.paper-link {
  color: var(--bc-cyan, #5ae9ff);
}

.dtype-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}

.methods-tables-stack {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
  max-width: 100%;
}

.methods-info-table-wrap {
  width: 100%;
  min-width: 0;
}

.methods-operator-table-wrap {
  width: 100%;
  min-width: 0;
  padding-top: 0.65rem;
  border-top: 1px solid var(--bc-border-subtle, rgba(120, 170, 255, 0.18));
}

.methods-operator-table-wrap:first-of-type {
  padding-top: 0.15rem;
}

.combined-pipeline-layout {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1rem;
  align-items: start;
}

.combined-pipeline-matrix {
  min-width: 0;
  max-width: 100%;
  grid-column: 1 / -1;
}

.automata-row {
  display: grid;
  grid-column: 1 / -1;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
  width: 100%;
  min-width: 0;
}

.pipeline-method-cell {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.18rem;
  width: 100%;
  min-width: 0;
}

.operator-automaton {
  --automaton-edge: var(--bc-automaton-edge, var(--bc-cyan, #5ae9ff));
  --automaton-edge-active: var(--bc-automaton-edge-active, #ffd664);
  --automaton-label-bg: var(--bc-automaton-label-bg, rgba(12, 18, 30, 0.88));
  --automaton-label-border: var(--bc-automaton-label-border, rgba(120, 170, 255, 0.22));
  --automaton-label-text: var(--bc-automaton-label-text, rgba(220, 230, 255, 0.88));
  --automaton-label-active-bg: var(--bc-automaton-label-active-bg, rgba(52, 38, 12, 0.9));
  --automaton-label-active-border: var(--bc-automaton-label-active-border, rgba(255, 210, 90, 0.62));
  --automaton-label-active-text: var(--bc-automaton-label-active-text, rgba(255, 239, 190, 0.98));
  --automaton-state-bg: var(--bc-automaton-state-bg, rgba(20, 34, 52, 0.96));
  --automaton-state-active-bg: var(--bc-automaton-state-active-bg, rgba(38, 34, 20, 0.98));
  --automaton-state-text: var(--bc-automaton-state-text, rgba(235, 242, 255, 0.96));
  --automaton-badge-bg: var(--bc-automaton-badge-bg, rgba(255, 214, 100, 0.96));
  --automaton-badge-text: var(--bc-automaton-badge-text, rgba(20, 24, 30, 0.96));
  --automaton-glow: var(--bc-automaton-glow, rgba(255, 214, 100, 0.38));
  min-width: 0;
  padding: 0.2rem 0 0;
}

.automaton-svg {
  display: block;
  width: 100%;
  max-width: none;
  height: auto;
  min-height: 320px;
  margin: 0 auto;
}

.byte-automaton-svg {
  max-width: 560px;
  min-height: 230px;
}

.automaton-edge path {
  fill: none;
  stroke: var(--automaton-edge);
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
  marker-end: url("#automaton-arrow");
  opacity: 0.82;
}

.byte-automaton-edge path {
  marker-end: url("#byte-automaton-arrow");
}

.byte-automaton-loop {
  marker-end: url("#byte-automaton-arrow") !important;
}

.automaton-edge.is-active path {
  stroke: var(--automaton-edge-active);
  stroke-width: 2.25;
  opacity: 0.95;
}

.automaton-loop {
  opacity: 0.72;
}

.automaton-edge-label {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 0.16rem 0.32rem;
  border: 1px solid var(--automaton-label-border);
  border-radius: 6px;
  background: var(--automaton-label-bg);
  color: var(--automaton-label-text);
  font-size: 0.58rem;
  line-height: 1.12;
  text-align: center;
  box-sizing: border-box;
}

.automaton-edge-label.is-active {
  border-color: var(--automaton-label-active-border);
  background: var(--automaton-label-active-bg);
  color: var(--automaton-label-active-text);
}

.automaton-run-edge path {
  fill: none;
  stroke: var(--automaton-edge-active);
  stroke-width: 4;
  stroke-linecap: round;
  stroke-linejoin: round;
  marker-end: url("#automaton-arrow");
  stroke-dasharray: 12 11;
  animation: automaton-flow 1.05s linear infinite;
  animation-delay: calc(var(--run-index) * 0.16s);
  filter: drop-shadow(0 0 4px var(--automaton-glow));
}

.byte-automaton-run-edge path {
  marker-end: url("#byte-automaton-arrow");
}

.automaton-run-badge {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  margin: 0 auto;
  border-radius: 999px;
  background: var(--automaton-badge-bg);
  color: var(--automaton-badge-text);
  font-size: 0.68rem;
  font-weight: 700;
  line-height: 1;
}

.automaton-run-label {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  padding: 0 0.34rem;
  border: 1px solid var(--automaton-label-active-border);
  border-radius: 6px;
  background: var(--automaton-label-active-bg);
  color: var(--automaton-label-active-text);
  font-size: 0.58rem;
  font-weight: 650;
  line-height: 1;
  text-align: center;
  box-sizing: border-box;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.automaton-state circle {
  fill: var(--automaton-state-bg);
  stroke: var(--automaton-edge);
  stroke-width: 2;
}

.automaton-state.is-active circle {
  fill: var(--automaton-state-active-bg);
  stroke: var(--automaton-edge-active);
  stroke-width: 3;
  filter: drop-shadow(0 0 5px var(--automaton-glow));
}

.automaton-state.is-start circle {
  stroke-dasharray: 4 3;
}

.automaton-state.is-final circle {
  stroke-width: 4;
}

.automaton-state text {
  fill: var(--automaton-state-text);
  font-size: 0.78rem;
  font-weight: 600;
}

.automaton-caption {
  min-height: 2.45rem;
  margin-top: 0.35rem;
  color: var(--bc-text-muted, rgba(220, 230, 255, 0.74));
  font-size: 0.76rem;
  line-height: 1.35;
}

.automaton-caption-label {
  color: var(--bc-text-primary, rgba(235, 242, 255, 0.92));
  font-weight: 600;
}

@keyframes automaton-flow {
  to {
    stroke-dashoffset: -23;
  }
}

.operators-matrix-title {
  margin: 0 0 0.25rem;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--bc-text-primary, rgba(235, 242, 255, 0.95));
}

.operators-matrix-desc {
  margin: 0 0 0.5rem;
  font-size: 0.82rem;
  line-height: 1.35;
  color: var(--bc-text-muted, rgba(220, 230, 255, 0.72));
}

.operators-reference-stack {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.operator-downloads {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 1rem;
  margin-top: 0.5rem;
  font-size: 0.88rem;
  line-height: 1.4;
}

.operator-downloads a {
  color: #2563eb;
  font-weight: 600;
  text-decoration: none;
}

.operator-downloads a:hover {
  text-decoration: underline;
}

.operator-downloads span {
  flex-basis: 100%;
  color: #606266;
}

.operators-reference-title {
  margin: 0 0 0.55rem;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--bc-text-muted, rgba(220, 230, 255, 0.88));
}

:deep(.operators-reference-stack .el-table .el-table__body-wrapper tbody td.el-table__cell),
:deep(.operators-reference-stack .el-table .el-table__fixed-body-wrapper tbody td.el-table__cell) {
  padding-top: 0.55rem;
  padding-bottom: 0.55rem;
  vertical-align: middle;
}

:deep(.operators-reference-stack .el-table .el-table__body-wrapper tbody td.el-table__cell .cell),
:deep(.operators-reference-stack .el-table .el-table__fixed-body-wrapper tbody td.el-table__cell .cell) {
  display: flex;
  align-items: center;
  min-height: 5.75rem;
}

.operator-header-vertical {
  display: inline-block;
  transform: rotate(-90deg);
  transform-origin: center center;
  white-space: nowrap;
  font-size: 0.875rem;
  line-height: 1.1;
  letter-spacing: 0.02em;
  color: var(--bc-text-muted, rgba(220, 230, 255, 0.88));
}

.operator-step-num {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  color: var(--bc-cyan, #5ae9ff);
}

:deep(.operator-pipeline-matrix-table.el-table th.el-table__cell),
:deep(.operator-pipeline-matrix-table.el-table td.el-table__cell) {
  text-align: center !important;
  vertical-align: middle !important;
}

:deep(.operator-pipeline-matrix-table.el-table th.el-table__cell .cell),
:deep(.operator-pipeline-matrix-table.el-table td.el-table__cell .cell) {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  width: 100% !important;
  height: 100% !important;
  padding: 0 !important;
  text-align: center !important;
  box-sizing: border-box !important;
}

:deep(.operator-pipeline-matrix-table.el-table th.operator-header-col.el-table__cell) {
  height: v-bind('`${OPERATOR_HEADER_HEIGHT_PX}px`') !important;
  padding: 0 !important;
}

:deep(.operator-pipeline-matrix-table.el-table th.operator-category-col.el-table__cell) {
  height: v-bind('`${OPERATOR_CATEGORY_HEADER_HEIGHT_PX}px`') !important;
  padding: 0 !important;
}

:deep(.operator-pipeline-matrix-table.el-table th.method-name-col.el-table__cell) {
  padding: 0 !important;
}

:deep(.operator-category-col .cell) {
  display: flex;
  align-items: center;
  justify-content: center;
  height: v-bind('`${OPERATOR_CATEGORY_HEADER_HEIGHT_PX}px`') !important;
  min-height: v-bind('`${OPERATOR_CATEGORY_HEADER_HEIGHT_PX}px`') !important;
  padding: 0.2rem 0.35rem;
  overflow: hidden;
  white-space: normal;
  word-break: break-word;
  overflow-wrap: anywhere;
  text-align: center;
  font-size: 0.72rem;
  line-height: 1.15;
  letter-spacing: 0;
  color: var(--bc-text-muted, rgba(220, 230, 255, 0.68));
}

:deep(.operator-header-col .cell) {
  height: v-bind('`${OPERATOR_HEADER_HEIGHT_PX}px`') !important;
  min-height: v-bind('`${OPERATOR_HEADER_HEIGHT_PX}px`') !important;
  overflow: hidden;
}

:deep(.operator-pipeline-matrix-table.el-table .el-table__body-wrapper tbody td.el-table__cell),
:deep(.operator-pipeline-matrix-table.el-table .el-table__fixed-body-wrapper tbody td.el-table__cell) {
  height: v-bind('`${PIPELINE_BODY_ROW_HEIGHT_PX}px`') !important;
  padding: 0 !important;
  line-height: 1 !important;
}

:deep(.operator-pipeline-matrix-table.el-table .el-table__body-wrapper .method-name-col .cell),
:deep(.operator-pipeline-matrix-table.el-table .el-table__fixed-body-wrapper .method-name-col .cell) {
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

:deep(.operator-pipeline-matrix-table.el-table .operator-step-num) {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
}

.method-desc-cell {
  max-height: 4.2rem;
  overflow-y: auto;
  overflow-x: hidden;
  line-height: 1.4;
  padding-right: 0.2rem;
  white-space: normal;
  word-break: break-word;
  scrollbar-color: var(--bc-scroll-thumb, rgba(120, 170, 255, 0.45))
    var(--bc-scroll-track, rgba(20, 26, 40, 0.5));
}

.operators-reference-stack .method-desc-cell {
  width: 100%;
  max-height: 6.4rem;
}

.method-paper-cell {
  max-height: 4.2rem;
  overflow-y: auto;
  overflow-x: hidden;
  line-height: 1.4;
  padding-right: 0.2rem;
  white-space: normal;
  word-break: break-word;
  scrollbar-color: var(--bc-scroll-thumb, rgba(120, 170, 255, 0.45))
    var(--bc-scroll-track, rgba(20, 26, 40, 0.5));
}

.method-paper-cell::-webkit-scrollbar {
  width: 6px;
}

.method-paper-cell::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: var(--bc-scroll-thumb, rgba(120, 170, 255, 0.35));
}

.method-paper-cell::-webkit-scrollbar-track {
  background: var(--bc-scroll-track, rgba(20, 26, 40, 0.45));
  border-radius: 999px;
}

.method-desc-cell::-webkit-scrollbar {
  width: 6px;
}

.method-desc-cell::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: var(--bc-scroll-thumb, rgba(120, 170, 255, 0.35));
}

.method-desc-cell::-webkit-scrollbar-track {
  background: var(--bc-scroll-track, rgba(20, 26, 40, 0.45));
  border-radius: 999px;
}

@media (max-width: 1100px) {
  .combined-pipeline-layout {
    grid-template-columns: 1fr;
  }

  .combined-pipeline-matrix {
    grid-column: auto;
  }

  .automata-row {
    grid-template-columns: 1fr;
  }

  .automaton-svg {
    max-width: 620px;
    min-height: 260px;
  }
}
</style>
