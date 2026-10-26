<template>
  <div class="operator-matrix-table">
    <el-table
      ref="tableRef"
      :data="methods"
      :loading="loading"
      :max-height="maxHeight"
      :empty-text="emptyText"
      :style="tableStyle"
      class="operator-matrix-el-table operator-pipeline-matrix-table"
      @cell-mouse-enter="handleCellMouseEnter"
      @mouseleave="emit('leave')"
    >
      <el-table-column
        label="Method"
        :width="methodColumnWidth"
        fixed="left"
        align="center"
        label-class-name="method-name-col"
        class-name="method-name-col"
        show-overflow-tooltip
      >
        <template #default="scope">
          {{ displayMethodCatalogName(scope.row.name) }}
        </template>
      </el-table-column>
      <el-table-column
        v-for="op in operators"
        :key="`${matrixKey}-op-${op.id}`"
        :prop="`op_${op.id}`"
        :column-key="op.id"
        :width="operatorColumnWidth"
        align="center"
        label-class-name="operator-header-col"
        class-name="operator-step-col"
      >
        <template #header>
          <span
            class="operator-header-vertical"
            :title="op.example || op.description || op.name"
          >
            {{ op.name }}
          </span>
        </template>
        <template #default="scope">
          <span
            v-if="stepForRow(scope.row, op.id)"
            class="operator-step-num"
            @mouseenter="emit('row-enter', scope.row)"
          >
            {{ stepForRow(scope.row, op.id) }}
          </span>
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { displayMethodCatalogName } from "@/utils/algorithmDisplayName.js";

const props = defineProps({
  matrixKey: {
    type: String,
    required: true,
  },
  methods: {
    type: Array,
    default: () => [],
  },
  operators: {
    type: Array,
    default: () => [],
  },
  loading: {
    type: Boolean,
    default: false,
  },
  emptyText: {
    type: String,
    default: "No methods in catalog.",
  },
  maxHeight: {
    type: Number,
    default: 450,
  },
  methodColumnWidth: {
    type: Number,
    default: 100,
  },
  operatorColumnWidth: {
    type: Number,
    default: 32,
  },
  stepForRow: {
    type: Function,
    required: true,
  },
});

const emit = defineEmits(["row-enter", "leave"]);

const tableRef = ref(null);
let bodyWrapper = null;

function handleCellMouseEnter(row) {
  emit("row-enter", row);
}

function syncHeaderScroll() {
  const root = tableRef.value?.$el;
  if (!root) {
    return;
  }
  const headerWrapper = root.querySelector(".el-table__header-wrapper");
  if (!bodyWrapper || !headerWrapper) {
    return;
  }
  headerWrapper.scrollLeft = bodyWrapper.scrollLeft;
}

function onBodyScroll() {
  syncHeaderScroll();
}

function bindBodyScrollListener() {
  if (bodyWrapper) {
    bodyWrapper.removeEventListener("scroll", onBodyScroll);
    bodyWrapper = null;
  }
  const root = tableRef.value?.$el;
  bodyWrapper = root?.querySelector(".el-table__body-wrapper") ?? null;
  bodyWrapper?.addEventListener("scroll", onBodyScroll, { passive: true });
}

watch(
  () => [props.methods.length, props.operators.length, props.loading],
  async () => {
    await nextTick();
    bindBodyScrollListener();
  },
);

onMounted(async () => {
  await nextTick();
  bindBodyScrollListener();
});

onBeforeUnmount(() => {
  if (bodyWrapper) {
    bodyWrapper.removeEventListener("scroll", onBodyScroll);
    bodyWrapper = null;
  }
});

const tableStyle = computed(() => {
  const totalWidth =
    props.methodColumnWidth + props.operators.length * props.operatorColumnWidth;
  return {
    width: `${totalWidth}px`,
    minWidth: "100%",
  };
});
</script>

<style scoped>
.operator-matrix-table {
  width: 100%;
  min-width: 0;
}

/* Let Element Plus own horizontal scroll so header and body stay aligned. */
:deep(.operator-matrix-el-table .el-table__body-wrapper) {
  overflow-x: auto;
}

:deep(.operator-matrix-el-table .el-table__header-wrapper) {
  overflow: hidden;
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
  height: 124px !important;
  padding: 0 !important;
}

:deep(.operator-pipeline-matrix-table.el-table th.method-name-col.el-table__cell) {
  padding: 0 !important;
}

:deep(.operator-header-col .cell) {
  height: 124px !important;
  min-height: 124px !important;
  overflow: hidden;
}

:deep(.operator-pipeline-matrix-table.el-table .el-table__body-wrapper tbody td.el-table__cell),
:deep(.operator-pipeline-matrix-table.el-table .el-table__fixed-body-wrapper tbody td.el-table__cell) {
  height: 24px !important;
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
</style>
