<template>
  <el-tag
    class="compression-type-tag"
    :type="methodCompressionModeTagType(resolvedMode)"
    size="small"
  >
    {{ methodCompressionModeLabel(resolvedMode) }}
  </el-tag>
</template>

<script setup>
import { computed } from "vue";
import { getAlgorithmCompressionMode } from "@/utils/algorithmCompressionMode.js";
import {
  methodCompressionModeLabel,
  methodCompressionModeTagType,
} from "@/utils/methodCatalogCompressionMode.js";

const props = defineProps({
  /** `lossy` or `lossless` when already known (Methods API). */
  mode: { type: String, default: "" },
  /** Benchmark algorithm key; used when mode is omitted (Home / TsFile). */
  algorithm: { type: String, default: "" },
});

const resolvedMode = computed(() => {
  const m = String(props.mode || "").trim().toLowerCase();
  if (m === "lossy" || m === "lossless") {
    return m;
  }
  return getAlgorithmCompressionMode(props.algorithm);
});
</script>
