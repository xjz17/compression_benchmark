<template>
  <el-image
    v-if="displaySrc"
    :src="displaySrc"
    :preview-src-list="preview ? [displaySrc] : []"
    :fit="fit"
    :preview-teleported="previewTeleported"
    :class="imageClass"
  />
  <span v-else-if="failed" class="cell-empty">Failed to load</span>
  <span v-else-if="src" class="cell-empty">…</span>
</template>

<script setup>
import { onBeforeUnmount, ref, watch } from "vue";
import axios from "@/utils/httpClient.js";

const props = defineProps({
  src: { type: String, default: "" },
  /** When true, fetch via axios (sends API token) for private user-upload assets. */
  requireAuth: { type: Boolean, default: false },
  fit: { type: String, default: "contain" },
  preview: { type: Boolean, default: true },
  previewTeleported: { type: Boolean, default: true },
  imageClass: { type: String, default: "" },
});

const displaySrc = ref("");
const failed = ref(false);
let objectUrl = "";

function revokeObjectUrl() {
  if (objectUrl) {
    window.URL.revokeObjectURL(objectUrl);
    objectUrl = "";
  }
}

async function loadImage() {
  revokeObjectUrl();
  displaySrc.value = "";
  failed.value = false;
  const url = String(props.src || "").trim();
  if (!url) {
    return;
  }
  if (!props.requireAuth) {
    displaySrc.value = url;
    return;
  }
  try {
    const { data } = await axios.get(url, { responseType: "blob" });
    objectUrl = window.URL.createObjectURL(data);
    displaySrc.value = objectUrl;
  } catch (error) {
    console.error(error);
    failed.value = true;
  }
}

watch(
  () => [props.src, props.requireAuth],
  () => {
    loadImage();
  },
  { immediate: true },
);

onBeforeUnmount(() => {
  revokeObjectUrl();
});
</script>
