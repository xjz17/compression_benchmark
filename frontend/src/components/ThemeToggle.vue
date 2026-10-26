<template>
  <button
    type="button"
    class="theme-toggle-btn"
    :aria-label="isDarkTheme ? 'Switch to light mode' : 'Switch to dark mode'"
    @click="toggleTheme"
  >
    <el-icon :size="16">
      <component :is="isDarkTheme ? Sunny : Moon" />
    </el-icon>
    <span class="theme-toggle-label">
      {{ isDarkTheme ? "Light" : "Dark" }}
    </span>
  </button>
</template>

<script setup>
import { computed } from "vue";
import { Moon, Sunny } from "@element-plus/icons-vue";
import { useTheme } from "@/composables/useTheme.js";

const { isDarkTheme, toggleTheme } = useTheme();
const themeText = computed(() => (isDarkTheme.value ? "Light" : "Dark"));
defineExpose({ themeText });
</script>

<style scoped>
.theme-toggle-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.45rem 0.75rem;
  border-radius: 10px;
  border: 1px solid var(--bc-border-strong, rgba(120, 170, 255, 0.22));
  background: var(--bc-control-bg, rgba(12, 16, 28, 0.72));
  color: var(--bc-text, #e8ecf4);
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  transition:
    border-color var(--bc-transition, 0.2s ease),
    background-color var(--bc-transition, 0.2s ease),
    color var(--bc-transition, 0.2s ease);
}

.theme-toggle-btn:hover {
  border-color: var(--bc-accent, #4da3ff);
  background: var(--bc-accent-soft, rgba(77, 163, 255, 0.12));
}

.theme-toggle-label {
  letter-spacing: 0.02em;
}
</style>
