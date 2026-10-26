<template>
  <div class="app-layout">
    <header class="nav-shell">
      <div class="nav-container">
        <div class="logo-container" @click="$router.push('/')" title="Home">
          <img src="@/assets/benchmark-logo.jpg" alt="Benchmark Logo" class="logo-img" />
        </div>
        <nav class="app-nav" aria-label="Main navigation">
          <router-link
            v-for="item in navItems"
            :key="item.to"
            :to="item.to"
            class="nav-item"
            active-class="is-active"
          >
            {{ item.label }}
          </router-link>
        </nav>
        <div class="nav-actions">
          <ThemeToggle />
          <UserMenu />
        </div>
      </div>
    </header>

    <div class="title-block">
      <div class="page-title">{{ pageTitle }}</div>
      <p v-if="subtitle" class="page-subtitle">{{ subtitle }}</p>
    </div>

    <div class="page-body">
      <slot></slot>
    </div>

    <el-backtop :right="28" :bottom="28" class="backtop-bench" />

    <AuthDialog
      v-model="authDialogOpen"
      :initial-tab="authDialogInitialTab"
    />
  </div>
</template>

<script setup>
import { computed } from "vue";
import { useRoute } from "vue-router";
import AuthDialog from "@/components/AuthDialog.vue";
import ThemeToggle from "@/components/ThemeToggle.vue";
import UserMenu from "@/components/UserMenu.vue";
import { authDialogInitialTab, authDialogOpen, useAuth } from "@/composables/useAuth.js";

const route = useRoute();
const { isAuthenticated } = useAuth();

const navItems = computed(() => {
  const items = [
    { to: "/", label: "Home" },
    // { to: "/tsfile", label: "TsFile" },
    { to: "/datasets", label: "Datasets" },
    { to: "/methods", label: "Methods" },
  ];
  if (isAuthenticated.value) {
    items.push({ to: "/upload", label: "Upload" });
  }
  return items;
});

const props = defineProps({
  title: {
    type: String,
    required: true,
  },
  /** Short line under the title; omit on simple pages */
  subtitle: {
    type: String,
    default: "",
  },
});

const currentRoute = computed(() => route.path);
const pageTitle = computed(() => props.title);
const subtitle = computed(() => props.subtitle);
</script>

<style scoped>
.app-layout {
  min-height: 100vh;
  position: relative;
  padding-bottom: 3rem;
}

.nav-shell {
  position: sticky;
  top: 0;
  z-index: 200;
  backdrop-filter: blur(16px) saturate(1.25);
  background: var(--bc-nav-bg, rgba(5, 7, 11, 0.78));
  border-bottom: 1px solid var(--bc-border, rgba(120, 170, 255, 0.14));
  box-shadow: var(--bc-nav-shadow, 0 4px 24px rgba(0, 0, 0, 0.35));
}

.nav-shell::before {
  content: '';
  display: block;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--bc-accent, #4da3ff), var(--bc-cyan, #5ae9ff), transparent);
  opacity: 0.6;
}

.nav-container {
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1.25rem;
  display: flex;
  align-items: center;
  min-height: 64px;
}

.nav-actions {
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
}

.logo-container {
  margin-right: 1rem;
  cursor: pointer;
  padding: 0.5rem 0.75rem 0.5rem 0;
  border-radius: 10px;
  transition: background 0.2s ease;
}

.logo-container:hover {
  background: var(--bc-accent-soft, rgba(77, 163, 255, 0.12));
}

.app-nav {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 0.25rem;
  border-bottom: none;
}

.nav-item {
  display: inline-flex;
  align-items: center;
  padding: 0 0.85rem;
  min-height: 40px;
  color: var(--bc-text, #e8ecf4);
  font-size: 0.95rem;
  font-weight: 600;
  letter-spacing: 0.03em;
  text-decoration: none;
  border-radius: 8px;
  margin: 0 4px;
  transition: background-color 0.15s ease, color 0.15s ease;
}

.nav-item:hover {
  background-color: var(--bc-accent-soft, rgba(77, 163, 255, 0.12));
  color: var(--bc-cyan, #5ae9ff);
}

.nav-item.is-active {
  color: var(--bc-cyan, #5ae9ff);
  border-bottom: 2px solid var(--bc-accent, #4da3ff);
  background-color: transparent;
}

.title-block {
  margin: 0 auto;
  padding: 2.5rem 1.25rem 1.5rem;
  max-width: 1200px;
  text-align: center;
}

.page-title {
  font-size: clamp(1.5rem, 4vw, 2.25rem);
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.2;
  margin: 0;
  padding: 0 0.5rem;
  background: var(
    --bc-title-gradient,
    linear-gradient(120deg, #f0f6ff 0%, #a8d4ff 55%, #5ae9ff 100%)
  );
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  filter: var(--bc-title-shadow, drop-shadow(0 0 24px rgba(90, 233, 255, 0.15)));
}

.page-subtitle {
  margin: 0.75rem 0 0;
  font-size: 0.95rem;
  font-weight: 500;
  color: var(--bc-text-muted, #9aa3b2);
  letter-spacing: 0.02em;
}

.page-body {
  max-width: 1200px;
  margin: 0 auto 3rem;
  padding: 1.75rem 1.25rem 2.25rem;
  background: var(
    --bc-page-body-bg,
    linear-gradient(145deg, rgba(18, 20, 30, 0.65) 0%, rgba(10, 12, 20, 0.85) 100%)
  );
  border-radius: var(--bc-radius-lg, 18px);
  border: 1px solid var(--bc-border-strong, rgba(120, 170, 255, 0.22));
  box-shadow: var(--bc-shadow, 0 8px 32px rgba(0, 0, 0, 0.4));
  backdrop-filter: blur(14px) saturate(1.2);
}

.logo-img {
  height: 42px;
  width: auto;
  transition: transform 0.25s ease, filter 0.25s ease;
}

.logo-container:hover .logo-img {
  transform: scale(1.04);
  filter: var(--bc-logo-hover-filter, brightness(1.08));
}

:deep(.backtop-bench) {
  --el-backtop-bg-color: var(--bc-backtop-bg, rgba(18, 24, 42, 0.92));
  --el-backtop-hover-bg-color: var(--bc-backtop-hover-bg, rgba(77, 163, 255, 0.2));
  border: 1px solid var(--bc-border, rgba(120, 170, 255, 0.2));
  box-shadow: var(--bc-backtop-shadow, 0 4px 20px rgba(0, 0, 0, 0.35));
}
</style>
