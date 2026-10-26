import { computed, ref } from "vue";

const THEME_STORAGE_KEY = "web_compression_theme_mode";
const THEME_DARK = "dark";
const THEME_LIGHT = "light";

const themeMode = ref(THEME_LIGHT);

function normalizeTheme(mode) {
  return mode === THEME_LIGHT ? THEME_LIGHT : THEME_DARK;
}

function applyThemeToDocument(mode) {
  if (typeof document === "undefined") return;
  const next = normalizeTheme(mode);
  const root = document.documentElement;
  root.setAttribute("data-theme", next);
  root.classList.toggle("dark", next === THEME_DARK);
  root.style.colorScheme = next;
}

function setTheme(mode) {
  const next = normalizeTheme(mode);
  themeMode.value = next;
  applyThemeToDocument(next);
  if (typeof localStorage !== "undefined") {
    localStorage.setItem(THEME_STORAGE_KEY, next);
  }
}

function toggleTheme() {
  setTheme(themeMode.value === THEME_DARK ? THEME_LIGHT : THEME_DARK);
}

function initThemeFromStorage() {
  let initial = THEME_LIGHT;
  if (typeof localStorage !== "undefined") {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved) {
      initial = normalizeTheme(saved);
    }
  }
  themeMode.value = initial;
  applyThemeToDocument(initial);
}

export function useTheme() {
  return {
    themeMode: computed(() => themeMode.value),
    isDarkTheme: computed(() => themeMode.value === THEME_DARK),
    setTheme,
    toggleTheme,
  };
}

export { initThemeFromStorage };
