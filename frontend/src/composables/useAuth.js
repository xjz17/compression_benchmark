import { computed, ref } from "vue";
import axios from "@/utils/httpClient.js";
import { setAuthToken } from "@/utils/httpClient.js";

const TOKEN_STORAGE_KEY = "web_compression_auth_token";

const token = ref("");
const user = ref(null);
const authReady = ref(false);

/** Global account dialog (mounted on AppLayout, not inside sticky nav). */
export const authDialogOpen = ref(false);
export const authDialogInitialTab = ref("login");

export function openAuthDialog(tab = "login") {
  authDialogInitialTab.value = tab === "register" ? "register" : "login";
  authDialogOpen.value = true;
}

export const isAuthenticated = computed(() => Boolean(token.value && user.value));

export function initAuthFromStorage() {
  const stored = localStorage.getItem(TOKEN_STORAGE_KEY);
  if (!stored) {
    authReady.value = true;
    return Promise.resolve();
  }
  token.value = stored;
  setAuthToken(stored);
  return fetchMe()
    .catch(() => {
      clearAuth();
    })
    .finally(() => {
      authReady.value = true;
    });
}

export function getAuthToken() {
  return token.value;
}

function applySession(tokenKey, userPayload) {
  token.value = tokenKey;
  user.value = userPayload;
  localStorage.setItem(TOKEN_STORAGE_KEY, tokenKey);
  setAuthToken(tokenKey);
}

export function clearAuth() {
  token.value = "";
  user.value = null;
  localStorage.removeItem(TOKEN_STORAGE_KEY);
  setAuthToken(null);
}

export async function fetchMe() {
  const { data } = await axios.get("/api/auth/me/");
  user.value = data.user;
  return data.user;
}

export async function registerAccount({ email, password, password_confirm }) {
  const { data } = await axios.post("/api/auth/register/", {
    email,
    password,
    password_confirm,
  });
  applySession(data.token, data.user);
  return data;
}

export async function loginAccount({ email, password }) {
  const { data } = await axios.post("/api/auth/login/", {
    email,
    password,
  });
  applySession(data.token, data.user);
  return data;
}

export async function logoutAccount() {
  try {
    if (token.value) {
      await axios.post("/api/auth/logout/");
    }
  } catch {
    /* still clear local session */
  } finally {
    clearAuth();
  }
}

export { authReady };

export function useAuth() {
  return {
    token,
    user,
    authReady,
    isAuthenticated,
    authDialogOpen,
    authDialogInitialTab,
    openAuthDialog,
    registerAccount,
    loginAccount,
    logoutAccount,
    fetchMe,
    clearAuth,
  };
}
