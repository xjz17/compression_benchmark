import { createRouter, createWebHistory } from "vue-router";
import { ElMessage } from "element-plus";
import Home from '../views/Home.vue';
import Upload from "../views/Upload.vue";
// import TsFile from "../views/TsFile.vue";
import Dataset from "../views/Dataset.vue";
import Methods from "../views/Methods.vue";
import { authReady, isAuthenticated, openAuthDialog } from "@/composables/useAuth.js";

const routes = [
  {
    path: "/",
    name: "Home",
    component: Home,
  },
  {
    path: "/upload",
    name: "Upload",
    component: Upload,
    meta: { requiresAuth: true },
  },
  // {
  //   path: "/tsfile",
  //   name: "TsFile",
  //   component: TsFile,
  // },
  {
    path: "/datasets",
    name: "Dataset",
    component: Dataset,
  },
  {
    path: "/methods",
    name: "Methods",
    component: Methods,
  },
];

const router = createRouter({
  history: createWebHistory("/benchmark/"),
  routes,
});

router.beforeEach((to) => {
  if (!to.meta.requiresAuth) {
    return true;
  }
  // initAuthFromStorage finishes before app mount; if not ready yet, allow once.
  if (!authReady?.value) {
    return true;
  }
  if (isAuthenticated.value) {
    return true;
  }
  openAuthDialog("login");
  ElMessage.warning("Sign in to access Upload & Benchmark.");
  return { name: "Home", replace: true };
});

export default router;
