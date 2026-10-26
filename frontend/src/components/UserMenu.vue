<template>
  <div class="user-menu">
    <template v-if="isAuthenticated">
      <el-dropdown trigger="click" placement="bottom-end" popper-class="user-menu-dropdown">
        <button type="button" class="user-menu-trigger is-signed-in" aria-label="Account menu">
          <el-icon :size="18"><UserFilled /></el-icon>
          <span class="user-menu-email">{{ user?.email }}</span>
          <el-icon class="user-menu-caret" :size="14"><ArrowDown /></el-icon>
        </button>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item disabled class="user-menu-dropdown-email">
              {{ user?.email }}
            </el-dropdown-item>
            <el-dropdown-item @click="goUpload">
              My uploads
            </el-dropdown-item>
            <el-dropdown-item divided @click="handleLogout">
              Sign out
            </el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
    </template>

    <button
      v-else
      type="button"
      class="user-menu-trigger"
      aria-label="Sign in or register"
      @click="openAuthDialog('login')"
    >
      <el-icon :size="18"><User /></el-icon>
      <span>Account</span>
    </button>

  </div>
</template>

<script setup>
import { ElMessage } from "element-plus";
import { useRouter } from "vue-router";
import { ArrowDown, User, UserFilled } from "@element-plus/icons-vue";
import { logoutAccount, openAuthDialog, useAuth } from "@/composables/useAuth.js";

const router = useRouter();
const { user, isAuthenticated } = useAuth();

function goUpload() {
  router.push("/upload");
}

async function handleLogout() {
  try {
    await logoutAccount();
    ElMessage.success("Signed out.");
    if (router.currentRoute.value.path === "/upload") {
      router.push("/");
    }
  } catch {
    ElMessage.error("Could not sign out. Please try again.");
  }
}

defineExpose({ openAuthDialog });
</script>

<style scoped>
.user-menu {
  flex-shrink: 0;
}

.user-menu-trigger {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.45rem 0.85rem;
  border-radius: 10px;
  border: 1px solid rgba(120, 170, 255, 0.22);
  background: var(--bc-control-bg, rgba(12, 16, 28, 0.72));
  color: var(--bc-text, #e8ecf4);
  font-size: 0.9rem;
  font-weight: 500;
  cursor: pointer;
  transition: border-color 0.15s ease, background 0.15s ease;
}

.user-menu-trigger:hover {
  border-color: rgba(77, 163, 255, 0.42);
  background: var(--bc-accent-soft, rgba(77, 163, 255, 0.12));
}

.user-menu-trigger.is-signed-in {
  max-width: min(280px, 42vw);
}

.user-menu-email {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 180px;
}

.user-menu-caret {
  opacity: 0.75;
}
</style>
