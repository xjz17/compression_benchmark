<template>
  <el-dialog
    v-model="visible"
    class="auth-dialog"
    modal-class="auth-dialog-overlay"
    width="420px"
    append-to-body
    :z-index="10000"
    :close-on-click-modal="false"
    destroy-on-close
    @closed="resetForms"
  >
    <template #header>
      <span class="auth-dialog-title">Account</span>
    </template>

    <el-tabs v-model="activeTab" class="auth-tabs" stretch>
      <el-tab-pane label="Sign in" name="login">
        <el-form
          ref="loginFormRef"
          :model="loginForm"
          :rules="loginRules"
          label-position="top"
          class="auth-form"
          @submit.prevent="submitLogin"
        >
          <el-form-item label="Email" prop="email">
            <el-input
              v-model="loginForm.email"
              type="email"
              autocomplete="email"
              placeholder="you@example.com"
              size="large"
            />
          </el-form-item>
          <el-form-item label="Password" prop="password">
            <el-input
              v-model="loginForm.password"
              type="password"
              autocomplete="current-password"
              placeholder="Your password"
              size="large"
              maxlength="20"
              show-password
            />
          </el-form-item>
          <el-button
            type="primary"
            class="auth-submit"
            size="large"
            :loading="submitting"
            @click="submitLogin"
          >
            Sign in
          </el-button>
        </el-form>
      </el-tab-pane>

      <el-tab-pane label="Register" name="register">
        <el-form
          ref="registerFormRef"
          :model="registerForm"
          :rules="registerRules"
          label-position="top"
          class="auth-form"
          @submit.prevent="submitRegister"
        >
          <el-form-item label="Email" prop="email">
            <el-input
              v-model="registerForm.email"
              type="email"
              autocomplete="email"
              placeholder="you@example.com"
              size="large"
            />
          </el-form-item>
          <el-form-item label="Password" prop="password">
            <el-input
              v-model="registerForm.password"
              type="password"
              autocomplete="new-password"
              placeholder="8–20 characters"
              size="large"
              maxlength="20"
              show-password
            />
          </el-form-item>
          <el-form-item label="Confirm password" prop="password_confirm">
            <el-input
              v-model="registerForm.password_confirm"
              type="password"
              autocomplete="new-password"
              placeholder="Repeat password"
              size="large"
              maxlength="20"
              show-password
            />
          </el-form-item>
          <el-button
            type="primary"
            class="auth-submit"
            size="large"
            :loading="submitting"
            @click="submitRegister"
          >
            Create account
          </el-button>
        </el-form>
      </el-tab-pane>
    </el-tabs>
  </el-dialog>
</template>

<script setup>
import { computed, reactive, ref, watch } from "vue";
import { ElMessage } from "element-plus";
import { loginAccount, registerAccount } from "@/composables/useAuth.js";

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  initialTab: { type: String, default: "login" },
});

const emit = defineEmits(["update:modelValue", "success"]);

const visible = computed({
  get: () => props.modelValue,
  set: (v) => emit("update:modelValue", v),
});

const activeTab = ref("login");
const submitting = ref(false);
const loginFormRef = ref(null);
const registerFormRef = ref(null);

const loginForm = reactive({ email: "", password: "" });
const registerForm = reactive({
  email: "",
  password: "",
  password_confirm: "",
});

const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const emailRule = [
  { required: true, message: "Email is required", trigger: "blur" },
  {
    validator: (_rule, value, callback) => {
      const v = String(value || "").trim();
      if (!v) {
        callback(new Error("Email is required"));
        return;
      }
      if (!emailPattern.test(v)) {
        callback(new Error("Enter a valid email address"));
        return;
      }
      callback();
    },
    trigger: "blur",
  },
];

const passwordLengthRule = [
  { required: true, message: "Password is required", trigger: "blur" },
  { min: 8, max: 20, message: "Use 8 to 20 characters", trigger: "blur" },
];

const loginRules = {
  email: emailRule,
  password: passwordLengthRule,
};

const registerRules = {
  email: emailRule,
  password: passwordLengthRule,
  password_confirm: [
    { required: true, message: "Confirm your password", trigger: "blur" },
    { min: 8, max: 20, message: "Use 8 to 20 characters", trigger: "blur" },
    {
      validator: (_rule, value, callback) => {
        if (value !== registerForm.password) {
          callback(new Error("Passwords do not match"));
          return;
        }
        callback();
      },
      trigger: "blur",
    },
  ],
};

watch(
  () => props.modelValue,
  (open) => {
    if (open) {
      activeTab.value = props.initialTab === "register" ? "register" : "login";
    }
  }
);

function resetForms() {
  loginForm.email = "";
  loginForm.password = "";
  registerForm.email = "";
  registerForm.password = "";
  registerForm.password_confirm = "";
  loginFormRef.value?.clearValidate();
  registerFormRef.value?.clearValidate();
}

function formatApiError(error) {
  const data = error?.response?.data;
  if (!data) return "Request failed. Please try again.";
  if (typeof data.detail === "string") return data.detail;
  const parts = [];
  for (const [key, val] of Object.entries(data)) {
    if (Array.isArray(val)) {
      parts.push(`${key}: ${val.join(" ")}`);
    } else if (typeof val === "string") {
      parts.push(val);
    }
  }
  return parts.length ? parts.join(" ") : "Request failed. Please try again.";
}

async function submitLogin() {
  const form = loginFormRef.value;
  if (!form) return;
  try {
    await form.validate();
  } catch {
    return;
  }
  submitting.value = true;
  try {
    await loginAccount({
      email: loginForm.email.trim(),
      password: loginForm.password,
    });
    ElMessage.success("Signed in successfully.");
    visible.value = false;
    emit("success", "login");
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    submitting.value = false;
  }
}

async function submitRegister() {
  const form = registerFormRef.value;
  if (!form) return;
  try {
    await form.validate();
  } catch {
    return;
  }
  submitting.value = true;
  try {
    await registerAccount({
      email: registerForm.email.trim(),
      password: registerForm.password,
      password_confirm: registerForm.password_confirm,
    });
    ElMessage.success("Account created. You are now signed in.");
    visible.value = false;
    emit("success", "register");
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    submitting.value = false;
  }
}
</script>

<style scoped>
.auth-dialog-title {
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--bc-text, #e8ecf4);
}

.auth-form {
  margin-top: 0.5rem;
}

.auth-submit {
  width: 100%;
  margin-top: 0.25rem;
}

:deep(.auth-tabs .el-tabs__item) {
  color: var(--bc-text-muted, #9aa3b2);
}

:deep(.auth-tabs .el-tabs__item.is-active) {
  color: var(--bc-accent, #4da3ff);
}

:deep(.auth-form .el-form-item__label) {
  color: var(--bc-text-muted, #9aa3b2);
  font-weight: 500;
}
</style>
