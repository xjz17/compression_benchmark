import { createApp } from 'vue'
import App from './App.vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import 'element-plus/theme-chalk/dark/css-vars.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import router from './router'
import { initAuthFromStorage } from './composables/useAuth.js'
import { initThemeFromStorage } from './composables/useTheme.js'

const app = createApp(App)
app.use(router)
app.use(ElementPlus)

for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
}

initThemeFromStorage()
initAuthFromStorage().finally(() => {
  app.mount('#app')
})
