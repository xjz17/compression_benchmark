// Use Axios CJS browser bundle to avoid webpack ESM interop regressions
// seen with some Axios 1.15+ package-resolution paths in Vue CLI dev mode.
import axios from "axios/dist/browser/axios.cjs";

const client = axios.create();

let authToken = null;

export function setAuthToken(value) {
  authToken = value && String(value).trim() ? String(value).trim() : null;
}

client.interceptors.request.use((config) => {
  if (authToken) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Token ${authToken}`;
  }
  return config;
});

export default client;
