<template>
  <div id="app">
    <router-view></router-view>
  </div>
</template>

<script>
</script>

<style>
@import '@/styles/main.css';
@import '@/styles/theme.css';

html, body {
  margin: 0;
  padding: 0;
  min-height: 100%;
  overflow-x: hidden;
  background-color: var(--bc-bg-deep, #060708);
  background-image: var(--bc-page-bg-image);
  background-attachment: fixed;
}

#app {
  min-height: 100vh;
  color: var(--bc-text, #e6eaf2);
  font-family: var(
    --bc-font-sans,
    system-ui,
    -apple-system,
    'Segoe UI',
    'Helvetica Neue',
    sans-serif
  );
  position: relative;
  -webkit-font-smoothing: antialiased;
}
</style>
