# frontend

Vue 3 + Vite + Element Plus. Production assets deploy under `/benchmark/` (see `vite.config.js` `base`).

## Setup

```shell
npm install
```

## Development

```shell
npm run dev
```

Opens **`http://localhost:8080/benchmark/`**. `/api` is proxied to **`http://127.0.0.1:8000`** (use `127.0.0.1`, not `localhost`, on Windows — Django often binds IPv4 only).

`npm run serve` is an alias for `npm run dev`.

## Production build

```shell
npm run build
```

Output: `dist/` (served by Django/nginx at `/benchmark/` — see root `note_deploy.md`).

## Preview production build locally

```shell
npm run preview
```
