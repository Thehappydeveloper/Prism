# Prism Web (Svelte + TypeScript)

This directory now contains the migrated frontend application.

## Stack

- Svelte 4
- TypeScript (strict)
- Vite
- Chart.js

## Layout

- `src/core/`: API client, route mapping, types, QoE/state helpers
- `src/pages/`: top-level pages (`home`, `control`, `batch`, `compare`)
- `src/features/advanced-compare/`: compare-specific logic
  - `plugins/`: typed chart plugins
  - `metrics.ts`: grouping + aggregation helpers

## Plugin/Registry Architecture (Advanced Compare)

New charts are added by implementing one plugin file in:

- `src/features/advanced-compare/plugins/*.ts`

Each plugin exports a `ChartPlugin`:

- `id`
- `title`
- `build(ctx) => ChartConfiguration | null`

Then register it in:

- `src/features/advanced-compare/plugins/registry.ts`

No other page-level code changes are required.

## Dev

From `src/web/`:

```bash
npm install
npm run dev
```

## Build

From `src/web/`:

```bash
npm run build
```

Build output is written to `src/web/dist` and served by `src/server.py` under `/static/*`.
If `dist` is missing, `python src/server.py` will attempt `npm install` and `npm run build` automatically.

Compatibility routes:

- `/` -> app entrypoint
- `/static/index.html` -> control center
- `/static/aggregate.html` -> batch planner
- `/static/compare.html` -> advanced compare
