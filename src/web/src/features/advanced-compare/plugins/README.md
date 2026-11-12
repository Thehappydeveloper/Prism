# Advanced Compare plugins

Each chart is a `ChartPlugin` registered in `registry.ts`. A plugin provides an id, title,
category, description, `requiredDatasets`, and a pure `build` function.

`requiredDatasets` is the data contract. The page unions these declarations and asks the
diagnostics API only for those compact metric families, so adding a graph does not require
editing the page's loading logic. Prefer existing sufficient statistics; add a new compact
server-side dataset only when the graph cannot be derived from an existing one. Raw tile or
frame maps are intentionally reserved for the single-run FoV debugger.

Publication styling and export are applied centrally after `build`. Dataset labels that match
predictor/group labels automatically receive deterministic color, marker, and dash identities,
so plugins must use `labelForRun` for predictor series. Do not assign semantic meaning to a
predictor's array position or palette index. Categorical predictor bar charts automatically
switch to horizontal layout for long/many labels. Set `sortDirection` to `asc` or `desc` when
the primary bar metric has a clear optimization direction; leave natural axes and parameter
sweeps as `none`.

Keep units in axis titles and keep raw data in the Chart.js configuration. The publication
exporter renders bar, line, and scatter configurations directly at the requested physical size
for SVG/PDF/PNG output. Avoid drawing essential information only from a private canvas plugin,
because it cannot be reproduced by generic vector export.
