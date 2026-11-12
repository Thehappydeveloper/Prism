# Navis Prism Compatibility Spec

This file is a technical implementation brief for the LLM or engineer upgrading Navis to support Prism exports.

## Goal

Implement a Navis receiver mode that accepts Prism run exports and opens a visualization session where:

- the original PLY sequence is loaded from the path sent by Prism,
- tiles can be recolored from ABR representation decisions,
- tiles can be recolored from FoV visibility scores,
- the navigation trace is rendered so Navis shows user position and view direction,
- the bandwidth trace is rendered in real time in the overlay,
- redirect back into the Navis run/session happens immediately after Prism exports the run.

## Prism Source Of Truth

Prism sends a single HTTP POST to the Navis export endpoint configured in Prism settings.

Current schema:

- `schema = "prism.navis.compat.v2"`

Prism-side configuration lookup:

1. UI settings saved at `results/_settings/navis.json`
2. Environment variables as fallback:
- `PRISM_NAVIS_URL`
- `PRISM_NAVIS_HEALTH_PATH`
- `PRISM_NAVIS_EXPORT_PATH`
- `PRISM_NAVIS_TIMEOUT_S`
- `PRISM_NAVIS_AUTO_EXPORT`
- `PRISM_NAVIS_AUTH_TOKEN`

## Required Navis Endpoints

Navis should expose:

1. `GET /health`
- Return HTTP `200` with a lightweight JSON body such as `{ "ok": true }`.
- This is optional but strongly recommended.

2. `POST /api/prism/runs`
- Accept Prism payloads defined below.
- Persist enough state to open a viewer session immediately.
- Respond with JSON containing one of:
- `redirect_url`
- `open_url`
- `url`

Prism uses the first of those keys that exists. If none exist, Prism falls back to `/?run_id=<run_id>` on the Navis base URL.

## Required Payload Handling

Navis must accept the following payload fields.

Top-level:

- `source`
- `schema`
- `exported_at`
- `run`
- `algorithms`
- `content`
- `traces`
- `playback`
- `tiling`
- `segments`

`run`:

- `run.run_id`
- `run.status`
- `run.started`
- `run.finished`

`algorithms`:

- `algorithms.abr_name`
- `algorithms.predictor_name`

`content`:

- `content.ply_dir_path`
- `content.mpd_path`

`traces`:

- `traces.navigation_trace_path`
- `traces.bandwidth_trace_path`

`playback`:

- `playback.segment_count`
- `playback.segment_duration_s`
- `playback.buffer_window_segments`
- `playback.rtt_s`

`tiling`:

- `tiling.tile_catalog`
- `tiling.component_max_rep_idx`

`segments[]`:

- `segment_idx`
- `time_s`
- `duration_s`
- `abr_requested`
- `abr_applied`
- `fov_prediction`
- `fov_ground_truth`
- `fov_visibility` (backward-compatible alias of `fov_prediction`)

Example:

```json
{
  "source": "prism",
  "schema": "prism.navis.compat.v2",
  "exported_at": "2026-03-06T16:00:00.000000",
  "run": {
    "run_id": "run_260306_160000_abcd",
    "status": "finished",
    "started": "2026-03-06T16:00:00.000000",
    "finished": "2026-03-06T16:00:10.000000"
  },
  "algorithms": {
    "abr_name": "FoVRoundRobinABR",
    "predictor_name": "AverageMotionPredictor"
  },
  "content": {
    "ply_dir_path": "/abs/path/to/Ply",
    "mpd_path": "/abs/path/to/content.mpd"
  },
  "traces": {
    "navigation_trace_path": "/abs/path/to/fov_trace.csv",
    "bandwidth_trace_path": "/abs/path/to/network_trace.csv"
  },
  "playback": {
    "segment_count": 75,
    "segment_duration_s": 2.0,
    "buffer_window_segments": 10,
    "rtt_s": 0.06
  },
  "tiling": {
    "tile_catalog": {
      "0": {
        "bounds": {
          "min_corner": [101.0, 0.0, 0.0],
          "max_corner": [291.0, 338.3333, 658.0],
          "center": [196.0, 169.1666, 329.0],
          "volume": 42298433.3333
        },
        "components": ["atlas", "attr", "geom", "occp"]
      }
    },
    "component_max_rep_idx": {
      "atlas": 0,
      "geom": 3,
      "occp": 0,
      "attr": 2
    }
  },
  "segments": [
    {
      "segment_idx": 0,
      "time_s": 0.0,
      "duration_s": 2.0,
      "abr_requested": {
        "0": { "atlas": 0, "geom": 3, "occp": 0, "attr": 2 }
      },
      "abr_applied": {
        "0": { "atlas": 0, "geom": 3, "occp": 0, "attr": 2 }
      },
      "fov_prediction": {
        "0": 1.0,
        "1": 0.34
      },
      "fov_ground_truth": {
        "0": 1.0,
        "1": 0.0
      },
      "fov_visibility": {
        "0": 1.0,
        "1": 0.34
      }
    }
  ]
}
```

## Navis Receiver Requirements

After receiving a Prism payload, Navis must:

1. Validate the schema.
- Reject unsupported schema versions with HTTP `400`.

2. Create or update a Navis session keyed by `run.run_id`.
- Persist the entire payload or a normalized equivalent.

3. Prepare content loading.
- Use `content.ply_dir_path` as the source directory for PLY playback.
- Enumerate `*.ply`, sort by frame number extracted from filename, and build the frame timeline.

4. Prepare trace loading.
- Load `traces.navigation_trace_path` as the navigation/head trace.
- Load `traces.bandwidth_trace_path` as the bandwidth trace.

5. Build a segment timeline.
- Use `segments[].segment_idx` and `segments[].duration_s`.
- If needed, map current playback time `t` to segment index with:
- `segment_idx = floor(t / playback.segment_duration_s)`

6. Build a tile lookup.
- Use `tiling.tile_catalog[tile_id].bounds` as the authoritative tile AABB.
- Navis should precompute point-to-tile membership for each PLY frame by testing each point against tile AABBs.
- Cache this mapping per frame so recoloring is fast during playback.

## Required Navis Visualization Features

Navis should implement three Prism-specific overlays/modes.

### 1. ABR representation coloring

Navis must support a color mode driven by `segments[].abr_applied`.

Behavior:

- For the active segment, compute a scalar quality score per tile from `abr_applied[tile_id]`.
- Suggested default score:
- average of component rep indices normalized by `tiling.component_max_rep_idx`

Formula:

```text
tile_quality(tile_id) =
mean(rep_idx(component) / max_rep_idx(component))
```

If `max_rep_idx(component) == 0`, treat that component as `1.0` when selected at `0`.

Apply the scalar to all points belonging to that tile.

Recommended palette:

- low quality = cool/dim
- high quality = warm/bright

Navis should also support toggling between:

- `abr_applied`
- `abr_requested`

This allows visualizing cancellations/downswitches when requested and applied differ.

### 2. FoV visibility coloring

Navis must support color modes driven by:

- `segments[].fov_prediction`
- `segments[].fov_ground_truth`
- `segments[].fov_visibility` as a backward-compatible fallback for older exports

Behavior:

- For the active segment, lookup the selected FoV channel by tile id.
- Apply the visibility score to the tile color or alpha.

Recommended defaults:

- `1.0` = highly visible / bright
- `0.0` = out of FoV / dimmed

Prediction and ground truth must remain distinct in Navis UI. If ground truth is absent, hide or disable the ground-truth mode instead of silently showing prediction data.

This mode should be independent from ABR mode, or combinable with it if Navis supports multi-channel styling.

### 3. Navigation + bandwidth overlays

Navigation overlay requirements:

- Parse the navigation trace CSV from `traces.navigation_trace_path`.
- Show user position in world space.
- Show viewing direction as a ray, arrow, frustum, or equivalent.
- Sync the shown trace state to the active playback time.

Bandwidth overlay requirements:

- Parse the bandwidth trace CSV from `traces.bandwidth_trace_path`.
- Show current bandwidth in real time during playback.
- Optionally show a compact chart with current sample and recent history.

## Expected Navis CSV Handling

Navigation trace:

- Use Prism FoV CSV directly.
- Minimum useful fields are `FrameNumber` and pose/orientation columns.
- Navis should support the column names used by Prism datasets.

Bandwidth trace:

- Expect CSV with:
- column 1 = time in seconds
- column 2 = bandwidth in Mbps

Navis should interpolate or step-hold bandwidth by current playback time.

## Recommended Session Model In Navis

Navis should create an internal `PrismRunSession` equivalent with:

- `run_id`
- `ply_dir_path`
- `navigation_trace_path`
- `bandwidth_trace_path`
- `segment_duration_s`
- `tile_catalog`
- `component_max_rep_idx`
- `segments`
- cached point-to-tile membership

This should be the only state needed to replay the run.

## Recommended HTTP Response From Navis

After successful ingest, return:

```json
{
  "redirect_url": "/runs/run_260306_160000_abcd"
}
```

That page should immediately launch the Prism compatibility viewer for the stored run.

## Implementation Notes For The Navis LLM

When modifying Navis for Prism compatibility, the LLM should:

1. Add `GET /health` if it does not exist.
2. Add `POST /api/prism/runs` with schema validation for `prism.navis.compat.v2`.
3. Add a Prism session store keyed by `run_id`.
4. Add a Prism run viewer route that loads:
- PLY directory
- navigation trace
- bandwidth trace
- tile catalog
- segment timeline
5. Add tile recoloring modes:
- `abr_applied`
- `abr_requested`
- `fov_prediction`
- `fov_ground_truth`
6. Add an overlay showing:
- current segment index
- current bandwidth
- current user position
- current viewing direction
7. Precompute and cache point-to-tile assignments from tile AABBs for performance.
