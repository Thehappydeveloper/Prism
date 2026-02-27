# Prism Developer Guide: Extending FoV Prediction and ABR

This document explains how Prism is structured so you can implement new **FoV predictors** and **ABR algorithms** without digging through code first.

## 1. What Prism Simulates

Prism is an offline simulator for tiled MPEG-I V3C/VPCC streaming. For each segment, it:

1. Reads available tile/component representations from an MPD.
2. Asks an ABR algorithm which representation index to request per tile component.
3. Simulates download under a network trace.
4. Enqueues the downloaded segment into a playback buffer.
5. Logs events and artifacts for analysis/QoE.

No decoding/rendering is done; this is a control-loop simulator.

## 2. Repo Structure (Developer-Facing)

- `src/app.py`
  - Main simulation orchestrator (`PlaybackConfig`, `run_playback`).
  - Instantiates network, video, FoV, ABR, player.
- `src/player/player.py`
  - Core control loop (`Player.play()`), ABR invocation, download simulation, buffer interaction.
- `src/abr/base.py`
  - ABR interface contract.
- `src/abr/baselines.py`
  - Built-in ABR algorithms.
  - New ABRs should be added here if you want UI auto-discovery.
- `src/fov_predictors/base.py`
  - FoV predictor interface + geometric helpers (`_points_in_fov`, `tile_fov_proportion`).
- `src/fov_predictors/baselines.py`
  - Built-in FoV predictors.
  - New predictors should be added here for UI auto-discovery.
- `src/core/video.py`
  - Converts MPD content into tile/component choices per segment.
- `src/core/media_segment.py`
  - Data classes for `TileComponentOptions`, `ComponentSegment`, `TileComponentSet`.
- `src/core/mpd_parser.py`
  - Parses MPD adaptation sets, segment templates, tile bounds.
- `src/core/network.py`
  - Trace-driven bandwidth and RTT model.
- `src/core/fov.py`
  - FoV trace loader (`FrameNumber`-indexed DataFrame) and predictor holder.
- `src/core/buffer.py`
  - Buffer occupancy + rebuffer accounting.
- `src/core/event_logger.py`
  - Persisted telemetry (`events.json`).
- `src/server.py`
  - Web API + algorithm class discovery for UI.

## 3. End-to-End Runtime Flow

The main call chain is:

`run_playback()` (`src/app.py`) -> `Player.play()` (`src/player/player.py`) -> per segment:

1. `video.list_segment_tiles(seg_idx)` builds ABR options.
2. `abr.decide(...)` returns desired rep index per tile component.
3. Player sanitizes/clamps decision indices.
4. Optional player-side augmentation:
   - FoV-centric upgrades (`enable_fov_upgrades`)
   - Mid-download cancellation/downswitch (`enable_mid_download_cancel`)
5. Player simulates RTT + component downloads using instantaneous trace bandwidth.
6. Segment is enqueued; consume/rebuffer events are logged.

Artifacts are written to `results/<run_id>/`:

- `config.json`
- `events.json`
- `<run_id>.log`
- `<run_id>.ndjson`

## 4. Core Data Contracts You Must Target

### 4.1 ABR input tile object

ABR receives `List[TileComponentOptions]` where each tile contains:

- `tile_id`, `segment_idx`
- geometric bounds (`min_corner`, `max_corner`, `center`)
- `component_options: Dict[str, List[ComponentSegment]]`

Each `ComponentSegment` includes:

- `rep_idx` (integer index expected in ABR output)
- `size_bits`, `duration_s`, `bandwidth_bps`
- `quality_id` and source file path metadata

### 4.2 ABR output

ABR must return:

```python
Dict[str, Dict[str, int]]
# {
#   tile_id: {
#     component_name: rep_idx
#   }
# }
```

Player behavior for imperfect output:

- Unknown tile IDs are ignored.
- Missing components default to lowest quality (`0`).
- Out-of-range indices are clamped.

### 4.3 FoV predictor output (runtime expectation)

At runtime, ABR uses `ABR.predict_fov(...)`, which expects predictor output shaped like:

```python
Dict[str, float]  # tile_id -> FoV proportion in [0, 1]
```

In built-ins, `> 0` means the tile is considered visible.

## 5. Implementing a New ABR Algorithm

### 5.1 Required interface

Inherit `ABR` and implement:

```python
def decide(self, tiles, segment_frames_path, network, fov, buffer, clock) -> Dict[str, Dict[str, int]]:
    ...
```

You can use:

- `network.get_instantaneous_bandwidth(clock.get_time())`
- `network.get_instantaneous_rtt(clock.get_time())`
- `buffer.buffer_duration_s()` / `buffer.has_space_for_one_more`
- `self.predict_fov(fov, tile_groups, segment_frames_path)` for tile visibility priors

### 5.2 Registration/discovery rules

- Add class in `src/abr/baselines.py`.
- Name must not start with `_`.
- Must subclass `ABR`.
- If your ABR needs FoV predictor output, set:

```python
abr_requires_predictor = True
```

This is enforced by CLI and shown in UI.

### 5.3 Minimal template

```python
from abr.base import ABR

class MyFoVAwareABR(ABR):
    abr_requires_predictor = True

    def __init__(self, safety_factor: float = 0.8):
        self.safety_factor = safety_factor

    def decide(self, tiles, segment_frames_path, network, fov, buffer, clock):
        tile_groups = self.group_tiles(tiles)
        fov_prop = self.predict_fov(fov, tile_groups, segment_frames_path)
        bw = network.get_instantaneous_bandwidth(clock.get_time()) * self.safety_factor

        decisions = {}
        for tile_id, tile in tile_groups.items():
            decisions[tile_id] = {}
            for comp, reps in tile.component_options.items():
                decisions[tile_id][comp] = len(reps) - 1 if fov_prop.get(tile_id, 0.0) > 0 else 0
        return decisions
```

## 6. Implementing a New FoV Predictor

### 6.1 Required interface

Inherit `FovPredictor` and implement:

```python
def predict(self, tiles, frame_paths, fov_data):
    ...
```

Runtime expectation: return `Dict[tile_id, proportion]`.

### 6.2 Typical predictor pattern

1. Determine which frame index to predict for using `frame_paths`.
2. Read pose from FoV trace (`fov_data.get_frame(...)`).
3. Read point cloud for the target frame (usually from last frame path in segment).
4. Build FoV mask with `_points_in_fov(...)`.
5. Convert mask to per-tile proportions via `tile_fov_proportion(...)`.

### 6.3 FoV trace assumptions in built-ins

Built-in predictors assume columns like:

- `FrameNumber` (index key)
- position: `HMDPX`, `HMDPY`, `HMDPZ`
- orientation: `HMDRX`, `HMDRY`, `HMDRZ`

If your dataset differs, normalize in your predictor.

### 6.4 Minimal template

```python
from fov_predictors.base import FovPredictor
from pathlib import Path
import numpy as np

class MyPredictor(FovPredictor):
    def predict(self, tiles, frame_paths, fov_data):
        if not frame_paths:
            return {t.unique_id: 1.0 for t in tiles}

        frame_path = Path(frame_paths[-1])
        frame_idx = int("".join(ch for ch in frame_path.stem if ch.isdigit()))
        row = fov_data.get_frame(frame_idx - fov_data.start_frame_index)
        if row is None:
            return {t.unique_id: 1.0 for t in tiles}

        pos = row[["HMDPX", "HMDPY", "HMDPZ"]].to_numpy(dtype=float)
        # Replace with your orientation/forward prediction logic.
        forward = np.array([0.0, 0.0, 1.0], dtype=float)

        points = self.ply_reader.read(frame_path)
        mask = self._points_in_fov(points, pos, forward, fov_h=self.fov_h, fov_v=self.fov_v)
        return self.tile_fov_proportion(tiles, mask, points)
```

## 7. Important Behavioral Details and Gotchas

- First segment ABR call gets `segment_frames_path = None` (player passes `next_segment - 1`). Handle this explicitly.
- `rep_idx` is not arbitrary; it maps to MPD-derived component quality ordering in `Video`.
- ABR output is per-component; not all tiles necessarily share component counts.
- Player may modify effective quality even after ABR decision if FoV upgrades/cancel flags are enabled.
- UI/server class discovery (`src/server.py`) only lists subclasses defined in `abr.baselines` / `fov_predictors.baselines` modules.
- `FovPredictor.predict` type hints in `src/fov_predictors/base.py` are stale; follow runtime usage (dict of tile proportions).

## 8. Where to Plug in Evaluation Logic

- Playback-level telemetry: `results/<run_id>/events.json` from `EventLogger`.
- QoE models: `src/qoe/heuristics.py` (`QoEFunction` subclasses).
- Diagnostics API: `POST /api/diagnostics` in `src/server.py` computes cross-run metrics.

If you add ABR/FoV metadata that should influence diagnostics, extend `events.json` schema via `EventLogger` and adjust server aggregation helpers.

## 9. Recommended Development Workflow for New Algorithms

1. Implement predictor in `src/fov_predictors/baselines.py`.
2. Implement ABR in `src/abr/baselines.py`.
3. Ensure ABR returns full tile/component decision dict.
4. Run a short experiment and inspect:
   - segment download times
   - rebuffering events
   - per-component representation choices in `events.json`
5. Compare against baseline ABRs (`BandwidthAwareABR`, `FoVAwareUniformABR`, etc.) under identical traces.
6. Validate sensitivity across both stable and volatile network traces.

This keeps algorithm research isolated while remaining compatible with Prism’s existing orchestration, UI discovery, and diagnostics stack.
