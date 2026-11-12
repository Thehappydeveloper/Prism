# Reachability Raycast FoV Predictor

This folder contains the advanced FoV predictor used by `ReachabilityRaycastPredictor`.
It is intentionally separated from baseline predictors.

## Pipeline (high level)

Per decision point, the predictor computes tile importance in four stages:

1. Voxelize scene geometry and build the reachable navigation volume.
2. Estimate user intention from pose history (movement + gaze mismatch).
3. Convert intention into a probability distribution over reachable voxels.
4. Raycast visibility from likely voxels to sampled tile points, then aggregate to tile probabilities.

Output:

- `{"tile_id": probability_in_[0,1], ...}` (normalized, sums to 1)

## File map

- `predictor.py`: predictor class, per-frame caches, orchestration.
- `reachability_pipeline.py`: core stage logic and probability computations.
- `pose_intention.py`: pose parsing and confidence model.
- `path_planning.py`: 26-connected Dijkstra and path-direction features.
- `voxel_grid.py`: voxel grid primitives + occupancy utilities.
- `raycast_visibility.py`: voxel traversal + tile visibility aggregation.

## Hyperparameter reference

Unless noted as dataset- or MPD-derived, parameters below are constructor args of `ReachabilityRaycastPredictor`.

### Navigation / geometry

- `voxel_size_l` (default `"auto"`)
  Controls voxel resolution. Smaller = finer spatial precision, but more voxels and higher runtime.
  Typical range: `0.3` to `0.8`.
  Can also be set to `"auto"`, in which case the predictor uses:
  `(min(D_w, D_h) * (1 - g)) / 4` with the configured tightest entry dimensions and a relative safety margin.
- `r_nav` (default `"auto"`)
  Extra AABB expansion around the point cloud before voxelization.
  Increase if users near scene boundaries are clipped.
  Can also be set to `"auto"`, in which case the predictor uses the registered dataset `max_speed * auto_r_nav_horizon_s` (default horizon: `3s`).
- Dataset `max_speed`
  Registered on the dataset profile and used to define reachable radius within one MPD segment.
- Segment duration
  Inferred from MPD representation metadata, not exposed as a FoV tuning hyperparameter.
- `use_obstacles` (default `True`)
  If `True`, path planning respects occupied voxels. If `False`, uses direct geometric distance/alignment.
- `k_path_dir` (default `3`)
  Number of initial path steps used to estimate intended travel direction.
  Larger values smooth direction but reduce local responsiveness.

### Intention model

- `history_frames` (default `15`)
  Number of recent pose frames used for speed/acceleration estimation.
  Lower = reactive/noisy, higher = stable/slower.
- Dataset motion limits: `max_speed` and `max_acceleration`
  Stored on the registered dataset profile, not as predictor hyperparameters.
  They normalize measured pose-trace speed and acceleration before confidence is computed.
- `stability_weight` (default `0.7`)
  Blend weight between stable predictable motion and any intense motion.
  Higher values favor steady movement; lower values react more to abrupt movement.
- `mismatch_penalty_strength` (default `1.0`)
  Strength of the gaze-vs-motion disagreement penalty.
- `lambda_max` (default `0.9`)
  Upper bound on alignment weight when blending distance cost and directional alignment.
- `softmax_temp` (default `1.0`)
  Temperature for voxel probability softmax.
  Lower (<1) sharpens peaks; higher (>1) flattens distribution.

### Visibility / aggregation

- `samples_per_tile` (default `"auto"`)
  Number of representative points sampled per tile AABB for visibility estimation.
  Increase for stability; decreases speed linearly.
  Can also be set to `"auto"` to allocate per-tile samples from a global ray budget.
- `max_voxels_eval` (default `"auto"`)
  Hard cap on number of voxels evaluated in visibility aggregation.
  Primary runtime control.
  Can also be set to `"auto"` to keep the smallest top-probability voxel set that reaches the configured mass target without blowing the ray budget.
- `min_voxel_prob` (default `"auto"`)
  Probability floor for pruning low-impact voxels before raycasting.
  Can also be set to `"auto"` to use the probability of the last voxel kept by the adaptive pruning step.
- `r_vis` (default `6.0`)
  Distance scale used in tile gating term.
- `sigma_g` (default `None`)
  Gaussian gating sigma. If `None`, uses `0.5 * r_vis`.
  Lower values focus strongly on near tiles.
- `max_rays_per_tile` (default `None`)
  Optional cap on rays per tile during visibility estimation.
  Use to bound runtime on dense tiles.

### Caching / debugging

- `rng_seed` (default `0`)
  Seed for deterministic tile-point sampling.
- `auto_entry_width` / `auto_entry_height` (defaults `0.9` / `2.0`)
  Smallest opening dimensions that must remain navigable when `voxel_size_l="auto"`.
- `auto_clearance_margin` (default `0.1`)
  Relative safety margin `g` used by the automatic voxel-size formula.
- `auto_body_width` / `auto_body_height` (defaults `0.6` / `1.8`)
  Deprecated compatibility fields. They are currently ignored by the automatic voxel-size rule.
- `auto_r_nav_horizon_s` (default `3.0`)
  Motion horizon used when `r_nav="auto"`.
- `auto_visibility_time_budget_ms` (default `20.0`)
  Wall-clock budget target for the `"auto"` visibility allocator.
- `auto_visibility_ms_per_ray_guess` (default `0.002`)
  Initial cost guess used before the predictor has enough history to estimate milliseconds per ray from recent frames.
- `auto_voxel_mass_target` (default `0.95`)
  Target cumulative voxel probability mass retained when voxel pruning is automatic.
- `auto_tile_confidence` (default `0.95`)
  Detection-confidence target for adaptive tile sampling.
- `auto_tile_support_points` (default `32`)
  Minimum visible support size, in tile points, that adaptive sampling tries not to miss.
- `auto_min_samples_per_tile` / `auto_max_samples_per_tile` (defaults `4` / `64`)
  Lower and upper bounds for adaptive per-tile sampling.
- `cache_size` (default `8`)
  LRU cache size for voxel contexts and tile samples.
- `debug` (default `False`)
  Enables logging of top tiles and voxel stats.

## Tuning guide (practical)

- To reduce runtime first:
  Decrease `max_voxels_eval`, then `samples_per_tile`, then set `max_rays_per_tile`.
- To make predictions less noisy:
  Increase `history_frames`, slightly increase `softmax_temp`, and/or increase `samples_per_tile`.
- To make predictions more proactive:
  Increase `lambda_max`, but verify this does not over-spread tile probabilities.
- If behavior is too myopic:
  Increase `r_vis` or `sigma_g`.

## Fallback behavior

If critical inputs are missing (pose, geometry, traces), predictor returns a uniform tile distribution.
This preserves playback stability and avoids hard failures.
