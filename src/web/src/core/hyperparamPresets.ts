import type { ParamMap, ScalarParam } from './types';

const PRESETS_KEY = 'prism_hyperparam_presets_v1';

// Six-neighbor reference used for the reachable-origin and ray-time sweeps.
// Variants below intentionally change one budget only.
const RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE: ParamMap = {
  navigation_voxel_size_l: '0.6',
  visibility_voxel_size_l: '0.35',
  r_nav: '1',
  use_obstacles: true,
  path_neighbor_connectivity: 6,
  k_path_dir: 3,
  history_horizons_s: '0.5,1.5,3.0',
  stability_weight: 0.4,
  mismatch_penalty_strength: 0.4,
  lambda_max: 0.95,
  softmax_temp: 1,
  samples_per_tile: 'auto',
  max_voxels_eval: 'auto',
  min_voxel_prob: '0',
  r_vis: 20,
  sigma_g: null,
  visibility_floor: 0.015,
  rng_seed: 0,
  max_rays_per_tile: 6,
  max_voxels_per_tile: 200,
  max_candidate_voxels: 500,
  voxel_selection_mode: 'probability_diverse',
  voxel_selection_core_fraction: 0.7,
  aggregation_mode: 'source_propagation',
  source_presence_sigma_l: null,
  source_score_tau: 1,
  source_min_candidates: 1,
  propagation_factor: 0.5,
  propagation_visibility_threshold: 1 / 3,
  raycast_processes: 1,
  auto_entry_width: 0.9,
  auto_entry_height: 2,
  auto_clearance_margin: 0.1,
  auto_r_nav_horizon_s: 3,
  auto_visibility_time_budget_ms: 3000,
  auto_visibility_ms_per_ray_guess: 0.007,
  auto_voxel_mass_target: 0.99,
  auto_tile_confidence: 0.9,
  auto_tile_support_points: 24,
  auto_min_samples_per_tile: 4,
  auto_max_samples_per_tile: 24,
  visual_debug_enabled: false,
  visual_debug_segments: '',
  visual_debug_max_tiles: 50,
  visual_debug_max_voxels: 400,
  visual_debug_max_contributors_per_tile: 8,
  visual_debug_max_rays_per_contributor: 4,
  visual_debug_trace_voxels: false,
  cache_size: 8,
  debug: false,
};

// Fixed ray-work counterpart for predictor-comparison and causal-ablation
// experiments. The 500-origin candidate budget and tile-point count stay
// fixed; the earlier support uses automatic mass-aware pruning so selection
// ablations can still draw from a probability tail.
const RAYCAST_FIXED_WORK_6_REFERENCE: ParamMap = {
  ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
  max_voxels_eval: 'auto',
  min_voxel_prob: 0,
  // Matches the stable sweep's observed mean of about six samples per active
  // tile, and the reference's six-ray-per-tile detail setting.
  samples_per_tile: 6,
};

const BUILTIN_PRESETS: HyperparamPresetStore = {
  abr: {
    ScoreThresholdBudgetABR: {
      paper_conservative: { safety_factor: 0.7 },
    },
    RankBudgetABR: {
      paper_conservative: { safety_factor: 0.7 },
      // Use the ground-truth bandwidth estimate without an additional
      // controller-side headroom reduction in the bandwidth-QoE study.
      perfect_bandwidth: { safety_factor: 1.0 },
    },
    FoVRoundRobinABR: {
      perfect_bandwidth: { safety_factor: 1.0 },
    },
    CoverageAwareUtilityABR: {
      prototype_spread: { safety_factor: 1.0, score_power: 0.75, coverage_strength: 1.5, fallback_score: 0.025 },
      // Cost-aware concave allocation for sparse, spatial score maps such as
      // Argus. It remains predictor-agnostic and must be evaluated fairly.
      cost_aware_coverage: { safety_factor: 1.0, score_power: 0.75, coverage_strength: 2.5, fallback_score: 0.025 },
    },
  },
  bandwidth: {},
  fov: {
    LinearExtrapolationPredictor: {
      linear_2s: { lookback_s: 2 },
    },
    LinearPoseExtrapolationPredictor: {
      linear_2s: { lookback_s: 2 },
    },
    DistanceBasedTileImportancePredictor: {
      sigma_8m: { sigma: 8 },
    },
    TrainedFovPredictor: {
      // Retrained 2026-07-23.  These are persistent Training-model IDs;
      // the preset labels remain stable for saved batch plans.
      lstm_10: { model_id: 'model_260723_171707_8df3' },
      mlp_10: { model_id: 'model_260723_171401_5ca2' },
    },
    ReachabilityRaycastPredictor: {
      broader_mass_26: {
        navigation_voxel_size_l: 0.5,
        visibility_voxel_size_l: 0.5,
        path_neighbor_connectivity: 26,
        r_nav: 'auto',
        use_obstacles: true,
        k_path_dir: 3,
        history_horizons_s: '0.5,1.5,3.0',
        stability_weight: 0.4,
        mismatch_penalty_strength: 0.4,
        lambda_max: 0.95,
        softmax_temp: 1.8,
        samples_per_tile: 'auto',
        max_voxels_eval: 256,
        min_voxel_prob: 0,
        r_vis: 100,
        sigma_g: null,
        visibility_floor: 0,
        rng_seed: 0,
        max_rays_per_tile: 3,
        auto_entry_width: 0.9,
        auto_entry_height: 2,
        auto_clearance_margin: 0.1,
        auto_r_nav_horizon_s: 3,
        auto_visibility_time_budget_ms: 200,
        auto_visibility_ms_per_ray_guess: 0.002,
        auto_voxel_mass_target: 0.98,
        auto_tile_confidence: 0.9,
        auto_tile_support_points: 24,
        auto_min_samples_per_tile: 2,
        auto_max_samples_per_tile: 48,
        cache_size: 8,
        debug: false,
      },
      aggressive_floor_26: {
        navigation_voxel_size_l: 0.5,
        visibility_voxel_size_l: 0.5,
        path_neighbor_connectivity: 26,
        r_nav: 'auto',
        use_obstacles: true,
        k_path_dir: 3,
        history_horizons_s: '0.5,1.5,3.0',
        stability_weight: 0.4,
        mismatch_penalty_strength: 0.4,
        lambda_max: 0.95,
        softmax_temp: 1.8,
        samples_per_tile: 'auto',
        max_voxels_eval: 256,
        min_voxel_prob: 0,
        r_vis: 100,
        sigma_g: null,
        visibility_floor: 0.04,
        rng_seed: 0,
        max_rays_per_tile: 3,
        auto_entry_width: 0.9,
        auto_entry_height: 2,
        auto_clearance_margin: 0.1,
        auto_r_nav_horizon_s: 3,
        auto_visibility_time_budget_ms: 200,
        auto_visibility_ms_per_ray_guess: 0.002,
        auto_voxel_mass_target: 0.98,
        auto_tile_confidence: 0.9,
        auto_tile_support_points: 24,
        auto_min_samples_per_tile: 2,
        auto_max_samples_per_tile: 48,
        cache_size: 8,
        debug: false,
      },
      two_second_spread_26: {
        navigation_voxel_size_l: 0.5,
        visibility_voxel_size_l: 0.5,
        path_neighbor_connectivity: 26,
        r_nav: 'auto',
        use_obstacles: true,
        k_path_dir: 3,
        history_horizons_s: '0.5,1.5,3.0',
        stability_weight: 0.4,
        mismatch_penalty_strength: 0.4,
        lambda_max: 0.95,
        softmax_temp: 2.2,
        samples_per_tile: 'auto',
        max_voxels_eval: 512,
        min_voxel_prob: 0,
        r_vis: 100,
        sigma_g: null,
        visibility_floor: 0.015,
        rng_seed: 0,
        max_rays_per_tile: 6,
        max_voxels_per_tile: 48,
        auto_entry_width: 0.9,
        auto_entry_height: 2,
        auto_clearance_margin: 0.1,
        auto_r_nav_horizon_s: 3,
        auto_visibility_time_budget_ms: 2000,
        auto_visibility_ms_per_ray_guess: 0.007,
        auto_voxel_mass_target: 0.99,
        auto_tile_confidence: 0.9,
        auto_tile_support_points: 24,
        auto_min_samples_per_tile: 4,
        auto_max_samples_per_tile: 24,
        cache_size: 8,
        debug: false,
      },
      // Candidate-origin budget sweep.  Each retains the 3 s ray-work budget.
      'raycast_3000ms_origins_500_6n_0.6': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
      },
      'raycast_3000ms_origins_64_6n_0.6': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        max_candidate_voxels: 64,
      },
      'raycast_3000ms_origins_128_6n_0.6': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        max_candidate_voxels: 128,
      },
      'raycast_3000ms_origins_250_6n_0.6': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        max_candidate_voxels: 250,
      },
      'raycast_3000ms_origins_750_6n_0.6': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        max_candidate_voxels: 750,
      },
      'raycast_3000ms_origins_1000_6n_0.6': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        max_candidate_voxels: 1000,
      },
      // Ray-work budget sweep.  The existing 3 s reference remains the upper
      // point because its measured end-to-end p95 is already about 4.1 s.
      'raycast_500ms_origins_500_6n_0.6': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        auto_visibility_time_budget_ms: 500,
      },
      'raycast_1000ms_origins_500_6n_0.6': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        auto_visibility_time_budget_ms: 1000,
      },
      'raycast_2000ms_origins_500_6n_0.6': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        auto_visibility_time_budget_ms: 2000,
      },
      // Mechanistic ablations and sensitivity controls, each anchored to the
      // 3 s / 500-origin / 6-neighbor reference unless stated otherwise.
      'raycast_3000ms_origins_500_26n_0.6': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        path_neighbor_connectivity: 26,
      },
      'raycast_3000ms_origins_500_6n_0.6_single_horizon': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        history_horizons_s: '1.5',
      },
      'raycast_3000ms_origins_500_6n_0.6_stability_0.8': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        stability_weight: 0.8,
      },
      'raycast_3000ms_origins_500_6n_0.6_topk': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        voxel_selection_mode: 'probability_topk',
      },
      // Per-tile ray-detail sweep. The 6-ray reference above is the middle point.
      'raycast_3000ms_origins_500_6n_0.6_rays_1': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        max_rays_per_tile: 1,
      },
      'raycast_3000ms_origins_500_6n_0.6_rays_3': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        max_rays_per_tile: 3,
      },
      'raycast_3000ms_origins_500_6n_0.6_rays_12': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        max_rays_per_tile: 12,
      },
      // Causal origin-distribution ablations: retain the same scene geometry,
      // ray work, and candidate budget as the reference.
      'raycast_3000ms_origins_500_6n_0.6_current_position': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        origin_distribution_mode: 'current_position',
      },
      'raycast_3000ms_origins_500_6n_0.6_uniform_reachable': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        origin_distribution_mode: 'uniform_reachable',
      },
      'raycast_3000ms_origins_500_6n_0.6_no_obstacles': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE,
        use_obstacles: false,
      },
      // Eight-process variants used by the raycast microbenchmarks.  They
      // retain the exact experimental setting above and alter only execution
      // parallelism inside the raycast visibility stage.
      'raycast_3000ms_origins_64_6n_0.6_processes_8': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE, max_candidate_voxels: 64, raycast_processes: 8,
      },
      'raycast_3000ms_origins_128_6n_0.6_processes_8': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE, max_candidate_voxels: 128, raycast_processes: 8,
      },
      'raycast_3000ms_origins_250_6n_0.6_processes_8': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE, max_candidate_voxels: 250, raycast_processes: 8,
      },
      'raycast_3000ms_origins_500_6n_0.6_processes_8': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE, raycast_processes: 8,
      },
      'raycast_3000ms_origins_750_6n_0.6_processes_8': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE, max_candidate_voxels: 750, raycast_processes: 8,
      },
      'raycast_3000ms_origins_1000_6n_0.6_processes_8': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE, max_candidate_voxels: 1000, raycast_processes: 8,
      },
      'raycast_500ms_origins_500_6n_0.6_processes_8': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE, auto_visibility_time_budget_ms: 500, raycast_processes: 8,
      },
      'raycast_1000ms_origins_500_6n_0.6_processes_8': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE, auto_visibility_time_budget_ms: 1000, raycast_processes: 8,
      },
      'raycast_2000ms_origins_500_6n_0.6_processes_8': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE, auto_visibility_time_budget_ms: 2000, raycast_processes: 8,
      },
      'raycast_3000ms_origins_500_6n_0.6_rays_1_processes_8': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE, max_rays_per_tile: 1, raycast_processes: 8,
      },
      'raycast_3000ms_origins_500_6n_0.6_rays_3_processes_8': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE, max_rays_per_tile: 3, raycast_processes: 8,
      },
      'raycast_3000ms_origins_500_6n_0.6_rays_12_processes_8': {
        ...RAYCAST_THREE_SECOND_SPREAD_6_REFERENCE, max_rays_per_tile: 12, raycast_processes: 8,
      },
      // Fixed-work variants for fair predictor and causal-ablation studies.
      // Do not use these to measure runtime adaptation; use the budget sweep
      // above for that purpose.
      'raycast_fixed_origins_500_samples_6_6n_0.6': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE,
      },
      // CPU-scaling controls.  Keep the deterministic reference work fixed
      // and vary only the number of processes used to split target-tile
      // visibility evaluation.
      'raycast_fixed_origins_500_samples_6_6n_0.6_processes_1': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE,
        raycast_processes: 1,
      },
      'raycast_fixed_origins_500_samples_6_6n_0.6_processes_2': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE,
        raycast_processes: 2,
      },
      'raycast_fixed_origins_500_samples_6_6n_0.6_processes_4': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE,
        raycast_processes: 4,
      },
      'raycast_fixed_origins_500_samples_6_6n_0.6_processes_8': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE,
        raycast_processes: 8,
      },
      // Reproducible visual-comparison variant: same fixed Argus work as the
      // eight-process reference, with compact ray/voxel debug information.
      'raycast_fixed_origins_500_samples_6_6n_0.6_processes_8_paper_vis': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE,
        raycast_processes: 8,
        visual_debug_enabled: true,
        visual_debug_trace_voxels: true,
      },
      // Direct expected-visibility control for the source-propagation
      // ablation.  It intentionally differs from the fixed-work reference
      // only in how direct ray evidence is aggregated into tile scores.
      'raycast_fixed_origins_500_samples_6_6n_0.6_expected_visibility_processes_8': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE,
        aggregation_mode: 'expected_visibility',
        raycast_processes: 8,
      },
      'raycast_fixed_origins_500_samples_6_6n_0.6_processes_16': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE,
        raycast_processes: 16,
      },
      'raycast_fixed_origins_500_samples_6_26n_0.6': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE,
        path_neighbor_connectivity: 26,
      },
      'raycast_fixed_origins_500_samples_6_6n_0.6_single_horizon': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE,
        history_horizons_s: '1.5',
      },
      'raycast_fixed_origins_500_samples_6_6n_0.6_stability_0.8': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE,
        stability_weight: 0.8,
      },
      'raycast_fixed_origins_500_samples_6_6n_0.6_topk': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE,
        voxel_selection_mode: 'probability_topk',
      },
      'raycast_fixed_origins_500_samples_6_6n_0.6_current_position': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE,
        origin_distribution_mode: 'current_position',
      },
      'raycast_fixed_origins_500_samples_6_6n_0.6_uniform_reachable': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE,
        origin_distribution_mode: 'uniform_reachable',
      },
      'raycast_fixed_origins_500_samples_6_6n_0.6_no_obstacles': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE,
        use_obstacles: false,
      },
      'raycast_fixed_origins_500_samples_6_26n_0.6_processes_8': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE, path_neighbor_connectivity: 26, raycast_processes: 8,
      },
      'raycast_fixed_origins_500_samples_6_6n_0.6_single_horizon_processes_8': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE, history_horizons_s: '1.5', raycast_processes: 8,
      },
      'raycast_fixed_origins_500_samples_6_6n_0.6_stability_0.8_processes_8': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE, stability_weight: 0.8, raycast_processes: 8,
      },
      'raycast_fixed_origins_500_samples_6_6n_0.6_topk_processes_8': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE, voxel_selection_mode: 'probability_topk', raycast_processes: 8,
      },
      'raycast_fixed_origins_500_samples_6_6n_0.6_current_position_processes_8': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE, origin_distribution_mode: 'current_position', raycast_processes: 8,
      },
      'raycast_fixed_origins_500_samples_6_6n_0.6_uniform_reachable_processes_8': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE, origin_distribution_mode: 'uniform_reachable', raycast_processes: 8,
      },
      'raycast_fixed_origins_500_samples_6_6n_0.6_no_obstacles_processes_8': {
        ...RAYCAST_FIXED_WORK_6_REFERENCE, use_obstacles: false, raycast_processes: 8,
      },
    },
  },
};

export type PresetScope = 'abr' | 'fov' | 'bandwidth';

export interface HyperparamPresetStore {
  abr: Record<string, Record<string, ParamMap>>;
  fov: Record<string, Record<string, ParamMap>>;
  bandwidth: Record<string, Record<string, ParamMap>>;
}

function isScalar(value: unknown): value is ScalarParam {
  return value === null || ['string', 'number', 'boolean'].includes(typeof value);
}

function sanitizeParamMap(raw: unknown): ParamMap {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return {};
  const out: ParamMap = {};
  Object.entries(raw as Record<string, unknown>).forEach(([k, v]) => {
    if (typeof k !== 'string') return;
    if (!isScalar(v)) return;
    out[k] = v;
  });
  return out;
}

function sanitizeScopeRecord(raw: unknown): Record<string, Record<string, ParamMap>> {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return {};
  const out: Record<string, Record<string, ParamMap>> = {};
  Object.entries(raw as Record<string, unknown>).forEach(([algoName, presetsRaw]) => {
    if (typeof algoName !== 'string') return;
    if (!presetsRaw || typeof presetsRaw !== 'object' || Array.isArray(presetsRaw)) return;
    const presets: Record<string, ParamMap> = {};
    Object.entries(presetsRaw as Record<string, unknown>).forEach(([presetName, paramsRaw]) => {
      if (typeof presetName !== 'string') return;
      presets[presetName] = sanitizeParamMap(paramsRaw);
    });
    out[algoName] = presets;
  });
  return out;
}

function normalizeStore(raw: unknown): HyperparamPresetStore {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return { abr: {}, fov: {}, bandwidth: {} };
  }
  const obj = raw as Record<string, unknown>;
  return {
    abr: sanitizeScopeRecord(obj.abr),
    fov: sanitizeScopeRecord(obj.fov),
    bandwidth: sanitizeScopeRecord(obj.bandwidth),
  };
}

function mergeBuiltinPresets(store: HyperparamPresetStore): HyperparamPresetStore {
  const storedRaycast = store.fov.ReachabilityRaycastPredictor || {};
  const builtinRaycast = BUILTIN_PRESETS.fov.ReachabilityRaycastPredictor || {};
  // These presets name frozen paper configurations.  A browser can retain an
  // older copy under the same name, which would otherwise make an imported
  // batch plan advertise one configuration while submitting another.  Keep
  // user-created presets available, but make the fixed-work family shipped
  // with Prism authoritative.
  const authoritativeFixedWork = Object.fromEntries(
    Object.entries(builtinRaycast).filter(([name]) => name.startsWith('raycast_fixed_')),
  );
  return {
    abr: { ...BUILTIN_PRESETS.abr, ...store.abr },
    bandwidth: { ...BUILTIN_PRESETS.bandwidth, ...store.bandwidth },
    fov: {
      ...BUILTIN_PRESETS.fov,
      ...store.fov,
      // ``linear_2s`` replaced the historical frame-count preset.  A browser
      // may retain an older local preset under the same name, which would make
      // an imported batch plan display ``linear_2s`` while actually sending
      // its stale ``lookback`` parameter.  Keep this built-in authoritative;
      // user presets remain available under their own names.
      LinearExtrapolationPredictor: {
        ...(store.fov.LinearExtrapolationPredictor || {}),
        ...(BUILTIN_PRESETS.fov.LinearExtrapolationPredictor || {}),
      },
      ReachabilityRaycastPredictor: {
        ...builtinRaycast,
        ...storedRaycast,
        ...authoritativeFixedWork,
      },
      // Model IDs change when a model is retrained.  Retain user-created
      // trained-model presets, but do not allow a browser's old saved copy of
      // these built-ins to point batch plans at a deleted model.
      TrainedFovPredictor: {
        ...(store.fov.TrainedFovPredictor || {}),
        ...(BUILTIN_PRESETS.fov.TrainedFovPredictor || {}),
      },
    },
  };
}

export function loadHyperparamPresets(): HyperparamPresetStore {
  if (typeof localStorage === 'undefined') return mergeBuiltinPresets({ abr: {}, fov: {}, bandwidth: {} });
  try {
    const parsed = JSON.parse(localStorage.getItem(PRESETS_KEY) || '{}');
    return mergeBuiltinPresets(normalizeStore(parsed));
  } catch {
    return mergeBuiltinPresets({ abr: {}, fov: {}, bandwidth: {} });
  }
}

export function saveHyperparamPresets(store: HyperparamPresetStore): void {
  if (typeof localStorage === 'undefined') return;
  localStorage.setItem(PRESETS_KEY, JSON.stringify(store));
}

export function listPresetNames(store: HyperparamPresetStore, scope: PresetScope, algoName: string): string[] {
  return Object.keys(store[scope][algoName] || {}).sort((a, b) => a.localeCompare(b));
}

export function getPresetParams(
  store: HyperparamPresetStore,
  scope: PresetScope,
  algoName: string,
  presetName: string,
): ParamMap | null {
  const params = store[scope][algoName]?.[presetName];
  return params ? { ...params } : null;
}

export function upsertPreset(
  store: HyperparamPresetStore,
  scope: PresetScope,
  algoName: string,
  presetName: string,
  params: ParamMap,
): HyperparamPresetStore {
  const next: HyperparamPresetStore = {
    abr: { ...store.abr },
    fov: { ...store.fov },
    bandwidth: { ...store.bandwidth },
  };
  const byAlgo = { ...(next[scope][algoName] || {}) };
  byAlgo[presetName] = { ...params };
  next[scope] = {
    ...next[scope],
    [algoName]: byAlgo,
  };
  return next;
}

export function deletePreset(
  store: HyperparamPresetStore,
  scope: PresetScope,
  algoName: string,
  presetName: string,
): HyperparamPresetStore {
  const next: HyperparamPresetStore = {
    abr: { ...store.abr },
    fov: { ...store.fov },
    bandwidth: { ...store.bandwidth },
  };
  const byAlgo = { ...(next[scope][algoName] || {}) };
  delete byAlgo[presetName];
  const scopeMap = { ...next[scope] };
  if (Object.keys(byAlgo).length === 0) {
    delete scopeMap[algoName];
  } else {
    scopeMap[algoName] = byAlgo;
  }
  next[scope] = scopeMap;
  return next;
}
