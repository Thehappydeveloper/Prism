export type Dict<T = unknown> = Record<string, T>;
export type ScalarParam = string | number | boolean | null;
export type ParamMap = Record<string, ScalarParam>;
export type LaunchPreflightMode = 'issues' | 'always' | 'never';
export interface LaunchPreflightCheck { severity: 'error' | 'warning'; title: string; detail?: string; affected_count: number; samples?: Array<Record<string, unknown>>; }
export interface LaunchPreflightResult { status: 'ready' | 'warning' | 'error'; run_count: number; valid_count: number; checks: LaunchPreflightCheck[]; fixed: Record<string, string>; varied: Array<{ label: string; values: string[]; count: number }>; }
export type RouteId = 'home' | 'datasets' | 'control' | 'batch' | 'compare' | 'reports' | 'settings' | 'training' | 'extra' | 'speedAcceleration' | 'pointCloudSequence' | 'traceHeightNormalizer';

export interface AlgorithmParamField {
  name: string;
  label: string;
  kind: 'number' | 'integer' | 'boolean' | 'string';
  default: ScalarParam;
  choices?: Array<{ value: string; label: string }>;
  annotation?: string;
  category?: string;
  help?: string;
  advanced?: boolean;
}

export interface QoEField {
  name: string;
  label: string;
  default: number;
  min?: number;
  max?: number;
  step?: number;
  help?: string;
}

export interface QoEModel {
  id: string;
  label: string;
  equation?: string;
  description?: string;
  implementation?: string;
  fields?: QoEField[];
  defaults?: Record<string, number>;
}

export interface AppOptions {
  mpds: string[];
  network_traces: string[];
  network_profiles?: NetworkProfile[];
  fov_traces: string[];
  network_root: string;
  fov_root: string;
  abr_classes: string[];
  fov_predictors: string[];
  fov_predictor_labels?: Record<string, string>;
  bandwidth_predictors: string[];
  abr_requires_predictor: Record<string, boolean>;
  abr_supports_predictor: Record<string, boolean>;
  abr_param_specs: Record<string, AlgorithmParamField[]>;
  fov_param_specs: Record<string, AlgorithmParamField[]>;
  bandwidth_param_specs: Record<string, AlgorithmParamField[]>;
  headset_presets: HeadsetPreset[];
  default_headset_preset?: string;
  angle_convention?: string;
  angle_conventions?: Array<{ id: string; label: string }>;
  qoe_functions: QoEModel[];
  navis?: NavisStatus;
  datasets?: DatasetProfile[];
}

export interface NetworkProfile {
  id: string;
  description: string;
  delay_ms: number;
  jitter_ms: number;
  loss_pct: number;
  rate_mbit: number | null;
  delay_distribution?: string;
}

export interface HeadsetPreset {
  id: string;
  label: string;
  fov_h_deg: number;
  fov_v_deg: number;
}

export interface HeadsetConfig {
  default_preset: string;
  angle_convention?: string;
  presets: HeadsetPreset[];
}

export interface HeadsetConfigResponse {
  settings: HeadsetConfig;
  angle_conventions?: Array<{ id: string; label: string }>;
  settings_path?: string;
}

export interface TraceRootConfig {
  network_root: string;
  fov_root: string;
}

export interface TraceRootConfigResponse {
  settings: TraceRootConfig;
  settings_path?: string;
}

export type TraceColumnAliasSettings = Record<string, Record<string, string[]>>;

export interface TraceColumnField {
  label: string;
  canonical: string;
  aliases: string[];
  required: boolean;
}

export interface TraceColumnConfigResponse {
  settings: TraceColumnAliasSettings;
  fields: Record<string, Record<string, TraceColumnField>>;
  settings_path?: string;
}

export interface PrismConfig {
  record_fov_stage_latency: boolean;
  batch_queue_coalesce_ms?: number;
  batch_queue_max_coalesce_ms?: number;
  batch_ply_io_workers?: number;
  batch_segment_workers?: number;
  batch_segment_workers_auto_cap?: number;
  batch_group_max_runs?: number;
  batch_prefetch_segments?: number;
  training_max_parallel_runs?: number;
  training_cpu_core_limit?: number;
  training_require_single_worker_for_parallel?: boolean;
}

export interface PrismConfigResponse {
  settings: PrismConfig;
  settings_path?: string;
}

export interface WorldTransform {
  scale: number;
  translation: [number, number, number];
  units?: string;
  source?: string;
  notes?: string;
}

export interface WorldTransformResponse {
  content_dir?: string;
  transform?: WorldTransform | null;
  detected_header_transform?: WorldTransform | null;
  transforms?: Array<{ content_dir: string; transform: WorldTransform }>;
  settings_path?: string;
}

export type DatasetScaleMode = 'already_normalized' | 'already_meters' | 'manual' | 'known_extent_manual' | 'known_extent_mpd';

export interface DatasetCalibration {
  mode?: DatasetScaleMode | string;
  dimension?: 'height' | 'width' | 'depth' | string;
  axis?: 'x' | 'y' | 'z' | string;
  real_extent_m?: number;
  source_extent_units?: number;
}

export interface DatasetExtentStats {
  source?: string;
  tile_bounds_count?: number;
  axis?: Record<string, { min: number; max: number; extent: number }>;
}

export interface DatasetMotionLimits {
  max_speed: number;
  max_acceleration: number;
}

export interface DatasetProfile {
  id: string;
  name: string;
  mpd_folder: string;
  mpd?: string;
  content_dir: string;
  scale_mode?: DatasetScaleMode | string;
  calibration?: DatasetCalibration;
  motion_limits?: DatasetMotionLimits;
  extent_stats?: DatasetExtentStats | null;
  transform?: WorldTransform;
  angle_convention?: string;
  apply_to_fov_trace?: boolean;
  already_normalized?: boolean;
  notes?: string;
  updated_at?: string;
}

export interface DatasetProfilesResponse {
  datasets: DatasetProfile[];
  settings_path?: string;
}

export interface TrainingDataset {
  id: string;
  family: 'fov' | 'bandwidth';
  name: string;
  trace_path: string;
  fps?: number;
  units?: string;
  prism_dataset_id?: string | null;
  prism_dataset_snapshot?: Partial<DatasetProfile> | null;
  transform?: WorldTransform | null;
  motion_limits?: DatasetMotionLimits | null;
  notes?: string;
  content_dir?: string;
  reference_ply?: string;
  angle_convention?: string;
  validation?: { file_count?: number; sample_file?: string };
  updated_at?: string;
}

export interface TrainingPlan {
  id: string;
  family: 'fov' | 'bandwidth';
  name: string;
  dataset_ids: string[];
  split_strategy?: string;
  split_assignments?: Record<string, string>;
  validation_fraction?: number;
  test_fraction?: number;
  mixing_strategy?: string;
  manual_dataset_weights?: Record<string, number>;
  max_samples_per_dataset?: number;
  seed?: number;
  history?: Record<string, number | boolean>;
  model?: Record<string, string | number | boolean>;
  training?: Record<string, string | number | boolean>;
  target?: string;
  updated_at?: string;
}

export interface TrainingRun {
  run_id: string;
  plan_id?: string;
  plan_name?: string | null;
  status: string;
  error?: string | null;
  started?: string | null;
  finished?: string | null;
  progress?: Record<string, string | number | boolean | null>;
  split_counts?: Record<string, number> | null;
  dataset_counts?: Array<Record<string, string | number>> | null;
  log?: string[];
  resources?: {
    num_workers?: number;
    estimated_core_usage?: number;
    parallel_eligible?: boolean;
  };
}

export interface TrainingModel {
  id: string;
  name?: string;
  source_model_id?: string;
  run_id?: string;
  plan_id?: string;
  family: 'fov' | 'bandwidth';
  model?: Record<string, unknown>;
  metrics?: Record<string, unknown>;
  checkpoint?: string;
  created_at?: string;
  exported?: boolean;
  exported_at?: string;
  exported_as?: string;
  predictor_class?: string;
  predictor_params?: Record<string, string>;
}

export interface TrainingMetrics {
  epochs?: Array<{ epoch: number; train_loss?: number; validation_mae?: number | null; elapsed_s?: number }>;
  test?: { mae?: number | null; evaluated_at?: string };
  graphs?: TrainingGraph[];
}

export interface TrainingGraph {
  id: string;
  title: string;
  type: 'line' | 'bar';
  x: Array<string | number>;
  series: Array<{ label: string; values: Array<number | null> }>;
}

export interface RunFormSession {
  name: string;
  config: RunConfig;
  updated_at?: string;
}

export interface RunFormSessionsResponse {
  sessions: RunFormSession[];
  settings_path?: string;
}

export interface RunSummary {
  run_id: string;
  status: string;
  error?: string;
  navis_export?: NavisExportResult;
  fov_comparison?: FovComparisonResult;
  queued_at?: string;
  queue_position?: number | null;
  started?: string;
  finished?: string;
  events_path?: string;
  log_path?: string;
  predictor_diagnostics_path?: string | null;
  artifact_fingerprint?: Record<string, { path: string; mtime_ns: number; size: number } | null>;
  progress?: {
    current: number;
    total: number;
    group_id?: string | null;
    group_size?: number;
    phase?: string;
    detail?: Record<string, string | number | boolean | null>;
    phase_started_at?: string;
    heartbeat_at?: string;
  } | null;
  config_summary?: {
    abr?: string;
    predictor?: string;
    bandwidth_predictor?: string;
    network_profile?: string | null;
    network_trace?: string;
    fov_trace?: string;
    ground_truth_mode?: string;
    oracle_mode?: string;
  };
  oracle_mode?: string;
  oracle_group_id?: string | null;
  baseline_run_id?: string | null;
  run_group_id?: string | null;
  run_group_name?: string | null;
  run_group_source?: string | null;
}

export interface RunGroup {
  id: string;
  name: string;
  description?: string;
  source?: string;
  plan_path?: string;
  created_at?: string;
  updated_at?: string;
  archived?: boolean;
  run_ids: string[];
}

export interface RunConfig {
  experiment_id?: string;
  experiment_name?: string;
  experiment_purpose?: string;
  experiment_kind?: string;
  experiment_primary_metrics?: string[];
  /** QoE model intended for post-run analysis; playback itself is unaffected. */
  analysis_qoe_function?: string;
  analysis_qoe_scoring_version?: string;
  /** Evaluation-only prefix excluded when deriving QoE from completed artifacts. */
  analysis_qoe_warmup_segments?: number;
  dataset_id?: string;
  mpd?: string;
  content_dir: string;
  network_trace?: string;
  network_profile?: string | null;
  fov_trace: string;
  abr: string;
  predictor?: string;
  bandwidth_predictor?: string;
  abr_params?: ParamMap;
  predictor_params?: ParamMap;
  bandwidth_predictor_params?: ParamMap;
  abr_preset_name?: string;
  predictor_preset_name?: string;
  bandwidth_predictor_preset_name?: string;
  headset_preset?: string;
  fov_h_deg?: number;
  fov_v_deg?: number;
  angle_convention?: string;
  rtt: number;
  block_size_bits?: number;
  enable_fov_upgrades?: boolean;
  enable_mid_download_cancel?: boolean;
  buffer_window_segments?: number;
  buffer_unbounded?: boolean;
  /** Process only the first N segments; 0 or omitted processes the full video. */
  segment_limit?: number;
  run_id?: string;
  oracle_gap_suite?: boolean;
  oracle_gap_modes?: string[];
  oracle_mode?: string;
  oracle_group_id?: string;
  baseline_run_id?: string;
  run_group_id?: string;
  run_group_name?: string;
  run_group_source?: string;
  ground_truth_mode?: string;
  ground_truth_visible_threshold?: number;
  ground_truth_peak_weight?: number;
  v3c_folder?: string;
  record_fov_stage_latency?: boolean;
  /** Optional grouped-run override: 0 selects auto, positive values fix workers. */
  segment_workers_override?: number | null;
}

export interface SegmentComponent {
  quality_id?: string;
  rep_idx?: number;
  size_bits?: number;
}

export interface SegmentTile {
  components?: Record<string, SegmentComponent>;
}

export interface EnqueuedSegment {
  time_s: number;
  duration_s: number;
  size_bits: number;
  tiles?: Record<string, SegmentTile>;
  tiles_in_fov?: string[];
  viewport_tiles?: string[];
}

export interface RebufferEvent {
  time_s: number;
  duration_s: number;
}

export interface EventLog {
  time_s: number[];
  buffer_occupancy_s: number[];
  enqueued_segments: EnqueuedSegment[];
  consumed_segments: unknown[];
  rebuffering_events: RebufferEvent[];
  abr_decisions?: Array<{
    segment_idx: number;
    time_s: number;
    requested: Record<string, Record<string, number>>;
    applied: Record<string, Record<string, number>>;
  }>;
  fov_predictions?: Array<{
    segment_idx: number;
    time_s: number;
    tile_visibility: Record<string, number>;
  }>;
  bandwidth_predictions?: Array<{
    segment_idx: number;
    time_s: number;
    prediction_time_s?: number;
    predictor?: string;
    profile?: string | null;
    window_s?: number;
    horizon_s?: number;
    sample_count?: number;
    window_start_s?: number;
    window_end_s?: number;
    window_min_bandwidth_bps?: number;
    window_max_bandwidth_bps?: number;
    measured_bandwidth_bps?: number;
    target_measured_bandwidth_bps?: number;
    actual_average_bandwidth_bps?: number;
    predicted_bandwidth_bps?: number;
    execution_time_ms?: number;
  }>;
  total_playback_time: number;
  total_rebuffering_time: number;
}

export interface RunMetrics {
  labels: Record<string, string>;
  qoe: Record<string, number[]>;
  qoe_series?: Record<string, Array<{ time_s: number; score: number }>>;
  fov_quality: Record<string, { fov: number[]; non: number[] }>;
  fov_prediction?: Record<string, {
    [key: string]: unknown;
    mae?: number[];
    rmse?: number[];
    iou?: number[];
    weighted_iou?: number[];
    top1_hit?: number[];
    topk_recall?: number[];
    false_positive_mass?: number[];
    false_negative_mass?: number[];
    playback_mae?: number[];
    playback_rmse?: number[];
    peak_miss_rate?: number[];
    visible_frame_recall?: number[];
    topk_recall_per_frame?: number[];
    wasted_visible_quality?: number[];
    missed_visible_quality?: number[];
    calibration?: Array<{ predicted: number; actual: number }>;
  }>;
  fov_segments?: Record<string, Array<{
    segment_idx: number;
    duration_s: number;
    predicted: Record<string, number>;
    actual: Record<string, number>;
    actual_mean?: Record<string, number>;
    actual_frames?: Array<{ frame: number; tile_visibility: Record<string, number> }>;
    ground_truth_mode?: string;
    tile_quality: Record<string, number>;
    tile_bits: Record<string, number>;
  }>>;
  delta_fov: Record<string, number[]>;
  stall: Record<string, { duration: number; count: number }>;
  component_bits: Record<string, Record<string, number>>;
  bitrate_vs_quality: Record<string, Array<[number, number]>>;
  buffer: Record<string, { mean: number; std: number; min: number; max: number }>;
  latency?: Record<string, {
    count: number;
    median_ms: number;
    mean_ms: number;
    p95_ms: number;
    max_ms: number;
    stages?: Array<{
      name: string;
      count: number;
      median_ms: number;
      mean_ms: number;
      p95_ms: number;
      max_ms: number;
    }>;
  }>;
  meta: Record<
    string,
    {
      duration_s?: number;
      fov_present?: boolean;
      abr?: string;
      predictor?: string;
      predictor_params?: Record<string, unknown>;
      predictor_preset_name?: string | null;
      bandwidth_predictor?: string;
      network_profile?: string | null;
      network_trace?: string;
      network_class?: string;
      fov_trace?: string;
      dataset_id?: string;
      buffer_window_segments?: number;
      error?: string;
      run_ids?: string[];
    }
  >;
}

export type QoEParams = Record<string, number>;

export interface DiagnosticsRequest {
  run_ids: string[];
  qoe_function?: string;
  qoe_params?: QoEParams;
  datasets?: Array<'qoe' | 'qoe_series' | 'fov_quality' | 'fov_prediction' | 'delta_fov' | 'stall' | 'component_bits' | 'bitrate_vs_quality' | 'buffer' | 'latency'>;
}

export interface OracleGapRequest extends DiagnosticsRequest {
  oracle_group_id?: string;
}

export interface OracleGapRun {
  run_id: string;
  mode: string;
  label: string;
  qoe: number | null;
  gap_from_actual: number | null;
  breakdown: Dict;
  config: {
    abr?: string;
    predictor?: string;
    bandwidth_predictor?: string;
  };
}

export interface OracleGapGroup {
  oracle_group_id: string;
  baseline_run_id?: string;
  validation: {
    valid: boolean;
    reason?: string;
    mismatches?: Array<{ run_id?: string; field: string }>;
  };
  runs: OracleGapRun[];
  derived_gaps: Dict<number | null>;
}

export interface OracleGapResponse {
  groups: OracleGapGroup[];
  metrics: RunMetrics;
}

export interface ExtraToolWarning {
  kind: string;
  message: string;
  path?: string;
}

export interface SpeedAccelerationDiscoverResponse {
  root_folder: string;
  csv_files: string[];
  file_count: number;
  files_loaded: number;
  files_skipped: number;
  columns: string[];
  numeric_columns: string[];
  warnings: ExtraToolWarning[];
}

export interface PointCloudSequenceDiscoverResponse {
  root_folder: string;
  recursive: boolean;
  ply_files: string[];
  file_count: number;
  first_frame_index: number | null;
  last_frame_index: number | null;
  warnings: ExtraToolWarning[];
}

export interface DistributionBin {
  label: string;
  lower_percent: number;
  upper_percent: number;
  lower: number;
  upper: number;
  count: number;
  percentage: number;
}

export interface NumericStats {
  min: number | null;
  max: number | null;
  median: number | null;
  mean: number | null;
  stddev: number | null;
  samples: number;
}

export interface OutlierSummary {
  total_samples: number;
  valid_samples: number;
  outlier_samples: number;
  percentage_removed: number;
  largest_removed: number | null;
  largest_retained: number | null;
  bounds: {
    lower: number | null;
    upper: number | null;
    q1: number | null;
    q3: number | null;
    iqr: number | null;
  };
}

export interface MotionMetricResult {
  stats: NumericStats;
  histogram: DistributionBin[];
  raw_stats: NumericStats;
  filtered_stats: NumericStats;
  raw_histogram: DistributionBin[];
  filtered_histogram: DistributionBin[];
  outlier_summary: OutlierSummary;
}

export interface PlyInspectionResponse {
  path: string;
  filename: string;
  reference?: {
    mode: 'file' | 'sequence' | string;
    reference_frame_number: number | null;
    sequence_folder: string | null;
    frame_count: number;
  };
  point_count: number;
  declared_vertex_count: number;
  bounds: {
    min_x: number;
    max_x: number;
    min_y: number;
    max_y: number;
    min_z: number;
    max_z: number;
  };
  extents: {
    x: number;
    y: number;
    z: number;
  };
}

export interface SpeedAccelerationAnalysisResponse {
  root_folder: string;
  selected_columns: {
    x: string;
    y: string;
    z: string;
  };
  timing: {
    fps: number;
    sample_interval_s: number;
  };
  scaling: {
    point_cloud: PlyInspectionResponse;
    height_axis: 'x' | 'y' | 'z' | string;
    measured_extent_units: number;
    real_height_m: number;
    scale_factor_m_per_unit: number;
    assumptions: string[];
    coordinate_example: {
      original: [number, number, number] | number[] | null;
      scaled_m: [number, number, number] | number[] | null;
    };
  };
  filtering: {
    filter_outliers: boolean;
    method: 'iqr' | 'percentile' | string;
    percentile_keep: number;
    histogram_source: 'raw' | 'filtered' | string;
    exclude_suspicious_jumps: boolean;
  };
  speed: MotionMetricResult;
  signed_acceleration: MotionMetricResult;
  quality: {
    total_samples: number;
    valid_samples: number;
    invalid_rows_skipped: number;
    suspicious_position_jumps: number;
    suspicious_position_jumps_excluded: number;
    speed_outliers_detected: number;
    acceleration_outliers_detected: number;
  };
  motion_overview: {
    recording_duration_s: number;
    total_distance_m: number;
    average_speed_mps: number | null;
    median_speed_mps: number | null;
    peak_speed_mps: number | null;
    median_acceleration_mps2: number | null;
    peak_acceleration_mps2: number | null;
    time_stationary_s: number;
    time_accelerating_s: number;
    time_decelerating_s: number;
    outliers_removed: number;
  };
  outlier_overview: {
    speed_outliers_removed: number;
    acceleration_outliers_removed: number;
    largest_removed_speed_mps: number | null;
    largest_removed_acceleration_mps2: number | null;
    largest_retained_speed_mps: number | null;
    largest_retained_acceleration_mps2: number | null;
  };
  summary: {
    files_discovered: number;
    files_read: number;
    files_skipped: number;
    rows_seen: number;
    valid_position_rows: number;
    invalid_position_rows: number;
    missing_position_rows: number;
    non_numeric_position_rows: number;
    speed_samples: number;
    raw_speed_samples: number;
    velocity_samples: number;
    signed_acceleration_samples: number;
    raw_signed_acceleration_samples: number;
    acceleration_samples: number;
  };
  warnings: ExtraToolWarning[];
}

export type PointCloudSequenceAggregation = 'minimum' | 'maximum' | 'mean' | 'median' | 'standard_deviation';

export interface PointCloudSequenceFrame {
  frame_index: number;
  file_name: string;
  path: string;
  declared_vertex_count: number;
  point_count: number;
  sampled_point_count: number;
  sample_stride: number;
  min_x: number;
  max_x: number;
  min_y: number;
  max_y: number;
  min_z: number;
  max_z: number;
  range_x: number;
  range_y: number;
  range_z: number;
  density: number | null;
  occupied_voxels: number;
  has_color: boolean;
  geometry_qp_safety: number;
  attribute_qp_safety: number | null;
  g1: number;
  g2: number;
  g3: number;
  g4: number;
  g5: number;
  a1: number | null;
  a2: number | null;
  a3: number | null;
  a4: number | null;
  a5: number | null;
  geometry_stability: number;
  attribute_stability: number | null;
  inverse_color_entropy: number | null;
  shared_color_voxels: number;
}

export interface PointCloudSequenceScoreStats {
  score: number | null;
  mean: number | null;
  median: number | null;
  min: number | null;
  max: number | null;
  stddev: number | null;
}

export interface PointCloudSequenceSegment {
  segment_id: number;
  start_frame_index: number;
  end_frame_index: number;
  frame_count: number;
  aggregation: PointCloudSequenceAggregation | string;
  min_x: number;
  max_x: number;
  min_y: number;
  max_y: number;
  min_z: number;
  max_z: number;
  range_x: number | null;
  range_y: number | null;
  range_z: number | null;
  point_count: number | null;
  density: number | null;
  geometry_qp_safety: number | null;
  attribute_qp_safety: number | null;
  geometry_score_stats: PointCloudSequenceScoreStats;
  attribute_score_stats: PointCloudSequenceScoreStats;
  geometry_worst_frame_index: number | null;
  attribute_worst_frame_index: number | null;
  geometry_stability: number | null;
  attribute_stability: number | null;
  inverse_color_entropy: number | null;
  g1: number | null;
  g2: number | null;
  g3: number | null;
  g4: number | null;
  g5: number | null;
  a1: number | null;
  a2: number | null;
  a3: number | null;
  a4: number | null;
  a5: number | null;
}

export interface PointCloudSequenceAnalysisResponse {
  root_folder: string;
  recursive: boolean;
  summary: {
    detected_ply_files: number;
    first_frame_index: number | null;
    last_frame_index: number | null;
    processed_frames: number;
    skipped_frames: number;
    color_frames: number;
    sampled_frames: number;
  };
  options: {
    segment_size: number;
    aggregation: PointCloudSequenceAggregation | string;
    voxel_bits: number;
    use_native_voxels: boolean;
    max_points_for_qp: number;
  };
  sequence_bounds: {
    min_x: number;
    max_x: number;
    min_y: number;
    max_y: number;
    min_z: number;
    max_z: number;
  };
  frames: PointCloudSequenceFrame[];
  segments: PointCloudSequenceSegment[];
  warnings: ExtraToolWarning[];
}

export interface TraceHeightOutputFile {
  source: string;
  output: string;
  source_filename: string;
  output_filename: string;
  renamed: boolean;
  rows_seen: number;
  rows_transformed: number;
  rows_skipped: number;
}

export interface TraceHeightNormalizerResponse {
  root_folder: string;
  output_folder: string;
  selected_columns: {
    x: string;
    y: string;
    z: string;
  };
  scaling: {
    point_cloud: PlyInspectionResponse;
    height_axis: 'x' | 'y' | 'z' | string;
    measured_extent_units: number;
    real_height_m: number;
    scale_factor_m_per_unit: number;
    axis_swap: {
      enabled: boolean;
      source_height_axis: 'x' | 'y' | 'z' | string;
      output_height_axis: 'x' | 'y' | 'z' | string;
    };
  };
  summary: {
    files_found: number;
    files_processed: number;
    files_skipped: number;
    rows_seen: number;
    rows_transformed: number;
    rows_skipped: number;
    output_files: number;
  };
  output_files: TraceHeightOutputFile[];
  warnings: ExtraToolWarning[];
}

export type DatasetExportMode = 'scaled' | 'voxelized' | 'both' | 'normalized';

export interface DatasetTransformationResponse {
  tool: string;
  output_folder: string;
  selected_columns: {
    x: string;
    y: string;
    z: string;
  };
  scaling: TraceHeightNormalizerResponse['scaling'] & {
    positive_coordinates: boolean;
    height_axis_starts_at_zero: boolean;
    height_axis_zero_shift_axis: 'x' | 'y' | 'z' | string;
    height_axis_zero_shift_offset_m: number;
    positive_translation_m: Record<'x' | 'y' | 'z', number>;
    scaled_reference_bounds_m?: Record<'x' | 'y' | 'z', [number, number] | { min: number; max: number }>;
    height_axis_zero_shifted_bounds_m?: Record<'x' | 'y' | 'z', [number, number] | { min: number; max: number }>;
    shifted_reference_bounds_m?: Record<'x' | 'y' | 'z', [number, number] | { min: number; max: number }>;
    final_exported_bounds_m?: Record<'x' | 'y' | 'z', [number, number] | { min: number; max: number }>;
    voxelization: {
      enabled: boolean;
      geometry_bit_depth: number | null;
      max_coordinate: number | null;
      scale: number | null;
    };
    debug_details?: {
      original_reference_extents_units: Record<'x' | 'y' | 'z', number>;
      swapped_reference_extents_units: Record<'x' | 'y' | 'z', number>;
      original_height_axis?: 'x' | 'y' | 'z';
      selected_height_axis_after_axis_swap: 'x' | 'y' | 'z';
      measured_height_extent_units: number;
      desired_real_world_height_m?: number;
      known_real_world_height_m: number;
      computed_scale_factor_m_per_unit: number;
      height_axis_minimum_before_zero_shift_m: number;
      height_axis_zero_shift_offset_m: number;
      height_axis_minimum_after_zero_shift_m: number;
      final_exported_extents_m: Record<'x' | 'y' | 'z', number>;
    };
  };
  export_mode: DatasetExportMode | string;
  export_label?: string;
  trace_resampling?: {
    enabled: boolean;
    target_samples?: number | null;
    inferred?: boolean;
    source?: string | null;
    source_frame_count?: number | null;
  };
  traces?: {
    output_folder: string;
    summary: TraceHeightNormalizerResponse['summary'];
    output_files: Array<TraceHeightOutputFile & { resampled?: boolean; resample_target_rows?: number | null }>;
  } | null;
  traces_voxelized?: {
    output_folder: string;
    summary: TraceHeightNormalizerResponse['summary'];
    output_files: Array<TraceHeightOutputFile & { resampled?: boolean; resample_target_rows?: number | null }>;
  } | null;
  point_clouds?: {
    output_folder: string;
    summary: {
      frames_found: number;
      frames_processed: number;
      points_written: number;
      points_skipped: number;
      worker_count: number;
    };
    output_files: Array<{
      source: string;
      output: string;
      frame_number: number;
      source_filename: string;
      output_filename: string;
      points_written: number;
      points_skipped: number;
    }>;
  } | null;
  point_clouds_voxelized?: DatasetTransformationResponse['point_clouds'];
  summary: {
    trace_outputs: number;
    point_cloud_outputs: number;
    worker_count: number;
  };
  warnings: ExtraToolWarning[];
}

export interface NetworkTracePoint {
  time_s: number;
  bandwidth_mbps: number;
}

export interface NetworkTraceResponse {
  samples: NetworkTracePoint[];
}

export interface NavisStatus {
  configured: boolean;
  reachable: boolean;
  base_url: string;
  health_path: string;
  export_path: string;
  auto_export: boolean;
  checked_url?: string;
  status_code?: number | null;
  latency_ms?: number | null;
  error?: string | null;
}

export interface NavisConfig {
  base_url: string;
  health_path: string;
  export_path: string;
  timeout_s: number;
  auto_export: boolean;
  auth_token?: string;
}

export interface NavisConfigResponse {
  settings: NavisConfig;
  status: NavisStatus;
  settings_path?: string;
}

export interface NavisExportResult {
  ok: boolean;
  run_id: string;
  redirect_url?: string;
  url?: string;
  status_code?: number | null;
  latency_ms?: number | null;
  request_bytes?: number | null;
  payload_summary?: {
    schema?: string;
    segment_count?: number;
    ply_dir_path?: string;
    navigation_trace_path?: string;
    bandwidth_trace_path?: string;
  };
  response?: unknown;
  response_text?: string;
  error?: string | null;
  navis_status?: NavisStatus;
}

export interface FovComparisonResult {
  ok?: boolean;
  run_id?: string;
  status?: 'done' | 'failed' | string;
  generated_at?: string;
  frame_count?: number;
  video_path?: string | null;
  data_path?: string | null;
  error?: string | null;
}

export interface PredictorDiagnosticsStageSummary {
  name: string;
  count: number;
  median_ms: number;
  mean_ms: number;
  p95_ms: number;
  max_ms: number;
}

export interface PredictorDiagnosticsMetadataFieldSummary {
  count: number;
  min: number;
  median: number;
  mean: number;
  p95: number;
  max: number;
}

export interface PredictorDiagnosticsSummary {
  enabled: boolean;
  predictor_name?: string | null;
  prediction_count: number;
  total: {
    count: number;
    median_ms: number;
    mean_ms: number;
    p95_ms: number;
    max_ms: number;
  };
  stages: PredictorDiagnosticsStageSummary[];
  metadata_summary?: Record<string, PredictorDiagnosticsMetadataFieldSummary>;
  metadata_samples?: Record<string, unknown>[];
}

export interface RunReadableOverview {
  run_id: string;
  status: string;
  error?: string | null;
  started?: string | null;
  finished?: string | null;
  abr: string;
  predictor: string;
  dataset_id?: string | null;
  network_trace?: string | null;
  network_profile?: string | null;
  fov_trace?: string | null;
  segment_count: number;
  playback_time_s: number;
  rebuffer_time_s: number;
  rebuffer_count: number;
  average_bitrate_mbps: number;
}

export interface RunReadableTimelineRow {
  segment_idx: number;
  decision_time_s: number;
  enqueue_time_s: number;
  duration_s: number;
  download_time_s: number;
  size_mbit: number;
  bitrate_mbps: number;
  buffer_after_s?: number | null;
  selection_summary: string;
  top_fov_tiles: Array<{ tile_id: string; visibility: number }>;
  notes: string[];
}

export interface RunLogStorySegment {
  segment_idx: number;
  component_count?: number;
  avg_component_bandwidth_mbps?: number | null;
  min_component_bandwidth_mbps?: number | null;
  max_component_bandwidth_mbps?: number | null;
  size_mbit?: number;
  download_time_s?: number;
}

export interface RunLogStoryEvent {
  kind: string;
  segment_idx?: number | null;
  message: string;
}

export interface RunLogStory {
  available: boolean;
  summary: string;
  segments: RunLogStorySegment[];
  events: RunLogStoryEvent[];
}

export interface PredictorReadableStory {
  available: boolean;
  predictor_name?: string | null;
  prediction_count?: number;
  total?: {
    median_ms: number;
    mean_ms: number;
    p95_ms: number;
    max_ms: number;
  };
  summary: string;
  story: string[];
  slowest_stages: Array<{
    name: string;
    mean_ms: number;
    p95_ms: number;
    max_ms: number;
    count: number;
  }>;
  metadata: Array<{
    name: string;
    median: number;
    min: number;
    max: number;
    count: number;
  }>;
}

export interface RunReadableSummary {
  overview: RunReadableOverview;
  timeline: RunReadableTimelineRow[];
  predictor_story: PredictorReadableStory;
  log_story: RunLogStory;
  raw_artifacts: {
    events_path?: string | null;
    log_path?: string | null;
    predictor_diagnostics_path?: string | null;
  };
}
