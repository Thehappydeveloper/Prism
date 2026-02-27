export type Dict<T = unknown> = Record<string, T>;

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
  fov_traces: string[];
  network_root: string;
  fov_root: string;
  abr_classes: string[];
  fov_predictors: string[];
  abr_requires_predictor: Record<string, boolean>;
  qoe_functions: QoEModel[];
}

export interface RunSummary {
  run_id: string;
  status: string;
  started?: string;
  finished?: string;
  events_path?: string;
  log_path?: string;
}

export interface RunConfig {
  mpd?: string;
  content_dir: string;
  network_trace: string;
  fov_trace: string;
  abr: string;
  predictor?: string;
  rtt: number;
  block_size_bits?: number;
  enable_fov_upgrades?: boolean;
  enable_mid_download_cancel?: boolean;
  buffer_window_segments?: number;
  buffer_unbounded?: boolean;
  run_id?: string;
  v3c_folder?: string;
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
  total_playback_time: number;
  total_rebuffering_time: number;
}

export interface RunMetrics {
  labels: Record<string, string>;
  qoe: Record<string, number[]>;
  fov_quality: Record<string, { fov: number[]; non: number[] }>;
  delta_fov: Record<string, number[]>;
  stall: Record<string, { duration: number; count: number }>;
  component_bits: Record<string, Record<string, number>>;
  bitrate_vs_quality: Record<string, Array<[number, number]>>;
  buffer: Record<string, { mean: number; std: number; min: number; max: number }>;
  meta: Record<
    string,
    {
      duration_s?: number;
      fov_present?: boolean;
      abr?: string;
      predictor?: string;
      network_trace?: string;
      network_class?: string;
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
}

export interface NetworkTracePoint {
  time_s: number;
  bandwidth_mbps: number;
}

export interface NetworkTraceResponse {
  samples: NetworkTracePoint[];
}

export type RouteId = 'home' | 'control' | 'batch' | 'compare';
