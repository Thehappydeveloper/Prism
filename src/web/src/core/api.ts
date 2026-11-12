import type {
  AppOptions,
  HeadsetConfig,
  HeadsetConfigResponse,
  PrismConfig,
  PrismConfigResponse,
  TraceRootConfig,
  TraceRootConfigResponse,
  TraceColumnAliasSettings,
  TraceColumnConfigResponse,
  NavisConfig,
  NavisConfigResponse,
  DiagnosticsRequest,
  DatasetProfile,
  DatasetTransformationResponse,
  DatasetProfilesResponse,
  EventLog,
  FovComparisonResult,
  NavisExportResult,
  NavisStatus,
  NetworkTraceResponse,
  OracleGapRequest,
  OracleGapResponse,
  PointCloudSequenceAnalysisResponse,
  PointCloudSequenceAggregation,
  PointCloudSequenceDiscoverResponse,
  PredictorDiagnosticsSummary,
  PredictorReadableStory,
  RunFormSessionsResponse,
  RunConfig,
  RunGroup,
  RunMetrics,
  RunLogStory,
  RunReadableSummary,
  RunSummary,
  PlyInspectionResponse,
  SpeedAccelerationAnalysisResponse,
  SpeedAccelerationDiscoverResponse,
  TraceHeightNormalizerResponse,
  TrainingDataset,
  TrainingGraph,
  TrainingMetrics,
  TrainingModel,
  TrainingPlan,
  TrainingRun,
  WorldTransform,
  WorldTransformResponse,
  LaunchPreflightResult,
} from './types';
import { loadCachedDiagnostics, saveCachedDiagnostics } from './diagnosticsCache';

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const text = await res.text();
  if (!res.ok) {
    const isHtml = /^\s*<!doctype html|^\s*<html\b/i.test(text);
    if (isHtml) {
      throw new Error(
        `The Prism API is unavailable (${res.status} at ${url}). ` +
        'This page appears to be served as static files; start Prism with "python src/server.py" and open http://localhost:8000/.',
      );
    }
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  try {
    return JSON.parse(text) as T;
  } catch (err) {
    const expectedBytes = res.headers.get('content-length');
    const expected = expectedBytes ? `; Content-Length ${expectedBytes}` : '';
    const message = err instanceof Error ? err.message : String(err);
    throw new Error(`Invalid JSON from ${url}: received ${text.length} characters${expected}. ${message}`);
  }
}

export async function getOptions(): Promise<AppOptions> {
  return fetchJson<AppOptions>('/api/options');
}

export async function listRuns(): Promise<RunSummary[]> {
  return fetchJson<RunSummary[]>('/api/runs');
}

export type ReportBatchPlan = { id: string; label: string; path: string };
export type ReportDefinition = { id: string; title: string; description: string; outputs: string[]; parent_report?: string; included_reports?: string[]; batch_plans?: ReportBatchPlan[]; render_options?: Record<string, { type?: string; label?: string; default?: boolean | number; min?: number; max?: number }> };
export type ReportResult = { status: 'ready' | 'warning' | 'error'; runs_found: number; runs_finished: number; issues: Array<{ severity: string; message: string }>; built?: boolean; build_id?: string; output_dir?: string; outputs?: string[]; output_urls?: Record<string, string> };
export type ReportBuild = { build_id: string; built_at?: string; runs?: number; outputs: string[] };

export async function listReports(): Promise<{ reports: ReportDefinition[] }> {
  return fetchJson('/api/reports');
}

export async function validateReport(reportId: string): Promise<ReportResult> {
  return fetchJson(`/api/reports/${encodeURIComponent(reportId)}/validate`, { method: 'POST' });
}

export async function buildReport(reportId: string, renderOptions?: Record<string, boolean>): Promise<ReportResult> {
  return fetchJson(`/api/reports/${encodeURIComponent(reportId)}/build`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ render_options: renderOptions || {} }),
  });
}

export async function loadReportBatchPlan(reportId: string, planId: string): Promise<{ report_id: string; plan_id: string; label: string; path: string; plan: unknown }> {
  return fetchJson(`/api/reports/${encodeURIComponent(reportId)}/batch-plans/${encodeURIComponent(planId)}`);
}

export async function updateReportFigureFontSize(reportId: string, fontSizePt: number): Promise<{ build_id: string; outputs: string[]; font_size_pt: number }> {
  return fetchJson(`/api/reports/${encodeURIComponent(reportId)}/svg-font-size`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ font_size_pt: fontSizePt }),
  });
}

export async function listReportBuilds(reportId: string): Promise<{ builds: ReportBuild[] }> {
  return fetchJson(`/api/reports/${encodeURIComponent(reportId)}/builds`);
}

export async function listRunsPage(
  offset = 0,
  limit = 200,
  filters: { search?: string; status?: string; groupId?: string; ungrouped?: boolean } = {},
): Promise<{ items: RunSummary[]; total: number; offset: number; limit: number }> {
  const query = new URLSearchParams({ offset: String(offset), limit: String(limit) });
  if (filters.search) query.set('search', filters.search);
  if (filters.status && filters.status !== 'all') query.set('status', filters.status);
  if (filters.groupId) query.set('group_id', filters.groupId);
  if (filters.ungrouped) query.set('ungrouped', '1');
  return fetchJson(`/api/runs?${query.toString()}`);
}

export type RunGroupSummary = { total: number; queued: number; running: number; paused: number; finished: number; failed: number; canceled: number };

export async function getRunGroupSummaries(filters: { search?: string; status?: string } = {}): Promise<{ groups: Record<string, RunGroupSummary> }> {
  const query = new URLSearchParams();
  if (filters.search) query.set('search', filters.search);
  if (filters.status && filters.status !== 'all') query.set('status', filters.status);
  return fetchJson(`/api/runs/group-summaries?${query.toString()}`);
}

export async function listRunGroups(): Promise<{ groups: RunGroup[] }> {
  return fetchJson<{ groups: RunGroup[] }>('/api/run-groups');
}

export async function saveRunGroup(payload: Partial<RunGroup>): Promise<{ groups: RunGroup[] }> {
  return fetchJson<{ groups: RunGroup[] }>('/api/run-groups', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function assignRunsToGroup(payload: { run_ids: string[]; id?: string; name?: string; description?: string; source?: string; plan_path?: string }): Promise<{ groups: RunGroup[] }> {
  return fetchJson<{ groups: RunGroup[] }>('/api/run-groups/assign', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function removeRunsFromGroups(runIds: string[]): Promise<{ groups: RunGroup[] }> {
  return fetchJson<{ groups: RunGroup[] }>('/api/run-groups/remove-runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_ids: runIds }),
  });
}

export async function archiveRunGroup(groupId: string, archived: boolean): Promise<{ groups: RunGroup[] }> {
  return fetchJson<{ groups: RunGroup[] }>('/api/run-groups/archive', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ group_id: groupId, archived }),
  });
}

export async function controlRunGroup(groupId: string, action: 'pause' | 'resume' | 'cancel' | 'force_cancel'): Promise<{ status: string; group_id: string; affected: number; groups: RunGroup[] }> {
  return fetchJson<{ status: string; group_id: string; affected: number; groups: RunGroup[] }>('/api/run-groups/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ group_id: groupId, action }),
  });
}

export async function deleteRunGroup(groupId: string): Promise<{ groups: RunGroup[] }> {
  return fetchJson<{ groups: RunGroup[] }>(`/api/run-groups/${encodeURIComponent(groupId)}`, { method: 'DELETE' });
}

export async function deleteRunGroupWithRuns(groupId: string): Promise<{ groups: RunGroup[]; deleted_run_ids: string[] }> {
  return fetchJson<{ groups: RunGroup[]; deleted_run_ids: string[] }>('/api/run-groups/delete-with-runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ group_id: groupId }),
  });
}

export async function startRun(payload: RunConfig): Promise<{ run_id: string; run_ids?: string[]; status: string }> {
  return fetchJson('/api/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function startRunsBatch(payloads: RunConfig[]): Promise<{ run_id: string; run_ids?: string[]; status: string }> {
  return fetchJson('/api/runs/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ runs: payloads }),
  });
}

export async function preflightRuns(payloads: RunConfig[]): Promise<LaunchPreflightResult> {
  return fetchJson<LaunchPreflightResult>('/api/preflight', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ runs: payloads }) });
}

export async function deleteRun(runId: string): Promise<{ status: string; run_id: string; deleted?: boolean; error?: string }> {
  return fetchJson(`/api/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' });
}

export async function getRunConfig(runId: string): Promise<RunConfig> {
  return fetchJson<RunConfig>(`/api/runs/${encodeURIComponent(runId)}/config`);
}

export async function getRunEvents(runId: string): Promise<EventLog> {
  return fetchJson<EventLog>(`/api/runs/${encodeURIComponent(runId)}/events`);
}

export async function getRunPredictorDiagnostics(runId: string): Promise<PredictorDiagnosticsSummary> {
  return fetchJson<PredictorDiagnosticsSummary>(`/api/runs/${encodeURIComponent(runId)}/predictor-diagnostics`);
}

export async function getRunReadableSummary(runId: string): Promise<RunReadableSummary> {
  return fetchJson<RunReadableSummary>(`/api/runs/${encodeURIComponent(runId)}/readable-summary`);
}

export async function getRunPredictorStory(runId: string): Promise<PredictorReadableStory> {
  return fetchJson<PredictorReadableStory>(`/api/runs/${encodeURIComponent(runId)}/predictor-story`);
}

export async function getRunLogStory(runId: string): Promise<RunLogStory> {
  return fetchJson<RunLogStory>(`/api/runs/${encodeURIComponent(runId)}/log-story`);
}

export async function getDiagnostics(payload: DiagnosticsRequest): Promise<RunMetrics> {
  let fingerprint = '';
  try {
    const response = await fetchJson<{ fingerprint: string }>('/api/diagnostics/fingerprint', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    fingerprint = response.fingerprint;
    const cached = await loadCachedDiagnostics(fingerprint);
    if (cached) return cached;
  } catch {
    // Continue through the authoritative endpoint if browser caching is unavailable.
  }
  const metrics = await fetchJson<RunMetrics>('/api/diagnostics', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (fingerprint) await saveCachedDiagnostics(fingerprint, metrics);
  return metrics;
}

export async function getOracleGap(payload: OracleGapRequest): Promise<OracleGapResponse> {
  return fetchJson<OracleGapResponse>('/api/oracle-gap', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function getTrace(path: string): Promise<NetworkTraceResponse> {
  const encoded = encodeURIComponent(path);
  return fetchJson<NetworkTraceResponse>(`/api/trace?path=${encoded}`);
}

export async function getNavisStatus(): Promise<NavisStatus> {
  return fetchJson<NavisStatus>('/api/navis/status');
}

export async function getNavisConfig(): Promise<NavisConfigResponse> {
  return fetchJson<NavisConfigResponse>('/api/navis/config');
}

export async function saveNavisConfig(settings: Partial<NavisConfig>): Promise<NavisConfigResponse> {
  return fetchJson<NavisConfigResponse>('/api/navis/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ settings }),
  });
}

export async function getHeadsetConfig(): Promise<HeadsetConfigResponse> {
  return fetchJson<HeadsetConfigResponse>('/api/headsets/config');
}

export async function getPrismConfig(): Promise<PrismConfigResponse> {
  return fetchJson<PrismConfigResponse>('/api/prism/config');
}

export async function savePrismConfig(settings: PrismConfig): Promise<PrismConfigResponse> {
  return fetchJson<PrismConfigResponse>('/api/prism/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ settings }),
  });
}

export async function saveHeadsetConfig(settings: HeadsetConfig): Promise<HeadsetConfigResponse> {
  return fetchJson<HeadsetConfigResponse>('/api/headsets/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ settings }),
  });
}

export async function getTraceRootConfig(): Promise<TraceRootConfigResponse> {
  return fetchJson<TraceRootConfigResponse>('/api/traces/config');
}

export async function saveTraceRootConfig(settings: TraceRootConfig): Promise<TraceRootConfigResponse> {
  return fetchJson<TraceRootConfigResponse>('/api/traces/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ settings }),
  });
}

export async function getTraceColumnConfig(): Promise<TraceColumnConfigResponse> {
  return fetchJson<TraceColumnConfigResponse>('/api/trace-columns/config');
}

export async function saveTraceColumnConfig(settings: TraceColumnAliasSettings): Promise<TraceColumnConfigResponse> {
  return fetchJson<TraceColumnConfigResponse>('/api/trace-columns/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ settings }),
  });
}

export async function getWorldTransform(contentDir: string): Promise<WorldTransformResponse> {
  return fetchJson<WorldTransformResponse>(`/api/world-transforms?content_dir=${encodeURIComponent(contentDir)}`);
}

export async function saveWorldTransform(contentDir: string, transform: WorldTransform): Promise<WorldTransformResponse> {
  return fetchJson<WorldTransformResponse>('/api/world-transforms', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content_dir: contentDir, transform }),
  });
}

export async function getDatasets(): Promise<DatasetProfilesResponse> {
  return fetchJson<DatasetProfilesResponse>('/api/datasets');
}

export async function saveDataset(profile: Partial<DatasetProfile> & Record<string, unknown>): Promise<DatasetProfilesResponse> {
  return fetchJson<DatasetProfilesResponse>('/api/datasets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(profile),
  });
}

export async function deleteDataset(datasetId: string): Promise<DatasetProfilesResponse> {
  return fetchJson<DatasetProfilesResponse>(`/api/datasets/${encodeURIComponent(datasetId)}`, {
    method: 'DELETE',
  });
}

export async function getTrainingDatasets(): Promise<{ datasets: TrainingDataset[]; settings_path?: string }> {
  return fetchJson('/api/training/datasets');
}

export async function saveTrainingDataset(payload: Partial<TrainingDataset> & Record<string, unknown>): Promise<{ datasets: TrainingDataset[] }> {
  return fetchJson('/api/training/datasets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function deleteTrainingDataset(datasetId: string): Promise<{ datasets: TrainingDataset[] }> {
  return fetchJson(`/api/training/datasets/${encodeURIComponent(datasetId)}`, { method: 'DELETE' });
}

export async function getTrainingPlans(): Promise<{ plans: TrainingPlan[]; settings_path?: string }> {
  return fetchJson('/api/training/plans');
}

export async function saveTrainingPlan(payload: Partial<TrainingPlan> & Record<string, unknown>): Promise<{ plans: TrainingPlan[] }> {
  return fetchJson('/api/training/plans', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function deleteTrainingPlan(planId: string): Promise<{ plans: TrainingPlan[] }> {
  return fetchJson(`/api/training/plans/${encodeURIComponent(planId)}`, { method: 'DELETE' });
}

export async function listTrainingRuns(): Promise<TrainingRun[]> {
  return fetchJson('/api/training/runs');
}

export async function startTrainingRun(planId: string): Promise<TrainingRun> {
  return fetchJson('/api/training/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ plan_id: planId }),
  });
}

export async function resumeTrainingRun(planId: string, runId: string): Promise<TrainingRun> {
  return fetchJson('/api/training/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ plan_id: planId, resume_run_id: runId }),
  });
}

export async function controlTrainingRun(runId: string, action: 'pause' | 'stop'): Promise<TrainingRun> {
  return fetchJson(`/api/training/runs/${encodeURIComponent(runId)}/control`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action }),
  });
}

export async function listTrainingModels(): Promise<{ models: TrainingModel[] }> {
  return fetchJson('/api/training/models');
}

export async function deleteTrainingRun(runId: string): Promise<{ run_id: string; deleted: boolean }> {
  return fetchJson(`/api/training/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' });
}

export async function deleteTrainingModel(modelId: string): Promise<{ model_id: string; deleted: boolean }> {
  return fetchJson(`/api/training/models/${encodeURIComponent(modelId)}`, { method: 'DELETE' });
}

export async function exportTrainingModel(modelId: string, exportName = ''): Promise<TrainingModel> {
  return fetchJson('/api/training/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId, export_name: exportName }),
  });
}

export async function evaluateTrainingModel(modelId: string, split = 'test'): Promise<Record<string, unknown>> {
  return fetchJson('/api/training/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId, split }),
  });
}

export async function getTrainingMetrics(runId: string): Promise<TrainingMetrics> {
  return fetchJson(`/api/training/runs/${encodeURIComponent(runId)}/metrics`);
}

export async function getTrainingResults(runId: string): Promise<{ run_id: string; metrics: TrainingMetrics; graphs: TrainingGraph[] }> {
  return fetchJson(`/api/training/runs/${encodeURIComponent(runId)}/results`);
}

export async function getRunFormSessions(): Promise<RunFormSessionsResponse> {
  return fetchJson<RunFormSessionsResponse>('/api/run-form-sessions');
}

export async function saveRunFormSession(name: string, config: RunConfig): Promise<RunFormSessionsResponse> {
  return fetchJson<RunFormSessionsResponse>('/api/run-form-sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, config }),
  });
}

export async function deleteRunFormSession(name: string): Promise<RunFormSessionsResponse> {
  return fetchJson<RunFormSessionsResponse>(`/api/run-form-sessions/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  });
}

export async function discoverSpeedAccelerationColumns(rootFolder: string): Promise<SpeedAccelerationDiscoverResponse> {
  return fetchJson<SpeedAccelerationDiscoverResponse>('/api/extra/speed-acceleration/discover', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ root_folder: rootFolder }),
  });
}

export async function inspectSpeedAccelerationPly(plyPath: string): Promise<PlyInspectionResponse> {
  return fetchJson<PlyInspectionResponse>('/api/extra/speed-acceleration/inspect-ply', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ply_path: plyPath }),
  });
}

export async function analyzeSpeedAcceleration(
  rootFolder: string,
  xColumn: string,
  yColumn: string,
  zColumn: string,
  fps: number,
  scaling: {
    plyPath: string;
    heightAxis: 'x' | 'y' | 'z';
    realHeightM: number;
  },
  filtering: {
    filterOutliers: boolean;
    filterMethod: 'iqr' | 'percentile';
    percentileKeep: number;
    histogramSource: 'raw' | 'filtered';
    excludeSuspiciousJumps: boolean;
  },
): Promise<SpeedAccelerationAnalysisResponse> {
  return fetchJson<SpeedAccelerationAnalysisResponse>('/api/extra/speed-acceleration/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      root_folder: rootFolder,
      x_column: xColumn,
      y_column: yColumn,
      z_column: zColumn,
      fps,
      ply_path: scaling.plyPath,
      height_axis: scaling.heightAxis,
      real_height_m: scaling.realHeightM,
      filter_outliers: filtering.filterOutliers,
      filter_method: filtering.filterMethod,
      percentile_keep: filtering.percentileKeep,
      histogram_source: filtering.histogramSource,
      exclude_suspicious_jumps: filtering.excludeSuspiciousJumps,
    }),
  });
}

export async function discoverPointCloudSequence(
  rootFolder: string,
  recursive: boolean,
): Promise<PointCloudSequenceDiscoverResponse> {
  return fetchJson<PointCloudSequenceDiscoverResponse>('/api/extra/point-cloud-sequence/discover', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ root_folder: rootFolder, recursive }),
  });
}

export async function analyzePointCloudSequence(payload: {
  rootFolder: string;
  recursive: boolean;
  segmentSize: number;
  aggregation: PointCloudSequenceAggregation;
  voxelBits: number;
  useNativeVoxels: boolean;
  maxPointsForQp: number;
}): Promise<PointCloudSequenceAnalysisResponse> {
  return fetchJson<PointCloudSequenceAnalysisResponse>('/api/extra/point-cloud-sequence/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      root_folder: payload.rootFolder,
      recursive: payload.recursive,
      segment_size: payload.segmentSize,
      aggregation: payload.aggregation,
      voxel_bits: payload.voxelBits,
      use_native_voxels: payload.useNativeVoxels,
      max_points_for_qp: payload.maxPointsForQp,
    }),
  });
}

export async function discoverTraceHeightColumns(rootFolder: string): Promise<SpeedAccelerationDiscoverResponse> {
  return fetchJson<SpeedAccelerationDiscoverResponse>('/api/extra/dataset-transformation/discover', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ root_folder: rootFolder }),
  });
}

export async function inspectTraceHeightPly(plyPath: string): Promise<PlyInspectionResponse> {
  return fetchJson<PlyInspectionResponse>('/api/extra/dataset-transformation/inspect-ply', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ply_path: plyPath }),
  });
}

export async function transformDataset(payload: {
  traceRootFolder: string;
  plyInputPath: string;
  outputFolder: string;
  xColumn: string;
  yColumn: string;
  zColumn: string;
  referencePlyPath: string;
  heightAxis: 'x' | 'y' | 'z';
  outputHeightAxis: 'x' | 'y' | 'z';
  realHeightM: number;
  transformTraces: boolean;
  transformPointClouds: boolean;
  heightAxisStartsAtZero: boolean;
  positiveCoordinates: boolean;
  voxelBits: number | null;
  exportMode: 'scaled' | 'voxelized' | 'both' | 'normalized';
  workerCount: number;
  fovResampleEnabled: boolean;
  fovResampleTarget: number | null;
}): Promise<DatasetTransformationResponse> {
  return fetchJson<DatasetTransformationResponse>('/api/extra/dataset-transformation/transform', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      trace_root_folder: payload.traceRootFolder,
      ply_input_path: payload.plyInputPath,
      output_folder: payload.outputFolder,
      x_column: payload.xColumn,
      y_column: payload.yColumn,
      z_column: payload.zColumn,
      reference_ply_path: payload.referencePlyPath,
      height_axis: payload.heightAxis,
      output_height_axis: payload.outputHeightAxis,
      real_height_m: payload.realHeightM,
      transform_traces: payload.transformTraces,
      transform_point_clouds: payload.transformPointClouds,
      height_axis_starts_at_zero: payload.heightAxisStartsAtZero,
      positive_coordinates: payload.positiveCoordinates,
      voxel_bits: payload.voxelBits,
      export_mode: payload.exportMode,
      worker_count: payload.workerCount,
      fov_resample_enabled: payload.fovResampleEnabled,
      fov_resample_target: payload.fovResampleTarget,
    }),
  });
}

export async function normalizeTraceHeight(
  rootFolder: string,
  outputFolder: string,
  xColumn: string,
  yColumn: string,
  zColumn: string,
  scaling: {
    plyPath: string;
    heightAxis: 'x' | 'y' | 'z';
    outputHeightAxis: 'x' | 'y' | 'z';
    realHeightM: number;
  },
): Promise<TraceHeightNormalizerResponse> {
  return fetchJson<TraceHeightNormalizerResponse>('/api/extra/trace-height-normalizer/normalize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      root_folder: rootFolder,
      output_folder: outputFolder,
      x_column: xColumn,
      y_column: yColumn,
      z_column: zColumn,
      ply_path: scaling.plyPath,
      height_axis: scaling.heightAxis,
      output_height_axis: scaling.outputHeightAxis,
      real_height_m: scaling.realHeightM,
    }),
  });
}

export async function exportRunToNavis(runId: string): Promise<NavisExportResult> {
  return fetchJson<NavisExportResult>('/api/navis/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_id: runId }),
  });
}

export async function generateRunFovComparison(runId: string): Promise<FovComparisonResult> {
  return fetchJson<FovComparisonResult>(`/api/runs/${encodeURIComponent(runId)}/fov-comparison`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
}

export async function getRunFovComparisonSegment(runId: string, segmentIdx: number): Promise<{ run_id: string; segment_idx: number; frame: any }> {
  return fetchJson(`/api/runs/${encodeURIComponent(runId)}/fov-comparison-segment`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ segment_idx: segmentIdx }),
  });
}

export function subscribeRuns(onRuns: (runs: RunSummary[]) => void, onError?: (error: unknown) => void): () => void {
  const source = new EventSource('/api/runs/stream');
  source.addEventListener('runs', (evt) => {
    try {
      const data = JSON.parse((evt as MessageEvent<string>).data) as RunSummary[];
      onRuns(data);
    } catch (error) {
      onError?.(error);
    }
  });
  source.onerror = (err) => {
    onError?.(err);
  };
  return () => source.close();
}
