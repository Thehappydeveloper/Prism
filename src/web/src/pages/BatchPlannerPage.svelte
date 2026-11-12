  <script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { gotoRoute } from '../core/router';
  import { fovPredictorLabel } from '../core/predictorLabels';
  import { getDiagnostics, getOptions, listRuns, loadReportBatchPlan, preflightRuns, startRunsBatch, subscribeRuns } from '../core/api';
  import { loadLaunchPreflightMode } from '../core/launchPreflight';
  import type { AlgorithmParamField, AppOptions, DatasetProfile, HeadsetPreset, LaunchPreflightResult, ParamMap, RunConfig, RunMetrics, RunSummary } from '../core/types';
  import {
    getPresetParams,
    listPresetNames,
    loadHyperparamPresets,
    type HyperparamPresetStore,
  } from '../core/hyperparamPresets';
  import {
    analyzeTracePaths,
    groupTraces,
    summarizeMultiTrace,
    traceDisplayName,
    type TraceFilterOption,
    type TraceGroupingOption,
    type TraceSortMode,
  } from '../core/traceBrowser';

  interface MediaPair {
    id: string;
    dataset_id: string;
    v3c_folder: string;
    content_dir: string;
    fov_traces: string[];
  }

  interface PlannedRun {
    id: string;
    config: RunConfig;
    status: 'planned' | 'queued' | 'failed';
    runId?: string;
    error?: string;
  }
  let preflight: LaunchPreflightResult | null = null;
  let showPreflight = false;
  let preflightLoading = false;
  let experimentPurpose = '';
  let experimentKind = 'exploration';

  interface BatchPlannerSession {
    name: string;
    config: BatchPlannerSnapshot;
    updated_at?: string;
  }

  interface BatchPlannerSnapshot {
    presetSelectionVersion?: number;
    selectedAbrs: string[];
    selectedPredictors: string[];
    selectedBandwidthPredictors: string[];
    selectedAbrPresetsByAlgo: Record<string, string[]>;
    selectedFovPresetsByPredictor: Record<string, string[]>;
    selectedBandwidthPresetsByPredictor: Record<string, string[]>;
    networkSource: 'trace' | 'profile';
    selectedNetworkTraces: string[];
    selectedNetworkProfiles: string[];
    networkTraceSamplingMode?: NetworkTraceSamplingMode;
    netSamples: number;
    fovSamples: number;
    seedText: string;
    rtt: number;
    bufferWindowSegments: number;
    segmentLimit?: number;
    enableMidCancel: boolean;
    headsetPreset: string;
    segmentWorkersOverride?: number | null;
    mediaPairs: MediaPair[];
    batchGroupName?: string;
    experimentPurpose?: string;
    experimentKind?: string;
    analysis_qoe_function?: string;
    analysis_qoe_scoring_version?: string;
    analysis_qoe_warmup_segments?: number;
  }

  let options: AppOptions | null = null;
  let loading = true;
  let error = '';

  let selectedAbrs = new Set<string>();
  let selectedPredictors = new Set<string>(['None']);
  let selectedBandwidthPredictors = new Set<string>(['None']);
  let presets: HyperparamPresetStore = loadHyperparamPresets();
  let selectedAbrPresetsByAlgo: Record<string, string[]> = {};
  let selectedFovPresetsByPredictor: Record<string, string[]> = {};
  let selectedBandwidthPresetsByPredictor: Record<string, string[]> = {};

  let networkSource: 'trace' | 'profile' = 'trace';
  let selectedNetworkTraces: string[] = [];
  let selectedNetworkProfiles: string[] = [];
  type NetworkTraceSamplingMode = 'proportional' | 'balanced_lumous';
  let networkTraceSamplingMode: NetworkTraceSamplingMode = 'proportional';
  let netSamples = 3;
  let fovSamples = 3;
  let seedText = '';

  let rtt = 0.06;
  let bufferWindowSegments = 2;
  let segmentLimit = 0;
  let enableMidCancel = false;
  let headsetPreset = '';
  // Undefined delegates to the server setting; zero means explicitly auto.
  let segmentWorkersOverride: number | null = null;

  let mediaPairs: MediaPair[] = [];
  let plannedRuns: PlannedRun[] = [];
  let planMessage = 'No plan yet.';
  let tracePickerMode: 'network' | 'fov' | null = null;
  let tracePickerPairId: string | null = null;
  let tracePickerSelection = new Set<string>();
  let traceGroupingId = '';
  let traceFilterText = '';
  let traceSortMode: TraceSortMode = 'name';
  let traceFilterSelections: Record<string, string[]> = {};
  let batchSessions: BatchPlannerSession[] = [];
  let selectedBatchSessionName = '';
  let batchGroupName = '';
  let planFileInput: HTMLInputElement | null = null;
  let lastPlannedNetworkInputs: string[] = [];
  let lastPlannedFovTracesByPair: Record<string, string[]> = {};

  let liveRuns: RunSummary[] = [];
  let diagnostics: RunMetrics | null = null;

  let unsubscribeRuns: (() => void) | null = null;
  const DEFAULT_BATCH_PRESET = '__default__';
  const BATCH_SESSION_STORAGE_KEY = 'prism_batch_planner_sessions_v1';
  const PAPER_QOE_FUNCTION = 'VisiblePlaybackUtility';
  const PAPER_QOE_SCORING_VERSION = 'mpd_stable_visible_utility_v1';
  // These defaults preserve the existing planner behavior. Imported plans
  // can carry an evaluation-specific QoE definition and warm-up prefix.
  let analysisQoeFunction = PAPER_QOE_FUNCTION;
  let analysisQoeScoringVersion = PAPER_QOE_SCORING_VERSION;
  let analysisQoeWarmupSegments = 0;

  function generateLocalId(): string {
    const rng = globalThis.crypto?.randomUUID;
    if (typeof rng === 'function') return rng.call(globalThis.crypto);
    return `id_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
  }

  function loadBatchSessions(): BatchPlannerSession[] {
    try {
      const parsed = JSON.parse(localStorage.getItem(BATCH_SESSION_STORAGE_KEY) || '[]');
      if (!Array.isArray(parsed)) return [];
      return parsed.filter((entry) => entry && typeof entry.name === 'string' && entry.config && typeof entry.config === 'object');
    } catch {
      return [];
    }
  }

  function persistBatchSessions(next: BatchPlannerSession[]): void {
    batchSessions = next
      .filter((entry) => entry.name.trim())
      .sort((a, b) => a.name.localeCompare(b.name));
    localStorage.setItem(BATCH_SESSION_STORAGE_KEY, JSON.stringify(batchSessions));
  }

  function captureBatchSnapshot(): BatchPlannerSnapshot {
    return {
      presetSelectionVersion: 2,
      selectedAbrs: Array.from(selectedAbrs).sort((a, b) => a.localeCompare(b)),
      selectedPredictors: Array.from(selectedPredictors).sort((a, b) => a.localeCompare(b)),
      selectedBandwidthPredictors: Array.from(selectedBandwidthPredictors).sort((a, b) => a.localeCompare(b)),
      selectedAbrPresetsByAlgo: { ...selectedAbrPresetsByAlgo },
      selectedFovPresetsByPredictor: { ...selectedFovPresetsByPredictor },
      selectedBandwidthPresetsByPredictor: { ...selectedBandwidthPresetsByPredictor },
      networkSource,
      selectedNetworkTraces: [...selectedNetworkTraces],
      selectedNetworkProfiles: [...selectedNetworkProfiles],
      networkTraceSamplingMode,
      netSamples,
      fovSamples,
      seedText,
      rtt,
      bufferWindowSegments,
      segmentLimit,
      enableMidCancel,
      headsetPreset,
      segmentWorkersOverride,
      mediaPairs: mediaPairs.map((pair) => ({ ...pair, fov_traces: [...pair.fov_traces] })),
      batchGroupName,
      experimentPurpose,
      experimentKind,
      analysis_qoe_function: analysisQoeFunction,
      analysis_qoe_scoring_version: analysisQoeScoringVersion,
      analysis_qoe_warmup_segments: analysisQoeWarmupSegments,
    };
  }

  function migrateLegacyPresetSelections(
    selections: Record<string, string[]> | undefined,
    presetSelectionVersion: number | undefined,
  ): Record<string, string[]> {
    const normalized = Object.fromEntries(
      Object.entries(selections || {}).map(([algorithm, presetNames]) => [
        algorithm,
        Array.from(new Set((presetNames || []).filter(Boolean))),
      ]),
    ) as Record<string, string[]>;
    if (presetSelectionVersion && presetSelectionVersion >= 2) return normalized;

    // Before version 2, selecting a named preset silently added Default. For
    // saved sessions and plan files from that version, retain Default only
    // when it was the sole selected variant.
    return Object.fromEntries(
      Object.entries(normalized).map(([algorithm, presetNames]) => {
        const namedPresets = presetNames.filter((presetName) => presetName !== DEFAULT_BATCH_PRESET);
        return [algorithm, namedPresets.length ? namedPresets : presetNames];
      }),
    );
  }

  function applyBatchSnapshot(snapshot: Partial<BatchPlannerSnapshot>): void {
    selectedAbrs = new Set(snapshot.selectedAbrs?.length ? snapshot.selectedAbrs : Array.from(selectedAbrs));
    selectedPredictors = new Set(snapshot.selectedPredictors?.length ? snapshot.selectedPredictors : Array.from(selectedPredictors));
    selectedBandwidthPredictors = new Set(snapshot.selectedBandwidthPredictors?.length ? snapshot.selectedBandwidthPredictors : Array.from(selectedBandwidthPredictors));
    selectedAbrPresetsByAlgo = migrateLegacyPresetSelections(
      snapshot.selectedAbrPresetsByAlgo,
      snapshot.presetSelectionVersion,
    );
    selectedFovPresetsByPredictor = migrateLegacyPresetSelections(
      snapshot.selectedFovPresetsByPredictor,
      snapshot.presetSelectionVersion,
    );
    selectedBandwidthPresetsByPredictor = migrateLegacyPresetSelections(
      snapshot.selectedBandwidthPresetsByPredictor,
      snapshot.presetSelectionVersion,
    );
    networkSource = snapshot.networkSource === 'profile' ? 'profile' : 'trace';
    selectedNetworkTraces = [...(snapshot.selectedNetworkTraces || selectedNetworkTraces)];
    selectedNetworkProfiles = [...(snapshot.selectedNetworkProfiles || selectedNetworkProfiles)];
    networkTraceSamplingMode = snapshot.networkTraceSamplingMode === 'balanced_lumous' ? 'balanced_lumous' : 'proportional';
    netSamples = Number(snapshot.netSamples || netSamples || 1);
    fovSamples = Number(snapshot.fovSamples || fovSamples || 1);
    seedText = String(snapshot.seedText || '');
    rtt = Number(snapshot.rtt ?? rtt);
    bufferWindowSegments = Number(snapshot.bufferWindowSegments ?? bufferWindowSegments);
    segmentLimit = Math.max(0, Math.floor(Number(snapshot.segmentLimit ?? segmentLimit) || 0));
    enableMidCancel = Boolean(snapshot.enableMidCancel);
    headsetPreset = String(snapshot.headsetPreset || headsetPreset);
    segmentWorkersOverride = snapshot.segmentWorkersOverride == null
      ? null
      : Math.max(0, Math.min(16, Math.floor(Number(snapshot.segmentWorkersOverride) || 0)));
    batchGroupName = String(snapshot.batchGroupName || batchGroupName || '');
    experimentPurpose = String(snapshot.experimentPurpose || '');
    experimentKind = String(snapshot.experimentKind || 'exploration');
    analysisQoeFunction = String(snapshot.analysis_qoe_function || PAPER_QOE_FUNCTION);
    analysisQoeScoringVersion = String(snapshot.analysis_qoe_scoring_version || PAPER_QOE_SCORING_VERSION);
    analysisQoeWarmupSegments = Math.max(0, Math.floor(Number(snapshot.analysis_qoe_warmup_segments) || 0));
    if (snapshot.mediaPairs?.length) {
      mediaPairs = snapshot.mediaPairs.map((pair) => ({
        id: pair.id || generateLocalId(),
        dataset_id: pair.dataset_id || '',
        v3c_folder: pair.v3c_folder || '',
        content_dir: pair.content_dir || '',
        fov_traces: [...(pair.fov_traces || [])],
      }));
    }
    plannedRuns = [];
    lastPlannedNetworkInputs = [];
    lastPlannedFovTracesByPair = {};
    planMessage = 'No plan yet.';
  }

  function defaultBatchGroupName(): string {
    const stamp = new Date().toISOString().slice(0, 16).replace('T', ' ');
    return batchGroupName.trim() || `Batch sweep ${stamp}`;
  }

  function setSegmentWorkersOverride(event: Event): void {
    const value = (event.currentTarget as HTMLSelectElement).value;
    segmentWorkersOverride = value === '' ? null : Number(value);
  }

  function safeGroupId(name: string): string {
    const cleaned = name.toLowerCase().replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '');
    return cleaned || `batch_${Date.now().toString(36)}`;
  }

  function exportBatchPlan(): void {
    const plan = {
      schema_version: 1,
      name: defaultBatchGroupName(),
      description: 'Prism batch plan export',
      experiment: { id: safeGroupId(defaultBatchGroupName()), name: defaultBatchGroupName(), purpose: experimentPurpose, kind: experimentKind, primary_metrics: ['qoe', 'weighted_iou', 'catastrophic_miss_rate'] },
      created_at: new Date().toISOString(),
      generator: 'prism_batch_planner',
      config: captureBatchSnapshot(),
      trace_selection: {
        network_source: networkSource,
        network_sampling_mode: networkSource === 'trace' ? networkTraceSamplingMode : 'profile',
        network_inputs: plannedRuns.length
          ? uniqueSortedStrings(plannedRuns.map((run) => run.config.network_profile || run.config.network_trace || ''))
          : [...lastPlannedNetworkInputs],
        network_trace_buckets: networkSource === 'trace'
          ? networkTraceBuckets(
            plannedRuns.length
              ? uniqueSortedStrings(plannedRuns.map((run) => run.config.network_trace || ''))
              : [...lastPlannedNetworkInputs],
          )
          : {},
        fov_traces_by_pair: plannedRuns.length ? fovTraceSelectionFromPlannedRuns() : { ...lastPlannedFovTracesByPair },
      },
      planned_runs: plannedRuns.map((run) => ({ config: run.config })),
    };
    const blob = new Blob([`${JSON.stringify(plan, null, 2)}\n`], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${safeGroupId(plan.name)}.prism-batch-plan.json`;
    link.click();
    URL.revokeObjectURL(url);
    planMessage = `Exported plan "${plan.name}".`;
  }

  async function importBatchPlan(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = '';
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      const snapshot = parsed?.config || parsed;
      if (!snapshot || typeof snapshot !== 'object') throw new Error('Plan file does not contain a config object.');
      applyBatchSnapshot(snapshot);
      if (parsed?.experiment && typeof parsed.experiment === 'object') {
        experimentPurpose = String(parsed.experiment.purpose || experimentPurpose);
        experimentKind = String(parsed.experiment.kind || experimentKind);
      }
      batchGroupName = String(parsed?.name || snapshot.batchGroupName || batchGroupName || file.name.replace(/\.json$/i, ''));
      if (Array.isArray(parsed?.planned_runs) && parsed.planned_runs.length) {
        plannedRuns = parsed.planned_runs
          .map((entry: { config?: RunConfig }) => entry?.config)
          .filter(Boolean)
          .map((config: RunConfig) => ({ id: generateLocalId(), status: 'planned' as const, config }));
        planMessage = `Loaded ${plannedRuns.length} planned run(s) from "${file.name}".`;
      } else {
        planMessage = `Loaded plan "${batchGroupName}". Review fields, then plan runs.`;
      }
    } catch (err) {
      planMessage = `Could not load plan: ${err instanceof Error ? err.message : String(err)}`;
    }
  }

  async function loadReportPlanFromLink(): Promise<void> {
    const query = new URLSearchParams(window.location.search);
    const reportId = query.get('report');
    const planId = query.get('plan');
    if (!reportId || !planId) return;
    const loaded = await loadReportBatchPlan(reportId, planId);
    const parsed = loaded.plan as { config?: Partial<BatchPlannerSnapshot>; experiment?: { purpose?: string; kind?: string }; name?: string; planned_runs?: Array<{ config?: RunConfig }> };
    const snapshot = (parsed?.config || parsed) as Partial<BatchPlannerSnapshot>;
    if (!snapshot || typeof snapshot !== 'object') throw new Error('The report plan does not contain a config object.');
    applyBatchSnapshot(snapshot);
    experimentPurpose = String(parsed?.experiment?.purpose || experimentPurpose);
    experimentKind = String(parsed?.experiment?.kind || experimentKind);
    batchGroupName = String(parsed?.name || snapshot.batchGroupName || batchGroupName);
    if (Array.isArray(parsed?.planned_runs) && parsed.planned_runs.length) {
      plannedRuns = parsed.planned_runs.map((entry) => entry?.config).filter(Boolean).map((config) => ({ id: generateLocalId(), status: 'planned' as const, config: config as RunConfig }));
    }
    planMessage = `Loaded report plan “${loaded.label}” from ${loaded.path}. Review it, then launch when ready.`;
  }

  function saveCurrentBatchSession(): void {
    const name = window.prompt('Session name:');
    if (name === null) return;
    const trimmed = name.trim();
    if (!trimmed) {
      planMessage = 'Session name is required.';
      return;
    }
    const next = batchSessions.filter((session) => session.name !== trimmed);
    next.push({ name: trimmed, config: captureBatchSnapshot(), updated_at: new Date().toISOString() });
    persistBatchSessions(next);
    selectedBatchSessionName = trimmed;
    planMessage = `Saved session "${trimmed}".`;
  }

  function loadSelectedBatchSession(): void {
    if (!selectedBatchSessionName) return;
    const found = batchSessions.find((session) => session.name === selectedBatchSessionName);
    if (!found) {
      planMessage = `Session "${selectedBatchSessionName}" not found.`;
      return;
    }
    applyBatchSnapshot(found.config);
    planMessage = `Loaded session "${found.name}".`;
  }

  function deleteSelectedBatchSession(): void {
    if (!selectedBatchSessionName) return;
    if (!window.confirm(`Delete saved batch session "${selectedBatchSessionName}"?`)) return;
    persistBatchSessions(batchSessions.filter((session) => session.name !== selectedBatchSessionName));
    planMessage = `Deleted session "${selectedBatchSessionName}".`;
    selectedBatchSessionName = '';
  }

  function headsetPresets(): HeadsetPreset[] {
    return options?.headset_presets || [];
  }

  function selectedHeadset(): HeadsetPreset | null {
    return headsetPresets().find((preset) => preset.id === headsetPreset) || null;
  }

  function datasetChoices(): DatasetProfile[] {
    return options?.datasets || [];
  }

  function datasetById(datasetId: string): DatasetProfile | null {
    return datasetChoices().find((dataset) => dataset.id === datasetId) || null;
  }

  function activeTracePool(): string[] {
    return tracePickerMode === 'network' ? (options?.network_traces || []) : (options?.fov_traces || []);
  }

  function activeTraceAnalysis() {
    return analyzeTracePaths(activeTracePool(), tracePickerMode === 'network' ? options?.network_root || '' : options?.fov_root || '');
  }

  function traceGroupingOptions(): TraceGroupingOption[] {
    return activeTraceAnalysis().groupingOptions;
  }

  function activeTraceFilters(): TraceFilterOption[] {
    return activeTraceAnalysis().filterOptions;
  }

  function resetTraceFilters(): void {
    traceFilterSelections = Object.fromEntries(activeTraceFilters().map((filter) => [filter.id, []]));
  }

  function groupedActiveTraces() {
    return groupTraces(
      activeTracePool(),
      tracePickerMode === 'network' ? options?.network_root || '' : options?.fov_root || '',
      traceGroupingId || activeTraceAnalysis().defaultGroupingId,
      traceFilterSelections,
      traceFilterText,
      traceSortMode,
    );
  }

  function abrRequiresPredictor(abr: string): boolean {
    return Boolean(options?.abr_requires_predictor?.[abr]);
  }

  function abrSupportsPredictor(abr: string): boolean {
    return Boolean(options?.abr_supports_predictor?.[abr]);
  }

  function isPredictorCompatible(abr: string, predictor: string): boolean {
    const hasPredictor = Boolean(predictor && predictor !== 'None');
    if (abrRequiresPredictor(abr)) return hasPredictor;
    if (!abrSupportsPredictor(abr)) return !hasPredictor;
    return true;
  }

  function coerceScalar(spec: AlgorithmParamField, value: unknown): string | number | boolean | null {
    if (spec.kind === 'boolean') return Boolean(value);
    if (spec.kind === 'integer') {
      if (value === null || value === undefined || value === '') return spec.default;
      const parsed = Number.parseInt(String(value), 10);
      return Number.isFinite(parsed) ? parsed : spec.default;
    }
    if (spec.kind === 'number') {
      if (value === null || value === undefined || value === '') return spec.default === null ? null : spec.default;
      const parsed = Number.parseFloat(String(value));
      return Number.isFinite(parsed) ? parsed : spec.default;
    }
    if (value === null || value === undefined) return '';
    return String(value);
  }

  function normalizeParams(specs: AlgorithmParamField[] | undefined | null, raw: ParamMap | null): ParamMap {
    const safeSpecs = specs || [];
    const out: ParamMap = {};
    safeSpecs.forEach((spec) => {
      out[spec.name] = spec.default;
    });
    if (!raw) return out;
    safeSpecs.forEach((spec) => {
      if (!(spec.name in raw)) return;
      out[spec.name] = coerceScalar(spec, raw[spec.name]);
    });
    return out;
  }

  function abrPresetParams(abr: string, presetName: string): ParamMap {
    if (!presetName || presetName === DEFAULT_BATCH_PRESET) return {};
    return normalizeParams(options?.abr_param_specs?.[abr], getPresetParams(presets, 'abr', abr, presetName));
  }

  function fovPresetParams(predictor: string, presetName: string): ParamMap {
    if (!presetName || presetName === DEFAULT_BATCH_PRESET || !predictor || predictor === 'None') return {};
    return normalizeParams(options?.fov_param_specs?.[predictor], getPresetParams(presets, 'fov', predictor, presetName));
  }

  function bandwidthPresetParams(predictor: string, presetName: string): ParamMap {
    if (!presetName || presetName === DEFAULT_BATCH_PRESET || !predictor || predictor === 'None') return {};
    return normalizeParams(options?.bandwidth_param_specs?.[predictor], getPresetParams(presets, 'bandwidth', predictor, presetName));
  }

  function selectedPresetNames(record: Record<string, string[]>, key: string): string[] {
    const selected = Array.from(new Set(record[key]?.filter(Boolean) || []));
    // An untouched algorithm has its baseline configuration selected.
    return selected.length ? selected : [DEFAULT_BATCH_PRESET];
  }

  function togglePresetName(record: Record<string, string[]>, key: string, presetName: string): Record<string, string[]> {
    const hasExplicitSelection = Object.prototype.hasOwnProperty.call(record, key);
    // A named preset replaces the initial, implicit Default. Once Default is
    // checked after that, it is stored explicitly and can be swept alongside
    // named presets.
    const current = new Set(hasExplicitSelection ? (record[key]?.filter(Boolean) || []) : []);
    if (current.has(presetName)) current.delete(presetName);
    else current.add(presetName);
    return { ...record, [key]: Array.from(current).sort((a, b) => a.localeCompare(b)) };
  }

  function toggleAbrPreset(abr: string, presetName: string): void {
    selectedAbrPresetsByAlgo = togglePresetName(selectedAbrPresetsByAlgo, abr, presetName);
  }

  function toggleFovPreset(predictor: string, presetName: string): void {
    selectedFovPresetsByPredictor = togglePresetName(selectedFovPresetsByPredictor, predictor, presetName);
  }

  function toggleBandwidthPreset(predictor: string, presetName: string): void {
    selectedBandwidthPresetsByPredictor = togglePresetName(selectedBandwidthPresetsByPredictor, predictor, presetName);
  }

  function isAbrPresetSelected(abr: string, presetName: string): boolean {
    return selectedPresetNames(selectedAbrPresetsByAlgo, abr).includes(presetName);
  }

  function isFovPresetSelected(predictor: string, presetName: string): boolean {
    return selectedPresetNames(selectedFovPresetsByPredictor, predictor).includes(presetName);
  }

  function isBandwidthPresetSelected(predictor: string, presetName: string): boolean {
    return selectedPresetNames(selectedBandwidthPresetsByPredictor, predictor).includes(presetName);
  }

  function presetLabel(presetName: string): string {
    return presetName === DEFAULT_BATCH_PRESET ? 'Default' : presetName;
  }

  function selectionSummary(values: Set<string>, fallback: string, maxChars = 44): string {
    const names = Array.from(values).filter(Boolean);
    if (!names.length) return fallback;

    let summary = '';
    for (let i = 0; i < names.length; i += 1) {
      const name = names[i]!;
      const candidate = summary ? `${summary}, ${name}` : name;
      if (candidate.length <= maxChars) {
        summary = candidate;
        continue;
      }
      if (!summary) {
        const reserve = i < names.length - 1 ? ' and more' : '';
        const trimmed = name.slice(0, Math.max(10, maxChars - reserve.length - 1)).trim();
        return `${trimmed}…${reserve}`;
      }
      return `${summary} and more`;
    }

    return summary;
  }

  function addPair(): void {
    mediaPairs = [
      ...mediaPairs,
      { id: generateLocalId(), dataset_id: '', v3c_folder: '', content_dir: '', fov_traces: [] },
    ];
  }

  function removePair(id: string): void {
    mediaPairs = mediaPairs.filter((p) => p.id !== id);
  }

  function updatePair(id: string, next: Partial<MediaPair>): void {
    mediaPairs = mediaPairs.map((p) => (p.id === id ? { ...p, ...next } : p));
  }

  function handlePairDatasetChange(id: string, event: Event): void {
    const target = event.currentTarget as HTMLSelectElement | null;
    const dataset = datasetById(target?.value || '');
    updatePair(id, {
      dataset_id: dataset?.id || '',
      v3c_folder: dataset?.mpd_folder || '',
      content_dir: dataset?.content_dir || '',
    });
  }

  function selectedTraceSummary(traces: string[], maxDatasets = 2): string {
    if (!traces.length) return 'Choose traces';
    return summarizeMultiTrace(traces, maxDatasets);
  }

  function lumousNetworkTraces(): string[] {
    return (options?.network_traces || [])
      .filter((trace) => String(trace || '').toLowerCase().includes('lumous'))
      .sort((a, b) => a.localeCompare(b));
  }

  function selectLumousNetworkTraces(): void {
    const traces = lumousNetworkTraces();
    if (!traces.length) {
      planMessage = 'No Lumous network traces were found under the configured network trace root.';
      return;
    }
    networkSource = 'trace';
    selectedNetworkTraces = traces;
    networkTraceSamplingMode = 'balanced_lumous';
    planMessage = `Selected ${traces.length} Lumous network traces. Pool: ${balancedLumousSummary(traces)}.`;
  }

  function networkProfileOptions() {
    return options?.network_profiles || [];
  }

  function selectedProfileSummary(profileIds: string[]): string {
    if (!profileIds.length) return 'All profiles';
    const labels = profileIds.map((id) => networkProfileOptions().find((profile) => profile.id === id)?.id || id);
    return labels.length <= 2 ? labels.join(', ') : `${labels.slice(0, 2).join(', ')} and ${labels.length - 2} more`;
  }

  function toggleNetworkProfile(profileId: string): void {
    const next = new Set(selectedNetworkProfiles);
    if (next.has(profileId)) next.delete(profileId);
    else next.add(profileId);
    selectedNetworkProfiles = Array.from(next).sort((a, b) => a.localeCompare(b));
  }

  function openNetworkTracePicker(): void {
    tracePickerMode = 'network';
    tracePickerPairId = null;
    traceGroupingId = activeTraceAnalysis().defaultGroupingId;
    traceFilterText = '';
    traceSortMode = 'name';
    tracePickerSelection = new Set(selectedNetworkTraces);
    resetTraceFilters();
  }

  function openFovTracePicker(pair: MediaPair): void {
    tracePickerMode = 'fov';
    tracePickerPairId = pair.id;
    traceGroupingId = activeTraceAnalysis().defaultGroupingId;
    traceFilterText = '';
    traceSortMode = 'name';
    tracePickerSelection = new Set(pair.fov_traces);
    resetTraceFilters();
  }

  function closeTracePicker(): void {
    tracePickerMode = null;
    tracePickerPairId = null;
    traceGroupingId = '';
    traceFilterText = '';
    tracePickerSelection = new Set();
    traceFilterSelections = {};
  }

  function activeTracePair(): MediaPair | null {
    return mediaPairs.find((pair) => pair.id === tracePickerPairId) || null;
  }

  function toggleTrace(trace: string): void {
    const next = new Set(tracePickerSelection);
    if (next.has(trace)) next.delete(trace);
    else next.add(trace);
    tracePickerSelection = next;
  }

  function areAllDatasetTracesSelected(traces: string[]): boolean {
    return traces.length > 0 && traces.every((trace) => tracePickerSelection.has(trace));
  }

  function toggleDatasetTraces(traces: string[]): void {
    const next = new Set(tracePickerSelection);
    const remove = areAllDatasetTracesSelected(traces);
    for (const trace of traces) {
      if (remove) next.delete(trace);
      else next.add(trace);
    }
    tracePickerSelection = next;
  }

  function saveTracePicker(): void {
    if (tracePickerMode === 'network') {
      selectedNetworkTraces = Array.from(tracePickerSelection).sort();
      closeTracePicker();
      return;
    }
    if (!tracePickerPairId) return;
    updatePair(tracePickerPairId, { fov_traces: Array.from(tracePickerSelection).sort() });
    closeTracePicker();
  }

  function toggleTraceFilter(filterId: string, value: string): void {
    const current = new Set(traceFilterSelections[filterId] || []);
    if (current.has(value)) current.delete(value);
    else current.add(value);
    traceFilterSelections = {
      ...traceFilterSelections,
      [filterId]: Array.from(current).sort((a, b) => a.localeCompare(b)),
    };
  }

  function clearTraceFilter(filterId: string): void {
    traceFilterSelections = {
      ...traceFilterSelections,
      [filterId]: [],
    };
  }

  function traceFilterCount(filterId: string): number {
    return traceFilterSelections[filterId]?.length || 0;
  }

  function toggleSelection(setRef: Set<string>, key: string): Set<string> {
    const next = new Set(setRef);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    return next;
  }

  function hashSeed(input: string): number {
    if (!input) return Date.now() >>> 0;
    let h = 2166136261;
    for (let i = 0; i < input.length; i += 1) {
      h ^= input.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function mulberry32(seed: number): () => number {
    return () => {
      let t = (seed += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function sampleArray<T>(arr: T[], count: number, rng: () => number): T[] {
    if (count >= arr.length) return [...arr];
    const pool = [...arr];
    const out: T[] = [];
    for (let i = 0; i < count && pool.length; i += 1) {
      const idx = Math.floor(rng() * pool.length);
      out.push(pool.splice(idx, 1)[0]!);
    }
    return out;
  }

  function uniqueSortedStrings(values: string[]): string[] {
    return Array.from(new Set(values.filter(Boolean))).sort((a, b) => a.localeCompare(b));
  }

  function networkTraceClass(path: string): string {
    const raw = String(path || '');
    const segments = raw.toLowerCase().split(/[\\/]+/).filter(Boolean);
    const fileName = segments[segments.length - 1] || '';
    const orderedParts = [fileName, ...segments.slice(0, -1).reverse()];
    for (const part of orderedParts) {
      const tokens = part.split(/[^a-z0-9]+/).filter(Boolean);
      if (tokens.includes('4g') || tokens.includes('lte')) return '4G';
      if (tokens.includes('5g')) return '5G';
    }
    const low = raw.toLowerCase();
    if (/(^|[^a-z0-9])(4g|lte)([^a-z0-9]|$)/.test(low)) return '4G';
    if (/(^|[^a-z0-9])5g([^a-z0-9]|$)/.test(low)) return '5G';
    return 'Other';
  }

  function networkTraceMobility(path: string): string {
    const low = String(path || '').toLowerCase();
    if (low.includes('walking') || low.includes('walk') || low.includes('pedestrian')) return 'Walking';
    if (low.includes('driving') || low.includes('drive') || low.includes('vehicle') || low.includes('vehicular') || low.includes('/car') || low.includes('_car')) return 'Driving';
    return 'Other';
  }

  function networkTraceLumousBucket(path: string): string {
    return `${networkTraceClass(path)} ${networkTraceMobility(path)}`;
  }

  function preferredLumousBucketOrder(): string[] {
    return ['4G Driving', '4G Walking', '5G Driving', '5G Walking'];
  }

  function sampleNetworkTraces(traces: string[], count: number, rng: () => number): string[] {
    const target = Math.max(1, Math.min(Math.floor(Number(count) || 1), traces.length));
    if (target >= traces.length) return [...traces];

    const buckets = new Map<string, string[]>();
    traces.forEach((trace) => {
      const key = networkTraceClass(trace);
      buckets.set(key, [...(buckets.get(key) || []), trace]);
    });
    const bucketEntries = Array.from(buckets.entries())
      .filter(([, values]) => values.length)
      .sort(([a], [b]) => a.localeCompare(b));
    if (bucketEntries.length <= 1) return sampleArray(traces, target, rng);

    const selected: string[] = [];
    const shuffledByClass = bucketEntries.map(([key, values]) => ({
      key,
      pool: sampleArray(values, values.length, rng),
      desired: (target * values.length) / traces.length,
      allocated: 0,
    }));

    shuffledByClass.forEach((bucket) => {
      if (selected.length >= target) return;
      const take = Math.min(bucket.pool.length, Math.floor(bucket.desired));
      selected.push(...bucket.pool.splice(0, take));
      bucket.allocated += take;
    });

    while (selected.length < target) {
      const next = shuffledByClass
        .filter((bucket) => bucket.pool.length)
        .sort((a, b) => ((b.desired - b.allocated) - (a.desired - a.allocated)) || a.key.localeCompare(b.key))[0];
      if (!next) break;
      selected.push(next.pool.shift()!);
      next.allocated += 1;
    }

    return selected.sort((a, b) => a.localeCompare(b));
  }

  function sampleBalancedLumousNetworkTraces(traces: string[], count: number, rng: () => number): string[] {
    const target = Math.max(1, Math.min(Math.floor(Number(count) || 1), traces.length));
    if (target >= traces.length) return [...traces];

    const buckets = new Map<string, string[]>();
    traces.forEach((trace) => {
      const key = networkTraceLumousBucket(trace);
      buckets.set(key, [...(buckets.get(key) || []), trace]);
    });

    const primaryBuckets = preferredLumousBucketOrder()
      .map((key) => ({ key, pool: sampleArray(buckets.get(key) || [], buckets.get(key)?.length || 0, rng), allocated: 0 }))
      .filter((bucket) => bucket.pool.length);
    if (primaryBuckets.length <= 1) return sampleNetworkTraces(traces, target, rng);

    const selected: string[] = [];
    const base = Math.floor(target / primaryBuckets.length);
    primaryBuckets.forEach((bucket) => {
      const take = Math.min(base, bucket.pool.length, target - selected.length);
      selected.push(...bucket.pool.splice(0, take));
      bucket.allocated += take;
    });

    while (selected.length < target) {
      const next = primaryBuckets
        .filter((bucket) => bucket.pool.length)
        .sort((a, b) => (a.allocated - b.allocated) || preferredLumousBucketOrder().indexOf(a.key) - preferredLumousBucketOrder().indexOf(b.key))[0];
      if (!next) break;
      selected.push(next.pool.shift()!);
      next.allocated += 1;
    }

    if (selected.length < target) {
      const leftovers = traces.filter((trace) => !selected.includes(trace));
      selected.push(...sampleArray(leftovers, target - selected.length, rng));
    }

    return selected.sort((a, b) => a.localeCompare(b));
  }

  function fovTraceSelectionFromPlannedRuns(): Record<string, string[]> {
    const byPair: Record<string, Set<string>> = {};
    for (const run of plannedRuns) {
      const key = run.config.dataset_id || run.config.content_dir || 'unknown';
      if (!byPair[key]) byPair[key] = new Set();
      if (run.config.fov_trace) byPair[key]!.add(run.config.fov_trace);
    }
    return Object.fromEntries(
      Object.entries(byPair).map(([key, traces]) => [key, Array.from(traces).sort((a, b) => a.localeCompare(b))]),
    );
  }

  function balancedLumousSummary(traces: string[]): string {
    const counts = new Map<string, number>();
    traces.forEach((trace) => counts.set(networkTraceLumousBucket(trace), (counts.get(networkTraceLumousBucket(trace)) || 0) + 1));
    return preferredLumousBucketOrder()
      .map((bucket) => `${bucket} ${counts.get(bucket) || 0}`)
      .join(', ');
  }

  function balancedLumousMissingBuckets(traces: string[]): string[] {
    const counts = new Map<string, number>();
    traces.forEach((trace) => counts.set(networkTraceLumousBucket(trace), (counts.get(networkTraceLumousBucket(trace)) || 0) + 1));
    return preferredLumousBucketOrder().filter((bucket) => !counts.get(bucket));
  }

  function networkTraceBuckets(traces: string[]): Record<string, string> {
    return Object.fromEntries(
      uniqueSortedStrings(traces).map((trace) => [trace, networkTraceLumousBucket(trace)]),
    );
  }

  function planRuns(): void {
    if (!options) return;

    const abrs = Array.from(selectedAbrs);
    const predictors = Array.from(selectedPredictors);
    const bandwidthPredictors = Array.from(selectedBandwidthPredictors);

    const rng = mulberry32(hashSeed(seedText));

    const networkPool = networkSource === 'profile'
      ? (selectedNetworkProfiles.length ? selectedNetworkProfiles : networkProfileOptions().map((profile) => profile.id))
      : (selectedNetworkTraces.length ? selectedNetworkTraces : (options.network_traces || []));
    const selectedNetwork = networkSource === 'trace'
      ? (
        networkTraceSamplingMode === 'balanced_lumous'
          ? sampleBalancedLumousNetworkTraces(networkPool, Math.max(1, netSamples), rng)
          : sampleNetworkTraces(networkPool, Math.max(1, netSamples), rng)
      )
      : sampleArray(networkPool, Math.max(1, netSamples), rng);

    const planned: PlannedRun[] = [];
    const plannedFovTracesByPair: Record<string, string[]> = {};
    let skippedIncompatible = 0;

    for (const pair of mediaPairs) {
      if (!pair.dataset_id) continue;
      const sampledFov = sampleArray(pair.fov_traces, Math.max(1, fovSamples), rng);
      plannedFovTracesByPair[pair.dataset_id] = [...sampledFov].sort((a, b) => a.localeCompare(b));
      for (const abr of abrs) {
        for (const predictor of predictors) {
          if (!isPredictorCompatible(abr, predictor)) {
            skippedIncompatible += selectedNetwork.length * sampledFov.length;
            continue;
          }
          const abrPresetNames = selectedPresetNames(selectedAbrPresetsByAlgo, abr);
          const fovPresetNames = predictor && predictor !== 'None'
            ? selectedPresetNames(selectedFovPresetsByPredictor, predictor)
            : [DEFAULT_BATCH_PRESET];
          const bandwidthNames = bandwidthPredictors.length ? bandwidthPredictors : ['None'];
          for (const abrPresetName of abrPresetNames) {
            for (const fovPresetName of fovPresetNames) {
              for (const bandwidthPredictor of bandwidthNames) {
                const bandwidthPresetNames = bandwidthPredictor && bandwidthPredictor !== 'None'
                  ? selectedPresetNames(selectedBandwidthPresetsByPredictor, bandwidthPredictor)
                  : [DEFAULT_BATCH_PRESET];
                for (const bandwidthPresetName of bandwidthPresetNames) {
                  const abrParams = abrPresetParams(abr, abrPresetName);
                  const predictorParams = fovPresetParams(predictor, fovPresetName);
                  const bandwidthParams = bandwidthPresetParams(bandwidthPredictor, bandwidthPresetName);
                  for (const networkInput of selectedNetwork) {
                    for (const fovTrace of sampledFov) {
                      planned.push({
                        id: generateLocalId(),
                        status: 'planned',
                        config: {
                          v3c_folder: pair.v3c_folder,
                          content_dir: pair.content_dir,
                          dataset_id: pair.dataset_id,
                          network_trace: networkSource === 'trace' ? networkInput : undefined,
                          network_profile: networkSource === 'profile' ? networkInput : undefined,
                          fov_trace: fovTrace,
                          abr,
                          predictor,
                          bandwidth_predictor: bandwidthPredictor,
                          abr_params: abrParams,
                          predictor_params: predictorParams,
                          bandwidth_predictor_params: bandwidthParams,
                          abr_preset_name: abrPresetName === DEFAULT_BATCH_PRESET ? undefined : abrPresetName,
                          predictor_preset_name: fovPresetName === DEFAULT_BATCH_PRESET ? undefined : fovPresetName,
                          bandwidth_predictor_preset_name: bandwidthPresetName === DEFAULT_BATCH_PRESET ? undefined : bandwidthPresetName,
                          headset_preset: selectedHeadset()?.id,
                          fov_h_deg: selectedHeadset()?.fov_h_deg,
                          fov_v_deg: selectedHeadset()?.fov_v_deg,
                          angle_convention: datasetById(pair.dataset_id)?.angle_convention,
                          rtt,
                          buffer_window_segments: bufferWindowSegments,
                          segment_limit: Math.max(0, Math.floor(Number(segmentLimit) || 0)),
                          enable_mid_download_cancel: enableMidCancel,
                          segment_workers_override: segmentWorkersOverride,
                          analysis_qoe_function: analysisQoeFunction,
                          analysis_qoe_scoring_version: analysisQoeScoringVersion,
                          analysis_qoe_warmup_segments: analysisQoeWarmupSegments,
                        },
                      });
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    plannedRuns = planned.sort((a, b) => plannedRunSortKey(a.config).localeCompare(plannedRunSortKey(b.config)));
    lastPlannedNetworkInputs = [...selectedNetwork].sort((a, b) => a.localeCompare(b));
    lastPlannedFovTracesByPair = plannedFovTracesByPair;
    if (planned.length) {
      const networkBalance = networkSource === 'trace' && networkTraceSamplingMode === 'balanced_lumous'
        ? ` Balanced Lumous network traces: ${balancedLumousSummary(selectedNetwork)}.`
        : '';
      const missing = networkSource === 'trace' && networkTraceSamplingMode === 'balanced_lumous'
        ? balancedLumousMissingBuckets(networkPool)
        : [];
      const missingMessage = missing.length
        ? ` Missing selected-pool buckets: ${missing.join(', ')}.`
        : '';
      planMessage = `${planned.length} runs planned.${networkBalance}${missingMessage}`;
      return;
    }
    planMessage = skippedIncompatible
      ? 'No runs planned. The selected ABRs and FoV predictors do not form any compatible pairs.'
      : 'No runs planned. Select ABRs, predictors, traces, and at least one media pair.';
  }

  function plannedRunSortKey(config: RunConfig): string {
    return [
      config.dataset_id || '',
      config.content_dir || '',
      config.fov_trace || '',
      config.network_profile ? `profile:${config.network_profile}` : config.network_trace || '',
      config.abr || '',
      config.predictor || '',
      config.bandwidth_predictor || '',
      config.abr_preset_name || '',
      config.predictor_preset_name || '',
      config.bandwidth_predictor_preset_name || '',
    ].join('\u0001');
  }

  function batchPayload(next: PlannedRun[]): RunConfig[] {
    const groupName = defaultBatchGroupName();
    const groupId = safeGroupId(`${groupName}_${Date.now().toString(36)}`);
    return next.map((plan) => ({ ...plan.config, run_group_id: groupId, run_group_name: groupName, run_group_source: 'batch_plan', experiment_id: safeGroupId(groupName), experiment_name: groupName, experiment_purpose: experimentPurpose, experiment_kind: experimentKind, experiment_primary_metrics: ['qoe', 'weighted_iou', 'catastrophic_miss_rate'], analysis_qoe_function: analysisQoeFunction, analysis_qoe_scoring_version: analysisQoeScoringVersion, analysis_qoe_warmup_segments: analysisQoeWarmupSegments }));
  }

  async function startBatch(skipPreflight = false): Promise<void> {
    if (!plannedRuns.length) return;
    const next = [...plannedRuns];
    const groupName = defaultBatchGroupName();
    const payload = batchPayload(next);
    if (!skipPreflight) {
      preflightLoading = true;
      try {
        const result = await preflightRuns(payload);
        preflight = result;
        const mode = loadLaunchPreflightMode();
        if (result.status === 'error' && mode === 'never') {
          planMessage = `Cannot start: ${result.checks.map((check) => check.title).join(' ')}`;
          return;
        }
        if (result.status === 'error' || result.status === 'warning' || mode === 'always') {
          showPreflight = true;
          return;
        }
      } catch (err) {
        planMessage = `Validation failed: ${err instanceof Error ? err.message : String(err)}`;
        return;
      } finally { preflightLoading = false; }
    }
    try {
      const res = await startRunsBatch(payload);
      const runIds = res.run_ids || [];
      plannedRuns = next.map((plan, index) => ({
        ...plan,
        status: 'queued',
        runId: runIds[index] || (index === 0 ? res.run_id : undefined),
      }));
      planMessage = `Queued ${plannedRuns.length} runs as "${groupName}" · inputs validated.`;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      plannedRuns = next.map((plan) => ({ ...plan, status: 'failed', error: message }));
      planMessage = `Batch start failed: ${message}`;
    }
  }

  async function refreshInsights(): Promise<void> {
    try {
      const finished = liveRuns.filter((r) => r.status === 'finished').map((r) => r.run_id);
      if (!finished.length) {
        diagnostics = null;
        return;
      }
      diagnostics = await getDiagnostics({ run_ids: finished });
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  async function refreshOptions(): Promise<void> {
    options = await getOptions();
  }

  function abrRows() {
    const metrics = diagnostics;
    if (!metrics) return [] as Array<{ abr: string; runs: number; avgQoe: number; avgStall: number; avgBitrate: number }>;
    const buckets: Record<string, { runs: number; qoe: number[]; stall: number[]; bitrate: number[] }> = {};

    Object.keys(metrics.qoe || {}).forEach((runId) => {
      const abr = metrics.meta?.[runId]?.abr || 'Unknown';
      if (!buckets[abr]) buckets[abr] = { runs: 0, qoe: [], stall: [], bitrate: [] };
      buckets[abr].runs += 1;
      buckets[abr].qoe.push(...(metrics.qoe?.[runId] || []));
      buckets[abr].stall.push(Number(metrics.stall?.[runId]?.duration || 0));
      const bq = metrics.bitrate_vs_quality?.[runId] || [];
      bq.forEach(([bitrate]) => buckets[abr]!.bitrate.push(Number(bitrate || 0)));
    });

    const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
    return Object.entries(buckets).map(([abr, b]) => ({
      abr,
      runs: b.runs,
      avgQoe: avg(b.qoe),
      avgStall: avg(b.stall),
      avgBitrate: avg(b.bitrate),
    }));
  }

  onMount(async () => {
    try {
      batchSessions = loadBatchSessions();
      await refreshOptions();
      selectedAbrs = new Set([options?.abr_classes[0] || 'BandwidthAwareABR']);
      headsetPreset = options?.default_headset_preset || options?.headset_presets?.[0]?.id || '';
      selectedNetworkTraces = [...(options?.network_traces || [])];
      selectedNetworkProfiles = [];
      mediaPairs = [{ id: generateLocalId(), dataset_id: '', v3c_folder: '', content_dir: '', fov_traces: (options?.fov_traces || []).slice(0, 3) }];
      await loadReportPlanFromLink();
      liveRuns = await listRuns();
      unsubscribeRuns = subscribeRuns((runs) => {
        liveRuns = runs;
      });
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  });

  onDestroy(() => {
    unsubscribeRuns?.();
  });
</script>

<header class="hero">
  <div class="hero-actions">
    <a class="button-link ghost" href="/" on:click|preventDefault={() => gotoRoute('home')}>Main menu</a>
    <a class="button-link ghost" href="/static/datasets.html" on:click|preventDefault={() => gotoRoute('datasets')}>Datasets</a>
    <a class="button-link ghost" href="/static/index.html" on:click|preventDefault={() => gotoRoute('control')}>Single Run</a>
    <a class="button-link active" aria-current="page" href="/static/aggregate.html" on:click|preventDefault={() => gotoRoute('batch')}>Batch Planner</a>
    <a class="button-link ghost" href="/static/compare.html" on:click|preventDefault={() => gotoRoute('compare')}>Compare Runs</a>
    <a class="button-link ghost" href="/static/reports.html" on:click|preventDefault={() => gotoRoute('reports')}>Reports</a>
    <a class="button-link ghost" href="/static/extra.html" on:click|preventDefault={() => gotoRoute('extra')}>Extra</a>
    <a class="button-link ghost" href="/static/settings.html" on:click|preventDefault={() => gotoRoute('settings')}>Settings</a>
  </div>
</header>

<main>
  <section class="card form-card aggregate-form">
    <div class="run-form-toolbar">
      <h2>Plan experiment set</h2>
      <details class="form-tools-menu">
        <summary>Form tools</summary>
        <div class="form-tools-popover">
          <label>Saved session
            <select bind:value={selectedBatchSessionName}>
              <option value="">Select saved session</option>
              {#each batchSessions as session}
                <option value={session.name}>{session.name}</option>
              {/each}
            </select>
          </label>
          <button type="button" class="ghost small" on:click={saveCurrentBatchSession}>Save current fields</button>
          <button type="button" class="ghost small" on:click={loadSelectedBatchSession} disabled={!selectedBatchSessionName}>Load selected session</button>
          <button type="button" class="ghost small" on:click={deleteSelectedBatchSession} disabled={!selectedBatchSessionName}>Delete selected session</button>
          <button type="button" class="ghost small" on:click={exportBatchPlan}>Export plan JSON</button>
          <button type="button" class="ghost small" on:click={() => planFileInput?.click()}>Load plan JSON</button>
          <input class="hidden-file-input" bind:this={planFileInput} type="file" accept="application/json,.json" on:change={importBatchPlan} />
          <button type="button" class="ghost small" on:click={refreshOptions}>Refresh datasets</button>
        </div>
      </details>
    </div>

    {#if loading}
      <p class="subtle">Loading options…</p>
    {:else}
      <div class="planner-grid">
        <div class="panel-block abr-panel">
          <div class="panel-topline">
            <p class="eyebrow micro">ABRs</p>
            <span class="panel-count">{selectedAbrs.size}</span>
          </div>
          <details class="compact-picker">
            <summary>{selectionSummary(selectedAbrs, 'Select ABRs', 54)}</summary>
            <div class="algorithm-choice-list">
              {#each options?.abr_classes || [] as abr}
                <div class:selected={selectedAbrs.has(abr)} class="algorithm-choice-row">
                  <label class="pill-toggle algorithm-toggle">
                    <input type="checkbox" checked={selectedAbrs.has(abr)} on:change={() => (selectedAbrs = toggleSelection(selectedAbrs, abr))} />
                    <span title={abr}>{abr}</span>
                  </label>
                  {#if selectedAbrs.has(abr)}
                    <div class="preset-checkbox-list" aria-label={`Presets for ${abr}`}>
                      {#each [DEFAULT_BATCH_PRESET, ...listPresetNames(presets, 'abr', abr)] as presetName}
                        <label class="mini-check">
                          <input type="checkbox" checked={isAbrPresetSelected(abr, presetName)} on:change={() => toggleAbrPreset(abr, presetName)} />
                          <span title={presetLabel(presetName)}>{presetLabel(presetName)}</span>
                        </label>
                      {/each}
                    </div>
                  {/if}
                </div>
              {/each}
            </div>
          </details>
        </div>

        <div class="panel-block fov-panel">
          <div class="panel-topline">
            <p class="eyebrow micro">FoV predictors</p>
            <span class="panel-count">{selectedPredictors.size}</span>
          </div>
          <details class="compact-picker">
            <summary>{selectionSummary(selectedPredictors, 'Select predictors', 54)}</summary>
            <div class="algorithm-choice-list">
              <div class:selected={selectedPredictors.has('None')} class="algorithm-choice-row">
                <label class="pill-toggle algorithm-toggle">
                  <input
                    type="checkbox"
                    checked={selectedPredictors.has('None')}
                    on:change={() => (selectedPredictors = toggleSelection(selectedPredictors, 'None'))}
                  />
                  <span>None</span>
                </label>
              </div>
              {#each options?.fov_predictors || [] as predictor}
                <div class:selected={selectedPredictors.has(predictor)} class="algorithm-choice-row">
                  <label class="pill-toggle algorithm-toggle">
                    <input
                      type="checkbox"
                      checked={selectedPredictors.has(predictor)}
                      on:change={() => (selectedPredictors = toggleSelection(selectedPredictors, predictor))}
                    />
                    <span title={predictor}>{fovPredictorLabel(predictor)}</span>
                  </label>
                  {#if selectedPredictors.has(predictor)}
                    <div class="preset-checkbox-list" aria-label={`Presets for ${predictor}`}>
                      {#each [DEFAULT_BATCH_PRESET, ...listPresetNames(presets, 'fov', predictor)] as presetName}
                        <label class="mini-check">
                          <input type="checkbox" checked={isFovPresetSelected(predictor, presetName)} on:change={() => toggleFovPreset(predictor, presetName)} />
                          <span title={presetLabel(presetName)}>{presetLabel(presetName)}</span>
                        </label>
                      {/each}
                    </div>
                  {/if}
                </div>
              {/each}
            </div>
          </details>
        </div>

        <div class="panel-block bandwidth-panel">
          <div class="panel-topline">
            <p class="eyebrow micro">Bandwidth predictors</p>
            <span class="panel-count">{selectedBandwidthPredictors.size}</span>
          </div>
          <details class="compact-picker">
            <summary>{selectionSummary(selectedBandwidthPredictors, 'Select bandwidth predictors', 54)}</summary>
            <div class="algorithm-choice-list">
              <div class:selected={selectedBandwidthPredictors.has('None')} class="algorithm-choice-row">
                <label class="pill-toggle algorithm-toggle">
                  <input
                    type="checkbox"
                    checked={selectedBandwidthPredictors.has('None')}
                    on:change={() => (selectedBandwidthPredictors = toggleSelection(selectedBandwidthPredictors, 'None'))}
                  />
                  <span>None</span>
                </label>
              </div>
              {#each options?.bandwidth_predictors || [] as predictor}
                <div class:selected={selectedBandwidthPredictors.has(predictor)} class="algorithm-choice-row">
                  <label class="pill-toggle algorithm-toggle">
                    <input
                      type="checkbox"
                      checked={selectedBandwidthPredictors.has(predictor)}
                      on:change={() => (selectedBandwidthPredictors = toggleSelection(selectedBandwidthPredictors, predictor))}
                    />
                    <span title={predictor}>{predictor}</span>
                  </label>
                  {#if selectedBandwidthPredictors.has(predictor)}
                    <div class="preset-checkbox-list" aria-label={`Presets for ${predictor}`}>
                      {#each [DEFAULT_BATCH_PRESET, ...listPresetNames(presets, 'bandwidth', predictor)] as presetName}
                        <label class="mini-check">
                          <input type="checkbox" checked={isBandwidthPresetSelected(predictor, presetName)} on:change={() => toggleBandwidthPreset(predictor, presetName)} />
                          <span title={presetLabel(presetName)}>{presetLabel(presetName)}</span>
                        </label>
                      {/each}
                    </div>
                  {/if}
                </div>
              {/each}
            </div>
          </details>
        </div>

        <div class="panel-block trace-panel">
          <div class="panel-topline">
            <p class="eyebrow micro">Network</p>
            <span class="panel-count">{networkSource === 'profile' ? selectedProfileSummary(selectedNetworkProfiles) : selectedNetworkTraces.length}</span>
          </div>
          <div class="trace-grid">
            <div class="trace-pool-field network-source-field">
              <span>Source</span>
              <div class="network-source-toggle">
                <button type="button" class:active={networkSource === 'trace'} on:click={() => (networkSource = 'trace')}>Trace</button>
                <button type="button" class:active={networkSource === 'profile'} on:click={() => (networkSource = 'profile')}>Profile</button>
              </div>
              {#if networkSource === 'trace'}
                <button type="button" class="ghost small trace-picker-button" on:click={openNetworkTracePicker}>
                  Browse traces
                </button>
                <button type="button" class="ghost small trace-picker-button" on:click={selectLumousNetworkTraces}>
                  Select Lumous 4G + 5G
                </button>
                <p class="pair-trace-summary">{selectedTraceSummary(selectedNetworkTraces)}</p>
              {:else}
                <details class="compact-picker network-profile-picker">
                  <summary>{selectedProfileSummary(selectedNetworkProfiles)}</summary>
                  <div class="network-profile-list">
                    {#each networkProfileOptions() as profile}
                      <label class="mini-check" title={profile.description}>
                        <input
                          type="checkbox"
                          checked={selectedNetworkProfiles.includes(profile.id)}
                          on:change={() => toggleNetworkProfile(profile.id)}
                        />
                        <span>{profile.id}</span>
                      </label>
                    {/each}
                  </div>
                </details>
                <p class="pair-trace-summary">Empty selection samples all profiles.</p>
              {/if}
            </div>
            <label>Network samples
              <input type="number" min="1" bind:value={netSamples} />
            </label>
            {#if networkSource === 'trace'}
              <label>Network sampling
                <select bind:value={networkTraceSamplingMode}>
                  <option value="proportional">Balanced by radio</option>
                  <option value="balanced_lumous">Balanced Lumous radio + mobility</option>
                </select>
                {#if networkTraceSamplingMode === 'balanced_lumous'}
                  <p class="pair-trace-summary">Selected pool: {balancedLumousSummary(selectedNetworkTraces.length ? selectedNetworkTraces : (options?.network_traces || []))}</p>
                {/if}
              </label>
            {/if}
            <label>FoV samples
              <input type="number" min="1" bind:value={fovSamples} />
            </label>
            <label>Seed
              <input bind:value={seedText} placeholder="e.g. 42" />
            </label>
          </div>
        </div>

        <div class="panel-block controls-panel">
          <div class="inline-grid">
            <label class="wide">Batch group name
              <input bind:value={batchGroupName} placeholder="Raycast paper sweep v1" />
            </label>
            <label>Experiment type
              <select bind:value={experimentKind}><option value="exploration">Exploration</option><option value="ablation">Ablation</option><option value="baseline_comparison">Baseline comparison</option><option value="paper_evaluation">Paper evaluation</option></select>
            </label>
            <label>Purpose
              <input bind:value={experimentPurpose} placeholder="What does this sweep test?" />
            </label>
            <label>RTT (s)
              <input type="number" step="0.01" bind:value={rtt} />
            </label>
            <label>Fetch window
              <input type="number" min="0" bind:value={bufferWindowSegments} />
            </label>
            <label title="Run only the first N segments. Set 0 for the full video.">Stop after segments
              <input type="number" min="0" step="1" bind:value={segmentLimit} />
            </label>
            <label title="Overrides the server setting for this imported/exported batch. Use 1 for controlled raycast experiments.">Segment workers
              <select
                value={segmentWorkersOverride == null ? '' : String(segmentWorkersOverride)}
                on:change={setSegmentWorkersOverride}
              >
                <option value="">Server default</option>
                <option value="0">Auto (server cap)</option>
                {#each Array.from({ length: 16 }, (_, index) => index + 1) as workerCount}
                  <option value={String(workerCount)}>{workerCount}</option>
                {/each}
              </select>
            </label>
            <label>Headset preset
              <select bind:value={headsetPreset}>
                {#each headsetPresets() as preset}
                  <option value={preset.id}>{preset.label}</option>
                {/each}
              </select>
            </label>
            <label>Adaptive abort
              <span class="check-row batch-check-row">
                <input type="checkbox" bind:checked={enableMidCancel} aria-label="Enable adaptive abort in batch runs" />
              </span>
            </label>
          </div>
        </div>

        <div class="panel-block media-panel">
          <div class="pair-list-head">
            <p class="eyebrow micro">Media pairs</p>
            <button class="ghost small" on:click={refreshOptions}>Refresh datasets</button>
            <button class="ghost small" on:click={addPair}>Add media row</button>
          </div>
          <div class="pair-list">
            {#each mediaPairs as pair, index}
              <div class="pair-card">
                <div class="pair-card-head">
                  <div>
                    <p class="pair-title">Dataset {index + 1}</p>
                  </div>
                  <button class="ghost small" on:click={() => removePair(pair.id)}>Remove</button>
                </div>
                <label>Dataset
                  <select value={pair.dataset_id} on:change={(e) => handlePairDatasetChange(pair.id, e)}>
                    <option value="">Select registered dataset</option>
                    {#each datasetChoices() as dataset}
                      <option value={dataset.id}>{dataset.name}</option>
                    {/each}
                  </select>
                  <p class="pair-trace-summary">
                    {#if pair.dataset_id}
                      {pair.v3c_folder} · {pair.content_dir}
                    {:else}
                      <a href="/static/datasets.html" on:click|preventDefault={() => gotoRoute('datasets')}>Manage datasets</a>
                    {/if}
                  </p>
                </label>
                <label class="pair-traces-field">FoV traces
                  <button type="button" class="ghost small trace-picker-button" on:click={() => openFovTracePicker(pair)}>
                    Browse
                  </button>
                  <p class="pair-trace-summary">{selectedTraceSummary(pair.fov_traces)}</p>
                </label>
              </div>
            {/each}
          </div>
        </div>
      </div>

      <div class="form-footer planner-actions">
        <button on:click={planRuns}>Plan runs</button>
        <button class="ghost" disabled={!plannedRuns.length || preflightLoading} on:click={() => startBatch()}>{preflightLoading ? 'Validating…' : 'Start batch'}</button>
        <span class="status">{planMessage}</span>
      </div>
    {/if}
  </section>

  <section class="card">
    <div class="section-header">
      <h2>Planned runs</h2>
    </div>
    <div class="status-list">
      {#if !plannedRuns.length}
        <p class="subtle">No planned runs yet.</p>
      {:else}
        {#each plannedRuns as plan}
          <div class="status-item">
            <div class="status-copy">
              <div class="status-title-row">
                <strong>{plan.config.abr}</strong>
                <span class="status-inline-meta">
                  {plan.config.abr_preset_name || 'Default'} · FoV {fovPredictorLabel(plan.config.predictor || 'None')} · {plan.config.predictor_preset_name || 'Default'} · BW {plan.config.bandwidth_predictor || 'None'} · {plan.config.bandwidth_predictor_preset_name || 'Default'} · {plan.config.headset_preset || 'No headset'} · {plan.config.dataset_id || 'No dataset'}
                </span>
              </div>
              <div class="status-trace-row">
                <code title={plan.config.network_profile || plan.config.network_trace || ''}>
                  {plan.config.network_profile ? `profile:${plan.config.network_profile}` : plan.config.network_trace}
                </code>
                <code title={plan.config.fov_trace}>{plan.config.fov_trace}</code>
              </div>
            </div>
            <span class="pill {plan.status === 'failed' ? 'failed' : plan.status === 'queued' ? 'success' : 'queued'}">{plan.status}</span>
          </div>
        {/each}
      {/if}
    </div>
  </section>

  <section class="card">
    <div class="section-header">
      <h2>Batch insights</h2>
      <div class="section-actions">
        <button class="ghost small" on:click={refreshInsights}>Refresh stats</button>
      </div>
    </div>

    {#if error}
      <p class="subtle">{error}</p>
    {/if}

    <div class="table-wrap">
      <table class="compact">
        <thead>
          <tr>
            <th>ABR</th>
            <th>Runs</th>
            <th>Avg QoE</th>
            <th>Avg stall (s)</th>
            <th>Avg bitrate (Mbps)</th>
          </tr>
        </thead>
        <tbody>
          {#each abrRows() as row}
            <tr>
              <td>{row.abr}</td>
              <td>{row.runs}</td>
              <td>{row.avgQoe.toFixed(3)}</td>
              <td>{row.avgStall.toFixed(2)}</td>
              <td>{row.avgBitrate.toFixed(2)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>
</main>

  {#if tracePickerMode}
  <div class="trace-picker-backdrop" on:pointerdown|self={closeTracePicker}>
    <section class="card trace-picker-modal">
      <div class="section-header">
        <div>
          <h3>{tracePickerMode === 'network' ? 'Select network traces' : 'Select FoV traces'}</h3>
          <p class="subtle">
            {#if tracePickerMode === 'network'}
              Choose one or more network traces from the detected folder structure.
            {:else}
              {activeTracePair()?.content_dir || activeTracePair()?.v3c_folder || 'Choose one or more FoV traces from the detected folder structure.'}
            {/if}
          </p>
        </div>
        <div class="section-actions">
          <button type="button" class="ghost small" on:click={closeTracePicker}>Close</button>
        </div>
      </div>

      <div class="trace-picker-toolbar">
        <label>Group by
          <select bind:value={traceGroupingId}>
            {#each traceGroupingOptions() as option}
              <option value={option.id}>{option.label}</option>
            {/each}
          </select>
        </label>
        <label>Filter
          <input bind:value={traceFilterText} placeholder="Search groups or paths" />
        </label>
        <label>Sort
          <select bind:value={traceSortMode}>
            <option value="name">Name</option>
            <option value="count">Count</option>
          </select>
        </label>
        <span class="micro subtle">{tracePickerSelection.size} selected</span>
        <div class="section-actions">
          <button type="button" class="ghost small" on:click={() => (tracePickerSelection = new Set(activeTracePool()))}>Select all</button>
          <button type="button" class="ghost small" on:click={() => (tracePickerSelection = new Set())}>Clear</button>
        </div>
      </div>

      {#if activeTraceFilters().length}
        <div class="trace-filter-strip">
          {#each activeTraceFilters() as filter}
            <details class="trace-filter-menu">
              <summary>
                {filter.label}
                {#if traceFilterCount(filter.id)}
                  <span>{traceFilterCount(filter.id)}</span>
                {/if}
              </summary>
              <div class="trace-filter-menu-body">
                <div class="trace-filter-chip-row">
                  {#each filter.values as value}
                    <button
                      type="button"
                      class:selected={traceFilterSelections[filter.id]?.includes(value)}
                      class="trace-filter-chip"
                      on:click={() => toggleTraceFilter(filter.id, value)}
                    >
                      {value}
                    </button>
                  {/each}
                </div>
                <button type="button" class="ghost small" on:click={() => clearTraceFilter(filter.id)}>Clear</button>
              </div>
            </details>
          {/each}
        </div>
      {/if}

      {#key `${tracePickerMode}:${traceGroupingId}:${traceFilterText}:${traceSortMode}:${JSON.stringify(traceFilterSelections)}`}
        <div class="trace-dataset-list">
          {#each groupedActiveTraces() as group}
            <section class="trace-dataset-card">
              <div class="trace-dataset-head">
                <div>
                  <h4>{group.name}</h4>
                  <p class="micro subtle">{group.traces.length} traces</p>
                </div>
                <button type="button" class="ghost small" on:click={() => toggleDatasetTraces(group.traces)}>
                  {areAllDatasetTracesSelected(group.traces) ? 'Clear' : 'Select all'}
                </button>
              </div>

              <div class="trace-checkbox-list">
                {#each group.traces as trace}
                  <label class="trace-option">
                    <input type="checkbox" checked={tracePickerSelection.has(trace)} on:change={() => toggleTrace(trace)} />
                    <span>{traceDisplayName(trace)}</span>
                  </label>
                {/each}
              </div>
            </section>
          {/each}
        </div>
      {/key}

      <div class="section-actions">
        <button type="button" class="ghost small" on:click={closeTracePicker}>Cancel</button>
        <button type="button" class="ghost small" on:click={saveTracePicker}>Use selected traces</button>
      </div>
    </section>
  </div>
{/if}

{#if showPreflight && preflight}
  <div class="preflight-backdrop" role="presentation">
    <section class="preflight-dialog" role="dialog" aria-modal="true" aria-labelledby="preflight-title">
      <h2 id="preflight-title">{preflight.status === 'error' ? 'Cannot queue this batch' : 'Review batch'}</h2>
      <p class="preflight-lede">{preflight.run_count} planned runs · {preflight.valid_count} valid configurations</p>
      {#if Object.keys(preflight.fixed).length}<p class="preflight-summary"><strong>Fixed:</strong> {Object.entries(preflight.fixed).map(([key, value]) => `${key}: ${value}`).join(' · ')}</p>{/if}
      {#if preflight.varied.length}<p class="preflight-summary"><strong>Varies:</strong> {preflight.varied.map((item) => `${item.label} (${item.count})`).join(' · ')}</p>{/if}
      {#each preflight.checks as check}
        <div class:preflight-error={check.severity === 'error'} class:preflight-warning={check.severity === 'warning'} class="preflight-check">
          <strong>{check.severity === 'error' ? '✕' : '⚠'} {check.title}</strong>
          <span>{check.affected_count} {check.affected_count === 1 ? 'run' : 'runs'} affected.</span>
          {#if check.detail}<small>{check.detail}</small>{/if}
        </div>
      {/each}
      {#if preflight.status === 'ready'}<p class="preflight-ready">✓ All inputs and configurations are compatible.</p>{/if}
      <div class="preflight-actions">
        <button class="ghost" on:click={() => { showPreflight = false; }}>Back</button>
        {#if preflight.status !== 'error'}<button on:click={() => { showPreflight = false; void startBatch(true); }}>Queue {preflight.run_count} runs</button>{/if}
      </div>
    </section>
  </div>
{/if}

<style>
  .preflight-backdrop { position: fixed; inset: 0; z-index: 50; display: grid; place-items: center; padding: 1rem; background: rgba(15, 23, 42, 0.42); }
  .preflight-dialog { width: min(720px, 100%); padding: 1.25rem; border: 1px solid var(--modal-border); border-radius: 14px; background: var(--modal-surface); box-shadow: 0 22px 60px rgba(0,0,0,.28); }
  .preflight-dialog h2 { margin: 0; }
  .preflight-lede, .preflight-summary { margin: .42rem 0; color: var(--muted-text); }
  .preflight-check { display: grid; gap: .18rem; margin-top: .7rem; padding: .7rem; border-left: 3px solid #e59b20; background: rgba(229,155,32,.08); }
  .preflight-check.preflight-error { border-left-color: #dc4c64; background: rgba(220,76,100,.08); }
  .preflight-check small { color: var(--muted-text); }
  .preflight-ready { color: #168a4d; }
  .preflight-actions { display: flex; justify-content: flex-end; gap: .55rem; margin-top: 1rem; }
  .hero-actions .button-link {
    white-space: nowrap;
  }

  .aggregate-form {
    padding: 0.75rem;
    background: var(--card-bg);
  }

  .run-form-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.8rem;
    margin: -0.75rem -0.75rem 0.55rem;
    padding: 0.4rem 0.62rem;
    border-bottom: 1px solid var(--card-border);
    background: rgba(148, 163, 184, 0.06);
    border-radius: 14px 14px 0 0;
  }

  .run-form-toolbar h2 {
    margin: 0;
    font-size: 0.98rem;
    white-space: nowrap;
  }

  .form-tools-menu {
    position: relative;
  }

  .form-tools-menu > summary {
    list-style: none;
    cursor: pointer;
    border: 1px solid var(--ghost-border);
    background: var(--soft-bg);
    color: var(--text-muted);
    border-radius: 10px;
    padding: 0.26rem 0.44rem;
    font-size: 0.8rem;
    line-height: 1.2;
    user-select: none;
  }

  .form-tools-menu > summary::-webkit-details-marker {
    display: none;
  }

  .form-tools-menu > summary::after {
    content: ' ▾';
    opacity: 0.8;
  }

  .form-tools-menu[open] > summary::after {
    content: ' ▴';
  }

  .form-tools-popover {
    position: absolute;
    top: calc(100% + 0.45rem);
    right: 0;
    z-index: 20;
    min-width: 220px;
    display: grid;
    gap: 0.45rem;
    padding: 0.55rem;
    border-radius: 12px;
    border: 1px solid var(--modal-border);
    background: var(--modal-surface);
    box-shadow: 0 16px 42px rgba(2, 6, 23, 0.65);
  }

  .form-tools-popover button {
    width: 100%;
    text-align: left;
  }

  .hidden-file-input {
    display: none;
  }

  .aggregate-form .planner-grid {
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0.65rem;
    align-items: start;
  }

  .aggregate-form .panel-block {
    display: grid;
    background: rgba(148, 163, 184, 0.035);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 10px;
    box-shadow: none;
    padding: 0.55rem 0.62rem;
    gap: 0.42rem;
  }

  .abr-panel,
  .fov-panel,
  .bandwidth-panel {
    grid-column: span 2;
    min-width: 0;
  }

  .trace-panel {
    grid-column: span 3;
    align-self: stretch;
  }

  .controls-panel {
    grid-column: span 3;
    align-self: stretch;
  }

  .media-panel {
    grid-column: 1 / -1;
  }

  .panel-topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
  }

  .panel-count {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 1.35rem;
    height: 1.35rem;
    padding: 0 0.45rem;
    border-radius: 999px;
    border: 1px solid rgba(56, 189, 248, 0.28);
    background: rgba(56, 189, 248, 0.1);
    color: #38bdf8;
    font-size: 0.72rem;
    font-weight: 700;
  }

  .compact-picker {
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 9px;
    background: var(--field-bg);
    padding: 0.12rem;
  }

  .compact-picker > summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    cursor: pointer;
    list-style: none;
    padding: 0.38rem 0.5rem;
    font-weight: 700;
    color: var(--text-main);
    min-height: 1.9rem;
    font-size: 0.86rem;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .compact-picker > summary::-webkit-details-marker {
    display: none;
  }

  .compact-picker > summary::after {
    content: '▾';
    flex: 0 0 auto;
    opacity: 0.8;
  }

  .compact-picker:not([open]) > summary::after {
    content: '▸';
  }

  .algorithm-choice-list {
    display: grid;
    gap: 0.28rem;
    margin-top: 0.12rem;
    padding: 0 0.18rem 0.18rem;
    max-height: 220px;
    overflow: auto;
  }

  .algorithm-choice-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 0.24rem;
    align-items: center;
    min-width: 0;
  }

  .algorithm-choice-row.selected {
    padding-bottom: 0.22rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12);
  }

  .algorithm-toggle {
    width: 100%;
    min-width: 0;
    border-radius: 8px;
    padding: 0.34rem 0.48rem;
  }

  .algorithm-toggle span {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .preset-checkbox-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    padding-left: 1.28rem;
  }

  .mini-check {
    min-width: 0;
    display: inline-flex;
    flex-direction: row;
    align-items: center;
    gap: 0.28rem;
    max-width: 100%;
    padding: 0.18rem 0.42rem;
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    background: rgba(255, 255, 255, 0.025);
    color: var(--text-muted);
    font-size: 0.76rem;
    line-height: 1.2;
  }

  .mini-check input {
    margin: 0;
    width: 0.82rem;
    height: 0.82rem;
  }

  .mini-check span {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .compact-picker .pill-toggle {
    border-radius: 12px;
    justify-content: space-between;
    width: 100%;
    padding: 0.48rem 0.65rem;
  }

  .trace-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.48rem 0.55rem;
    padding: 0;
  }

  .trace-pool-field {
    grid-column: span 2;
    min-width: 0;
  }

  .network-source-field {
    display: grid;
    gap: 0.28rem;
    align-content: start;
  }

  .network-source-toggle {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.25rem;
  }

  .network-source-toggle button {
    min-height: 1.8rem;
    padding: 0.25rem 0.45rem;
    border-radius: 8px;
    font-size: 0.78rem;
  }

  .network-source-toggle button.active {
    border-color: rgba(56, 189, 248, 0.45);
    background: rgba(56, 189, 248, 0.13);
    color: #38bdf8;
  }

  .network-profile-picker {
    min-width: 0;
  }

  .network-profile-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    max-height: 150px;
    overflow: auto;
    padding: 0 0.18rem 0.18rem;
  }

  .trace-grid label:last-child {
    grid-column: auto;
  }

  .controls-panel .inline-grid {
    flex: 1 1 auto;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.48rem 0.55rem;
    align-items: start;
  }

  .batch-check-row {
    min-height: 2.05rem;
    justify-content: center;
    padding: 0.25rem 0.55rem;
    border: 1px solid var(--field-border);
    border-radius: 10px;
    background: var(--field-bg);
  }

  .batch-check-row input {
    width: 1rem;
    height: 1rem;
  }

  .pair-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 0.55rem;
  }

  .pair-card {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.48rem 0.55rem;
    padding: 0.62rem;
    border-radius: 10px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(255, 255, 255, 0.03);
  }

  .pair-card-head {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.8rem;
  }

  .pair-title {
    margin: 0;
    font-weight: 700;
    color: var(--text-main);
  }

  .pair-traces-field {
    grid-column: 1 / -1;
  }

  .pair-card label {
    min-width: 0;
  }

  .trace-picker-button {
    justify-self: flex-start;
  }

  .pair-trace-summary {
    margin: 0.14rem 0 0;
    color: var(--text-muted);
    font-size: 0.76rem;
    line-height: 1.25;
  }

  .trace-picker-backdrop {
    position: fixed;
    inset: 0;
    z-index: 1400;
    background: var(--overlay-backdrop);
    backdrop-filter: blur(4px);
    display: grid;
    place-items: center;
    padding: 1rem;
  }

  .trace-picker-modal {
    width: min(920px, 96vw);
    max-height: 88vh;
    overflow: auto;
    display: grid;
    gap: 0.9rem;
    background: var(--modal-surface);
    border: 1px solid var(--modal-border);
    box-shadow: 0 22px 64px rgba(2, 6, 23, 0.8);
  }

  .trace-picker-toolbar {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    flex-wrap: wrap;
  }

  .trace-filter-strip {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
  }

  .trace-filter-menu {
    min-width: 0;
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.025);
  }

  .trace-filter-menu > summary {
    list-style: none;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.52rem 0.7rem;
    font-size: 0.84rem;
    color: var(--text-main);
  }

  .trace-filter-menu > summary::-webkit-details-marker {
    display: none;
  }

  .trace-filter-menu > summary span {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 1.1rem;
    height: 1.1rem;
    padding: 0 0.28rem;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    font-size: 0.72rem;
    font-weight: 700;
  }

  .trace-filter-menu-body {
    display: grid;
    gap: 0.55rem;
    padding: 0 0.7rem 0.7rem;
  }

  .trace-filter-chip-row {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
  }

  .trace-filter-chip {
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(255, 255, 255, 0.025);
    color: var(--text-main);
    border-radius: 999px;
    padding: 0.38rem 0.6rem;
    font-size: 0.78rem;
  }

  .trace-filter-chip.selected {
    border-color: rgba(56, 189, 248, 0.35);
    background: rgba(56, 189, 248, 0.12);
    color: #67e8f9;
  }

  .trace-dataset-list {
    display: grid;
    gap: 0.8rem;
  }

  .trace-dataset-card {
    display: grid;
    gap: 0.75rem;
    padding: 0.9rem;
    border-radius: 14px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    background: rgba(255, 255, 255, 0.03);
  }

  .trace-dataset-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.8rem;
  }

  .trace-dataset-head h4 {
    margin: 0;
  }

  .trace-dataset-head p {
    margin: 0.18rem 0 0;
  }

  .trace-checkbox-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.55rem 0.7rem;
  }

  .trace-option {
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    min-width: 0;
    padding: 0.55rem 0.65rem;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(255, 255, 255, 0.025);
  }

  .trace-option span {
    min-width: 0;
    overflow-wrap: anywhere;
  }

  .planner-actions {
    align-items: center;
    gap: 0.7rem;
    padding-top: 0.9rem;
    margin-top: 0.15rem;
    border-top: 1px solid rgba(148, 163, 184, 0.14);
  }

  .planner-actions .status {
    flex: 1 1 260px;
    min-width: 0;
  }

  .status-list {
    display: grid;
    gap: 0.55rem;
  }

  .status-item {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 0.75rem;
    align-items: center;
    padding: 0.7rem 0.8rem;
    border-radius: 14px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(255, 255, 255, 0.03);
  }

  .status-copy {
    display: grid;
    gap: 0.35rem;
    min-width: 0;
  }

  .status-title-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem 0.7rem;
    align-items: baseline;
  }

  .status-inline-meta {
    color: var(--text-muted);
    font-size: 0.9rem;
  }

  .status-trace-row {
    display: grid;
    gap: 0.35rem;
  }

  .status-trace-row code {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    padding: 0.28rem 0.5rem;
    border-radius: 8px;
    border: 1px solid var(--trace-chip-border);
    background: var(--trace-chip-bg);
    color: var(--trace-chip-text);
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.01em;
  }

  .table-wrap {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.18);
    background: rgba(255, 255, 255, 0.03);
  }

  @media (max-width: 1180px) {
    .abr-panel,
    .fov-panel,
    .bandwidth-panel,
    .trace-panel,
    .controls-panel {
      grid-column: span 6;
    }

    .trace-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .trace-pool-field {
      grid-column: 1 / -1;
    }
  }

  @media (max-width: 860px) {
    .aggregate-form .planner-grid,
    .trace-grid,
    .controls-panel .inline-grid,
    .pair-card {
      grid-template-columns: 1fr;
    }

    .abr-panel,
    .fov-panel,
    .bandwidth-panel,
    .trace-panel,
    .controls-panel {
      grid-column: 1 / -1;
    }

    .status-item {
      grid-template-columns: 1fr;
    }

    .pair-list {
      grid-template-columns: 1fr;
    }

    .trace-checkbox-list {
      grid-template-columns: 1fr;
    }

    .trace-filter-strip {
      flex-direction: column;
    }
  }
</style>
