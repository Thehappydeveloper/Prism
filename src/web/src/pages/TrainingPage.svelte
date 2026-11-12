<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import {
    controlTrainingRun,
    deleteTrainingModel,
    deleteTrainingDataset,
    deleteTrainingPlan,
    deleteTrainingRun,
    evaluateTrainingModel,
    exportTrainingModel,
    getOptions,
    getTraceColumnConfig,
    getTrainingDatasets,
    getTrainingMetrics,
    getTrainingResults,
    getTrainingPlans,
    listTrainingModels,
    listTrainingRuns,
    saveTrainingDataset,
    saveTrainingPlan,
    saveTraceColumnConfig,
    resumeTrainingRun,
    startTrainingRun,
  } from '../core/api';
  import ChartCanvas from '../components/ChartCanvas.svelte';
  import { gotoRoute } from '../core/router';
  import type { AppOptions, DatasetProfile, TraceColumnAliasSettings, TraceColumnField, TrainingDataset, TrainingGraph, TrainingMetrics, TrainingModel, TrainingPlan, TrainingRun } from '../core/types';
  import type { ChartConfiguration } from 'chart.js';

  let datasets: TrainingDataset[] = [];
  let plans: TrainingPlan[] = [];
  let runs: TrainingRun[] = [];
  let models: TrainingModel[] = [];
  let options: AppOptions | null = null;
  let metrics: TrainingMetrics | null = null;
  let graphs: TrainingGraph[] = [];
  let selectedRunId = '';
  let loading = true;
  let saving = false;
  let error = '';
  let statusMessage = '';
  let traceColumnStatusMessage = '';
  let traceColumnSettingsPath = '';
  let savingTraceColumns = false;
  let pollHandle: number | null = null;
  let activeTab: 'info' | 'settings' | 'datasets' | 'plans' | 'runs' | 'models' | 'results' = 'info';

  let datasetId = '';
  let datasetFamily: 'fov' | 'bandwidth' = 'fov';
  let datasetName = '';
  let tracePath = '';
  let fps = 30;
  let units = 'frames';
  let prismDatasetId = '';
  let datasetAngleConvention = 'unity_zxy';

  let planId = '';
  let planFamily: 'fov' | 'bandwidth' = 'fov';
  let planName = '';
  let selectedDatasetIds: string[] = [];
  let modelType = 'mlp';
  let target = 'pose';
  let hiddenSize = 64;
  let epochs = 5;
  let batchSize = 64;
  let learningRate = 0.001;
  let historyDuration = 2;
  let historyStep = 0.5;
  let sampleStride = 0.5;
  let horizon = 1;
  let seed = 0;
  let maxSamples = 0;
  let splitStrategy = 'by_trace';
  let validationFraction = 0.15;
  let testFraction = 0.15;
  let mixingStrategy = 'proportional';
  let manualWeights = '';
  let defaultSplitRole: 'train' | 'validation' | 'test' = 'train';
  let datasetSplitRoles: Record<string, 'default' | 'train' | 'validation' | 'test'> = {};
  let includeCurrent = true;
  let fixedHistorySamples = 0;
  let numLayers = 1;
  let dropout = 0;
  let optimizer = 'adam';
  let scheduler = 'none';
  let earlyStoppingPatience = 0;
  let checkpointFrequency = 1;
  let device = 'auto';
  let numWorkers = 0;
  let mixedPrecision = false;
  let traceColumnAliases: TraceColumnAliasSettings = { fov: {}, network: {} };
  let traceColumnFields: Record<string, Record<string, TraceColumnField>> = {};

  $: compatibleDatasets = datasets.filter((dataset) => dataset.family === planFamily);
  $: activeRuns = runs.some((run) => ['queued', 'running', 'stopping'].includes(run.status));
  $: modelRequiresPointCloud = ['pointnet', 'pointnet2', 'sparse_conv', 'sparseconv'].includes(modelType);
  $: selectedPrismDataset = options?.datasets?.find((dataset) => dataset.id === prismDatasetId) || null;
  $: selectedRun = runs.find((run) => run.run_id === selectedRunId) || null;
  $: selectedPlan = plans.find((plan) => plan.id === selectedRun?.plan_id) || null;
  $: currentSplitPreview = splitPreviewForPlan();
  $: if (planFamily === 'fov' && !modelRequiresPointCloud && target === 'tile_visibility') {
    target = 'pose';
  }

  type DatasetMetadata = {
    name?: string;
    transform?: DatasetProfile['transform'] | null;
    motion_limits?: DatasetProfile['motion_limits'] | null;
  } | null | undefined;

  function datasetTransformSummary(dataset: DatasetMetadata): string {
    const transform = dataset?.transform;
    if (!transform) return 'No transform recorded';
    const translation = Array.isArray(transform.translation) ? transform.translation.map((value) => Number(value).toPrecision(4)).join(', ') : '0, 0, 0';
    return `scale ${Number(transform.scale ?? 1).toPrecision(5)}, translation [${translation}]`;
  }

  function datasetMotionSummary(dataset: DatasetMetadata): string {
    const limits = dataset?.motion_limits;
    if (!limits) return 'No motion limits recorded';
    return `${Number(limits.max_speed ?? 0).toPrecision(3)} m/s, ${Number(limits.max_acceleration ?? 0).toPrecision(3)} m/s^2`;
  }

  function prismDatasetName(dataset: TrainingDataset): string {
    const linked = options?.datasets?.find((item) => item.id === dataset.prism_dataset_id);
    return linked?.name || dataset.prism_dataset_snapshot?.name || dataset.prism_dataset_id || '—';
  }

  function datasetById(datasetIdValue: string): TrainingDataset | undefined {
    return datasets.find((dataset) => dataset.id === datasetIdValue);
  }

  function datasetDisplayName(datasetIdValue: string): string {
    return datasetById(datasetIdValue)?.name || datasetIdValue;
  }

  function normalizeRole(value: unknown): 'train' | 'validation' | 'test' {
    const role = String(value || '').toLowerCase();
    return role === 'validation' || role === 'test' ? role : 'train';
  }

  function splitAssignmentsFromRoles(): Record<string, string> {
    const assignments: Record<string, string> = { '*': defaultSplitRole };
    for (const datasetIdValue of selectedDatasetIds) {
      const role = datasetSplitRoles[datasetIdValue];
      if (!role || role === 'default') continue;
      assignments[datasetDisplayName(datasetIdValue)] = role;
    }
    return assignments;
  }

  function roleForDataset(datasetIdValue: string): 'default' | 'train' | 'validation' | 'test' {
    return datasetSplitRoles[datasetIdValue] || 'default';
  }

  function updateDatasetRole(datasetIdValue: string, event: Event): void {
    const value = ((event.currentTarget as HTMLSelectElement | null)?.value || 'default') as 'default' | 'train' | 'validation' | 'test';
    datasetSplitRoles = { ...datasetSplitRoles, [datasetIdValue]: value };
  }

  function splitPreviewForPlan(plan: TrainingPlan | null | undefined = null): Record<'train' | 'validation' | 'test', string[]> {
    const sourceDatasetIds = plan ? plan.dataset_ids || [] : selectedDatasetIds;
    const assignments = plan ? plan.split_assignments || {} : splitAssignmentsFromRoles();
    const defaultRole = normalizeRole(assignments['*'] || 'train');
    const preview: Record<'train' | 'validation' | 'test', string[]> = { train: [], validation: [], test: [] };
    for (const datasetIdValue of sourceDatasetIds) {
      const dataset = datasetById(datasetIdValue);
      const name = dataset?.name || datasetIdValue;
      const role = normalizeRole(assignments[datasetIdValue] || assignments[name] || assignments[name.toLowerCase()] || defaultRole);
      preview[role].push(name);
    }
    return preview;
  }

  function compactNames(names: string[]): string {
    return names.length ? names.join(', ') : 'none';
  }

  function planSummary(plan: TrainingPlan): string {
    const preview = splitPreviewForPlan(plan);
    return `Train ${preview.train.length} / Val ${preview.validation.length} / Test ${preview.test.length}`;
  }

  function planTitle(planIdValue: string | undefined): string {
    return plans.find((plan) => plan.id === planIdValue)?.name || planIdValue || '-';
  }

  function runPlanTitle(run: TrainingRun | null | undefined): string {
    return String(run?.plan_name || planTitle(run?.plan_id));
  }

  function numberValue(value: unknown, fallback = 0): number {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function progressPercent(run: TrainingRun, keyCurrent: string, keyTotal: string): number {
    const current = numberValue(run.progress?.[keyCurrent]);
    const total = numberValue(run.progress?.[keyTotal]);
    if (total <= 0) return 0;
    return Math.max(0, Math.min(100, (current / total) * 100));
  }

  function progressLabel(run: TrainingRun): string {
    const phase = String(run.progress?.phase || run.status || 'queued');
    const epoch = numberValue(run.progress?.epoch);
    const epochs = numberValue(run.progress?.epochs);
    const batch = numberValue(run.progress?.batch);
    const batches = numberValue(run.progress?.batches);
    const indexCurrent = numberValue(run.progress?.index_current);
    const indexTotal = numberValue(run.progress?.index_total);
    if (phase === 'queued') return String(run.progress?.wait_reason || 'queued');
    if (phase === 'indexing' && indexTotal > 0) return `indexing samples · file ${indexCurrent}/${indexTotal}`;
    if (epochs > 0 && batches > 0) return `${phase} · epoch ${epoch}/${epochs} · batch ${batch}/${batches}`;
    if (epochs > 0) return `${phase} · epoch ${epoch}/${epochs}`;
    return phase;
  }

  function latestLogLine(run: TrainingRun): string {
    const log = run.log || [];
    return log.length ? String(log[log.length - 1] || '') : '';
  }

  function indexDetail(run: TrainingRun): string {
    const dataset = String(run.progress?.index_dataset || '');
    const trace = String(run.progress?.index_trace || '');
    const fileSamples = numberValue(run.progress?.index_file_samples);
    const datasetSamples = numberValue(run.progress?.index_dataset_samples);
    const parts = [dataset, trace].filter(Boolean);
    if (fileSamples || datasetSamples) parts.push(`${fileSamples} file samples`, `${datasetSamples} dataset samples`);
    return parts.join(' · ');
  }

  function metricText(value: unknown): string {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric.toPrecision(5) : '-';
  }

  function fixedText(value: unknown, digits = 2, suffix = ''): string {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? `${numeric.toFixed(digits)}${suffix}` : '-';
  }

  function durationText(value: unknown): string {
    const seconds = Number(value);
    if (!Number.isFinite(seconds)) return '-';
    if (seconds < 60) return `${Math.max(0, seconds).toFixed(0)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    if (minutes < 60) return `${minutes}m ${remainingSeconds}s`;
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    return `${hours}h ${remainingMinutes}m`;
  }

  function recentLog(run: TrainingRun | null): string[] {
    return [...(run?.log || [])].slice(-80).reverse();
  }

  function modelDisplayName(model: TrainingModel): string {
    return String(model.name || model.id);
  }

  function defaultExportName(model: TrainingModel): string {
    return String(model.name || model.model?.model_type || model.id);
  }

  function navClass(routeName: string): string {
    return routeName === 'training' ? 'button-link active' : 'button-link ghost';
  }

  async function refreshAll(): Promise<void> {
    try {
      const [datasetPayload, planPayload, runPayload, modelPayload] = await Promise.all([
        getTrainingDatasets(),
        getTrainingPlans(),
        listTrainingRuns(),
        listTrainingModels(),
      ]);
      datasets = datasetPayload.datasets || [];
      plans = planPayload.plans || [];
      runs = runPayload || [];
      models = modelPayload.models || [];
      if (selectedRunId) {
        try {
          const resultsPayload = await getTrainingResults(selectedRunId);
          metrics = resultsPayload.metrics;
          graphs = resultsPayload.graphs || [];
        } catch {
          metrics = null;
          graphs = [];
        }
      }
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  function applyTraceColumnSettings(
    settings: TraceColumnAliasSettings | null | undefined,
    fields: Record<string, Record<string, TraceColumnField>> | null | undefined,
  ): void {
    traceColumnFields = fields || {};
    const next: TraceColumnAliasSettings = {};
    for (const traceType of Object.keys(traceColumnFields)) {
      next[traceType] = {};
      for (const fieldId of Object.keys(traceColumnFields[traceType] || {})) {
        const field = traceColumnFields[traceType]?.[fieldId];
        next[traceType][fieldId] = [...(settings?.[traceType]?.[fieldId] || field?.aliases || [])];
      }
    }
    traceColumnAliases = next;
  }

  function aliasesText(traceType: string, fieldId: string): string {
    return (traceColumnAliases[traceType]?.[fieldId] || []).join(', ');
  }

  function updateTraceColumnAliases(traceType: string, fieldId: string, event: Event): void {
    const target = event.currentTarget as HTMLInputElement | null;
    const aliases = (target?.value || '').split(',').map((alias) => alias.trim()).filter(Boolean);
    traceColumnAliases = {
      ...traceColumnAliases,
      [traceType]: {
        ...(traceColumnAliases[traceType] || {}),
        [fieldId]: aliases,
      },
    };
  }

  async function saveTraceColumns(): Promise<void> {
    if (savingTraceColumns) return;
    savingTraceColumns = true;
    error = '';
    traceColumnStatusMessage = '';
    try {
      const result = await saveTraceColumnConfig(traceColumnAliases);
      traceColumnSettingsPath = result.settings_path || traceColumnSettingsPath;
      applyTraceColumnSettings(result.settings, result.fields);
      traceColumnStatusMessage = 'Shared trace column mappings saved.';
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      savingTraceColumns = false;
    }
  }

  function resetDataset(): void {
    datasetId = '';
    datasetName = '';
    tracePath = datasetFamily === 'fov' ? options?.fov_root || '' : options?.network_root || '';
    fps = datasetFamily === 'fov' ? 30 : 1;
    units = datasetFamily === 'fov' ? 'frames' : 'mbps';
    prismDatasetId = '';
    datasetAngleConvention = options?.angle_convention || 'unity_zxy';
  }

  function editDataset(dataset: TrainingDataset): void {
    datasetId = dataset.id;
    datasetFamily = dataset.family;
    datasetName = dataset.name;
    tracePath = dataset.trace_path;
    fps = Number(dataset.fps || (dataset.family === 'fov' ? 30 : 1));
    units = dataset.units || (dataset.family === 'fov' ? 'frames' : 'mbps');
    prismDatasetId = dataset.prism_dataset_id || '';
    datasetAngleConvention = dataset.angle_convention || options?.angle_convention || 'unity_zxy';
  }

  async function saveDatasetForm(): Promise<void> {
    saving = true;
    error = '';
    statusMessage = '';
    try {
      const payload = await saveTrainingDataset({
        id: datasetId,
        family: datasetFamily,
        name: datasetName,
        trace_path: tracePath,
        fps,
        units,
        prism_dataset_id: prismDatasetId || undefined,
        angle_convention: datasetAngleConvention,
      });
      datasets = payload.datasets || [];
      statusMessage = 'Training dataset saved.';
      resetDataset();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      saving = false;
    }
  }

  function resetPlan(): void {
    planId = '';
    planName = '';
    selectedDatasetIds = [];
    modelType = 'mlp';
    target = planFamily === 'fov' ? 'pose' : 'future_bandwidth';
    hiddenSize = 64;
    epochs = 5;
    batchSize = 64;
    learningRate = 0.001;
    historyDuration = 2;
    historyStep = 0.5;
    sampleStride = 0.5;
    horizon = 1;
    seed = 0;
    maxSamples = 0;
    splitStrategy = 'by_trace';
    validationFraction = 0.15;
    testFraction = 0.15;
    mixingStrategy = 'proportional';
    manualWeights = '';
    defaultSplitRole = 'train';
    datasetSplitRoles = {};
    includeCurrent = true;
    fixedHistorySamples = 0;
    numLayers = 1;
    dropout = 0;
    optimizer = 'adam';
    scheduler = 'none';
    earlyStoppingPatience = 0;
    checkpointFrequency = 1;
    device = 'auto';
    numWorkers = 0;
    mixedPrecision = false;
  }

  function editPlan(plan: TrainingPlan): void {
    planId = plan.id;
    planFamily = plan.family;
    planName = plan.name;
    selectedDatasetIds = [...(plan.dataset_ids || [])];
    modelType = String(plan.model?.model_type || 'mlp');
    target = String(plan.target || (plan.family === 'fov' ? 'pose' : 'future_bandwidth'));
    hiddenSize = Number(plan.model?.hidden_size || 64);
    epochs = Number(plan.training?.epochs || 5);
    batchSize = Number(plan.training?.batch_size || 64);
    learningRate = Number(plan.training?.learning_rate || 0.001);
    historyDuration = Number(plan.history?.history_duration_s || 2);
    historyStep = Number(plan.history?.history_step_s || 0.5);
    sampleStride = Number(plan.history?.sample_stride_s || plan.history?.history_step_s || 0.5);
    horizon = Number(plan.history?.prediction_horizon_s || 1);
    seed = Number(plan.seed || 0);
    maxSamples = Number(plan.max_samples_per_dataset || 0);
    splitStrategy = String(plan.split_strategy || 'by_trace');
    validationFraction = Number(plan.validation_fraction ?? 0.15);
    testFraction = Number(plan.test_fraction ?? 0.15);
    mixingStrategy = String(plan.mixing_strategy || 'proportional');
    manualWeights = Object.entries(plan.manual_dataset_weights || {}).map(([key, value]) => `${key}:${value}`).join(', ');
    const assignments = plan.split_assignments || {};
    defaultSplitRole = normalizeRole(assignments['*'] || 'train');
    const nextRoles: Record<string, 'default' | 'train' | 'validation' | 'test'> = {};
    for (const datasetIdValue of plan.dataset_ids || []) {
      const dataset = datasetById(datasetIdValue);
      const name = dataset?.name || datasetIdValue;
      const explicit = assignments[datasetIdValue] || assignments[name] || assignments[name.toLowerCase()];
      nextRoles[datasetIdValue] = explicit ? normalizeRole(explicit) : 'default';
    }
    datasetSplitRoles = nextRoles;
    includeCurrent = plan.history?.include_current !== false;
    fixedHistorySamples = Number(plan.history?.fixed_history_samples || 0);
    numLayers = Number(plan.model?.num_layers || 1);
    dropout = Number(plan.model?.dropout || 0);
    optimizer = String(plan.training?.optimizer || 'adam');
    scheduler = String(plan.training?.scheduler || 'none');
    earlyStoppingPatience = Number(plan.training?.early_stopping_patience || 0);
    checkpointFrequency = Number(plan.training?.checkpoint_frequency || 1);
    device = String(plan.training?.device || 'auto');
    numWorkers = Number(plan.training?.num_workers || 0);
    mixedPrecision = Boolean(plan.training?.mixed_precision || false);
  }

  function togglePlanDataset(datasetIdValue: string): void {
    if (selectedDatasetIds.includes(datasetIdValue)) {
      selectedDatasetIds = selectedDatasetIds.filter((value) => value !== datasetIdValue);
      const nextRoles = { ...datasetSplitRoles };
      delete nextRoles[datasetIdValue];
      datasetSplitRoles = nextRoles;
    } else {
      selectedDatasetIds = [...selectedDatasetIds, datasetIdValue];
      datasetSplitRoles = { ...datasetSplitRoles, [datasetIdValue]: 'default' };
    }
  }

  async function savePlanForm(): Promise<void> {
    saving = true;
    error = '';
    statusMessage = '';
    try {
      const parsedWeights: Record<string, number> = {};
      manualWeights.split(',').map((item) => item.trim()).filter(Boolean).forEach((item) => {
        const [key, value] = item.split(':').map((part) => part.trim());
        const numeric = Number(value);
        if (key && Number.isFinite(numeric)) parsedWeights[key] = numeric;
      });
      const parsedAssignments = splitStrategy === 'manual' ? splitAssignmentsFromRoles() : {};
      const payload = await saveTrainingPlan({
        id: planId,
        family: planFamily,
        name: planName,
        dataset_ids: selectedDatasetIds,
        split_strategy: splitStrategy,
        split_assignments: parsedAssignments,
        validation_fraction: validationFraction,
        test_fraction: testFraction,
        mixing_strategy: mixingStrategy,
        manual_dataset_weights: parsedWeights,
        max_samples_per_dataset: maxSamples,
        seed,
        target,
        history: {
          history_duration_s: historyDuration,
          history_step_s: historyStep,
          sample_stride_s: sampleStride,
          prediction_horizon_s: horizon,
          include_current: includeCurrent,
          fixed_history_samples: fixedHistorySamples,
        },
        model: {
          model_type: modelType,
          hidden_size: hiddenSize,
          num_layers: numLayers,
          dropout,
        },
        training: {
          epochs,
          batch_size: batchSize,
          learning_rate: learningRate,
          optimizer,
          loss: 'mse',
          scheduler,
          early_stopping_patience: earlyStoppingPatience,
          checkpoint_frequency: checkpointFrequency,
          device,
          num_workers: numWorkers,
          mixed_precision: mixedPrecision,
        },
      });
      plans = payload.plans || [];
      statusMessage = 'Training plan saved.';
      resetPlan();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      saving = false;
    }
  }

  async function startPlan(plan: TrainingPlan): Promise<void> {
    try {
      const run = await startTrainingRun(plan.id);
      selectedRunId = run.run_id;
      runs = [run, ...runs.filter((item) => item.run_id !== run.run_id)];
      statusMessage = `Started ${run.run_id}.`;
      await refreshAll();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  async function loadMetrics(run: TrainingRun): Promise<void> {
    selectedRunId = run.run_id;
    try {
      const payload = await getTrainingResults(run.run_id);
      metrics = payload.metrics;
      graphs = payload.graphs || [];
    } catch {
      metrics = null;
      graphs = [];
    }
  }

  async function exportModel(model: TrainingModel): Promise<void> {
    const exportName = window.prompt('Export model as', defaultExportName(model));
    if (exportName === null) {
      return;
    }
    try {
      const exported = await exportTrainingModel(model.id, exportName.trim());
      const exportedId = exported.id || exportName.trim() || model.id;
      statusMessage = `Exported ${modelDisplayName(model)}. Use ${model.family === 'fov' ? 'TrainedFovPredictor' : 'TrainedBandwidthPredictor'} with model_id=${exportedId}.`;
      await refreshAll();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  async function evaluateModel(model: TrainingModel): Promise<void> {
    try {
      const result = await evaluateTrainingModel(model.id, 'test');
      statusMessage = `Evaluation saved for ${model.id}: MAE ${Number(result.mae || 0).toPrecision(5)}.`;
      await refreshAll();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  async function resumeRun(run: TrainingRun): Promise<void> {
    if (!run.plan_id) return;
    try {
      const resumed = await resumeTrainingRun(run.plan_id, run.run_id);
      selectedRunId = resumed.run_id;
      statusMessage = `Resumed ${run.run_id}.`;
      await refreshAll();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  function chartConfig(graph: TrainingGraph): ChartConfiguration {
    return {
      type: graph.type,
      data: {
        labels: graph.x,
        datasets: graph.series.map((series, index) => ({
          label: series.label,
          data: series.values,
          borderColor: index === 0 ? '#38bdf8' : '#f472b6',
          backgroundColor: index === 0 ? 'rgba(56, 189, 248, 0.18)' : 'rgba(244, 114, 182, 0.18)',
          tension: 0.25,
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: true } },
        scales: { y: { beginAtZero: true } },
      },
    };
  }

  async function loadInitial(): Promise<void> {
    try {
      const [optionsPayload, traceColumnPayload] = await Promise.all([getOptions(), getTraceColumnConfig()]);
      options = optionsPayload;
      traceColumnSettingsPath = traceColumnPayload.settings_path || '';
      applyTraceColumnSettings(traceColumnPayload.settings, traceColumnPayload.fields);
      resetDataset();
      await refreshAll();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      loading = false;
    }
  }

  onMount(() => {
    loadInitial();
    pollHandle = window.setInterval(() => {
      if (activeRuns) refreshAll();
    }, 2500);
  });

  onDestroy(() => {
    if (pollHandle !== null) window.clearInterval(pollHandle);
  });
</script>

<header class="hero">
  <div class="hero-actions training-topbar">
    <a class="button-link" class:active={activeTab === 'info'} class:ghost={activeTab !== 'info'} href="/static/training.html" on:click|preventDefault={() => activeTab = 'info'}>Info</a>
    <a class="button-link" class:active={activeTab === 'settings'} class:ghost={activeTab !== 'settings'} href="/static/training.html" on:click|preventDefault={() => activeTab = 'settings'}>Settings</a>
    <a class="button-link" class:active={activeTab === 'datasets'} class:ghost={activeTab !== 'datasets'} href="/static/training.html" on:click|preventDefault={() => activeTab = 'datasets'}>Datasets</a>
    <a class="button-link" class:active={activeTab === 'plans'} class:ghost={activeTab !== 'plans'} href="/static/training.html" on:click|preventDefault={() => activeTab = 'plans'}>Plans</a>
    <a class="button-link" class:active={activeTab === 'runs'} class:ghost={activeTab !== 'runs'} href="/static/training.html" on:click|preventDefault={() => activeTab = 'runs'}>Runner</a>
    <a class="button-link" class:active={activeTab === 'models'} class:ghost={activeTab !== 'models'} href="/static/training.html" on:click|preventDefault={() => activeTab = 'models'}>Models</a>
    <a class="button-link" class:active={activeTab === 'results'} class:ghost={activeTab !== 'results'} href="/static/training.html" on:click|preventDefault={() => activeTab = 'results'}>Results</a>
    <a class="button-link ghost exit-link" href="/" on:click|preventDefault={() => gotoRoute('home')}>Exit Training Setup</a>
  </div>
</header>

<main>
  {#if loading}
    <section class="card"><p>Loading training module...</p></section>
  {:else}
    {#if error}<section class="alert error">{error}</section>{/if}
    {#if statusMessage}<section class="alert success">{statusMessage}</section>{/if}
    {#if activeTab === 'info'}
      <section class="training-home-panel">
        <div class="workflow-head">
          <div>
            <p class="eyebrow micro">Workflow</p>
            <h2>Train predictors, then export them back to Prism</h2>
          </div>
        </div>
        <div class="workflow-list">
          <article class="workflow-step">
            <div class="step-number">1</div>
            <div class="step-copy"><h3>Register Dataset</h3><p>Add FoV or bandwidth traces and bind FoV datasets to reference PLY/calibration metadata.</p></div>
            <button class="button-link" on:click={() => activeTab = 'datasets'}>Open</button>
          </article>
          <article class="workflow-step">
            <div class="step-number">2</div>
            <div class="step-copy"><h3>Prepare Plan</h3><p>Choose target, history, model, splits, mixing, checkpoints, and output settings.</p></div>
            <button class="button-link" on:click={() => activeTab = 'plans'}>Open</button>
          </article>
          <article class="workflow-step">
            <div class="step-number">3</div>
            <div class="step-copy"><h3>Train / Test</h3><p>Run training, pause/resume, evaluate checkpoints, and keep reproducible indexes/results.</p></div>
            <button class="button-link" on:click={() => activeTab = 'runs'}>Open</button>
          </article>
          <article class="workflow-step">
            <div class="step-number">4</div>
            <div class="step-copy"><h3>Inspect Results</h3><p>Review loss curves, test metrics, tile visibility metrics, and exported model metadata.</p></div>
            <button class="button-link" on:click={() => activeTab = 'results'}>Open</button>
          </article>
          <article class="workflow-step">
            <div class="step-number">5</div>
            <div class="step-copy"><h3>Export Predictor</h3><p>Export trained FoV or bandwidth predictors so the main Prism simulation can consume them.</p></div>
            <button class="button-link" on:click={() => activeTab = 'models'}>Open</button>
          </article>
        </div>
      </section>
    {/if}

    {#if activeTab === 'settings'}
      <section class="card form-card">
        <div class="section-header">
          <div>
            <h2>Training Settings</h2>
            <p class="subtle">Training uses the same saved trace column mappings as main Prism.</p>
          </div>
          <button type="button" on:click={saveTraceColumns} disabled={savingTraceColumns}>
            {savingTraceColumns ? 'Saving...' : 'Save column mappings'}
          </button>
        </div>
        <div class="settings-grid">
          <div><strong>FoV trace root</strong><span class="mono">{options?.fov_root || '-'}</span></div>
          <div><strong>Network trace root</strong><span class="mono">{options?.network_root || '-'}</span></div>
          <div><strong>Default Euler convention</strong><span>{options?.angle_convention || 'unity_zxy'}</span></div>
          <div><strong>Bandwidth format</strong><span>Lumos-compatible Time/Bandwidth CSV</span></div>
        </div>
        <div class="notice">
          FoV traces may provide either quaternion columns or Euler-angle columns. If both are present, Training and Prism prefer quaternions. If only Euler columns are present, they are converted to quaternions using the dataset registration convention.
        </div>

        {#if Object.keys(traceColumnFields).length === 0}
          <p class="subtle">Loading trace column mappings...</p>
        {:else}
          <div class="trace-column-editor">
            {#each Object.entries(traceColumnFields) as [traceType, fields]}
              <div class="trace-column-group">
                <h3>{traceType === 'fov' ? 'FoV traces' : 'Network traces'}</h3>
                <div class="trace-column-grid">
                  {#each Object.entries(fields) as [fieldId, field]}
                    <label class:preferred-quat={traceType === 'fov' && fieldId.startsWith('quaternion_')}>
                      <span>{field.label}{field.required ? '' : ' (optional)'}</span>
                      <small>Canonical: <code>{field.canonical}</code>{traceType === 'fov' && fieldId.startsWith('quaternion_') ? ' · preferred' : ''}</small>
                      <input
                        value={aliasesText(traceType, fieldId)}
                        on:input={(e) => updateTraceColumnAliases(traceType, fieldId, e)}
                        placeholder="alias_a, alias_b"
                      />
                    </label>
                  {/each}
                </div>
              </div>
            {/each}
          </div>
        {/if}
        {#if traceColumnSettingsPath}
          <p class="micro subtle saved-path">Saved in: <code>{traceColumnSettingsPath}</code></p>
        {/if}
        {#if traceColumnStatusMessage}
          <p class="status">{traceColumnStatusMessage}</p>
        {/if}
      </section>
    {/if}

    {#if activeTab === 'datasets'}
    <section class="training-grid">
      <div class="card">
        <div class="section-header">
          <h2>{datasetId ? 'Edit Training Dataset' : 'Register Training Dataset'}</h2>
          <button class="ghost small" on:click={resetDataset}>New</button>
        </div>
        <div class="form-grid">
          <label>Family
            <select bind:value={datasetFamily} on:change={resetDataset}>
              <option value="fov">FoV</option>
              <option value="bandwidth">Bandwidth</option>
            </select>
          </label>
          <label>Name <input bind:value={datasetName} /></label>
          <label class="wide">Trace file or folder <input bind:value={tracePath} /></label>
          <label>FPS / sampling <input type="number" min="0.001" step="0.001" bind:value={fps} /></label>
          <label>Units <input bind:value={units} /></label>
          {#if datasetFamily === 'fov'}
            <label class="wide">Source Prism dataset
              <select bind:value={prismDatasetId}>
                <option value="">Select registered Prism dataset</option>
                {#each options?.datasets || [] as dataset}
                  <option value={dataset.id}>{dataset.name}</option>
                {/each}
              </select>
            </label>
            {#if selectedPrismDataset}
              <div class="wide inherited-dataset-card">
                <div><span>Content dir</span><code>{selectedPrismDataset.content_dir || 'Not set'}</code></div>
                <div><span>Angle convention</span><strong>{selectedPrismDataset.angle_convention || 'unity_zxy'}</strong></div>
                <div><span>Transform</span><strong>{datasetTransformSummary(selectedPrismDataset)}</strong></div>
                <div><span>Motion limits</span><strong>{datasetMotionSummary(selectedPrismDataset)}</strong></div>
              </div>
            {:else}
              <p class="wide subtle">
                FoV training datasets inherit geometry, transform, motion limits, and angle convention from a registered Prism dataset.
              </p>
            {/if}
          {/if}
        </div>
        <button disabled={saving || (datasetFamily === 'fov' && !prismDatasetId)} on:click={saveDatasetForm}>Save dataset</button>
      </div>
    </section>

    <section class="card">
      <h2>Datasets</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Name</th><th>Family</th><th>Source</th><th>Files</th><th>Path</th><th></th></tr></thead>
          <tbody>
            {#each datasets as dataset}
              <tr>
                <td>
                  <div>{dataset.name}</div>
                  {#if dataset.family === 'fov'}
                    <div class="micro muted">{datasetTransformSummary(dataset.prism_dataset_snapshot || dataset)}</div>
                  {/if}
                </td>
                <td>{dataset.family}</td>
                <td>{dataset.family === 'fov' ? prismDatasetName(dataset) : 'Trace only'}</td>
                <td>{dataset.validation?.file_count || 0}</td>
                <td class="mono">{dataset.trace_path}</td>
                <td><button class="small ghost" on:click={() => editDataset(dataset)}>Edit</button> <button class="small danger" on:click={async () => { datasets = (await deleteTrainingDataset(dataset.id)).datasets || []; }}>Delete</button></td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>
    {/if}

    {#if activeTab === 'plans'}
    <section class="training-grid plan-editor-grid">
      <div class="card form-card">
        <div class="section-header">
          <h2>{planId ? 'Edit Training Plan' : 'Create Training Plan'}</h2>
          <button class="ghost small" on:click={resetPlan}>New</button>
        </div>
        <div class="form-grid">
          <label>Family
            <select bind:value={planFamily}>
              <option value="fov">FoV</option>
              <option value="bandwidth">Bandwidth</option>
            </select>
          </label>
          <label>Name <input bind:value={planName} /></label>
          <label>Model
            <select bind:value={modelType}>
              <option value="mlp">MLP</option>
              <option value="gru">GRU</option>
              <option value="lstm">LSTM</option>
            </select>
          </label>
          <label>Target
            <select bind:value={target}>
              {#if planFamily === 'fov'}
                <option value="pose">XYZ + quaternion</option>
                {#if modelRequiresPointCloud}
                  <option value="tile_visibility">Tile visibility</option>
                {/if}
              {:else}
                <option value="future_bandwidth">Future bandwidth</option>
              {/if}
            </select>
          </label>
          <label>Hidden <input type="number" min="4" bind:value={hiddenSize} /></label>
          <label>Layers <input type="number" min="1" bind:value={numLayers} /></label>
          <label>Dropout <input type="number" min="0" max="0.9" step="0.05" bind:value={dropout} /></label>
          <label>Epochs <input type="number" min="1" bind:value={epochs} /></label>
          <label>Batch <input type="number" min="1" bind:value={batchSize} /></label>
          <label>Learning rate <input type="number" min="0.000001" step="0.0001" bind:value={learningRate} /></label>
          <label>Optimizer
            <select bind:value={optimizer}>
              <option value="adam">Adam</option>
            </select>
          </label>
          <label>Scheduler
            <select bind:value={scheduler}>
              <option value="none">None</option>
              <option value="step">Step</option>
              <option value="plateau">Plateau</option>
            </select>
          </label>
          <label>History seconds <input type="number" min="0.001" step="0.1" bind:value={historyDuration} /></label>
          <label>History step <input type="number" min="0.001" step="0.1" bind:value={historyStep} /></label>
          <label>Sample stride <input type="number" min="0" step="0.1" bind:value={sampleStride} /></label>
          <label>Fixed history samples <input type="number" min="0" bind:value={fixedHistorySamples} /></label>
          <label>Horizon <input type="number" min="0" step="0.1" bind:value={horizon} /></label>
          <label>Split
            <select bind:value={splitStrategy}>
              <option value="by_trace">By trace</option>
              <option value="by_dataset">By dataset</option>
              <option value="by_participant">By participant</option>
              <option value="by_scenario">By scenario/content</option>
              <option value="manual">Manual assignments</option>
              <option value="random_sample">Random sample</option>
            </select>
          </label>
          <label>Validation fraction <input type="number" min="0" max="0.9" step="0.05" bind:value={validationFraction} /></label>
          <label>Test fraction <input type="number" min="0" max="0.9" step="0.05" bind:value={testFraction} /></label>
          <label>Mixing
            <select bind:value={mixingStrategy}>
              <option value="proportional">Proportional</option>
              <option value="equal_dataset">Equal dataset</option>
              <option value="manual">Manual weights</option>
            </select>
          </label>
          <label class="wide">Manual weights <input bind:value={manualWeights} placeholder="S1:1, S2:0.5" /></label>
          <label>Seed <input type="number" bind:value={seed} /></label>
          <label>Sample cap <input type="number" min="0" bind:value={maxSamples} /></label>
          <label>Early stop patience <input type="number" min="0" bind:value={earlyStoppingPatience} /></label>
          <label>Checkpoint freq <input type="number" min="1" bind:value={checkpointFrequency} /></label>
          <label>Device <input bind:value={device} /></label>
          <label>Workers <input type="number" min="0" bind:value={numWorkers} /></label>
          <label class="check-row"><input type="checkbox" bind:checked={includeCurrent} /> Include current sample</label>
          <label class="check-row"><input type="checkbox" bind:checked={mixedPrecision} /> Mixed precision</label>
        </div>
        <button disabled={saving} on:click={savePlanForm}>Save plan</button>
      </div>

      <div class="card">
        <div class="section-header">
          <h2>Datasets And Split</h2>
          {#if splitStrategy === 'manual'}
            <label class="inline-field">Default
              <select bind:value={defaultSplitRole}>
                <option value="train">Train</option>
                <option value="validation">Validation</option>
                <option value="test">Test</option>
              </select>
            </label>
          {/if}
        </div>
        <div class="dataset-role-list">
          {#each compatibleDatasets as dataset}
            <div class="dataset-role-row">
              <label class="check-row">
                <input type="checkbox" checked={selectedDatasetIds.includes(dataset.id)} on:change={() => togglePlanDataset(dataset.id)} />
                <span>{dataset.name}</span>
              </label>
              {#if splitStrategy === 'manual' && selectedDatasetIds.includes(dataset.id)}
                <select value={roleForDataset(dataset.id)} on:change={(event) => updateDatasetRole(dataset.id, event)}>
                  <option value="default">Default ({defaultSplitRole})</option>
                  <option value="train">Train</option>
                  <option value="validation">Validation</option>
                  <option value="test">Test</option>
                </select>
              {/if}
            </div>
          {/each}
        </div>
        <div class="split-preview">
          <h3>Preview</h3>
          <p><strong>Train:</strong> {compactNames(currentSplitPreview.train)}</p>
          <p><strong>Validation:</strong> {compactNames(currentSplitPreview.validation)}</p>
          <p><strong>Test:</strong> {compactNames(currentSplitPreview.test)}</p>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Plans</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Plan</th><th>Model</th><th>Target</th><th>Datasets</th><th>Split</th><th>Horizon</th><th>Mixing</th><th></th></tr></thead>
          <tbody>
            {#each plans as plan}
              <tr>
                <td>{plan.name}</td>
                <td>{plan.model?.model_type || 'mlp'}</td>
                <td>{plan.target || '-'}</td>
                <td>{plan.dataset_ids?.length || 0}</td>
                <td>{planSummary(plan)}</td>
                <td>{Number(plan.history?.prediction_horizon_s || 0).toFixed(1)} s</td>
                <td>{plan.mixing_strategy || 'proportional'}</td>
                <td><button class="small" on:click={() => startPlan(plan)}>Start</button> <button class="small ghost" on:click={() => editPlan(plan)}>Edit</button> <button class="small danger" on:click={async () => { plans = (await deleteTrainingPlan(plan.id)).plans || []; }}>Delete</button></td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>
    {/if}

    {#if activeTab === 'runs'}
    <section class="card">
      <div class="section-header">
        <h2>Runner</h2>
        <button class="ghost small" on:click={refreshAll}>Refresh</button>
      </div>
      <h3>Start A Plan</h3>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Plan</th><th>Family</th><th>Split</th><th>Workers</th><th></th></tr></thead>
          <tbody>
            {#each plans as plan}
              <tr>
                <td>{plan.name}</td>
                <td>{plan.family}</td>
                <td>{planSummary(plan)}</td>
                <td>{Number(plan.training?.num_workers || 0)}</td>
                <td><button class="small" on:click={() => startPlan(plan)}>Start</button> <button class="small ghost" on:click={() => { editPlan(plan); activeTab = 'plans'; }}>Edit</button></td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>

      <h3>Runs</h3>
      <div class="runner-grid">
        {#each runs as run}
          <article class="run-card" class:selected={selectedRunId === run.run_id}>
            <button class="run-select" on:click={() => loadMetrics(run)}>
              <strong>{run.run_id}</strong>
              <span>{runPlanTitle(run)}</span>
              <span class="status-pill">{run.status}</span>
            </button>
            <div class="progress-block">
              <div class="progress-label"><span>{progressLabel(run)}</span><span>{Math.round(progressPercent(run, 'epoch', 'epochs'))}% epochs</span></div>
              <div class="progress-track"><span style={`width: ${progressPercent(run, 'epoch', 'epochs')}%`}></span></div>
              {#if run.progress?.phase === 'indexing' || numberValue(run.progress?.index_total) > 0}
                <div class="progress-label"><span>Indexing</span><span>{Math.round(progressPercent(run, 'index_current', 'index_total'))}% files</span></div>
                <div class="progress-track"><span style={`width: ${progressPercent(run, 'index_current', 'index_total')}%`}></span></div>
                {#if indexDetail(run)}
                  <div class="micro muted truncate">{indexDetail(run)}</div>
                {/if}
              {/if}
              <div class="progress-label"><span>Batch</span><span>{Math.round(progressPercent(run, 'batch', 'batches'))}%</span></div>
              <div class="progress-track"><span style={`width: ${progressPercent(run, 'batch', 'batches')}%`}></span></div>
            </div>
            {#if latestLogLine(run)}
              <div class="latest-log">{latestLogLine(run)}</div>
            {/if}
            <div class="run-stats">
              <span>loss {metricText(run.progress?.current_loss)}</span>
              <span>train {metricText(run.progress?.latest_train_loss)}</span>
              <span>val {metricText(run.progress?.latest_validation_mae)}</span>
              <span>ETA {durationText(run.progress?.eta_s)}</span>
              <span>data {fixedText(run.progress?.data_time_s, 2, 's')}</span>
              <span>compute {fixedText(run.progress?.compute_time_s, 2, 's')}</span>
              <span>infer {fixedText(run.progress?.inference_ms_per_batch, 2, ' ms/batch')}</span>
              <span>{fixedText(run.progress?.samples_per_s, 1, ' samples/s')}</span>
              <span>{String(run.progress?.device_name || run.progress?.device || 'device -')}</span>
              <span>cores {run.resources?.estimated_core_usage ?? run.progress?.estimated_core_usage ?? 1}</span>
              <span>workers {run.resources?.num_workers ?? run.progress?.num_workers ?? 0}</span>
            </div>
            <div class="run-actions">
              {#if run.status === 'running'}
                <button class="small ghost" on:click={() => controlTrainingRun(run.run_id, 'pause')}>Pause</button>
                <button class="small danger" on:click={() => controlTrainingRun(run.run_id, 'stop')}>Stop</button>
              {:else if run.status === 'queued' || run.status === 'stopping'}
                <button class="small danger" on:click={() => controlTrainingRun(run.run_id, 'stop')}>Stop</button>
              {/if}
              {#if run.status === 'paused'}
                <button class="small" on:click={() => resumeRun(run)}>Resume</button>
              {/if}
              {#if !['running', 'stopping'].includes(run.status)}
                <button class="small danger" on:click={async () => { await deleteTrainingRun(run.run_id); await refreshAll(); }}>Delete</button>
              {/if}
            </div>
          </article>
        {/each}
      </div>
      {#if selectedRun}
        <div class="run-detail">
          <h3>Selected Run</h3>
          <div class="settings-grid">
            <div><strong>Plan</strong><span>{runPlanTitle(selectedRun)}</span></div>
            <div><strong>Status</strong><span>{selectedRun.status}</span></div>
            <div><strong>Phase</strong><span>{selectedRun.progress?.phase || '-'}</span></div>
            <div><strong>Device</strong><span>{String(selectedRun.progress?.device_name || selectedRun.progress?.device || '-')}</span></div>
            <div><strong>ETA</strong><span>{durationText(selectedRun.progress?.eta_s)}</span></div>
            <div><strong>Parallel eligible</strong><span>{String(selectedRun.resources?.parallel_eligible ?? selectedRun.progress?.parallel_eligible ?? true)}</span></div>
            <div><strong>Split counts</strong><span>train {selectedRun.split_counts?.train ?? '-'} · val {selectedRun.split_counts?.validation ?? '-'} · test {selectedRun.split_counts?.test ?? '-'}</span></div>
            <div><strong>Latest metrics</strong><span>loss {metricText(selectedRun.progress?.current_loss)} · train {metricText(selectedRun.progress?.latest_train_loss)} · val {metricText(selectedRun.progress?.latest_validation_mae)}</span></div>
            <div><strong>Batch timing</strong><span>data {fixedText(selectedRun.progress?.data_time_s, 3, 's')} · compute {fixedText(selectedRun.progress?.compute_time_s, 3, 's')} · avg {fixedText(selectedRun.progress?.avg_batch_time_s, 3, 's')}</span></div>
            <div><strong>Inference</strong><span>{fixedText(selectedRun.progress?.inference_ms_per_batch, 3, ' ms/batch')} · {fixedText(selectedRun.progress?.inference_ms_per_sample, 4, ' ms/sample')}</span></div>
          </div>
          <div class="live-log">
            <div class="section-header compact">
              <h3>Live Log</h3>
              <span class="micro muted">{selectedRun.log?.length || 0} entries</span>
            </div>
            {#if recentLog(selectedRun).length}
              <pre>{recentLog(selectedRun).join('\n')}</pre>
            {:else}
              <p class="muted">No log entries reported yet.</p>
            {/if}
          </div>
        </div>
      {/if}
    </section>
    {/if}

    {#if activeTab === 'models'}
    <section class="card">
        <h2>Models</h2>
        {#each models as model}
          <div class="model-row">
            <div>
              <strong>{modelDisplayName(model)}</strong>
              <div class="muted">{model.id} · {model.family} · {model.exported ? `exported as ${model.exported_as || model.id}` : 'not exported'}</div>
            </div>
            <div class="model-actions">
              <button class="small ghost" on:click={() => evaluateModel(model)}>Evaluate</button>
              <button class="small" on:click={() => exportModel(model)}>Export</button>
              <button class="small danger" on:click={async () => { await deleteTrainingModel(model.id); await refreshAll(); }}>Delete</button>
            </div>
          </div>
        {/each}
    </section>
    {/if}

    {#if activeTab === 'results'}
    <section class="card">
        <h2>Results</h2>
        <div class="run-list">
          {#each runs as run}
            <button class:selected={selectedRunId === run.run_id} on:click={() => loadMetrics(run)}>
              {run.run_id} · {run.status} · {runPlanTitle(run)}
            </button>
          {/each}
        </div>
        {#if metrics?.epochs?.length}
          <div class="table-wrap">
            <table>
              <thead><tr><th>Epoch</th><th>Train loss</th><th>Validation MAE</th><th>Seconds</th></tr></thead>
              <tbody>
                {#each metrics.epochs as epoch}
                  <tr>
                    <td>{epoch.epoch}</td>
                    <td>{Number(epoch.train_loss || 0).toPrecision(5)}</td>
                    <td>{epoch.validation_mae == null ? '-' : Number(epoch.validation_mae).toPrecision(5)}</td>
                    <td>{Number(epoch.elapsed_s || 0).toFixed(1)}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
          <p class="muted">Test MAE: {metrics.test?.mae == null ? '-' : Number(metrics.test.mae).toPrecision(5)}</p>
          {#each graphs as graph}
            <h3>{graph.title}</h3>
            <ChartCanvas config={chartConfig(graph)} height={260} />
          {/each}
        {:else}
          <p class="muted">Select a run with saved metrics.</p>
        {/if}
    </section>
    {/if}
  {/if}
</main>

<style>
  .training-topbar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    width: 100%;
    justify-content: flex-start;
  }
  .exit-link {
    margin-left: auto;
  }
  .training-home-panel {
    border: 1px solid var(--card-border);
    border-radius: 14px;
    background: var(--card-bg);
    padding: 1.15rem;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.24);
  }
  .workflow-head {
    padding-bottom: 0.9rem;
    border-bottom: 1px solid var(--card-border);
  }
  .workflow-head h2 {
    margin: 0.1rem 0 0;
  }
  .workflow-list {
    display: grid;
    gap: 0.85rem;
    margin-top: 1rem;
  }
  .workflow-step {
    display: grid;
    grid-template-columns: 44px minmax(0, 1fr) auto;
    gap: 0.9rem;
    align-items: center;
    padding: 0.95rem;
    border: 1px solid var(--card-border);
    border-radius: 10px;
    background: var(--soft-bg);
  }
  .step-number {
    display: grid;
    place-items: center;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--btn-main);
    color: var(--btn-text);
    font-weight: 800;
  }
  .step-copy h3,
  .step-copy p {
    margin: 0;
  }
  .step-copy p {
    color: var(--text-muted);
    margin-top: 0.2rem;
  }
  .settings-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
  }
  .settings-grid div {
    padding: 12px;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    background: var(--soft-bg);
  }
  .settings-grid span {
    display: block;
    margin-top: 6px;
    color: var(--text-muted);
  }
  .notice {
    margin: 1rem 0;
    padding: 0.85rem;
    border: 1px solid var(--chip-border);
    border-radius: 8px;
    background: var(--chip-bg);
    color: var(--text-main);
  }
  .trace-column-editor {
    display: grid;
    gap: 1rem;
    margin-top: 1rem;
  }
  .trace-column-group h3 {
    margin: 0 0 0.65rem;
    font-size: 0.95rem;
  }
  .trace-column-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.75rem;
  }
  .trace-column-grid label {
    min-width: 0;
  }
  .trace-column-grid small {
    display: block;
    margin: 0.25rem 0 0.35rem;
    color: var(--text-muted);
  }
  .preferred-quat {
    border-color: var(--chip-border);
  }
  .saved-path {
    margin-top: 0.6rem;
  }
  .training-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 16px;
    margin-bottom: 16px;
  }
  .form-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
  }
  .wide {
    grid-column: 1 / -1;
  }
  .inherited-dataset-card {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.55rem;
    padding: 0.65rem;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    background: var(--soft-bg-alt);
  }
  .inherited-dataset-card > div {
    display: grid;
    gap: 0.16rem;
    min-width: 0;
  }
  .inherited-dataset-card span {
    color: var(--text-muted);
    font-size: 0.78rem;
  }
  .inherited-dataset-card code,
  .inherited-dataset-card strong {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .run-list {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0;
  }
  .plan-editor-grid {
    align-items: start;
  }
  .inline-field {
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }
  .dataset-role-list {
    display: grid;
    gap: 8px;
    margin-top: 10px;
  }
  .dataset-role-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 160px;
    gap: 8px;
    align-items: center;
  }
  .split-preview {
    margin-top: 14px;
    padding: 12px;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    background: var(--soft-bg);
  }
  .split-preview h3,
  .split-preview p {
    margin: 0;
  }
  .split-preview p {
    margin-top: 6px;
    color: var(--text-muted);
  }
  .runner-grid {
    display: grid;
    gap: 12px;
    margin-top: 12px;
  }
  .run-card {
    display: grid;
    gap: 10px;
    padding: 12px;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    background: var(--soft-bg);
  }
  .run-select {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) auto;
    gap: 10px;
    align-items: center;
    width: 100%;
    text-align: left;
    background: transparent;
    color: inherit;
    border: 0;
    padding: 0;
  }
  .status-pill {
    justify-self: end;
    padding: 3px 8px;
    border: 1px solid var(--chip-border);
    border-radius: 999px;
    background: var(--chip-bg);
    font-size: 0.78rem;
  }
  .progress-block {
    display: grid;
    gap: 5px;
  }
  .progress-label,
  .run-stats,
  .run-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    justify-content: space-between;
    color: var(--text-muted);
    font-size: 0.82rem;
  }
  .progress-track {
    height: 8px;
    overflow: hidden;
    border-radius: 999px;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
  }
  .progress-track span {
    display: block;
    height: 100%;
    background: var(--accent-main);
  }
  .latest-log {
    overflow: hidden;
    padding: 8px 10px;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    background: var(--card-bg);
    color: var(--text-muted);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.76rem;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .run-detail {
    margin-top: 16px;
  }
  .section-header.compact {
    margin-top: 14px;
    margin-bottom: 6px;
  }
  .live-log pre {
    max-height: 360px;
    overflow: auto;
    margin: 0;
    padding: 12px;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    background: var(--soft-bg);
    color: var(--text-main);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.78rem;
    line-height: 1.45;
    white-space: pre-wrap;
  }
  .check-row {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    border: 1px solid var(--card-border);
    border-radius: 8px;
  }
  .model-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid var(--card-border);
  }
  .model-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: flex-end;
  }
  .mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.85rem;
  }
  .muted {
    color: var(--text-muted);
  }
  .micro {
    font-size: 0.76rem;
  }
  .truncate {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .selected {
    outline: 2px solid var(--accent-main);
  }
  @media (max-width: 900px) {
    .training-grid,
    .form-grid,
    .dataset-role-row,
    .run-select {
      grid-template-columns: 1fr;
    }
  }
</style>
