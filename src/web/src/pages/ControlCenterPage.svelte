<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import type { ChartConfiguration } from 'chart.js';
  import RunLaunchForm from '../features/control-center/components/RunLaunchForm.svelte';
  import RunsExplorer from '../features/control-center/components/RunsExplorer.svelte';
  import RunInsightsPanel from '../features/control-center/components/RunInsightsPanel.svelte';
  import { toChartConfig } from '../features/control-center/chartConfig';
  import {
    basenamePath,
    buildRunRows,
    filterAndSortRunRows,
    formatDateTime,
    formatRuntime,
    statusDotClass,
    type RunRow,
    type RunRuntimeFilter,
    type RunStatusFilter,
  } from '../features/control-center/runExplorerUtils';
  import { invertVisibleRowsSelection, selectRangeRows, selectVisibleRows } from '../features/control-center/runSelection';
  import { deleteRun, getDiagnostics, getOptions, getRunConfig, getRunEvents, getTrace, listRuns, startRun, subscribeRuns } from '../core/api';
  import { loadQoeState, normalizeQoeModels, sanitizeQoeParams, saveQoeState, type QoEState } from '../core/qoe';
  import { loadRunAliases, saveRunAliases, saveSelectedCompareRuns } from '../core/storage';
  import type { AppOptions, EventLog, QoEModel, RunConfig, RunSummary } from '../core/types';
  import { aggregateBounds, buildRunLayers, computeBounds, type LayerVisibility } from '../features/control-center/runChart';

  const FORM_CACHE_KEY = 'prism_form_defaults_v2';
  const QOE_STORAGE_KEY = 'control_qoe_state_v4';

  let options: AppOptions | null = null;
  let runs: RunSummary[] = [];
  let statusMessage = '';
  let loading = true;
  let submitting = false;
  let error = '';

  let selectedRunIds = new Set<string>();
  let runAliases = loadRunAliases();

  let selectedRunId = '';
  let selectedRunConfig: RunConfig | null = null;
  let selectedRunEvents: EventLog | null = null;

  let qoeModels: QoEModel[] = [];
  let qoeState: QoEState = { selectedId: '', paramsById: {} };
  let showQoeControls = false;

  let showBandwidth = true;
  let showBitrate = true;
  let showBuffer = true;
  let showQoe = true;
  let tailMode: 'drain' | 'playback' = 'drain';

  let chartConfig: ChartConfiguration | null = null;
  let compareMode = false;
  let comparing = false;
  let compareCards: Array<{ runId: string; label: string; chart: ChartConfiguration | null; error?: string }> = [];

  let form: RunConfig = {
    v3c_folder: '',
    content_dir: '',
    network_trace: '',
    fov_trace: '',
    abr: 'BandwidthAwareABR',
    predictor: 'None',
    rtt: 0.06,
    buffer_window_segments: 10,
    enable_mid_download_cancel: false,
  };

  let unsubscribeRuns: (() => void) | null = null;
  const traceCache = new Map<string, Array<{ time_s: number; bandwidth_mbps: number }>>();
  let expandedRunIds = new Set<string>();
  let loadingDetailIds = new Set<string>();
  let runConfigCache: Record<string, RunConfig | null> = {};
  let runSearch = '';
  let statusFilter: RunStatusFilter = 'all';
  let runtimeFilter: RunRuntimeFilter = 'all';
  let lastSelectedRowIndex: number | null = null;

  let runRows: RunRow[] = [];
  let filteredRunRows: RunRow[] = [];

  function defaultAbr(): string {
    return options?.abr_classes.includes('BandwidthAwareABR') ? 'BandwidthAwareABR' : options?.abr_classes[0] || 'BandwidthAwareABR';
  }

  function defaultPredictor(): string {
    return 'None';
  }

  function saveFormToCache(): void {
    localStorage.setItem(FORM_CACHE_KEY, JSON.stringify(form));
    // Backward compatibility with legacy cache key.
    localStorage.setItem('prism_form_defaults', JSON.stringify(form));
  }

  function loadFormFromCache(): void {
    try {
      const raw = localStorage.getItem(FORM_CACHE_KEY) || localStorage.getItem('prism_form_defaults') || '{}';
      const parsed = JSON.parse(raw);
      form = { ...form, ...parsed };
      statusMessage = 'Loaded cached form params.';
    } catch {
      statusMessage = 'No valid form cache found.';
    }
  }

  function clearFormCache(): void {
    localStorage.removeItem(FORM_CACHE_KEY);
    localStorage.removeItem('prism_form_defaults');
    statusMessage = 'Cleared cached form params.';
  }

  function activeQoeModel(): QoEModel | null {
    return qoeModels.find((m) => m.id === qoeState.selectedId) || qoeModels[0] || null;
  }

  function activeQoeParams(): Record<string, number> {
    const model = activeQoeModel();
    if (!model) return {};
    const params = sanitizeQoeParams(model, qoeState.paramsById[model.id] || {});
    qoeState.paramsById[model.id] = params;
    return params;
  }

  function ensureRunConfigLoaded(runId: string): void {
    if (runId in runConfigCache || loadingDetailIds.has(runId)) return;
    loadingDetailIds.add(runId);
    loadingDetailIds = new Set(loadingDetailIds);
    getRunConfig(runId)
      .then((cfg) => {
        runConfigCache[runId] = cfg;
      })
      .catch(() => {
        runConfigCache[runId] = null;
      })
      .finally(() => {
        loadingDetailIds.delete(runId);
        runConfigCache = { ...runConfigCache };
        loadingDetailIds = new Set(loadingDetailIds);
      });
  }

  function toggleRunDetails(runId: string): void {
    if (expandedRunIds.has(runId)) {
      expandedRunIds.delete(runId);
    } else {
      expandedRunIds.add(runId);
      ensureRunConfigLoaded(runId);
    }
    expandedRunIds = new Set(expandedRunIds);
  }

  function clearSelection(): void {
    selectedRunIds = new Set();
  }

  function selectVisibleRuns(): void {
    selectedRunIds = selectVisibleRows(filteredRunRows);
  }

  function invertVisibleSelection(): void {
    selectedRunIds = invertVisibleRowsSelection(selectedRunIds, filteredRunRows);
  }

  function toggleSelectionByRow(rowId: string, idx: number, event: MouseEvent | KeyboardEvent): void {
    if (event.shiftKey && lastSelectedRowIndex !== null) {
      selectedRunIds = selectRangeRows(filteredRunRows, lastSelectedRowIndex, idx, selectedRunIds, event.metaKey || event.ctrlKey);
    } else if (event.metaKey || event.ctrlKey) {
      const next = new Set(selectedRunIds);
      if (next.has(rowId)) next.delete(rowId);
      else next.add(rowId);
      selectedRunIds = next;
    } else {
      selectedRunIds = new Set([rowId]);
    }
    lastSelectedRowIndex = idx;
  }

  function onRowSelectClick(rowId: string, idx: number, event: MouseEvent): void {
    toggleSelectionByRow(rowId, idx, event);
  }

  function onRowSelectKeydown(rowId: string, idx: number, event: KeyboardEvent): void {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    toggleSelectionByRow(rowId, idx, event);
  }

  function onRowCheckboxClick(rowId: string, idx: number, event: MouseEvent): void {
    event.stopPropagation();
    if (event.shiftKey && lastSelectedRowIndex !== null) {
      selectedRunIds = selectRangeRows(filteredRunRows, lastSelectedRowIndex, idx, selectedRunIds, true);
      lastSelectedRowIndex = idx;
      return;
    }
    const next = new Set(selectedRunIds);
    if (next.has(rowId)) next.delete(rowId);
    else next.add(rowId);
    selectedRunIds = next;
    lastSelectedRowIndex = idx;
  }

  async function refreshRuns(): Promise<void> {
    runs = await listRuns();
  }

  async function submitRun(): Promise<void> {
    if (submitting) return;
    submitting = true;
    error = '';
    try {
      const payload: RunConfig = {
        ...form,
        abr: form.abr || defaultAbr(),
        predictor: form.predictor || 'None',
      };
      const response = await startRun(payload);
      statusMessage = `Run queued: ${response.run_id}`;
      saveFormToCache();
      await refreshRuns();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      submitting = false;
    }
  }

  async function removeRun(runId: string): Promise<void> {
    await deleteRun(runId);
    selectedRunIds.delete(runId);
    if (selectedRunId === runId) {
      selectedRunId = '';
      selectedRunConfig = null;
      selectedRunEvents = null;
      chartConfig = null;
    }
    await refreshRuns();
  }

  async function inspectRun(runId: string): Promise<void> {
    compareMode = false;
    compareCards = [];
    selectedRunId = runId;
    error = '';
    try {
      const [cfg, events] = await Promise.all([getRunConfig(runId), getRunEvents(runId)]);
      selectedRunConfig = cfg;
      selectedRunEvents = events;
      await rebuildRunChart();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      selectedRunConfig = null;
      selectedRunEvents = null;
      chartConfig = null;
    }
  }

  function runLabel(runId: string): string {
    const alias = runAliases[runId];
    return alias ? `${alias} (${runId})` : runId;
  }

  function layerVisibility(): LayerVisibility {
    return {
      bandwidth: showBandwidth,
      bitrate: showBitrate,
      buffer: showBuffer,
      qoe: showQoe,
    };
  }

  async function getQoeByRun(runIds: string[]): Promise<Record<string, number[]>> {
    if (!showQoe || !runIds.length) return {};
    const model = activeQoeModel();
    if (!model) return {};
    const diag = await getDiagnostics({ run_ids: runIds, qoe_function: model.id, qoe_params: activeQoeParams() });
    return diag.qoe || {};
  }

  async function fetchTraceSamples(path: string | undefined): Promise<Array<{ time_s: number; bandwidth_mbps: number }>> {
    if (!path) return [];
    if (traceCache.has(path)) return traceCache.get(path)!;
    try {
      const trace = await getTrace(path);
      const samples = trace.samples || [];
      traceCache.set(path, samples);
      return samples;
    } catch {
      traceCache.set(path, []);
      return [];
    }
  }


  async function buildChartForRun(
    runId: string,
    cfg: RunConfig,
    events: EventLog,
    qoeByRun: Record<string, number[]>,
  ): Promise<ChartConfiguration> {
    const traceSamples = await fetchTraceSamples(cfg.network_trace);
    const qoeValues = qoeByRun[runId] || [];
    const built = buildRunLayers(events, cfg, traceSamples, qoeValues, tailMode);
    const visibility = layerVisibility();
    const bounds = computeBounds(built.layers, visibility);
    return toChartConfig(built.layers, built.stalls, bounds, visibility);
  }

  async function rebuildRunChart(): Promise<void> {
    if (!selectedRunId || !selectedRunEvents || !selectedRunConfig) {
      chartConfig = null;
      return;
    }
    const qoeByRun = await getQoeByRun([selectedRunId]);
    chartConfig = await buildChartForRun(selectedRunId, selectedRunConfig, selectedRunEvents, qoeByRun);
  }

  function compareSelected(): void {
    const ids = Array.from(selectedRunIds);
    if (!ids.length) return;
    saveSelectedCompareRuns(ids);
    const query = encodeURIComponent(ids.join(','));
    window.location.href = `/static/compare.html?runs=${query}`;
  }

  async function compareSelectedInControl(): Promise<void> {
    const ids = Array.from(selectedRunIds);
    if (!ids.length) return;

    comparing = true;
    compareMode = true;
    selectedRunId = '';
    chartConfig = null;
    compareCards = [];
    error = '';

    try {
      const qoeByRun = await getQoeByRun(ids);
      const loaded = await Promise.all(
        ids.map(async (runId) => {
          try {
            const [cfg, events] = await Promise.all([getRunConfig(runId), getRunEvents(runId)]);
            const traceSamples = await fetchTraceSamples(cfg.network_trace);
            const built = buildRunLayers(events, cfg, traceSamples, qoeByRun[runId] || [], tailMode);
            return { runId, label: runLabel(runId), built };
          } catch (err) {
            return {
              runId,
              label: runLabel(runId),
              built: null,
              error: err instanceof Error ? err.message : String(err),
            };
          }
        }),
      );
      const bounds = aggregateBounds(
        loaded.filter((entry) => entry.built).map((entry) => computeBounds(entry.built!.layers, layerVisibility())),
      );
      const visibility = layerVisibility();
      compareCards = loaded.map((entry) => ({
        runId: entry.runId,
        label: entry.label,
        chart: entry.built ? toChartConfig(entry.built.layers, entry.built.stalls, bounds, visibility, entry.label) : null,
        error: entry.error,
      }));
    } finally {
      comparing = false;
    }
  }

  function exitCompareMode(): void {
    compareMode = false;
    compareCards = [];
    comparing = false;
  }

  async function deleteSelected(): Promise<void> {
    const ids = Array.from(selectedRunIds);
    for (const runId of ids) {
      // eslint-disable-next-line no-await-in-loop
      await deleteRun(runId);
    }
    selectedRunIds = new Set();
    await refreshRuns();
  }

  async function selectAllRuns(): Promise<void> {
    const visibleIds = filteredRunRows.map((row) => row.run.run_id);
    if (visibleIds.length && visibleIds.every((id) => selectedRunIds.has(id))) {
      selectedRunIds = new Set();
    } else {
      selectedRunIds = new Set(visibleIds);
    }
  }

  function updateAlias(runId: string, alias: string): void {
    const trimmed = alias.trim();
    if (trimmed) runAliases[runId] = trimmed;
    else delete runAliases[runId];
    runAliases = { ...runAliases };
    saveRunAliases(runAliases);
  }

  function handleRunFormSubmit(): void {
    submitRun();
  }

  function handleAliasInputChange(runId: string, event: Event): void {
    const target = event.currentTarget as HTMLInputElement | null;
    updateAlias(runId, target?.value || '');
  }

  function onQoeModelChange(modelId: string): void {
    qoeState.selectedId = modelId;
    const model = activeQoeModel();
    if (model && !qoeState.paramsById[model.id]) {
      qoeState.paramsById[model.id] = sanitizeQoeParams(model, {});
    }
    saveQoeState(QOE_STORAGE_KEY, qoeState);
  }

  function handleQoeModelSelectChange(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement | null;
    onQoeModelChange(target?.value || '');
  }

  function onQoeFieldChange(field: string, value: string): void {
    const model = activeQoeModel();
    if (!model) return;
    const current = { ...(qoeState.paramsById[model.id] || {}) };
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      current[field] = numeric;
      qoeState.paramsById[model.id] = sanitizeQoeParams(model, current);
      saveQoeState(QOE_STORAGE_KEY, qoeState);
    }
  }

  function handleQoeFieldInputChange(field: string, event: Event): void {
    const target = event.currentTarget as HTMLInputElement | null;
    onQoeFieldChange(field, target?.value || '');
  }

  function toggleQoeControlsPanel(): void {
    showQoeControls = !showQoeControls;
  }

  function toggleTailMode(): void {
    tailMode = tailMode === 'drain' ? 'playback' : 'drain';
    refreshCurrentViewCharts();
  }

  async function applyQoeToChart(): Promise<void> {
    await refreshCurrentViewCharts();
  }

  async function resetQoeForChart(): Promise<void> {
    const model = activeQoeModel();
    if (!model) return;
    qoeState.paramsById[model.id] = sanitizeQoeParams(model, {});
    saveQoeState(QOE_STORAGE_KEY, qoeState);
    await refreshCurrentViewCharts();
  }

  async function refreshCurrentViewCharts(): Promise<void> {
    if (compareMode) {
      await compareSelectedInControl();
      return;
    }
    if (selectedRunId) {
      await rebuildRunChart();
    }
  }

  $: runRows = buildRunRows(runs, runAliases);

  $: {
    // Keep explicit reactive references so Svelte recomputes when filters/cache change.
    runSearch;
    statusFilter;
    runtimeFilter;
    runConfigCache;
    filteredRunRows = filterAndSortRunRows(runRows, runSearch, statusFilter, runtimeFilter, runConfigCache);
  }

  $: {
    const validIds = new Set(runs.map((run) => run.run_id));
    const nextSelected = new Set(Array.from(selectedRunIds).filter((id) => validIds.has(id)));
    if (nextSelected.size !== selectedRunIds.size) selectedRunIds = nextSelected;
    const nextExpanded = new Set(Array.from(expandedRunIds).filter((id) => validIds.has(id)));
    if (nextExpanded.size !== expandedRunIds.size) expandedRunIds = nextExpanded;
  }

  onMount(async () => {
    try {
      options = await getOptions();
      qoeModels = normalizeQoeModels(options.qoe_functions);
      qoeState = loadQoeState(QOE_STORAGE_KEY, qoeModels);
      form.abr = form.abr || defaultAbr();
      form.predictor = form.predictor || defaultPredictor();
      await refreshRuns();
      unsubscribeRuns = subscribeRuns((next) => {
        runs = next;
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
  <div>
    <p class="eyebrow">Prism VPCC Lab</p>
    <h1>Control Center</h1>
    <p class="lede">Run individual experiments and inspect per-run behavior.</p>
  </div>
  <div class="hero-actions">
    <a class="button-link ghost" href="/">Main menu</a>
    <a class="button-link primary" href="/static/aggregate.html">Plan batch runs</a>
  </div>
</header>

<main>
  <RunLaunchForm
    {options}
    {form}
    {statusMessage}
    {submitting}
    {loading}
    onSubmit={handleRunFormSubmit}
    onLoadCache={loadFormFromCache}
    onClearCache={clearFormCache}
  />

  <RunsExplorer
    {runs}
    {filteredRunRows}
    {selectedRunIds}
    {selectedRunId}
    {expandedRunIds}
    {loadingDetailIds}
    {runConfigCache}
    {runAliases}
    bind:runSearch
    bind:statusFilter
    bind:runtimeFilter
    onSelectAllRuns={selectAllRuns}
    onSelectVisibleRuns={selectVisibleRuns}
    onInvertVisibleSelection={invertVisibleSelection}
    onClearSelection={clearSelection}
    onCompareSelectedInControl={compareSelectedInControl}
    onAdvancedCompare={compareSelected}
    onDeleteSelected={deleteSelected}
    onRowCheckboxClick={onRowCheckboxClick}
    onRowSelectClick={onRowSelectClick}
    onRowSelectKeydown={onRowSelectKeydown}
    onInspectRun={inspectRun}
    onToggleRunDetails={toggleRunDetails}
    onRemoveRun={removeRun}
    onAliasChange={handleAliasInputChange}
    {formatRuntime}
    {formatDateTime}
    {statusDotClass}
    {basenamePath}
  />

  <RunInsightsPanel
    bind:showBandwidth
    bind:showBitrate
    bind:showBuffer
    bind:showQoe
    bind:tailMode
    bind:showQoeControls
    {compareMode}
    {comparing}
    {compareCards}
    {selectedRunId}
    {error}
    {qoeModels}
    {qoeState}
    activeQoeModel={activeQoeModel()}
    {chartConfig}
    onRefreshCharts={refreshCurrentViewCharts}
    onToggleTailMode={toggleTailMode}
    onExitCompareMode={exitCompareMode}
    onToggleQoeControlsPanel={toggleQoeControlsPanel}
    onQoeModelSelectChange={handleQoeModelSelectChange}
    onQoeFieldInputChange={handleQoeFieldInputChange}
    onApplyQoeToChart={applyQoeToChart}
    onResetQoeForChart={resetQoeForChart}
  />
</main>
