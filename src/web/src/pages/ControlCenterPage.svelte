<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import type { ChartConfiguration } from 'chart.js';
  import { gotoRoute } from '../core/router';
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
    type RunRowSort,
    type RunStatusFilter,
  } from '../features/control-center/runExplorerUtils';
  import { selectRangeRows } from '../features/control-center/runSelection';
  import {
    assignRunsToGroup,
    controlRunGroup,
    deleteRunFormSession,
    deleteRun,
    deleteRunGroup,
    deleteRunGroupWithRuns,
    exportRunToNavis,
    generateRunFovComparison,
    getDiagnostics,
    getNavisStatus,
    getOptions,
    getRunFormSessions,
    getRunConfig,
    getRunEvents,
    getRunGroupSummaries,
    getTrace,
    listRunGroups,
    listRunsPage,
    preflightRuns,
    removeRunsFromGroups,
    saveRunGroup,
    saveRunFormSession,
    startRun,
    subscribeRuns,
  } from '../core/api';
  import { loadLaunchPreflightMode } from '../core/launchPreflight';
  import { GraphCache, scheduleGraphWork, stableStringify, timed } from '../core/graphCache';
  import { loadQoeState, normalizeQoeModels, sanitizeQoeParams, saveQoeState, type QoEState } from '../core/qoe';
  import { loadRunAliases, saveRunAliases, saveSelectedCompareRuns } from '../core/storage';
  import type { AppOptions, EventLog, FovComparisonResult, NavisStatus, QoEModel, RunConfig, RunFormSession, RunGroup, RunSummary } from '../core/types';
  import { aggregateBounds, buildRunLayers, computeBounds, type LayerVisibility } from '../features/control-center/runChart';

  const FORM_CACHE_KEY = 'prism_form_defaults_v2';
  const QOE_STORAGE_KEY = 'control_qoe_state_v4';

  let options: AppOptions | null = null;
  let runs: RunSummary[] = [];
  let indexedRunTotal = 0;
  let matchingRunTotal = 0;
  let loadingMoreRuns = false;
  let globalRunOffset = 0;
  let appliedRunSearch = '';
  let appliedRunStatus: RunStatusFilter = 'all';
  let runFilterTimer: ReturnType<typeof setTimeout> | null = null;
  let runIndexReady = false;
  let loadedRunGroupIds = new Set<string>();
  let loadingRunGroupIds = new Set<string>();
  let runGroupSummaries: Record<string, import('../core/api').RunGroupSummary> = {};
  const RUN_PAGE_SIZE = 200;
  let statusMessage = '';
  let loading = true;
  let submitting = false;
  let error = '';
  let navisStatus: NavisStatus | null = null;
  let navisExportingRunIds = new Set<string>();
  let fovComparisonGeneratingRunIds = new Set<string>();
  let fovComparisonPreview: { runId: string; result: FovComparisonResult } | null = null;
  let fovComparisonData: any | null = null;
  let fovComparisonDataPath = '';
  let fovComparisonDataLoading = false;
  let fovComparisonDataError = '';
  let fovPreviewMode: 'video' | 'debug' = 'video';
  let fovDebugFrameIndex = 0;
  let fovDebugSelectedTileId = '';
  let fovDebugSelectedStartKey = '';
  let fovDebugTileColorMode: 'prediction' | 'truth' | 'combined' | 'error' = 'prediction';
  let fovOverlayTrace = true;
  let fovOverlayHorizons = true;
  let fovOverlayReachable = true;
  let fovOverlayPointSlice = false;
  let fovOverlayVoxelStarts = true;
  let fovOverlayDijkstraPaths = false;
  let fovOverlayRays = true;
  let fovOverlayHits = true;
  let navisStatusTimer: ReturnType<typeof setInterval> | null = null;

  let selectedRunIds = new Set<string>();
  let runAliases = loadRunAliases();
  let runGroups: RunGroup[] = [];
  let collapsedRunGroupIds = new Set<string>();
  let initializedRunGroupCollapse = false;

  let selectedRunId = '';
  let selectedRunConfig: RunConfig | null = null;
  let selectedRunEvents: EventLog | null = null;

  let qoeModels: QoEModel[] = [];
  let qoeState: QoEState = { selectedId: '', paramsById: {} };
  let showQoeControls = false;

  let showBandwidth = true;

  $: fovDebugFrames = Array.isArray(fovComparisonData?.frames) ? fovComparisonData.frames : [];
  $: fovDebugFrame = fovDebugFrames[fovDebugFrameIndex] || null;
  $: fovDebugTileIds = fovDebugFrame?.tile_ids || [];
  $: if (fovDebugFrame && !fovDebugSelectedTileId && fovDebugTileIds.length) {
    fovDebugSelectedTileId = String(fovDebugTileIds[0]);
  }
  $: if (fovDebugFrame && fovDebugSelectedTileId && !fovDebugTileIds.includes(fovDebugSelectedTileId)) {
    fovDebugSelectedTileId = String(fovDebugTileIds[0] || '');
  }
  $: fovDebugPayload = fovDebugFrame?.prediction_debug || null;
  $: fovDebugSelectedTile = fovDebugPayload?.tiles?.[fovDebugSelectedTileId] || null;
  $: fovDebugContributors = fovDebugSelectedTile?.contributors || [];
  $: fovDebugDistributionStats = fovDebugPayload?.distribution_stats || {};
  $: fovDebugSelectedTileStats = fovDebugSelectedTile?.stats || {};
  $: fovDebugPointSlice = fovComparisonData?.point_cloud_slice || null;
  $: fovDebugBounds = computeFovDebugBounds(fovDebugFrame, fovDebugPayload, fovDebugPointSlice);
  $: fovDebugScaleReference = buildFovDebugScaleReference(fovDebugFrame, fovDebugPayload);
  $: fovDebugStartVoxels = fovDebugPayload?.selected_voxels || [];
  $: fovDebugPathStart = fovDebugStartVoxels.find((voxel: any) => startVoxelKey(voxel) === fovDebugSelectedStartKey) || null;
  $: if (fovOverlayDijkstraPaths && fovDebugStartVoxels.length) {
    const selectedStillValid = fovDebugStartVoxels.some((voxel: any) => startVoxelKey(voxel) === fovDebugSelectedStartKey && (voxel?.path || []).length);
    if (!selectedStillValid) {
      const firstWithPath = fovDebugStartVoxels.find((voxel: any) => (voxel?.path || []).length);
      fovDebugSelectedStartKey = firstWithPath ? startVoxelKey(firstWithPath) : '';
    }
  }
  $: fovDebugViewBox = fovDebugBounds
    ? `${fovDebugBounds.minX} ${fovDebugBounds.minZ} ${fovDebugBounds.width} ${fovDebugBounds.height}`
    : '0 0 1 1';
  let showBitrate = true;
  let showBuffer = true;
  let showQoe = true;
  let tailMode: 'drain' | 'playback' = 'drain';

  let chartConfig: ChartConfiguration | null = null;
  let chartLoading = false;
  let chartBuildSerial = 0;
  let compareMode = false;
  let comparing = false;
  let compareLayout: 'horizontal' | 'vertical' = 'horizontal';
  let compareCollapseQoe = false;
  let compareCards: Array<{ runId: string; label: string; chart: ChartConfiguration | null; error?: string }> = [];
  type OracleCounterpart = {
    runId: string;
    type: 'fov' | 'bandwidth' | 'both';
    label: string;
    reason: string;
  };
  let oracleCounterparts: OracleCounterpart[] = [];
  let oracleCounterpartsLoading = false;
  let selectedOracleCounterpartRunId = '';

  let form: RunConfig = {
    v3c_folder: '',
    content_dir: '',
    network_trace: '',
    network_profile: null,
    fov_trace: '',
    abr: 'BandwidthAwareABR',
    predictor: 'None',
    bandwidth_predictor: 'None',
    rtt: 0.06,
    buffer_window_segments: 2,
    segment_limit: 0,
    enable_mid_download_cancel: false,
  };
  let savedSessions: RunFormSession[] = [];

  let unsubscribeRuns: (() => void) | null = null;
  const traceCache = new Map<string, Array<{ time_s: number; bandwidth_mbps: number }>>();
  const qoeSeriesCache = new GraphCache<Record<string, Array<{ time_s: number; score: number }>>>('single-run-qoe', 24);
  const chartConfigCache = new GraphCache<ChartConfiguration>('single-run-chart', 48);
  let expandedRunIds = new Set<string>();
  let loadingDetailIds = new Set<string>();
  let runConfigCache: Record<string, RunConfig | null> = {};
  let runSearch = '';
  let statusFilter: RunStatusFilter = 'all';
  let rowSort: RunRowSort = 'newest';
  let groupView: 'all' | 'batches' | 'ungrouped' = 'all';
  let groupSort: 'newest' | 'oldest' | 'name' | 'size' = 'newest';
  let lastSelectedRowIndex: number | null = null;

  let runRows: RunRow[] = [];
  let filteredRunRows: RunRow[] = [];
  const RUN_ALIAS_MAX_LENGTH = 32;

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

  async function refreshRunFormSessions(): Promise<void> {
    try {
      const payload = await getRunFormSessions();
      savedSessions = payload.sessions || [];
    } catch {
      savedSessions = [];
    }
  }

  async function saveCurrentFormSession(): Promise<void> {
    const name = window.prompt('Session name:');
    if (name === null) return;
    const trimmed = name.trim();
    if (!trimmed) {
      statusMessage = 'Session name is required.';
      return;
    }
    try {
      const payload = await saveRunFormSession(trimmed, form);
      savedSessions = payload.sessions || [];
      statusMessage = `Saved session "${trimmed}".`;
    } catch (err) {
      statusMessage = err instanceof Error ? err.message : String(err);
    }
  }

  function loadNamedFormSession(name: string): void {
    const found = savedSessions.find((session) => session.name === name);
    if (!found) {
      statusMessage = `Session "${name}" not found.`;
      return;
    }
    form = { ...form, ...found.config };
    statusMessage = `Loaded session "${name}".`;
  }

  async function deleteNamedFormSession(name: string): Promise<void> {
    if (!name) return;
    if (!window.confirm(`Delete saved session "${name}"?`)) return;
    try {
      const payload = await deleteRunFormSession(name);
      savedSessions = payload.sessions || [];
      statusMessage = `Deleted session "${name}".`;
    } catch (err) {
      statusMessage = err instanceof Error ? err.message : String(err);
    }
  }

  function activeQoeModel(): QoEModel | null {
    return qoeModels.find((m) => m.id === qoeState.selectedId) || qoeModels[0] || null;
  }

  function ensureValidQoeSelection(): void {
    if (!qoeModels.length) return;
    const hasSelected = qoeModels.some((m) => m.id === qoeState.selectedId);
    if (hasSelected) return;
    qoeState = {
      ...qoeState,
      selectedId: qoeModels[0]?.id || '',
      paramsById: { ...(qoeState.paramsById || {}) },
    };
  }

  function activeQoeParams(): Record<string, number> {
    const model = activeQoeModel();
    if (!model) return {};
    const params = sanitizeQoeParams(model, qoeState.paramsById[model.id] || {});
    qoeState = {
      ...qoeState,
      paramsById: {
        ...(qoeState.paramsById || {}),
        [model.id]: params,
      },
    };
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
    const [page, summaries] = await Promise.all([
      listRunsPage(0, RUN_PAGE_SIZE, { search: appliedRunSearch, status: appliedRunStatus, ungrouped: true }),
      getRunGroupSummaries({ search: appliedRunSearch, status: appliedRunStatus }),
    ]);
    runs = page.items;
    indexedRunTotal = page.total;
    runGroupSummaries = summaries.groups || {};
    // The paginated list intentionally contains only ungrouped runs. Add the
    // matching batch-summary counts so the UI still reports the real number
    // of runs matching the current search/status filters.
    matchingRunTotal = page.total + Object.values(runGroupSummaries)
      .reduce((total, summary) => total + Number(summary.total || 0), 0);
    globalRunOffset = page.items.length;
    loadedRunGroupIds = new Set();
    // Keep the user's expanded batches open and refill them under the new
    // query, rather than leaving an expanded section temporarily empty.
    for (const group of runGroups) {
      if (!collapsedRunGroupIds.has(group.id)) void loadRunGroup(group.id);
    }
  }

  async function loadMoreRuns(): Promise<void> {
    if (loadingMoreRuns || globalRunOffset >= indexedRunTotal) return;
    loadingMoreRuns = true;
    try {
    const page = await listRunsPage(globalRunOffset, RUN_PAGE_SIZE, { search: appliedRunSearch, status: appliedRunStatus, ungrouped: true });
      const existing = new Set(runs.map((run) => run.run_id));
      runs = [...runs, ...page.items.filter((run) => !existing.has(run.run_id))];
      indexedRunTotal = page.total;
      globalRunOffset += page.items.length;
    } finally { loadingMoreRuns = false; }
  }

  async function loadRunGroup(groupId: string): Promise<void> {
    if (!groupId || loadedRunGroupIds.has(groupId) || loadingRunGroupIds.has(groupId)) return;
    loadingRunGroupIds = new Set(loadingRunGroupIds).add(groupId);
    try {
      const groupRuns: RunSummary[] = [];
      let offset = 0;
      let total = Infinity;
      // A group is loaded only when opened.  The server remains paginated so
      // unusually large experiment groups do not create one huge response.
      while (offset < total) {
        const page = await listRunsPage(offset, 500, { groupId, search: appliedRunSearch, status: appliedRunStatus });
        groupRuns.push(...page.items);
        total = page.total;
        offset += page.items.length;
        if (!page.items.length) break;
      }
      const existing = new Set(runs.map((run) => run.run_id));
      runs = [...runs, ...groupRuns.filter((run) => !existing.has(run.run_id))];
      loadedRunGroupIds = new Set(loadedRunGroupIds).add(groupId);
    } finally {
      const next = new Set(loadingRunGroupIds);
      next.delete(groupId);
      loadingRunGroupIds = next;
    }
  }

  async function refreshRunGroups(): Promise<void> {
    try {
      const payload = await listRunGroups();
      runGroups = payload.groups || [];
      if (!initializedRunGroupCollapse) {
        collapsedRunGroupIds = new Set(runGroups.map((group) => group.id));
        initializedRunGroupCollapse = true;
      }
    } catch {
      runGroups = [];
    }
  }

  function selectedRunIdList(): string[] {
    return Array.from(selectedRunIds);
  }

  function groupById(groupId: string): RunGroup | undefined {
    return runGroups.find((group) => group.id === groupId);
  }

  function groupIdByName(name: string): string {
    const normalized = name.trim().toLowerCase();
    return runGroups.find((group) => group.name.trim().toLowerCase() === normalized)?.id || '';
  }

  async function refreshRunsAndGroups(): Promise<void> {
    await Promise.all([refreshRuns(), refreshRunGroups()]);
  }

  async function toggleRunGroup(groupId: string): Promise<void> {
    const next = new Set(collapsedRunGroupIds);
    const opening = next.has(groupId);
    if (opening) next.delete(groupId);
    else next.add(groupId);
    collapsedRunGroupIds = next;
    if (opening) await loadRunGroup(groupId);
  }

  async function createGroupFromSelected(): Promise<void> {
    const ids = selectedRunIdList();
    if (!ids.length) return;
    const name = window.prompt('Group name:');
    if (name === null) return;
    const trimmed = name.trim();
    if (!trimmed) {
      statusMessage = 'Group name is required.';
      return;
    }
    try {
      await assignRunsToGroup({ run_ids: ids, name: trimmed, source: 'manual' });
      statusMessage = `Grouped ${ids.length} run${ids.length === 1 ? '' : 's'} as "${trimmed}".`;
      await refreshRunsAndGroups();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  async function moveSelectedToGroup(): Promise<void> {
    const ids = selectedRunIdList();
    if (!ids.length) return;
    const existing = runGroups.filter((group) => !group.archived).map((group) => group.name).join(', ');
    const name = window.prompt(existing ? `Move to group name:\nExisting: ${existing}` : 'Move to group name:');
    if (name === null) return;
    const trimmed = name.trim();
    if (!trimmed) {
      statusMessage = 'Group name is required.';
      return;
    }
    try {
      const existingId = groupIdByName(trimmed);
      await assignRunsToGroup({ run_ids: ids, id: existingId || undefined, name: trimmed, source: existingId ? groupById(existingId)?.source || 'manual' : 'manual' });
      statusMessage = `Moved ${ids.length} run${ids.length === 1 ? '' : 's'} to "${trimmed}".`;
      await refreshRunsAndGroups();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  async function removeSelectedFromGroups(): Promise<void> {
    const ids = selectedRunIdList();
    if (!ids.length) return;
    try {
      await removeRunsFromGroups(ids);
      statusMessage = `Removed ${ids.length} run${ids.length === 1 ? '' : 's'} from groups.`;
      await refreshRunsAndGroups();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  async function renameRunGroup(groupId: string): Promise<void> {
    const group = groupById(groupId);
    if (!group) return;
    const name = window.prompt('Group name:', group.name);
    if (name === null) return;
    const trimmed = name.trim();
    if (!trimmed) {
      statusMessage = 'Group name is required.';
      return;
    }
    try {
      await saveRunGroup({ ...group, name: trimmed });
      await assignRunsToGroup({ run_ids: group.run_ids, id: group.id, name: trimmed, source: group.source || 'manual' });
      statusMessage = `Renamed group to "${trimmed}".`;
      await refreshRunsAndGroups();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  async function removeRunGroup(groupId: string): Promise<void> {
    const group = groupById(groupId);
    const label = group?.name || groupId;
    if (!window.confirm(`Delete group "${label}"? Runs will not be deleted.`)) return;
    try {
      await deleteRunGroup(groupId);
      statusMessage = `Deleted group "${label}".`;
      await refreshRunsAndGroups();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  async function removeRunGroupWithRuns(groupId: string): Promise<void> {
    const group = groupById(groupId);
    const label = group?.name || groupId;
    const count = group?.run_ids?.length || 0;
    if (!window.confirm(`Permanently delete group "${label}" and its ${count} run${count === 1 ? '' : 's'}? This cannot be undone.`)) return;
    try {
      const result = await deleteRunGroupWithRuns(groupId);
      statusMessage = `Deleted group "${label}" and ${result.deleted_run_ids.length} run${result.deleted_run_ids.length === 1 ? '' : 's'}.`;
      await refreshRunsAndGroups();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  function compareRunGroup(runIds: string[], advanced = false): void {
    if (!runIds.length) return;
    selectedRunIds = new Set(runIds);
    if (advanced) {
      saveSelectedCompareRuns(runIds);
      const query = encodeURIComponent(runIds.join(','));
      window.location.href = `/static/compare.html?runs=${query}`;
      return;
    }
    void compareSelectedInControl();
  }

  async function compareWithOracleCounterpart(runId: string): Promise<void> {
    if (!selectedRunId || !runId) return;
    selectedRunIds = new Set([selectedRunId, runId]);
    await compareSelectedInControl();
  }

  async function dropRunOnGroup(runId: string, groupId: string): Promise<void> {
    const group = groupById(groupId);
    if (!group) return;
    try {
      await assignRunsToGroup({ run_ids: [runId], id: group.id, name: group.name, source: group.source || 'manual' });
      statusMessage = `Moved ${runId} to "${group.name}".`;
      await refreshRunsAndGroups();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  async function dropGroupOnGroup(sourceGroupId: string, targetGroupId: string): Promise<void> {
    const source = groupById(sourceGroupId);
    const target = groupById(targetGroupId);
    if (!source || !target || source.id === target.id || !source.run_ids.length) return;
    try {
      await assignRunsToGroup({ run_ids: source.run_ids, id: target.id, name: target.name, source: target.source || 'manual' });
      statusMessage = `Moved ${source.run_ids.length} runs from "${source.name}" to "${target.name}".`;
      await refreshRunsAndGroups();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  async function refreshOptions(): Promise<void> {
    const loaded = await getOptions();
    options = loaded;
    navisStatus = loaded.navis || navisStatus;
    qoeModels = normalizeQoeModels(loaded.qoe_functions);
    ensureValidQoeSelection();
  }

  function navisAvailable(): boolean {
    return Boolean(navisStatus?.configured && navisStatus?.reachable);
  }

  async function refreshNavisStatus(): Promise<void> {
    try {
      navisStatus = await getNavisStatus();
    } catch {
      navisStatus = null;
    }
  }

  async function openRunInNavis(runId: string): Promise<void> {
    if (navisExportingRunIds.has(runId)) return;

    navisExportingRunIds = new Set(navisExportingRunIds).add(runId);
    runs = runs.map((run) => (
      run.run_id === runId
        ? {
            ...run,
            navis_export: {
              ok: false,
              run_id: runId,
              error: 'Sending export request to Prism backend…',
            },
          }
        : run
    ));
    try {
      const sent = await exportRunToNavis(runId);
      runs = runs.map((run) => (run.run_id === runId ? { ...run, navis_export: sent } : run));
      if (!sent.ok) {
        const details = [
          sent.error || `Failed to export ${runId} to Navis.`,
          sent.url ? `POST ${sent.url}` : '',
          sent.status_code ? `HTTP ${sent.status_code}` : '',
          sent.latency_ms != null ? `${sent.latency_ms} ms` : '',
          sent.request_bytes != null ? `${sent.request_bytes} B` : '',
        ].filter(Boolean);
        statusMessage = details.join(' | ');
        return;
      }
      const target = sent.redirect_url || sent.navis_status?.base_url || navisStatus?.base_url;
      if (!target) {
        const details = [
          `Run ${runId} exported to Navis.`,
          sent.url ? `POST ${sent.url}` : '',
          sent.status_code ? `HTTP ${sent.status_code}` : '',
          sent.latency_ms != null ? `${sent.latency_ms} ms` : '',
          sent.request_bytes != null ? `${sent.request_bytes} B` : '',
        ].filter(Boolean);
        statusMessage = details.join(' | ');
        return;
      }
      const navisBase = sent.navis_status?.base_url || navisStatus?.base_url || window.location.origin;
      const resolvedTarget = new URL(target, navisBase).toString();
      const details = [
        `Opening ${runId} in Navis…`,
        sent.url ? `POST ${sent.url}` : '',
        sent.status_code ? `HTTP ${sent.status_code}` : '',
        sent.latency_ms != null ? `${sent.latency_ms} ms` : '',
        sent.request_bytes != null ? `${sent.request_bytes} B` : '',
      ].filter(Boolean);
      statusMessage = details.join(' | ');
      window.open(resolvedTarget, '_blank', 'noopener,noreferrer');
    } catch (err) {
      statusMessage = err instanceof Error ? err.message : String(err);
    } finally {
      const next = new Set(navisExportingRunIds);
      next.delete(runId);
      navisExportingRunIds = next;
    }
  }

  async function generateFovComparison(runId: string): Promise<void> {
    if (fovComparisonGeneratingRunIds.has(runId)) return;
    fovComparisonGeneratingRunIds = new Set(fovComparisonGeneratingRunIds).add(runId);
    runs = runs.map((run) => (
      run.run_id === runId
        ? {
            ...run,
            fov_comparison: {
              ok: false,
              run_id: runId,
              status: 'running',
              error: 'Generating FoV comparison video…',
              video_path: run.fov_comparison?.video_path || null,
              data_path: run.fov_comparison?.data_path || null,
            },
          }
        : run
    ));
    try {
      const result = await generateRunFovComparison(runId);
      runs = runs.map((run) => (run.run_id === runId ? { ...run, fov_comparison: result } : run));
      statusMessage = result.ok
        ? `Generated FoV comparison for ${runId}.`
        : result.error || `Failed to generate FoV comparison for ${runId}.`;
      if (result.video_path) {
        fovComparisonPreview = { runId, result };
      }
    } catch (err) {
      statusMessage = err instanceof Error ? err.message : String(err);
    } finally {
      const next = new Set(fovComparisonGeneratingRunIds);
      next.delete(runId);
      fovComparisonGeneratingRunIds = next;
      await refreshRuns();
    }
  }

  function previewFovComparison(runId: string): void {
    const run = runs.find((item) => item.run_id === runId);
    const result = run?.fov_comparison;
    if (!result?.video_path && !result?.data_path) {
      statusMessage = result?.error || 'No FoV comparison artifact is available for this run.';
      return;
    }
    fovComparisonPreview = { runId, result };
    fovPreviewMode = result.video_path ? 'video' : 'debug';
    fovDebugFrameIndex = 0;
    fovDebugSelectedTileId = '';
    void loadFovComparisonData(result.data_path || '');
  }

  function closeFovComparisonPreview(): void {
    fovComparisonPreview = null;
    fovPreviewMode = 'video';
    fovComparisonDataError = '';
  }

  async function loadFovComparisonData(path: string): Promise<void> {
    if (!path) {
      fovComparisonData = null;
      fovComparisonDataPath = '';
      fovComparisonDataError = 'No comparison data artifact is available.';
      return;
    }
    if (fovComparisonDataPath === path && fovComparisonData) return;
    fovComparisonDataLoading = true;
    fovComparisonDataError = '';
    try {
      const response = await fetch(path);
      if (!response.ok) throw new Error(`Failed to load comparison data (${response.status}).`);
      fovComparisonData = await response.json();
      fovComparisonDataPath = path;
      fovDebugFrameIndex = 0;
      fovDebugSelectedTileId = '';
    } catch (err) {
      fovComparisonData = null;
      fovComparisonDataPath = '';
      fovComparisonDataError = err instanceof Error ? err.message : String(err);
    } finally {
      fovComparisonDataLoading = false;
    }
  }

  function selectFovPreviewMode(mode: 'video' | 'debug'): void {
    fovPreviewMode = mode;
    if (mode === 'debug') {
      void loadFovComparisonData(fovComparisonPreview?.result.data_path || '');
    }
  }

  function scoreColor(value: unknown, alpha = 1): string {
    const raw = Number(value);
    const t = Number.isFinite(raw) ? Math.max(0, Math.min(1, raw)) : 0;
    const r = 255;
    const g = Math.round(245 - 215 * t);
    const b = Math.round(238 - 225 * t);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function errorColor(value: unknown, alpha = 1): string {
    const raw = Number(value);
    const t = Number.isFinite(raw) ? Math.max(0, Math.min(1, Math.abs(raw))) : 0;
    const r = Math.round(239 - 20 * (1 - t));
    const g = Math.round(246 - 208 * t);
    const b = Math.round(255 - 237 * t);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function tileColor(frame: any, tileId: unknown, alpha = 0.72): string {
    const key = String(tileId);
    if (fovDebugTileColorMode === 'truth') return scoreColor(frame?.actual?.[key], alpha);
    if (fovDebugTileColorMode === 'error') {
      return errorColor(Number(frame?.predicted?.[key] || 0) - Number(frame?.actual?.[key] || 0), alpha);
    }
    return scoreColor(frame?.predicted?.[key], alpha);
  }

  function horizonColor(idx: unknown, alpha = 0.95): string {
    const colors: Array<[number, number, number]> = [
      [124, 58, 237],
      [8, 145, 178],
      [219, 39, 119],
    ];
    const value = Number(idx);
    const colorIdx = Number.isFinite(value) ? Math.abs(Math.trunc(value)) % colors.length : 0;
    const rgb: [number, number, number] = colors[colorIdx] || [124, 58, 237];
    return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
  }

  function startVoxelKey(voxel: any): string {
    const raw = voxel?.voxel;
    if (Array.isArray(raw) && raw.length >= 3) return `${Number(raw[0])},${Number(raw[1])},${Number(raw[2])}`;
    const center = voxel?.center;
    if (Array.isArray(center) && center.length >= 3) return `${Number(center[0])}:${Number(center[1])}:${Number(center[2])}`;
    return '';
  }

  function pathHeightRange(path: any[]): { minY: number; maxY: number } {
    const ys = (path || [])
      .map((point) => Number(point?.[1]))
      .filter(Number.isFinite);
    if (!ys.length) return { minY: 0, maxY: 1 };
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    return maxY > minY ? { minY, maxY } : { minY, maxY: minY + 1 };
  }

  function pathHeightColor(y: unknown, minY: number, maxY: number, alpha = 0.92): string {
    const raw = Number(y);
    const t = Number.isFinite(raw) ? Math.max(0, Math.min(1, (raw - minY) / Math.max(maxY - minY, 1e-9))) : 0;
    const r = Math.round(220 * (1 - t) + 37 * t);
    const g = Math.round(38 * (1 - t) + 99 * t);
    const b = Math.round(38 * (1 - t) + 235 * t);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function buildFovDebugScaleReference(frame: any, debug: any): any {
    const firstTileId = (frame?.tile_ids || [])[0];
    const tileBounds = firstTileId != null ? frame?.tiles?.[String(firstTileId)]?.bounds_xz : null;
    let tileWidth = 0;
    let tileDepth = 0;
    if (Array.isArray(tileBounds) && tileBounds.length >= 4) {
      tileWidth = Math.abs(Number(tileBounds[2]) - Number(tileBounds[0]));
      tileDepth = Math.abs(Number(tileBounds[3]) - Number(tileBounds[1]));
    }
    const scale = debug?.scale_reference || {};
    return {
      tileWidth: Number.isFinite(tileWidth) ? tileWidth : 0,
      tileDepth: Number.isFinite(tileDepth) ? tileDepth : 0,
      geometry: Number(scale.geometry_voxel_size_l || 0),
      navigation: Number(scale.navigation_voxel_size_l || 0),
      visibility: Number(scale.visibility_voxel_size_l || debug?.voxel_grid?.voxel_size || 0),
    };
  }

  function fmtMeters(value: unknown): string {
    const raw = Number(value);
    if (!Number.isFinite(raw) || raw <= 0) return '—';
    return raw >= 1 ? `${raw.toFixed(2)}m` : `${raw.toFixed(3)}m`;
  }

  function fmtScore(value: unknown): string {
    const raw = Number(value);
    return Number.isFinite(raw) ? raw.toFixed(3) : '—';
  }

  function fmtCount(value: unknown): string {
    const raw = Number(value);
    return Number.isFinite(raw) ? Math.round(raw).toString() : '—';
  }

  function tracePolyline(samples: any[] | undefined): string {
    return (samples || [])
      .filter((sample) => Array.isArray(sample?.position) && sample.position.length >= 3)
      .map((sample) => `${Number(sample.position[0])},${Number(sample.position[2])}`)
      .join(' ');
  }

  function arrowEnd(sample: any, scale: number): { x: number; z: number } | null {
    if (!Array.isArray(sample?.position) || sample.position.length < 3) return null;
    if (!Array.isArray(sample?.forward) || sample.forward.length < 3) return null;
    const x = Number(sample.position[0]);
    const z = Number(sample.position[2]);
    const fx = Number(sample.forward[0]);
    const fz = Number(sample.forward[2]);
    if (![x, z, fx, fz].every(Number.isFinite)) return null;
    return { x: x + fx * scale, z: z + fz * scale };
  }

  function horizonEnd(horizon: any, maxDim: number): { x1: number; z1: number; x2: number; z2: number } | null {
    const origin = Array.isArray(horizon?.origin) ? horizon.origin : fovDebugPayload?.user_position;
    const direction = horizon?.direction;
    if (!Array.isArray(origin) || origin.length < 3 || !Array.isArray(direction) || direction.length < 3) return null;
    const x = Number(origin[0]);
    const z = Number(origin[2]);
    const dx = Number(direction[0]);
    const dz = Number(direction[2]);
    if (![x, z, dx, dz].every(Number.isFinite)) return null;
    const rawLength = Number(horizon?.distance_m);
    const capped = Math.min(maxDim * 0.26, Math.max(maxDim * 0.07, Number.isFinite(rawLength) ? rawLength : maxDim * 0.12));
    return { x1: x, z1: z, x2: x + dx * capped, z2: z + dz * capped };
  }

  function hitColor(kind: unknown): string {
    const key = String(kind || '');
    if (key === 'occluder') return 'rgba(220, 38, 38, 0.92)';
    if (key === 'target_tile' || key === 'target_voxel') return 'rgba(22, 163, 74, 0.9)';
    if (key === 'out_of_grid') return 'rgba(124, 58, 237, 0.9)';
    return 'rgba(71, 85, 105, 0.85)';
  }

  function computeFovDebugBounds(frame: any, debug: any, pointSlice: any): { minX: number; minZ: number; width: number; height: number } | null {
    const xs: number[] = [];
    const zs: number[] = [];
    for (const tile of Object.values(frame?.tiles || {}) as any[]) {
      const bounds = tile?.bounds_xz;
      if (!Array.isArray(bounds) || bounds.length < 4) continue;
      xs.push(Number(bounds[0]), Number(bounds[2]));
      zs.push(Number(bounds[1]), Number(bounds[3]));
    }
    for (const voxel of debug?.selected_voxels || []) {
      if (!Array.isArray(voxel?.center) || voxel.center.length < 3) continue;
      xs.push(Number(voxel.center[0]));
      zs.push(Number(voxel.center[2]));
      for (const point of voxel?.path || []) {
        if (!Array.isArray(point) || point.length < 3) continue;
        xs.push(Number(point[0]));
        zs.push(Number(point[2]));
      }
    }
    for (const tileDebug of Object.values(debug?.tiles || {}) as any[]) {
      for (const contributor of tileDebug?.contributors || []) {
        for (const ray of contributor?.rays || []) {
          if (Array.isArray(ray?.start) && ray.start.length >= 3) {
            xs.push(Number(ray.start[0]));
            zs.push(Number(ray.start[2]));
          }
          if (Array.isArray(ray?.target) && ray.target.length >= 3) {
            xs.push(Number(ray.target[0]));
            zs.push(Number(ray.target[2]));
          }
        }
      }
    }
    for (const sample of [...(frame?.user_trace?.observed || []), ...(frame?.user_trace?.actual || [])]) {
      if (!Array.isArray(sample?.position) || sample.position.length < 3) continue;
      xs.push(Number(sample.position[0]));
      zs.push(Number(sample.position[2]));
    }
    if (Array.isArray(frame?.user_trace?.current?.position) && frame.user_trace.current.position.length >= 3) {
      xs.push(Number(frame.user_trace.current.position[0]));
      zs.push(Number(frame.user_trace.current.position[2]));
    }
    for (const horizon of debug?.intentions?.horizons || []) {
      if (Array.isArray(horizon?.origin) && horizon.origin.length >= 3) {
        xs.push(Number(horizon.origin[0]));
        zs.push(Number(horizon.origin[2]));
      }
      if (Array.isArray(horizon?.endpoint) && horizon.endpoint.length >= 3) {
        xs.push(Number(horizon.endpoint[0]));
        zs.push(Number(horizon.endpoint[2]));
      }
    }
    const reachable = debug?.intentions?.reachable;
    if (Array.isArray(reachable?.center) && reachable.center.length >= 3) {
      const radius = Number(reachable.radius_m || 0);
      xs.push(Number(reachable.center[0]) - radius, Number(reachable.center[0]) + radius);
      zs.push(Number(reachable.center[2]) - radius, Number(reachable.center[2]) + radius);
    }
    for (const point of reachable?.sampled_free_voxels || []) {
      if (!Array.isArray(point) || point.length < 3) continue;
      xs.push(Number(point[0]));
      zs.push(Number(point[2]));
    }
    for (const point of pointSlice?.points || []) {
      if (!Array.isArray(point) || point.length < 2) continue;
      xs.push(Number(point[0]));
      zs.push(Number(point[1]));
    }
    const cleanX = xs.filter(Number.isFinite);
    const cleanZ = zs.filter(Number.isFinite);
    if (!cleanX.length || !cleanZ.length) return null;
    let minX = Math.min(...cleanX);
    let maxX = Math.max(...cleanX);
    let minZ = Math.min(...cleanZ);
    let maxZ = Math.max(...cleanZ);
    const pad = Math.max(maxX - minX, maxZ - minZ, 1) * 0.05;
    minX -= pad;
    maxX += pad;
    minZ -= pad;
    maxZ += pad;
    return { minX, minZ, width: Math.max(maxX - minX, 1e-6), height: Math.max(maxZ - minZ, 1e-6) };
  }

  function handleFovDebugTileKeydown(event: KeyboardEvent, tileId: unknown): void {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    fovDebugSelectedTileId = String(tileId);
  }

  async function submitRun(nextConfig?: RunConfig, skipPreflight = false): Promise<void> {
    if (submitting) return;
    submitting = true;
    error = '';
    try {
      const payload: RunConfig = {
        ...(nextConfig || form),
        abr: (nextConfig || form).abr || defaultAbr(),
        predictor: (nextConfig || form).predictor || 'None',
        bandwidth_predictor: (nextConfig || form).bandwidth_predictor || 'None',
      };
      form = payload;
      if (!skipPreflight) {
        const review = await preflightRuns([payload]);
        const mode = loadLaunchPreflightMode();
        if (review.status === 'error') {
          error = review.checks.map((check) => check.title).join(' ');
          return;
        }
        if ((review.status === 'warning' || mode === 'always') && mode !== 'never') {
          const lines = review.checks.map((check) => `${check.severity === 'warning' ? '⚠' : '✓'} ${check.title}`).join('\n');
          if (!window.confirm(`Review run\n\n${lines || '✓ All inputs are compatible.'}\n\nStart this run?`)) return;
        }
      }
      const response = await startRun(payload);
      const queuedIds = response.run_ids || [response.run_id].filter(Boolean);
      statusMessage = queuedIds.length > 1 ? `Run suite queued: ${queuedIds.length} runs` : `Run queued: ${response.run_id}`;
      saveFormToCache();
      await refreshRuns();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      submitting = false;
    }
  }

  async function removeRun(runId: string): Promise<void> {
    error = '';
    try {
      await deleteRun(runId);
      selectedRunIds.delete(runId);
      if (selectedRunId === runId) {
        selectedRunId = '';
        selectedRunConfig = null;
        selectedRunEvents = null;
        chartConfig = null;
      }
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      await refreshRuns();
    }
  }

  async function inspectRun(runId: string): Promise<void> {
    compareMode = false;
    compareCards = [];
    selectedRunId = runId;
    oracleCounterparts = [];
    selectedOracleCounterpartRunId = '';
    error = '';
    try {
      const [cfg, events] = await Promise.all([getRunConfig(runId), getRunEvents(runId)]);
      selectedRunConfig = cfg;
      runConfigCache[runId] = cfg;
      runConfigCache = { ...runConfigCache };
      selectedRunEvents = events;
      await rebuildRunChart();
      await refreshOracleCounterparts(runId, cfg);
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      selectedRunConfig = null;
      selectedRunEvents = null;
      chartConfig = null;
      oracleCounterparts = [];
      selectedOracleCounterpartRunId = '';
    }
  }

  function normalizedPath(value: unknown): string {
    return String(value || '').replace(/\\/g, '/').replace(/\/+$/g, '');
  }

  function comparableParams(value: unknown): string {
    if (!value || typeof value !== 'object') return '{}';
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, entryValue]) => entryValue !== undefined && entryValue !== '')
      .sort(([a], [b]) => a.localeCompare(b));
    return JSON.stringify(Object.fromEntries(entries));
  }

  function scalarConfigValue(value: unknown): string | number | boolean | null {
    if (value === undefined || value === '') return null;
    if (typeof value === 'number') return Number.isFinite(value) ? Number(value.toFixed(8)) : null;
    if (typeof value === 'boolean') return value;
    if (value === null) return null;
    return String(value);
  }

  function scenarioSignature(cfg: RunConfig): string {
    const networkSource = cfg.network_profile ? `profile:${cfg.network_profile}` : `trace:${normalizedPath(cfg.network_trace)}`;
    return stableStringify({
      dataset_id: scalarConfigValue(cfg.dataset_id),
      mpd: normalizedPath(cfg.mpd),
      content_dir: normalizedPath(cfg.content_dir),
      v3c_folder: normalizedPath(cfg.v3c_folder),
      fov_trace: normalizedPath(cfg.fov_trace),
      network_source: networkSource,
      abr: cfg.abr || '',
      abr_params: comparableParams(cfg.abr_params),
      headset_preset: scalarConfigValue(cfg.headset_preset),
      fov_h_deg: scalarConfigValue(cfg.fov_h_deg),
      fov_v_deg: scalarConfigValue(cfg.fov_v_deg),
      angle_convention: scalarConfigValue(cfg.angle_convention),
      rtt: scalarConfigValue(cfg.rtt),
      block_size_bits: scalarConfigValue(cfg.block_size_bits),
      enable_fov_upgrades: Boolean(cfg.enable_fov_upgrades),
      enable_mid_download_cancel: Boolean(cfg.enable_mid_download_cancel),
      buffer_window_segments: scalarConfigValue(cfg.buffer_window_segments),
      buffer_unbounded: Boolean(cfg.buffer_unbounded),
      segment_limit: scalarConfigValue(cfg.segment_limit),
    });
  }

  function isFovOraclePredictor(name: string | undefined): boolean {
    return ['GroundTruthFovPredictor', 'GroundTruthTileFovPredictor'].includes(String(name || ''));
  }

  function isBandwidthOraclePredictor(name: string | undefined): boolean {
    return String(name || '') === 'GroundTruthBandwidthPredictor';
  }

  function samePredictorDimension(aName: string | undefined, aParams: unknown, bName: string | undefined, bParams: unknown): boolean {
    return String(aName || 'None') === String(bName || 'None') && comparableParams(aParams) === comparableParams(bParams);
  }

  function classifyOracleCounterpart(target: RunConfig, candidate: RunConfig): OracleCounterpart['type'] | null {
    if (scenarioSignature(target) !== scenarioSignature(candidate)) return null;

    const targetFovOracle = isFovOraclePredictor(target.predictor);
    const candidateFovOracle = isFovOraclePredictor(candidate.predictor);
    const targetBandwidthOracle = isBandwidthOraclePredictor(target.bandwidth_predictor);
    const candidateBandwidthOracle = isBandwidthOraclePredictor(candidate.bandwidth_predictor);
    const addsFovOracle = candidateFovOracle && !targetFovOracle;
    const addsBandwidthOracle = candidateBandwidthOracle && !targetBandwidthOracle;
    if (!addsFovOracle && !addsBandwidthOracle) return null;

    if (!addsFovOracle && !samePredictorDimension(target.predictor, target.predictor_params, candidate.predictor, candidate.predictor_params)) {
      return null;
    }
    if (!addsBandwidthOracle && !samePredictorDimension(target.bandwidth_predictor, target.bandwidth_predictor_params, candidate.bandwidth_predictor, candidate.bandwidth_predictor_params)) {
      return null;
    }
    if (addsFovOracle && addsBandwidthOracle) return 'both';
    return addsFovOracle ? 'fov' : 'bandwidth';
  }

  function oracleCounterpartLabel(type: OracleCounterpart['type'], runId: string): string {
    const base = type === 'both' ? 'FoV + bandwidth oracle' : type === 'fov' ? 'FoV oracle' : 'Bandwidth oracle';
    return `${base} - ${runLabel(runId)}`;
  }

  function oracleCounterpartReason(type: OracleCounterpart['type']): string {
    const changed = type === 'both' ? 'FoV and bandwidth predictors' : type === 'fov' ? 'FoV predictor' : 'bandwidth predictor';
    return `Same group, content, FoV trace, network source, ABR, buffer, RTT, and playback settings; only the ${changed} is replaced by ground truth.`;
  }

  async function cachedRunConfig(runId: string): Promise<RunConfig | null> {
    if (runConfigCache[runId]) return runConfigCache[runId];
    try {
      const cfg = await getRunConfig(runId);
      runConfigCache[runId] = cfg;
      runConfigCache = { ...runConfigCache };
      return cfg;
    } catch {
      runConfigCache[runId] = null;
      runConfigCache = { ...runConfigCache };
      return null;
    }
  }

  async function refreshOracleCounterparts(runId: string, cfg: RunConfig): Promise<void> {
    const targetRun = runs.find((run) => run.run_id === runId);
    const groupId = targetRun?.run_group_id || cfg.run_group_id || '';
    if (!groupId) {
      oracleCounterparts = [];
      selectedOracleCounterpartRunId = '';
      return;
    }
    oracleCounterpartsLoading = true;
    try {
      const candidates = runs.filter((run) => run.run_id !== runId && run.run_group_id === groupId);
      const matches: OracleCounterpart[] = [];
      for (const candidate of candidates) {
        // eslint-disable-next-line no-await-in-loop
        const candidateCfg = await cachedRunConfig(candidate.run_id);
        if (!candidateCfg) continue;
        const type = classifyOracleCounterpart(cfg, candidateCfg);
        if (!type) continue;
        matches.push({
          runId: candidate.run_id,
          type,
          label: oracleCounterpartLabel(type, candidate.run_id),
          reason: oracleCounterpartReason(type),
        });
      }
      const order = { fov: 0, bandwidth: 1, both: 2 };
      oracleCounterparts = matches.sort((a, b) => order[a.type] - order[b.type] || runLabel(a.runId).localeCompare(runLabel(b.runId)));
      selectedOracleCounterpartRunId = oracleCounterparts[0]?.runId || '';
    } finally {
      oracleCounterpartsLoading = false;
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

  function runFingerprint(runId: string): string {
    const run = runs.find((item) => item.run_id === runId);
    if (!run) return runId;
    const artifactFingerprint = {
      config: run.artifact_fingerprint?.['config.json'],
      events: run.artifact_fingerprint?.['events.json'],
    };
    return stableStringify({
      runId,
      inProgress: run.status === 'queued' || run.status === 'running',
      artifactFingerprint,
    });
  }

  function qoeCacheKey(runIds: string[]): string {
    const model = activeQoeModel();
    return stableStringify({
      runIds: [...runIds].sort(),
      fingerprints: [...runIds].sort().map((runId) => runFingerprint(runId)),
      qoeModel: model?.id || '',
      qoeParams: model ? activeQoeParams() : {},
    });
  }

  function chartCacheKey(runId: string, cfg: RunConfig): string {
    const model = activeQoeModel();
    return stableStringify({
      runId,
      fingerprint: runFingerprint(runId),
      networkTrace: cfg.network_trace || '',
      networkProfile: cfg.network_profile || '',
      tailMode,
      visibility: layerVisibility(),
      qoeModel: model?.id || '',
      qoeParams: model ? activeQoeParams() : {},
    });
  }

  function buildCombinedQoeCompareChart(
    ids: string[],
    qoeByRun: Record<string, Array<{ time_s: number; score: number }>>,
  ): ChartConfiguration | null {
    const colors = ['#22d3ee', '#f472b6', '#a3e635', '#f59e0b', '#38bdf8', '#c084fc', '#fb7185', '#14b8a6'];
    const datasets = ids
      .map((runId, idx) => {
        const data = (qoeByRun[runId] || [])
          .map((point) => ({ x: Number(point.time_s), y: Number(point.score) }))
          .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
        if (!data.length) return null;
        const color = colors[idx % colors.length];
        return {
          label: runLabel(runId),
          data,
          borderColor: color,
          backgroundColor: color,
          borderWidth: 2,
          tension: 0.12,
          pointRadius: 0,
          parsing: false,
        };
      })
      .filter(Boolean) as any[];
    if (!datasets.length) return null;
    return {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
          legend: { display: true, position: 'bottom' },
          tooltip: {
            callbacks: {
              label: (ctx: any) => `${ctx.dataset.label}: ${Number(ctx.raw?.y || 0).toFixed(3)} @ ${Number(ctx.raw?.x || 0).toFixed(2)}s`,
            },
          },
        },
        scales: {
          x: { type: 'linear', title: { display: true, text: 'Time (s)' }, min: 0 },
          y: { title: { display: true, text: 'QoE score' }, min: 0, max: 1 },
        },
      },
    };
  }

  async function getQoeByRun(runIds: string[], force = false): Promise<Record<string, Array<{ time_s: number; score: number }>>> {
    if ((!showQoe && !force) || !runIds.length) return {};
    const model = activeQoeModel();
    if (!model) return {};
    const key = qoeCacheKey(runIds);
    const cached = qoeSeriesCache.get(key);
    if (cached) return cached;
    const started = performance.now();
    const diag = await getDiagnostics({ run_ids: runIds, qoe_function: model.id, qoe_params: activeQoeParams() });
    console.debug('[graph-timing] single-run qoe diagnostics fetch', { elapsedMs: Math.round(performance.now() - started), runs: runIds.length });
    const qoe = diag.qoe_series || {};
    qoeSeriesCache.set(key, qoe);
    return qoe;
  }

  async function fetchTraceSamples(path: string | undefined): Promise<Array<{ time_s: number; bandwidth_mbps: number }>> {
    if (!path) return [];
    if (traceCache.has(path)) return traceCache.get(path)!;
    try {
      const started = performance.now();
      const trace = await getTrace(path);
      const samples = trace.samples || [];
      console.debug('[graph-timing] single-run trace fetch', { path, samples: samples.length, elapsedMs: Math.round(performance.now() - started) });
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
    qoeByRun: Record<string, Array<{ time_s: number; score: number }>>,
  ): Promise<ChartConfiguration> {
    const key = chartCacheKey(runId, cfg);
    const cached = chartConfigCache.get(key);
    if (cached) return cached;
    const traceSamples = await fetchTraceSamples(cfg.network_trace);
    const qoeValues = qoeByRun[runId] || [];
    const config = await scheduleGraphWork(() => timed(`single-run build ${runId}`, () => {
      const built = buildRunLayers(events, cfg, traceSamples, qoeValues, tailMode);
      const visibility = layerVisibility();
      const bounds = computeBounds(built.layers, visibility);
      return toChartConfig(built.layers, built.stalls, bounds, visibility);
    }));
    chartConfigCache.set(key, config);
    return config;
  }

  async function rebuildRunChart(): Promise<void> {
    if (!selectedRunId || !selectedRunEvents || !selectedRunConfig) {
      chartConfig = null;
      return;
    }
    const token = ++chartBuildSerial;
    chartLoading = true;
    try {
      const qoeByRun = await getQoeByRun([selectedRunId]);
      const nextChart = await buildChartForRun(selectedRunId, selectedRunConfig, selectedRunEvents, qoeByRun);
      if (token === chartBuildSerial) chartConfig = nextChart;
    } finally {
      if (token === chartBuildSerial) chartLoading = false;
    }
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
      const qoeByRun = await getQoeByRun(ids, compareCollapseQoe);
      if (compareCollapseQoe) {
        const chart = await scheduleGraphWork(() => timed('single-run compare combined QoE build', () =>
          buildCombinedQoeCompareChart(ids, qoeByRun)
        ));
        compareCards = [{
          runId: 'combined-qoe',
          label: 'QoE score over time',
          chart,
          error: chart ? '' : 'No QoE series available for the selected runs.',
        }];
        return;
      }
      const loaded = await Promise.all(
        ids.map(async (runId) => {
          try {
            const [cfg, events] = await Promise.all([getRunConfig(runId), getRunEvents(runId)]);
            const traceSamples = await fetchTraceSamples(cfg.network_trace);
            const built = await scheduleGraphWork(() => timed(`single-run compare build ${runId}`, () =>
              buildRunLayers(events, cfg, traceSamples, qoeByRun[runId] || [], tailMode)
            ));
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
    const failures: string[] = [];
    for (const runId of ids) {
      try {
        // eslint-disable-next-line no-await-in-loop
        await deleteRun(runId);
      } catch (err) {
        failures.push(`${runId}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
    if (failures.length) {
      error = failures.join(' | ');
    } else {
      error = '';
      selectedRunIds = new Set();
    }
    await refreshRuns();
  }

  async function controlGroup(groupId: string, action: 'pause' | 'resume' | 'cancel' | 'force_cancel'): Promise<void> {
    if (action === 'cancel' && !window.confirm('Cancel all queued/running runs in this group?')) return;
    if (action === 'force_cancel' && !window.confirm('Force-cancel this group? Active raycast/PLY worker processes will be terminated immediately.')) return;
    try {
      const result = await controlRunGroup(groupId, action);
      runGroups = result.groups || runGroups;
      const actionLabel = `${action.charAt(0).toUpperCase()}${action.slice(1)}`;
      statusMessage = `${actionLabel} requested for ${result.affected} run${result.affected === 1 ? '' : 's'}.`;
      error = '';
      await refreshRuns();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  function validateAlias(runId: string, alias: string): { ok: boolean; value: string; reason?: string } {
    const trimmed = alias.trim();
    if (!trimmed) return { ok: true, value: '' };
    if (trimmed.length > RUN_ALIAS_MAX_LENGTH) {
      return {
        ok: false,
        value: trimmed.slice(0, RUN_ALIAS_MAX_LENGTH),
        reason: `Alias must be at most ${RUN_ALIAS_MAX_LENGTH} characters.`,
      };
    }
    const nextKey = trimmed.toLowerCase();
    const duplicate = Object.entries(runAliases).some(([otherRunId, otherAlias]) => (
      otherRunId !== runId && otherAlias.trim().toLowerCase() === nextKey
    ));
    if (duplicate) {
      return { ok: false, value: trimmed, reason: 'Alias already exists. Please choose a unique name.' };
    }
    return { ok: true, value: trimmed };
  }

  function updateAlias(runId: string, alias: string): { ok: boolean; reason?: string } {
    const checked = validateAlias(runId, alias);
    if (!checked.ok) return { ok: false, reason: checked.reason };
    if (checked.value) runAliases[runId] = checked.value;
    else delete runAliases[runId];
    runAliases = { ...runAliases };
    saveRunAliases(runAliases);
    return { ok: true };
  }

  function quickRenameRun(runId: string): void {
    const current = runAliases[runId] || '';
    const next = window.prompt('Run alias (leave empty to clear):', current);
    if (next === null) return;
    const result = updateAlias(runId, next);
    if (!result.ok && result.reason) {
      window.alert(result.reason);
    }
  }

  function handleRunFormSubmit(nextConfig: RunConfig): void {
    submitRun(nextConfig);
  }

  function handleAliasInputChange(runId: string, event: Event): void {
    const target = event.currentTarget as HTMLInputElement | null;
    const value = target?.value || '';
    const result = updateAlias(runId, value);
    if (!result.ok) {
      if (target) target.value = runAliases[runId] || '';
      if (result.reason) window.alert(result.reason);
    }
  }

  function onQoeModelChange(modelId: string): void {
    qoeState = {
      ...qoeState,
      selectedId: modelId,
      paramsById: { ...(qoeState.paramsById || {}) },
    };
    const model = activeQoeModel();
    if (model && !qoeState.paramsById[model.id]) {
      qoeState = {
        ...qoeState,
        paramsById: {
          ...(qoeState.paramsById || {}),
          [model.id]: sanitizeQoeParams(model, {}),
        },
      };
    }
    saveQoeState(QOE_STORAGE_KEY, qoeState);
  }

  async function handleQoeModelSelectChange(event: Event): Promise<void> {
    const target = event.currentTarget as HTMLSelectElement | null;
    onQoeModelChange(target?.value || '');
    await refreshCurrentViewCharts();
  }

  function onQoeFieldChange(field: string, value: string): void {
    const model = activeQoeModel();
    if (!model) return;
    const current = { ...(qoeState.paramsById[model.id] || {}) };
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      current[field] = numeric;
      qoeState = {
        ...qoeState,
        paramsById: {
          ...(qoeState.paramsById || {}),
          [model.id]: sanitizeQoeParams(model, current),
        },
      };
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
    qoeState = {
      ...qoeState,
      paramsById: {
        ...(qoeState.paramsById || {}),
        [model.id]: sanitizeQoeParams(model, {}),
      },
    };
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
    rowSort;
    runConfigCache;
    filteredRunRows = filterAndSortRunRows(runRows, runSearch, statusFilter, runConfigCache, rowSort);
  }

  $: {
    // The initial page is compact, but search/status must cover the complete
    // SQLite catalogue. Debouncing prevents one request per keystroke.
    const nextSearch = runSearch.trim();
    const nextStatus = statusFilter;
    if (runIndexReady && (nextSearch !== appliedRunSearch || nextStatus !== appliedRunStatus)) {
      if (runFilterTimer) clearTimeout(runFilterTimer);
      runFilterTimer = setTimeout(() => {
        appliedRunSearch = nextSearch;
        appliedRunStatus = nextStatus;
        void refreshRuns();
      }, 220);
    }
  }

  $: {
    const validIds = new Set(runs.map((run) => run.run_id));
    const nextExpanded = new Set(Array.from(expandedRunIds).filter((id) => validIds.has(id)));
    if (nextExpanded.size !== expandedRunIds.size) expandedRunIds = nextExpanded;
  }

  onMount(async () => {
    try {
      await refreshOptions();
      navisStatus = options?.navis || null;
      qoeModels = normalizeQoeModels(options?.qoe_functions || []);
      qoeState = loadQoeState(QOE_STORAGE_KEY, qoeModels);
      ensureValidQoeSelection();
      form.abr = form.abr || defaultAbr();
      form.predictor = form.predictor || defaultPredictor();
      await refreshRunFormSessions();
      await refreshNavisStatus();
      navisStatusTimer = setInterval(() => {
        void refreshNavisStatus();
      }, 5000);
      await refreshRunsAndGroups();
      runIndexReady = true;
      unsubscribeRuns = subscribeRuns((next) => {
        const updates = new Map(next.map((run) => [run.run_id, run]));
        const visibleActiveRunEnded = runs.some((run) => (
          ['queued', 'running', 'paused', 'canceling'].includes(run.status) && !updates.has(run.run_id)
        ));
        if (visibleActiveRunEnded) {
          // The stream intentionally contains active work only. Once a job
          // disappears, refresh its compact indexed row so it becomes a final
          // result without keeping the full catalogue in the stream.
          void refreshRuns();
          return;
        }
        // The stream contains all active work, including batches launched
        // after this page mounted. Merge those new IDs as well as refreshing
        // existing rows; replacing only existing rows hid externally launched
        // batch queues from the Control Center.
        const merged = new Map(runs.map((run) => [run.run_id, run]));
        for (const [runId, update] of updates) merged.set(runId, update);
        runs = Array.from(merged.values());
      });
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  });

  onDestroy(() => {
    unsubscribeRuns?.();
    if (runFilterTimer) clearTimeout(runFilterTimer);
    if (navisStatusTimer) clearInterval(navisStatusTimer);
  });

  $: {
    qoeModels;
    qoeState.selectedId;
    ensureValidQoeSelection();
  }
</script>

<header class="hero">
  <div class="hero-actions">
    <a class="button-link ghost" href="/" on:click|preventDefault={() => gotoRoute('home')}>Main menu</a>
    <a class="button-link ghost" href="/static/datasets.html" on:click|preventDefault={() => gotoRoute('datasets')}>Datasets</a>
    <a class="button-link active" aria-current="page" href="/static/index.html" on:click|preventDefault={() => gotoRoute('control')}>Single Run</a>
    <a class="button-link ghost" href="/static/aggregate.html" on:click|preventDefault={() => gotoRoute('batch')}>Batch Planner</a>
    <a class="button-link ghost" href="/static/compare.html" on:click|preventDefault={() => gotoRoute('compare')}>Compare Runs</a>
    <a class="button-link ghost" href="/static/reports.html" on:click|preventDefault={() => gotoRoute('reports')}>Reports</a>
    <a class="button-link ghost" href="/static/extra.html" on:click|preventDefault={() => gotoRoute('extra')}>Extra</a>
    <a class="button-link ghost" href="/static/settings.html" on:click|preventDefault={() => gotoRoute('settings')}>Settings</a>
  </div>
</header>

<main>
  <RunLaunchForm
    {options}
    bind:form
    {submitting}
    {loading}
    {savedSessions}
    onSubmit={handleRunFormSubmit}
    onRefreshOptions={refreshOptions}
    onSaveSession={saveCurrentFormSession}
    onLoadSession={loadNamedFormSession}
    onDeleteSession={deleteNamedFormSession}
  />

  <RunsExplorer
    {runs}
    totalRuns={matchingRunTotal}
    canLoadMore={globalRunOffset < indexedRunTotal}
    {loadingMoreRuns}
    onLoadMoreRuns={loadMoreRuns}
    {filteredRunRows}
    {selectedRunIds}
    {selectedRunId}
    {expandedRunIds}
    {loadingDetailIds}
    {runConfigCache}
    {runAliases}
    {navisExportingRunIds}
    {fovComparisonGeneratingRunIds}
    {runGroups}
    {collapsedRunGroupIds}
    {loadingRunGroupIds}
    {runGroupSummaries}
    bind:runSearch
    bind:statusFilter
    bind:rowSort
    bind:groupView
    bind:groupSort
    onClearSelection={clearSelection}
    onCompareSelectedInControl={compareSelectedInControl}
    onAdvancedCompare={compareSelected}
    onDeleteSelected={deleteSelected}
    onCreateGroupFromSelected={createGroupFromSelected}
    onMoveSelectedToGroup={moveSelectedToGroup}
    onRemoveSelectedFromGroups={removeSelectedFromGroups}
    onToggleRunGroup={toggleRunGroup}
    onRenameRunGroup={renameRunGroup}
    onDeleteRunGroup={removeRunGroup}
    onDeleteRunGroupWithRuns={removeRunGroupWithRuns}
    onControlRunGroup={controlGroup}
    onCompareRunGroup={compareRunGroup}
    onDropRunOnGroup={dropRunOnGroup}
    onDropGroupOnGroup={dropGroupOnGroup}
    onRowCheckboxClick={onRowCheckboxClick}
    onRowSelectClick={onRowSelectClick}
    onRowSelectKeydown={onRowSelectKeydown}
    onInspectRun={inspectRun}
    onToggleRunDetails={toggleRunDetails}
    onRemoveRun={removeRun}
    onOpenInNavis={openRunInNavis}
    onGenerateFovComparison={generateFovComparison}
    onPreviewFovComparison={previewFovComparison}
    onQuickRenameRun={quickRenameRun}
    onAliasChange={handleAliasInputChange}
    aliasMaxLength={RUN_ALIAS_MAX_LENGTH}
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
    bind:compareLayout
    bind:compareCollapseQoe
    {compareCards}
    {selectedRunId}
    {error}
    {chartLoading}
    {oracleCounterparts}
    {oracleCounterpartsLoading}
    bind:selectedOracleCounterpartRunId
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
    onCompareWithOracleCounterpart={compareWithOracleCounterpart}
  />
</main>

{#if fovComparisonPreview?.result.video_path || fovComparisonPreview?.result.data_path}
  <div class="fov-preview-backdrop" on:pointerdown|self={closeFovComparisonPreview}>
    <section class="fov-preview-modal">
      <header class="fov-preview-header">
        <div>
          <h2>FoV Comparison</h2>
          <p class="subtle">{fovComparisonPreview.runId}</p>
        </div>
        <div class="section-actions">
          <button
            type="button"
            class:active={fovPreviewMode === 'video'}
            class="ghost small"
            on:click={() => selectFovPreviewMode('video')}
            disabled={!fovComparisonPreview.result.video_path}
          >
            Video
          </button>
          <button
            type="button"
            class:active={fovPreviewMode === 'debug'}
            class="ghost small"
            on:click={() => selectFovPreviewMode('debug')}
            disabled={!fovComparisonPreview.result.data_path}
          >
            Debug
          </button>
          {#if fovComparisonPreview.result.data_path}
            <a class="button-link ghost small" href={fovComparisonPreview.result.data_path} target="_blank">Data</a>
          {/if}
          {#if fovComparisonPreview.result.video_path}
            <a class="button-link ghost small" href={fovComparisonPreview.result.video_path} target="_blank">Open</a>
          {/if}
          <button type="button" class="ghost small" on:click={closeFovComparisonPreview}>Close</button>
        </div>
      </header>
      {#if fovPreviewMode === 'video'}
        {#if fovComparisonPreview.result.video_path}
          <video
            class="fov-preview-video"
            src={fovComparisonPreview.result.video_path}
            controls
            autoplay
            playsinline
          >
            <track kind="captions" />
          </video>
        {:else}
          <p class="subtle">No FoV comparison video is available for this run.</p>
        {/if}
      {:else}
        <div class="fov-debug-shell">
          {#if fovComparisonDataLoading}
            <p class="subtle">Loading comparison data…</p>
          {:else if fovComparisonDataError}
            <p class="navis-export-error">{fovComparisonDataError}</p>
          {:else if !fovDebugFrame}
            <p class="subtle">No comparison frames are available.</p>
          {:else}
            <aside class="fov-debug-sidebar">
              <label>
                <span class="micro subtle">Segment</span>
                <select bind:value={fovDebugFrameIndex}>
                  {#each fovDebugFrames as frame, idx}
                    <option value={idx}>
                      {frame.segment_idx} · {fmtScore(frame.metrics?.soft_iou)} IoU
                    </option>
                  {/each}
                </select>
              </label>
              <div class="fov-debug-stat-grid">
                <div><span class="micro subtle">MAE</span><strong>{fmtScore(fovDebugFrame.metrics?.mae)}</strong></div>
                <div><span class="micro subtle">Tiles</span><strong>{fovDebugTileIds.length}</strong></div>
                <div><span class="micro subtle">Voxels</span><strong>{fovDebugPayload?.selected_voxels?.length || 0}</strong></div>
                <div><span class="micro subtle">Debug tiles</span><strong>{Object.keys(fovDebugPayload?.tiles || {}).length}</strong></div>
              </div>
              <div class="fov-debug-overlays">
                <span class="micro subtle">Overlays</span>
                <label><input type="checkbox" bind:checked={fovOverlayTrace} />Trace</label>
                <label><input type="checkbox" bind:checked={fovOverlayHorizons} />Horizons</label>
                <label><input type="checkbox" bind:checked={fovOverlayReachable} />Reachable</label>
                <label><input type="checkbox" bind:checked={fovOverlayPointSlice} />Point cloud</label>
                <label><input type="checkbox" bind:checked={fovOverlayVoxelStarts} />Starts</label>
                <label><input type="checkbox" bind:checked={fovOverlayDijkstraPaths} />Paths</label>
                <label><input type="checkbox" bind:checked={fovOverlayRays} />Rays</label>
                <label><input type="checkbox" bind:checked={fovOverlayHits} />Hits</label>
              </div>
              <label>
                <span class="micro subtle">Tile color</span>
                <select bind:value={fovDebugTileColorMode}>
                  <option value="prediction">Prediction</option>
                  <option value="truth">Truth</option>
                  <option value="combined">Pred | Truth</option>
                  <option value="error">Abs error</option>
                </select>
              </label>
              {#if fovDebugPayload}
                <div class="fov-debug-stat-grid">
                  <div><span class="micro subtle">Entropy</span><strong>{fmtScore(fovDebugDistributionStats.entropy_bits)}</strong></div>
                  <div><span class="micro subtle">Eff. voxels</span><strong>{fmtScore(fovDebugDistributionStats.effective_sample_size)}</strong></div>
                  <div><span class="micro subtle">Core mass</span><strong>{fmtScore(fovDebugDistributionStats.core_mass)}</strong></div>
                  <div><span class="micro subtle">Tail mass</span><strong>{fmtScore(fovDebugDistributionStats.tail_mass)}</strong></div>
                  <div><span class="micro subtle">No samples</span><strong>{fmtCount(fovDebugDistributionStats.active_tiles_without_sample_points)}</strong></div>
                </div>
                {#if fovDebugPayload.intentions?.horizons?.length}
                  <div class="fov-debug-horizons">
                    <span class="micro subtle">Prediction horizons</span>
                    {#each fovDebugPayload.intentions.horizons as horizon}
                      <div>
                        <b class="legend-dot" style={`background: ${horizonColor(horizon.idx, 1)}`}></b>
                        <span>{fmtScore(horizon.horizon_s)}s</span>
                        <span>conf {fmtScore(horizon.effective_confidence)}</span>
                        <span>mismatch {fmtScore(horizon.mismatch)}</span>
                      </div>
                    {/each}
                  </div>
                {/if}
                {#if fovDebugPayload.intentions?.reachable}
                  {@const reachable = fovDebugPayload.intentions.reachable}
                  <div class="fov-debug-stat-grid">
                    <div><span class="micro subtle">Reach r</span><strong>{fmtScore(reachable.radius_m)}</strong></div>
                    <div><span class="micro subtle">Horizon</span><strong>{fmtScore(reachable.horizon_s)}</strong></div>
                    <div><span class="micro subtle">Max speed</span><strong>{fmtScore(reachable.max_speed_m_s)}</strong></div>
                    <div><span class="micro subtle">Free/cand.</span><strong>{fmtCount(reachable.free_voxels)}/{fmtCount(reachable.candidate_voxels)}</strong></div>
                  </div>
                {/if}
              {/if}
              <label>
                <span class="micro subtle">Clicked tile</span>
                <select bind:value={fovDebugSelectedTileId}>
                  {#each fovDebugTileIds as tileId}
                    <option value={String(tileId)}>
                      T{tileId} · p {fmtScore(fovDebugFrame.predicted?.[tileId])} · t {fmtScore(fovDebugFrame.actual?.[tileId])}
                    </option>
                  {/each}
                </select>
              </label>
              {#if !fovDebugPayload}
                <p class="subtle">This run did not record visual predictor debug for this segment. Enable `visual_debug_enabled` in the predictor preset and regenerate the run.</p>
              {:else if !fovDebugSelectedTile}
                <p class="subtle">This tile was not among the retained top debug tiles. Increase `visual_debug_max_tiles` for broader inspection.</p>
              {:else}
                <div class="fov-debug-stat-grid">
                  <div><span class="micro subtle">Pred</span><strong>{fmtScore(fovDebugFrame.predicted?.[fovDebugSelectedTileId])}</strong></div>
                  <div><span class="micro subtle">Truth</span><strong>{fmtScore(fovDebugFrame.actual?.[fovDebugSelectedTileId])}</strong></div>
                  <div><span class="micro subtle">Gate</span><strong>{fmtScore(fovDebugSelectedTile.gate)}</strong></div>
                  <div><span class="micro subtle">Raw score</span><strong>{fmtScore(fovDebugSelectedTileStats.score)}</strong></div>
                  <div><span class="micro subtle">Contrib.</span><strong>{fmtCount(fovDebugSelectedTileStats.contributor_count)}</strong></div>
                  <div><span class="micro subtle">Vis rays</span><strong>{fmtScore(fovDebugSelectedTileStats.visibility_ratio)}</strong></div>
                  <div><span class="micro subtle">Top share</span><strong>{fmtScore(fovDebugSelectedTileStats.top_contributor_share)}</strong></div>
                  <div><span class="micro subtle">Blocked</span><strong>{fmtCount(fovDebugSelectedTileStats.blocked_rays)}</strong></div>
                </div>
                <div class="fov-debug-contributors">
                  <h3>Voxel Contributors</h3>
                  {#each fovDebugContributors as contributor}
                    <button
                      type="button"
                      class="fov-debug-contributor"
                      title={`voxel ${contributor.voxel?.join(', ')}`}
                    >
                      <span>T{fovDebugSelectedTileId} ← [{contributor.voxel?.join(', ')}]</span>
                      <span>{fmtScore(contributor.contribution)}</span>
                      <small>
                        p {fmtScore(contributor.probability)}
                        · w {fmtScore(contributor.estimator_weight)}
                        · {contributor.visible_rays}/{contributor.tested_rays} rays
                        · {contributor.stratum}
                      </small>
                    </button>
                  {/each}
                </div>
              {/if}
            </aside>
            <section class="fov-debug-map-panel">
              <svg class="fov-debug-map" viewBox={fovDebugViewBox} preserveAspectRatio="xMidYMid meet">
                {#each fovDebugTileIds as tileId}
                  {@const tile = fovDebugFrame.tiles?.[tileId]}
                  {@const bounds = tile?.bounds_xz || [0, 0, 0, 0]}
                  {@const x0 = Math.min(bounds[0], bounds[2])}
                  {@const y0 = Math.min(bounds[1], bounds[3])}
                  {@const tileW = Math.abs(bounds[2] - bounds[0])}
                  {@const tileH = Math.abs(bounds[3] - bounds[1])}
                  {#if fovDebugTileColorMode === 'combined'}
                    <rect
                      x={x0}
                      y={y0}
                      width={tileW * 0.5}
                      height={tileH}
                      fill={scoreColor(fovDebugFrame.predicted?.[tileId], String(tileId) === fovDebugSelectedTileId ? 0.92 : 0.72)}
                      pointer-events="none"
                    />
                    <rect
                      x={x0 + tileW * 0.5}
                      y={y0}
                      width={tileW * 0.5}
                      height={tileH}
                      fill={scoreColor(fovDebugFrame.actual?.[tileId], String(tileId) === fovDebugSelectedTileId ? 0.92 : 0.72)}
                      pointer-events="none"
                    />
                  {/if}
                  <rect
                    role="button"
                    tabindex="0"
                    aria-label={`Select tile ${tileId}`}
                    class:selected={String(tileId) === fovDebugSelectedTileId}
                    x={x0}
                    y={y0}
                    width={tileW}
                    height={tileH}
                    fill={fovDebugTileColorMode === 'combined' ? 'rgba(255, 255, 255, 0)' : tileColor(fovDebugFrame, tileId, String(tileId) === fovDebugSelectedTileId ? 0.92 : 0.72)}
                    stroke="rgba(15, 23, 42, 0.35)"
                    stroke-width={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.0015}
                    on:click={() => { fovDebugSelectedTileId = String(tileId); }}
                    on:keydown={(event) => handleFovDebugTileKeydown(event, tileId)}
                  />
                {/each}
                {#if fovOverlayPointSlice && fovDebugPointSlice?.points?.length}
                  {#each fovDebugPointSlice.points as point}
                    <circle
                      cx={point?.[0] || 0}
                      cy={point?.[1] || 0}
                      r={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.0026}
                      fill="rgba(15, 23, 42, 0.22)"
                    />
                  {/each}
                {/if}
                {#if fovOverlayReachable && fovDebugPayload?.intentions?.reachable}
                  {@const reachable = fovDebugPayload.intentions.reachable}
                  {#if reachable.center}
                    <circle
                      cx={reachable.center?.[0] || 0}
                      cy={reachable.center?.[2] || 0}
                      r={Math.max(0, Number(reachable.radius_m || 0))}
                      fill="rgba(34, 197, 94, 0.08)"
                      stroke="rgba(22, 163, 74, 0.62)"
                      stroke-width={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.004}
                      stroke-dasharray={`${Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.018} ${Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.012}`}
                    />
                  {/if}
                  {#each (reachable.sampled_free_voxels || []) as point}
                    <circle
                      cx={point?.[0] || 0}
                      cy={point?.[2] || 0}
                      r={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.0028}
                      fill="rgba(22, 163, 74, 0.22)"
                    />
                  {/each}
                {/if}
                {#if fovOverlayTrace}
                  <polyline
                    points={tracePolyline(fovDebugFrame.user_trace?.observed)}
                    fill="none"
                    stroke="rgba(37, 99, 235, 0.88)"
                    stroke-width={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.005}
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  />
                  <polyline
                    points={tracePolyline(fovDebugFrame.user_trace?.actual)}
                    fill="none"
                    stroke="rgba(245, 158, 11, 0.9)"
                    stroke-width={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.005}
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  />
                  {#each (fovDebugFrame.user_trace?.actual || []) as sample}
                    <circle
                      cx={sample.position?.[0] || 0}
                      cy={sample.position?.[2] || 0}
                      r={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.0045}
                      fill="rgba(245, 158, 11, 0.8)"
                    />
                  {/each}
                  {#if fovDebugFrame.user_trace?.current}
                    {@const current = fovDebugFrame.user_trace.current}
                    {@const end = arrowEnd(current, Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.08)}
                    <circle
                      cx={current.position?.[0] || 0}
                      cy={current.position?.[2] || 0}
                      r={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.0075}
                      fill="rgba(37, 99, 235, 0.92)"
                      stroke="rgba(15, 23, 42, 0.72)"
                      stroke-width={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.0018}
                    />
                    {#if end}
                      <line
                        x1={current.position?.[0] || 0}
                        y1={current.position?.[2] || 0}
                        x2={end.x}
                        y2={end.z}
                        stroke="rgba(30, 64, 175, 0.95)"
                        stroke-width={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.006}
                        stroke-linecap="round"
                      />
                    {/if}
                  {/if}
                {/if}
                {#if fovOverlayHorizons}
                  {#each (fovDebugPayload?.intentions?.horizons || []) as horizon}
                    {@const h = horizonEnd(horizon, Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1))}
                    {#if h}
                      <line
                        x1={h.x1}
                        y1={h.z1}
                        x2={h.x2}
                        y2={h.z2}
                        stroke={horizonColor(horizon.idx, 0.95)}
                        stroke-width={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.0055}
                        stroke-linecap="round"
                      />
                      <circle
                        cx={h.x2}
                        cy={h.z2}
                        r={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.008}
                        fill={horizonColor(horizon.idx, 0.95)}
                        stroke="white"
                        stroke-width={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.0018}
                      />
                    {/if}
                  {/each}
                {/if}
                {#if fovOverlayDijkstraPaths && fovDebugPathStart?.path?.length > 1}
                  {@const path = fovDebugPathStart.path}
                  {@const heightRange = pathHeightRange(path)}
                  {#each path.slice(1) as point, idx}
                    {@const prev = path[idx]}
                    <line
                      x1={prev?.[0] || 0}
                      y1={prev?.[2] || 0}
                      x2={point?.[0] || 0}
                      y2={point?.[2] || 0}
                      stroke={pathHeightColor((Number(prev?.[1]) + Number(point?.[1])) * 0.5, heightRange.minY, heightRange.maxY, 0.95)}
                      stroke-width={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.006}
                      stroke-linecap="round"
                    />
                  {/each}
                {/if}
                {#if fovOverlayVoxelStarts}
                  {#each (fovDebugPayload?.selected_voxels || []) as voxel}
                    {#if !fovOverlayDijkstraPaths || startVoxelKey(voxel) === fovDebugSelectedStartKey}
                      <circle
                        role="button"
                        tabindex="0"
                        class:start-point={true}
                        class:selected-start={startVoxelKey(voxel) === fovDebugSelectedStartKey}
                        aria-label={`Select start voxel ${startVoxelKey(voxel)}`}
                        cx={voxel.center?.[0] || 0}
                        cy={voxel.center?.[2] || 0}
                        r={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * (startVoxelKey(voxel) === fovDebugSelectedStartKey ? 0.009 : 0.006)}
                        fill={startVoxelKey(voxel) === fovDebugSelectedStartKey ? 'rgba(30, 64, 175, 0.92)' : 'rgba(30, 64, 175, 0.45)'}
                        stroke="rgba(30, 64, 175, 0.9)"
                        stroke-width={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.0015}
                        on:click={() => { fovDebugSelectedStartKey = startVoxelKey(voxel); }}
                        on:keydown={(event) => {
                          if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault();
                            fovDebugSelectedStartKey = startVoxelKey(voxel);
                          }
                        }}
                      />
                    {/if}
                  {/each}
                  {#each fovDebugContributors as contributor}
                    {#if !fovOverlayDijkstraPaths || startVoxelKey(contributor) === fovDebugSelectedStartKey}
                      <circle
                        cx={contributor.center?.[0] || 0}
                        cy={contributor.center?.[2] || 0}
                        r={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.010}
                        fill="rgba(2, 132, 199, 0.88)"
                        stroke="white"
                        stroke-width={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.002}
                      />
                    {/if}
                  {/each}
                {/if}
                {#each fovDebugContributors as contributor}
                  {#if !fovOverlayDijkstraPaths || startVoxelKey(contributor) === fovDebugSelectedStartKey}
                    {#if fovOverlayRays}
                      {#each (contributor.rays || []) as ray}
                        <line
                          x1={ray.start?.[0] || contributor.center?.[0] || 0}
                          y1={ray.start?.[2] || contributor.center?.[2] || 0}
                          x2={ray.target?.[0] || 0}
                          y2={ray.target?.[2] || 0}
                          stroke={ray.visible ? 'rgba(22, 163, 74, 0.82)' : 'rgba(220, 38, 38, 0.72)'}
                          stroke-width={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.003}
                          stroke-linecap="round"
                        />
                      {/each}
                    {/if}
                    {#if fovOverlayHits}
                      {#each (contributor.rays || []) as ray}
                        {#if ray.hit_center}
                          <circle
                            cx={ray.hit_center?.[0] || 0}
                            cy={ray.hit_center?.[2] || 0}
                            r={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.007}
                            fill={hitColor(ray.hit_kind)}
                            stroke="white"
                            stroke-width={Math.max(fovDebugBounds?.width || 1, fovDebugBounds?.height || 1) * 0.0018}
                          />
                        {/if}
                      {/each}
                    {/if}
                  {/if}
                {/each}
              </svg>
              {#if fovDebugScaleReference}
                {@const refMax = Math.max(fovDebugScaleReference.tileWidth || 0, fovDebugScaleReference.tileDepth || 0, fovDebugScaleReference.geometry || 0, fovDebugScaleReference.navigation || 0, fovDebugScaleReference.visibility || 0, 1e-6)}
                <div class="fov-debug-scale-reference">
                  <span class="micro subtle">Reference size</span>
                  <div>
                    <svg viewBox={`0 0 ${refMax} ${refMax}`} aria-hidden="true">
                      <rect x="0" y="0" width={fovDebugScaleReference.geometry || 0} height={fovDebugScaleReference.geometry || 0} class="scale-geometry" />
                    </svg>
                    <span>geometry voxel {fmtMeters(fovDebugScaleReference.geometry)} edge</span>
                  </div>
                  <div>
                    <svg viewBox={`0 0 ${refMax} ${refMax}`} aria-hidden="true">
                      <rect x="0" y="0" width={fovDebugScaleReference.navigation || 0} height={fovDebugScaleReference.navigation || 0} class="scale-navigation" />
                    </svg>
                    <span>navigation voxel {fmtMeters(fovDebugScaleReference.navigation)} edge</span>
                  </div>
                  <div>
                    <svg viewBox={`0 0 ${refMax} ${refMax}`} aria-hidden="true">
                      <rect x="0" y="0" width={fovDebugScaleReference.visibility || 0} height={fovDebugScaleReference.visibility || 0} class="scale-visibility" />
                    </svg>
                    <span>visibility voxel {fmtMeters(fovDebugScaleReference.visibility)} edge</span>
                  </div>
                  <div>
                    <svg viewBox={`0 0 ${refMax} ${refMax}`} aria-hidden="true">
                      <rect x="0" y="0" width={fovDebugScaleReference.tileWidth || 0} height={fovDebugScaleReference.tileDepth || 0} class="scale-tile" />
                    </svg>
                    <span>tile {fmtMeters(fovDebugScaleReference.tileWidth)} x {fmtMeters(fovDebugScaleReference.tileDepth)}</span>
                  </div>
                </div>
              {/if}
              <div class="fov-debug-map-footer">
                <span>Tile T{fovDebugSelectedTileId}</span>
                <span>pred {fmtScore(fovDebugFrame.predicted?.[fovDebugSelectedTileId])}</span>
                <span>truth {fmtScore(fovDebugFrame.actual?.[fovDebugSelectedTileId])}</span>
                {#if fovDebugTileColorMode === 'combined'}
                  <span>split left pred / right truth</span>
                {/if}
                <span>debug score {fmtScore(fovDebugSelectedTile?.score)}</span>
                <span><b class="legend-dot observed"></b>observed</span>
                <span><b class="legend-dot current"></b>current</span>
                <span><b class="legend-dot actual"></b>actual</span>
                <span><b class="legend-dot horizon"></b>horizon</span>
                <span><b class="legend-dot reachable"></b>reachable</span>
                {#if fovDebugPointSlice}
                  <span><b class="legend-dot point-slice"></b>mid slice {fmtCount(fovDebugPointSlice.point_count)}</span>
                {/if}
                <span><b class="legend-dot dijkstra-path"></b>Dijkstra path</span>
                <span><b class="legend-dot free-voxel"></b>free voxel</span>
                <span><b class="legend-dot start"></b>start</span>
                <span><b class="legend-dot occluder"></b>occluder</span>
              </div>
            </section>
          {/if}
        </div>
      {/if}
    </section>
  </div>
{/if}

<style>
  .fov-preview-backdrop {
    position: fixed;
    inset: 0;
    z-index: 1000;
    display: grid;
    place-items: center;
    padding: 1.25rem;
    background: rgba(15, 23, 42, 0.62);
  }

  .fov-preview-modal {
    width: min(1180px, 96vw);
    max-height: 92vh;
    display: grid;
    gap: 0.75rem;
    padding: 1rem;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    background: var(--card-bg);
    box-shadow: 0 24px 80px rgba(15, 23, 42, 0.28);
  }

  .fov-preview-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }

  .fov-preview-header h2 {
    margin: 0;
    font-size: 1rem;
  }

  .fov-preview-video {
    width: 100%;
    max-height: calc(92vh - 6rem);
    border-radius: 6px;
    background: #111827;
  }

  .fov-debug-shell {
    display: grid;
    grid-template-columns: minmax(250px, 320px) minmax(0, 1fr);
    gap: 1rem;
    min-height: min(720px, calc(92vh - 6rem));
  }

  .fov-debug-sidebar,
  .fov-debug-map-panel {
    min-width: 0;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    background: var(--card-bg-muted, rgba(248, 250, 252, 0.72));
  }

  .fov-debug-sidebar {
    display: grid;
    align-content: start;
    gap: 0.85rem;
    padding: 0.85rem;
    overflow: auto;
  }

  .fov-debug-sidebar label {
    display: grid;
    gap: 0.35rem;
  }

  .fov-debug-sidebar select {
    width: 100%;
  }

  .fov-debug-stat-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.5rem;
  }

  .fov-debug-stat-grid > div {
    display: grid;
    gap: 0.15rem;
    padding: 0.5rem;
    border: 1px solid var(--card-border);
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.72);
  }

  .fov-debug-overlays {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.45rem 0.65rem;
    padding: 0.55rem;
    border: 1px solid var(--card-border);
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.72);
  }

  .fov-debug-overlays > span {
    grid-column: 1 / -1;
  }

  .fov-debug-overlays label {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.8rem;
    color: var(--text);
  }

  .fov-debug-horizons {
    display: grid;
    gap: 0.35rem;
    padding: 0.55rem;
    border: 1px solid var(--card-border);
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.72);
  }

  .fov-debug-horizons > div {
    display: grid;
    grid-template-columns: auto 3.2rem 1fr 1fr;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.76rem;
    color: var(--text-subtle);
    font-variant-numeric: tabular-nums;
  }

  .fov-debug-contributors {
    display: grid;
    gap: 0.45rem;
  }

  .fov-debug-contributors h3 {
    margin: 0.25rem 0 0;
    font-size: 0.82rem;
  }

  .fov-debug-contributor {
    display: grid;
    gap: 0.2rem;
    width: 100%;
    padding: 0.55rem;
    border: 1px solid var(--card-border);
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.86);
    color: var(--text);
    text-align: left;
  }

  .fov-debug-contributor > span:first-child {
    font-weight: 650;
  }

  .fov-debug-contributor > span:nth-child(2) {
    color: #b91c1c;
    font-variant-numeric: tabular-nums;
  }

  .fov-debug-contributor small {
    color: var(--text-subtle);
    line-height: 1.35;
  }

  .fov-debug-map-panel {
    display: grid;
    grid-template-rows: minmax(0, 1fr) auto;
    padding: 0.75rem;
  }

  .fov-debug-map {
    width: 100%;
    height: 100%;
    min-height: 520px;
    border-radius: 6px;
    background: #ffffff;
  }

  .fov-debug-map rect {
    cursor: pointer;
  }

  .fov-debug-map *:focus,
  .fov-debug-map *:focus-visible {
    outline: none;
  }

  .fov-debug-map circle,
  .fov-debug-map line,
  .fov-debug-map polyline {
    pointer-events: none;
  }

  .fov-debug-map circle.start-point {
    pointer-events: auto;
    cursor: pointer;
  }

  .fov-debug-map rect.selected {
    stroke: #0f172a;
  }

  .fov-debug-scale-reference {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.45rem 0.8rem;
    padding-top: 0.65rem;
    color: var(--text-subtle);
    font-size: 0.78rem;
    font-variant-numeric: tabular-nums;
  }

  .fov-debug-scale-reference > div {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }

  .fov-debug-scale-reference svg {
    width: 2.2rem;
    height: 2.2rem;
    border: 1px solid rgba(148, 163, 184, 0.65);
    background: rgba(248, 250, 252, 0.9);
  }

  .fov-debug-scale-reference rect {
    vector-effect: non-scaling-stroke;
  }

  .fov-debug-scale-reference .scale-geometry {
    fill: rgba(100, 116, 139, 0.36);
    stroke: rgba(51, 65, 85, 0.9);
  }

  .fov-debug-scale-reference .scale-navigation {
    fill: rgba(37, 99, 235, 0.26);
    stroke: rgba(37, 99, 235, 0.9);
  }

  .fov-debug-scale-reference .scale-visibility {
    fill: rgba(22, 163, 74, 0.26);
    stroke: rgba(22, 163, 74, 0.9);
  }

  .fov-debug-scale-reference .scale-tile {
    fill: rgba(245, 158, 11, 0.22);
    stroke: rgba(217, 119, 6, 0.9);
  }

  .fov-debug-map-footer {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1rem;
    padding-top: 0.65rem;
    color: var(--text-subtle);
    font-size: 0.8rem;
    font-variant-numeric: tabular-nums;
  }

  .legend-dot {
    display: inline-block;
    width: 0.65rem;
    height: 0.65rem;
    margin-right: 0.3rem;
    border-radius: 50%;
    vertical-align: -0.05rem;
  }

  .legend-dot.observed {
    background: #2563eb;
  }

  .legend-dot.current {
    background: #1d4ed8;
    border: 1px solid #0f172a;
  }

  .legend-dot.actual {
    background: #f59e0b;
  }

  .legend-dot.horizon {
    background: #7c3aed;
  }

  .legend-dot.reachable {
    background: #16a34a;
  }

  .legend-dot.point-slice {
    background: rgba(15, 23, 42, 0.35);
    border: 1px solid rgba(15, 23, 42, 0.55);
  }

  .legend-dot.dijkstra-path {
    background: linear-gradient(90deg, #dc2626, #2563eb);
  }

  .legend-dot.free-voxel {
    background: rgba(22, 163, 74, 0.35);
    border: 1px solid rgba(22, 163, 74, 0.75);
  }

  .legend-dot.start {
    background: #0891b2;
  }

  .legend-dot.occluder {
    background: #dc2626;
  }

  .section-actions .active {
    border-color: var(--accent, #2563eb);
    color: var(--accent, #2563eb);
  }

  @media (max-width: 900px) {
    .fov-debug-shell {
      grid-template-columns: 1fr;
    }

    .fov-debug-map {
      min-height: 420px;
    }
  }
</style>
