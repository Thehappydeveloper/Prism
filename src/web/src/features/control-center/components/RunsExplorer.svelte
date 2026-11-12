<script lang="ts">
  import { onDestroy, tick } from 'svelte';
  import { fade, slide } from 'svelte/transition';
  import { getRunReadableSummary } from '../../../core/api';
  import { fovPredictorLabel } from '../../../core/predictorLabels';
  import type { RunReadableSummary, NavisExportResult, RunConfig, RunGroup, RunSummary } from '../../../core/types';
  import type { RunRow } from '../runExplorerUtils';

  export let runs: RunSummary[] = [];
  export let totalRuns = 0;
  export let canLoadMore = false;
  export let loadingMoreRuns = false;
  export let onLoadMoreRuns: () => void = () => {};
  export let filteredRunRows: RunRow[] = [];
  export let selectedRunIds = new Set<string>();
  export let selectedRunId = '';
  export let expandedRunIds = new Set<string>();
  export let loadingDetailIds = new Set<string>();
  export let runConfigCache: Record<string, RunConfig | null> = {};
  export let runAliases: Record<string, string> = {};
  export let navisExportingRunIds = new Set<string>();
  export let fovComparisonGeneratingRunIds = new Set<string>();
  export let runGroups: RunGroup[] = [];
  export let collapsedRunGroupIds = new Set<string>();
  export let loadingRunGroupIds = new Set<string>();
  export let runGroupSummaries: Record<string, GroupStats> = {};

  export let runSearch = '';
  export let statusFilter: 'all' | 'queued' | 'running' | 'paused' | 'finished' | 'failed' | 'canceled' = 'all';
  export let rowSort: 'newest' | 'oldest' | 'name' = 'newest';
  export let groupView: 'all' | 'batches' | 'ungrouped' = 'all';
  export let groupSort: 'newest' | 'oldest' | 'name' | 'size' = 'newest';

  export let onClearSelection: () => void = () => {};
  export let onCompareSelectedInControl: () => void = () => {};
  export let onAdvancedCompare: () => void = () => {};
  export let onDeleteSelected: () => void = () => {};
  export let onCreateGroupFromSelected: () => void = () => {};
  export let onMoveSelectedToGroup: () => void = () => {};
  export let onRemoveSelectedFromGroups: () => void = () => {};
  export let onToggleRunGroup: (groupId: string) => void = () => {};
  export let onRenameRunGroup: (groupId: string) => void = () => {};
  export let onDeleteRunGroup: (groupId: string) => void = () => {};
  export let onDeleteRunGroupWithRuns: (groupId: string) => void = () => {};
  export let onControlRunGroup: (groupId: string, action: 'pause' | 'resume' | 'cancel' | 'force_cancel') => void = () => {};
  export let onCompareRunGroup: (runIds: string[], advanced?: boolean) => void = () => {};
  export let onDropRunOnGroup: (runId: string, groupId: string) => void = () => {};
  export let onDropGroupOnGroup: (sourceGroupId: string, targetGroupId: string) => void = () => {};

  export let onRowCheckboxClick: (rowId: string, idx: number, event: MouseEvent) => void = () => {};
  export let onRowSelectClick: (rowId: string, idx: number, event: MouseEvent) => void = () => {};
  export let onRowSelectKeydown: (rowId: string, idx: number, event: KeyboardEvent) => void = () => {};
  export let onInspectRun: (runId: string) => void = () => {};
  export let onToggleRunDetails: (runId: string) => void = () => {};
  export let onRemoveRun: (runId: string) => void = () => {};
  export let onOpenInNavis: (runId: string) => void = () => {};
  export let onGenerateFovComparison: (runId: string) => void = () => {};
  export let onPreviewFovComparison: (runId: string) => void = () => {};
  export let onQuickRenameRun: (runId: string) => void = () => {};
  export let onAliasChange: (runId: string, event: Event) => void = () => {};
  export let aliasMaxLength = 32;

  export let formatRuntime: (sec: number | null) => string = () => '—';
  export let formatDateTime: (value: string | undefined) => string = () => '—';
  export let statusDotClass: (status: string) => string = () => 'status-dot unknown';
  export let basenamePath: (path: string | null | undefined) => string = () => '—';

  type ProgressRow = {
    id: string;
    label: string;
    current: number;
    total: number;
    size: number;
    startedTs: number | null;
    etaText: string;
    phase: string;
    detailText: string;
    detailItems: string[];
    phaseCounts: Record<string, number>;
    heartbeatText: string;
  };

  type GroupedRunSection = {
    id: string;
    name: string;
    source: string;
    archived: boolean;
    accent: string;
    stats: GroupStats;
    rows: Array<{ row: RunRow; idx: number }>;
  };

  type GroupStats = {
    total: number;
    queued: number;
    running: number;
    paused: number;
    finished: number;
    failed: number;
    canceled: number;
    progressCurrent?: number;
    progressTotal?: number;
  };

  const ungroupedSectionId = '__ungrouped__';

  let listHeightPx = 240;
  let queueHeightPx = 72;
  const minListHeightPx = 96;
  const maxListHeightPx = 1200;
  const minQueueHeightPx = 42;
  const maxQueueHeightPx = 360;
  const defaultVisibleRows = 3.5;
  const resizeHandleHeightPx = 12;
  let hasManualResize = false;
  let listEl: HTMLDivElement | null = null;
  let resizing = false;
  let queueResizing = false;
  let resizeStartY = 0;
  let resizeStartHeight = 0;
  let navisLogRunId = '';
  let navisLogResult: NavisExportResult | null = null;
  let readableRunId = '';
  let readableResult: RunReadableSummary | null = null;
  let readableError = '';
  let readableLoading = false;
  let readableMode: 'run' | 'predictor' = 'run';
  let draggedRunId = '';
  let draggedGroupId = '';
  let dragOverGroupId = '';
  let pendingDrop: { kind: 'run' | 'group'; sourceId: string; targetId: string } | null = null;
  $: runGroupById = new Map(runGroups.map((group) => [group.id, group]));
  $: groupedRunSections = buildGroupedRunSections(filteredRunRows, runGroups, runGroupSummaries, false, groupView, groupSort);
  $: queueRuns = [...runs]
    .filter((run) => ['queued', 'running', 'paused'].includes(String(run.status || '').toLowerCase()))
    .sort((a, b) => {
      const aRunning = String(a.status || '').toLowerCase() === 'running';
      const bRunning = String(b.status || '').toLowerCase() === 'running';
      if (aRunning !== bRunning) return aRunning ? -1 : 1;
      const aPaused = String(a.status || '').toLowerCase() === 'paused';
      const bPaused = String(b.status || '').toLowerCase() === 'paused';
      if (aPaused !== bPaused) return aPaused ? -1 : 1;
      return (a.queue_position ?? 999_999) - (b.queue_position ?? 999_999);
    });
  $: progressRows = buildProgressRows(queueRuns);

  function clampHeight(nextHeight: number): number {
    const viewportCap = typeof window !== 'undefined'
      ? Math.max(minListHeightPx, window.innerHeight - 220)
      : maxListHeightPx;
    const maxHeight = Math.min(maxListHeightPx, viewportCap);
    return Math.max(minListHeightPx, Math.min(maxHeight, nextHeight));
  }

  function onListScroll(event: Event): void {
    if (!canLoadMore || loadingMoreRuns) return;
    const element = event.currentTarget as HTMLElement;
    if (element.scrollHeight - element.scrollTop - element.clientHeight < 160) onLoadMoreRuns();
  }

  function clampQueueHeight(nextHeight: number): number {
    const viewportCap = typeof window !== 'undefined'
      ? Math.max(minQueueHeightPx, Math.round(window.innerHeight * 0.34))
      : maxQueueHeightPx;
    const maxHeight = Math.min(maxQueueHeightPx, viewportCap);
    return Math.max(minQueueHeightPx, Math.min(maxHeight, nextHeight));
  }

  function stopResizing(): void {
    if (!resizing) return;
    resizing = false;
    window.removeEventListener('pointermove', onResizePointerMove);
    window.removeEventListener('pointerup', stopResizing);
    window.removeEventListener('pointercancel', stopResizing);
    document.body.classList.remove('runs-resizing');
  }

  function stopQueueResizing(): void {
    if (!queueResizing) return;
    queueResizing = false;
    window.removeEventListener('pointermove', onQueueResizePointerMove);
    window.removeEventListener('pointerup', stopQueueResizing);
    window.removeEventListener('pointercancel', stopQueueResizing);
    document.body.classList.remove('runs-resizing');
  }

  function onResizePointerMove(event: PointerEvent): void {
    if (!resizing) return;
    const deltaY = event.clientY - resizeStartY;
    listHeightPx = clampHeight(resizeStartHeight + deltaY);
  }

  function onResizePointerDown(event: PointerEvent): void {
    if (event.button !== 0) return;
    hasManualResize = true;
    resizing = true;
    resizeStartY = event.clientY;
    resizeStartHeight = listHeightPx;
    window.addEventListener('pointermove', onResizePointerMove);
    window.addEventListener('pointerup', stopResizing);
    window.addEventListener('pointercancel', stopResizing);
    document.body.classList.add('runs-resizing');
    event.preventDefault();
  }

  function onQueueResizePointerMove(event: PointerEvent): void {
    if (!queueResizing) return;
    const deltaY = event.clientY - resizeStartY;
    queueHeightPx = clampQueueHeight(resizeStartHeight + deltaY);
  }

  function onQueueResizePointerDown(event: PointerEvent): void {
    if (event.button !== 0) return;
    queueResizing = true;
    resizeStartY = event.clientY;
    resizeStartHeight = queueHeightPx;
    window.addEventListener('pointermove', onQueueResizePointerMove);
    window.addEventListener('pointerup', stopQueueResizing);
    window.addEventListener('pointercancel', stopQueueResizing);
    document.body.classList.add('runs-resizing');
    event.preventDefault();
  }

  function onResizeHandleKeydown(event: KeyboardEvent): void {
    if (event.key === 'ArrowUp') {
      hasManualResize = true;
      listHeightPx = clampHeight(listHeightPx - 24);
      event.preventDefault();
    } else if (event.key === 'ArrowDown') {
      hasManualResize = true;
      listHeightPx = clampHeight(listHeightPx + 24);
      event.preventDefault();
    }
  }

  function onQueueResizeHandleKeydown(event: KeyboardEvent): void {
    if (event.key === 'ArrowUp') {
      queueHeightPx = clampQueueHeight(queueHeightPx - 18);
      event.preventDefault();
    } else if (event.key === 'ArrowDown') {
      queueHeightPx = clampQueueHeight(queueHeightPx + 18);
      event.preventDefault();
    }
  }

  function computeAutoHeight(): number {
    if (!listEl) return clampHeight(240);
    const style = window.getComputedStyle(listEl);
    const gap = Number.parseFloat(style.gap || style.rowGap || '8') || 8;
    const padTop = Number.parseFloat(style.paddingTop || '0') || 0;
    const padBottom = Number.parseFloat(style.paddingBottom || '0') || 0;
    const firstCard = listEl.querySelector('.run-card-compact') as HTMLElement | null;
    const rowHeight = firstCard?.getBoundingClientRect().height || 52;

    if (!filteredRunRows.length) {
      return clampHeight(padTop + padBottom + rowHeight + resizeHandleHeightPx);
    }

    const visibleRows = Math.min(defaultVisibleRows, filteredRunRows.length);
    const gapCount = Math.max(0, Math.ceil(visibleRows) - 1);
    const contentHeight = padTop + padBottom + (rowHeight * visibleRows) + (gap * gapCount);
    return clampHeight(contentHeight + resizeHandleHeightPx);
  }

  $: {
    if (hasManualResize) {
      filteredRunRows;
    } else {
      filteredRunRows;
      void tick().then(() => {
        listHeightPx = computeAutoHeight();
      });
    }
  }

  onDestroy(() => {
    stopResizing();
    stopQueueResizing();
  });

  function canOpenInNavis(run: RunSummary): boolean {
    const status = String(run.status || '').toLowerCase();
    if (status === 'finished') return true;
    if (status === 'unknown' && run.events_path) return true;
    return false;
  }

  function canGenerateFovComparison(run: RunSummary): boolean {
    const status = String(run.status || '').toLowerCase();
    // A live status update can mark a run finished before its list-row
    // metadata is refreshed with events_path.  The server is authoritative
    // about the event artifact (including compressed events.json.gz logs), so
    // do not leave a completed run's comparison action incorrectly disabled.
    if (status === 'finished') return true;
    if (status === 'unknown' && run.events_path) return true;
    return false;
  }

  function navisExportState(result: NavisExportResult | undefined): string {
    if (!result) return 'idle';
    return result.ok ? 'success' : 'failed';
  }

  function navisExportTitle(result: NavisExportResult | undefined): string {
    if (!result) return 'No Navis export recorded.';
    return result.ok ? 'Last Navis export succeeded.' : 'Last Navis export failed.';
  }

  function fovComparisonTitle(run: RunSummary): string {
    if (fovComparisonGeneratingRunIds.has(run.run_id)) return 'Generating FoV comparison video.';
    const result = run.fov_comparison;
    if (result?.error) return result.error;
    if (result?.video_path) return 'FoV comparison video is available.';
    if (result?.data_path) return 'FoV comparison data is available; video encoding did not finish.';
    if (!canGenerateFovComparison(run)) return 'FoV comparison requires a completed run with an event log.';
    return 'Generate FoV comparison video.';
  }

  function fovComparisonState(run: RunSummary): string {
    if (fovComparisonGeneratingRunIds.has(run.run_id)) return 'running';
    if (run.fov_comparison?.error || run.fov_comparison?.ok === false) return 'failed';
    if (run.fov_comparison?.video_path || run.fov_comparison?.data_path) return 'success';
    return 'idle';
  }

  function openNavisLog(runId: string, result: NavisExportResult | undefined): void {
    if (!result) return;
    navisLogRunId = runId;
    navisLogResult = result;
  }

  function closeNavisLog(): void {
    navisLogRunId = '';
    navisLogResult = null;
  }

  function navisLogText(result: NavisExportResult | null): string {
    if (!result) return '';
    return JSON.stringify(result, null, 2);
  }

  async function openReadableSummary(runId: string, mode: 'run' | 'predictor' = 'run'): Promise<void> {
    if (readableLoading) return;
    readableLoading = true;
    readableRunId = runId;
    readableMode = mode;
    readableResult = null;
    readableError = '';
    try {
      const result = await getRunReadableSummary(runId);
      if (readableRunId !== runId) return;
      readableResult = result;
    } catch (error) {
      if (readableRunId !== runId) return;
      readableError = error instanceof Error ? error.message : String(error);
    } finally {
      if (readableRunId === runId) {
        readableLoading = false;
      }
    }
  }

  function closeReadableSummary(): void {
    readableRunId = '';
    readableResult = null;
    readableError = '';
    readableLoading = false;
    readableMode = 'run';
  }

  function fixed(value: number | null | undefined, digits = 2): string {
    if (value === null || value === undefined || !Number.isFinite(value)) return '—';
    return value.toFixed(digits);
  }

  function seconds(value: number | null | undefined): string {
    if (value === null || value === undefined || !Number.isFinite(value)) return '—';
    return `${value.toFixed(2)}s`;
  }

  function fovTileText(tiles: Array<{ tile_id: string; visibility: number }>): string {
    if (!tiles.length) return '—';
    return tiles.map((tile) => `T${tile.tile_id} ${fixed(tile.visibility, 2)}`).join(', ');
  }

  function queueRunTitle(run: RunSummary): string {
    const cfg = run.config_summary || {};
    const parts = [
      cfg.abr || 'ABR?',
      cfg.predictor && cfg.predictor !== 'None' ? fovPredictorLabel(cfg.predictor) : 'No FoV predictor',
      cfg.bandwidth_predictor && cfg.bandwidth_predictor !== 'None' ? cfg.bandwidth_predictor : 'No bandwidth predictor',
    ];
    if (cfg.oracle_mode && cfg.oracle_mode !== 'none') parts.push(`oracle: ${cfg.oracle_mode}`);
    return parts.join(' · ');
  }

  function queueTraceText(run: RunSummary): string {
    const cfg = run.config_summary || {};
    const network = cfg.network_profile ? `profile:${cfg.network_profile}` : basenamePath(cfg.network_trace);
    return `${network} / ${basenamePath(cfg.fov_trace)}`;
  }

  function queueCompactText(run: RunSummary): string {
    const cfg = run.config_summary || {};
    const alias = runAliases[run.run_id] || run.run_id;
    const status = String(run.status || '').toLowerCase() === 'running' ? 'Run' : `Q${run.queue_position ?? '?'}`;
    const abr = cfg.abr || 'ABR?';
    const predictor = cfg.predictor && cfg.predictor !== 'None' ? fovPredictorLabel(cfg.predictor) : 'NoFoV';
    const network = cfg.network_profile ? `P:${cfg.network_profile}` : basenamePath(cfg.network_trace);
    return `${status} · ${alias} · ${abr} · ${predictor} · ${network}`;
  }

  function progressPercent(current: number, total: number): number {
    if (!Number.isFinite(total) || total <= 0) return 0;
    return Math.max(0, Math.min(100, (Number(current || 0) / Number(total)) * 100));
  }

  function parseRunTime(value: string | undefined): number | null {
    if (!value) return null;
    const ts = Date.parse(value);
    return Number.isFinite(ts) ? ts : null;
  }

  function formatEta(secondsLeft: number | null): string {
    if (secondsLeft === null || !Number.isFinite(secondsLeft) || secondsLeft < 0) return 'ETA —';
    if (secondsLeft < 60) return `ETA ${Math.max(1, Math.round(secondsLeft))}s`;
    const minutes = Math.floor(secondsLeft / 60);
    const seconds = Math.round(secondsLeft % 60);
    if (minutes < 60) return `ETA ${minutes}m ${seconds}s`;
    const hours = Math.floor(minutes / 60);
    const remMinutes = minutes % 60;
    return `ETA ${hours}h ${remMinutes}m`;
  }

  function estimateEtaText(current: number, total: number, startedTs: number | null): string {
    if (!startedTs || !Number.isFinite(current) || !Number.isFinite(total) || current <= 0 || total <= current) {
      return total > 0 && current >= total ? 'Done' : 'ETA —';
    }
    const elapsedS = Math.max(0, (Date.now() - startedTs) / 1000);
    if (elapsedS <= 0) return 'ETA —';
    const segmentRate = current / elapsedS;
    if (segmentRate <= 0) return 'ETA —';
    return formatEta((total - current) / segmentRate);
  }

  function phaseLabel(value: string | undefined | null): string {
    return String(value || 'running').replace(/_/g, ' ');
  }

  function heartbeatAgeText(value: string | undefined | null): string {
    const ts = parseRunTime(value || undefined);
    if (ts === null) return 'heartbeat —';
    const ageS = Math.max(0, (Date.now() - ts) / 1000);
    if (ageS < 60) return `heartbeat ${Math.round(ageS)}s ago`;
    const ageM = Math.floor(ageS / 60);
    const remS = Math.round(ageS % 60);
    return `heartbeat ${ageM}m ${remS}s ago`;
  }

  function progressDetailText(progress: RunSummary['progress']): string {
    return progressDetailItems(progress).join(' · ');
  }

  function formatProgressDetailKey(value: string): string {
    return value.replace(/_/g, ' ');
  }

  function formatProgressDetailValue(value: string | number | boolean | null | undefined): string {
    if (value === undefined) return 'unknown';
    if (value === null) return 'null';
    if (typeof value === 'boolean') return value ? 'yes' : 'no';
    return String(value);
  }

  function progressDetailItems(progress: RunSummary['progress']): string[] {
    const detail = progress?.detail || {};
    const preferredKeys = [
      'dataset_id',
      'segment_idx',
      'segment_total',
      'segment_runs',
      'segment_workers',
      'frame_count',
      'needs_ply',
      'predictor',
      'abr',
      'network',
      'reason',
      'previous_status',
      'status',
    ];
    const keys = [
      ...preferredKeys.filter((key) => detail[key] !== undefined),
      ...Object.keys(detail).filter((key) => !preferredKeys.includes(key)).sort(),
    ];
    return keys.map((key) => {
      const value = detail[key];
      if (key === 'segment_idx') return `segment ${formatProgressDetailValue(value)}`;
      if (key === 'segment_total') return `total ${formatProgressDetailValue(value)}`;
      if (key === 'segment_runs') return `${formatProgressDetailValue(value)} runs in step`;
      if (key === 'segment_workers') return `${formatProgressDetailValue(value)} workers`;
      if (key === 'frame_count') return `${formatProgressDetailValue(value)} frames`;
      if (key === 'needs_ply') return `PLY ${formatProgressDetailValue(value)}`;
      return `${formatProgressDetailKey(key)} ${formatProgressDetailValue(value)}`;
    });
  }

  function phaseCountsText(phaseCounts: Record<string, number>): string {
    return Object.entries(phaseCounts)
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([phase, count]) => `${phase} ${count}`)
      .join(' · ');
  }

  function buildProgressRows(activeRuns: RunSummary[]): ProgressRow[] {
    const rows = new Map<string, ProgressRow>();
    for (const run of activeRuns) {
      if (!['running', 'paused'].includes(String(run.status || '').toLowerCase())) continue;
      const progress = run.progress;
      if (!progress || !progress.total) continue;
      const groupId = progress.group_id || run.run_id;
      const current = Number(progress.current || 0);
      const total = Number(progress.total || 0);
      const startedTs = parseRunTime(run.started);
      const phase = phaseLabel(progress.phase);
      const detailItems = progressDetailItems(progress);
      const detailText = detailItems.join(' · ');
      const heartbeatText = heartbeatAgeText(progress.heartbeat_at);
      const existing = rows.get(groupId);
      if (existing) {
        existing.current = Math.min(existing.current, current);
        existing.total = Math.max(existing.total, total);
        existing.size = Math.max(existing.size, Number(progress.group_size || existing.size));
        existing.phaseCounts[phase] = (existing.phaseCounts[phase] || 0) + 1;
        existing.phase = phaseCountsText(existing.phaseCounts) || phase;
        if (!existing.detailText || detailItems.length > existing.detailItems.length) {
          existing.detailText = detailText;
          existing.detailItems = detailItems;
        }
        existing.heartbeatText = heartbeatText;
        if (startedTs !== null) {
          existing.startedTs = existing.startedTs === null ? startedTs : Math.min(existing.startedTs, startedTs);
        }
      } else {
        rows.set(groupId, {
          id: groupId,
          label: progress.group_id ? `Grouped run (${progress.group_size || 1})` : (runAliases[run.run_id] || run.run_id),
          current,
          total,
          size: Number(progress.group_size || 1),
          startedTs,
          etaText: 'ETA —',
          phase,
          detailText,
          detailItems,
          phaseCounts: { [phase]: 1 },
          heartbeatText,
        });
      }
    }
    return Array.from(rows.values())
      .map((row) => ({ ...row, etaText: estimateEtaText(row.current, row.total, row.startedTs) }))
      .sort((a, b) => a.label.localeCompare(b.label));
  }

  function buildGroupedRunSections(
    rows: RunRow[], runGroups: RunGroup[], summaries: Record<string, GroupStats>, includeArchived: boolean,
    view: 'all' | 'batches' | 'ungrouped', sort: 'newest' | 'oldest' | 'name' | 'size',
  ): GroupedRunSection[] {
    const sections: GroupedRunSection[] = [];
    const byId = new Map<string, GroupedRunSection>();
    const groups = new Map(runGroups.map((group) => [group.id, group]));
    // Group metadata has the authoritative membership list, so headers show
    // the correct batch size before its runs are retrieved on expansion.
    runGroups.forEach((group) => {
      // Batch planning has historically left harmless, empty registry entries
      // behind (for example repeated "Sweep_1" drafts). They are not runs and
      // should not look like empty experiment batches in the browser.
      const summary = summaries[group.id];
      if (view === 'ungrouped' || !summary?.total || (group.archived && !includeArchived)) return;
      const section = {
        id: group.id, name: group.name, source: group.source || 'manual', archived: Boolean(group.archived),
        accent: groupAccent(group.id), stats: { ...summary }, rows: [],
      };
      byId.set(group.id, section);
      sections.push(section);
    });
    rows.forEach((row, idx) => {
      const groupId = row.run.run_group_id || '';
      if ((view === 'batches' && !groupId) || (view === 'ungrouped' && groupId)) return;
      const group = groupId ? groups.get(groupId) : undefined;
      const archived = Boolean(group?.archived);
      const sectionId = groupId && (!archived || includeArchived) ? groupId : ungroupedSectionId;
      const name = sectionId === ungroupedSectionId ? 'Ungrouped runs' : (group?.name || row.run.run_group_name || groupId);
      const source = sectionId === ungroupedSectionId ? '' : (group?.source || row.run.run_group_source || 'manual');
      if (!byId.has(sectionId)) {
        const section = { id: sectionId, name, source, archived, accent: groupAccent(sectionId), stats: emptyGroupStats(), rows: [] };
        byId.set(sectionId, section);
        sections.push(section);
      }
      const section = byId.get(sectionId)!;
      section.rows.push({ row, idx });
      if (!summaries[sectionId]) section.stats = addRunToGroupStats(section.stats, row.run);
    });
    return sections.sort((a, b) => {
      const aGroup = groups.get(a.id);
      const bGroup = groups.get(b.id);
      if (sort === 'name') return a.name.localeCompare(b.name);
      if (sort === 'size') return groupRunIds(a).length - groupRunIds(b).length;
      // An ungrouped section has no group creation time, so order it by its
      // newest contained run. This keeps single runs and batch groups in one
      // coherent timeline for newest/oldest views.
      const sectionTime = (section: GroupedRunSection, group: RunGroup | undefined): number => {
        const groupTime = Date.parse(group?.created_at || '');
        if (Number.isFinite(groupTime)) return groupTime;
        return Math.max(0, ...section.rows.map((entry) => entry.row.startedTs ?? 0));
      };
      const aTime = sectionTime(a, aGroup);
      const bTime = sectionTime(b, bGroup);
      return sort === 'oldest' ? aTime - bTime : bTime - aTime;
    });
  }

  function emptyGroupStats(): GroupStats {
    return {
      total: 0,
      queued: 0,
      running: 0,
      paused: 0,
      finished: 0,
      failed: 0,
      canceled: 0,
      progressCurrent: 0,
      progressTotal: 0,
    };
  }

  function addRunToGroupStats(stats: GroupStats, run: RunSummary): GroupStats {
    const next = {
      ...stats,
      total: stats.total + 1,
      progressCurrent: stats.progressCurrent ?? 0,
      progressTotal: stats.progressTotal ?? 0,
    };
    const status = String(run.status || '').toLowerCase();
    if (status === 'queued') next.queued += 1;
    else if (status === 'running') next.running += 1;
    else if (status === 'paused') next.paused += 1;
    else if (status === 'finished') next.finished += 1;
    else if (status === 'failed') next.failed += 1;
    else if (status === 'canceled') next.canceled += 1;
    if (run.progress?.total) {
      next.progressCurrent += Number(run.progress.current || 0);
      next.progressTotal += Number(run.progress.total || 0);
    }
    return next;
  }

  function groupAccent(groupId: string): string {
    let hash = 0;
    for (const char of groupId) hash = ((hash << 5) - hash + char.charCodeAt(0)) | 0;
    const hues = [188, 204, 142, 32, 330, 262, 166, 16];
    const hue = hues[Math.abs(hash) % hues.length];
    return `${hue} 78% 58%`;
  }

  function groupStyle(section: GroupedRunSection): string {
    return `--group-accent: ${section.accent};`;
  }

  function groupProgressPercent(section: GroupedRunSection): number {
    const total = section.stats.progressTotal ?? 0;
    if (!total) return 0;
    return Math.max(0, Math.min(100, ((section.stats.progressCurrent ?? 0) / total) * 100));
  }

  function groupStatusText(stats: GroupStats): string {
    const parts = [
      stats.running ? `${stats.running} running` : '',
      stats.paused ? `${stats.paused} paused` : '',
      stats.queued ? `${stats.queued} queued` : '',
      stats.finished ? `${stats.finished} finished` : '',
      stats.failed ? `${stats.failed} failed` : '',
      stats.canceled ? `${stats.canceled} canceled` : '',
    ].filter(Boolean);
    return parts.join(' · ') || `${stats.total} run${stats.total === 1 ? '' : 's'}`;
  }

  function groupHasActive(stats: GroupStats): boolean {
    return Boolean(stats.running || stats.paused || stats.queued);
  }

  function groupRunIds(section: GroupedRunSection): string[] {
    return runGroupById.get(section.id)?.run_ids || section.rows.map((entry) => entry.row.run.run_id);
  }

  function groupRunCount(section: GroupedRunSection): number {
    return runGroupSummaries[section.id]?.total ?? groupRunIds(section).length;
  }

  function onRunDragStart(runId: string, event: DragEvent): void {
    event.stopPropagation();
    draggedRunId = runId;
    draggedGroupId = '';
    event.dataTransfer?.setData('text/plain', runId);
    if (event.dataTransfer) event.dataTransfer.effectAllowed = 'move';
  }

  function onGroupDragStart(groupId: string, event: DragEvent): void {
    draggedGroupId = groupId;
    draggedRunId = '';
    event.dataTransfer?.setData('application/x-prism-run-group', groupId);
    if (!event.dataTransfer) return;
    event.dataTransfer.effectAllowed = 'move';

    // Do not let the browser snapshot the whole grouped-list layout as the
    // drag image.  A deliberately tiny preview makes it clear that only this
    // batch is moving.
    const group = runGroupById.get(groupId);
    const preview = document.createElement('div');
    const dot = document.createElement('span');
    const label = document.createElement('strong');
    const chip = document.createElement('span');
    label.textContent = group?.name || groupId;
    chip.textContent = `${group?.run_ids.length || 0} runs`;
    dot.style.cssText = 'width:9px;height:9px;border-radius:999px;background:#22d3ee;border:1px solid rgba(255,255,255,.3)';
    label.style.cssText = 'min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap';
    chip.style.cssText = 'padding:2px 7px;border:1px solid rgba(255,255,255,.15);border-radius:999px;color:#cbd5e1;font-size:11px;white-space:nowrap';
    preview.append(dot, label, chip);
    preview.style.cssText = [
      'position:fixed', 'top:-1000px', 'left:-1000px', 'z-index:-1',
      'display:grid', 'grid-template-columns:auto minmax(0,1fr) auto', 'align-items:center', 'gap:7px',
      'box-sizing:border-box', 'width:360px', 'padding:8px 10px', 'border-radius:10px',
      'background:#172033', 'border:1px solid rgba(255,255,255,.12)', 'border-left:3px solid #22d3ee', 'color:#e2e8f0',
      'font:600 13px system-ui,sans-serif', 'box-shadow:0 5px 16px rgba(0,0,0,.28)',
    ].join(';');
    document.body.appendChild(preview);
    event.dataTransfer.setDragImage(preview, 12, 14);
    requestAnimationFrame(() => preview.remove());
  }

  function onGroupDrop(groupId: string, event: DragEvent): void {
    event.preventDefault();
    const sourceGroupId = event.dataTransfer?.getData('application/x-prism-run-group') || draggedGroupId;
    const runId = event.dataTransfer?.getData('text/plain') || draggedRunId;
    draggedRunId = '';
    draggedGroupId = '';
    dragOverGroupId = '';
    if (!groupId || groupId === ungroupedSectionId) return;
    if (sourceGroupId && sourceGroupId !== groupId) {
      pendingDrop = { kind: 'group', sourceId: sourceGroupId, targetId: groupId };
    } else if (runId) {
      pendingDrop = { kind: 'run', sourceId: runId, targetId: groupId };
    }
  }

  function onGroupDragEnter(groupId: string): void {
    if ((draggedRunId || draggedGroupId) && groupId !== ungroupedSectionId) dragOverGroupId = groupId;
  }

  function confirmPendingDrop(): void {
    if (!pendingDrop) return;
    const action = pendingDrop;
    pendingDrop = null;
    if (action.kind === 'group') onDropGroupOnGroup(action.sourceId, action.targetId);
    else onDropRunOnGroup(action.sourceId, action.targetId);
  }

  function pendingDropText(): string {
    if (!pendingDrop) return '';
    const target = runGroupById.get(pendingDrop.targetId)?.name || pendingDrop.targetId;
    if (pendingDrop.kind === 'run') return `Move run ${pendingDrop.sourceId} into “${target}”?`;
    const source = runGroupById.get(pendingDrop.sourceId);
    const count = source?.run_ids.length || 0;
    return `Move all ${count} runs from “${source?.name || pendingDrop.sourceId}” into “${target}”?`;
  }

  function onGroupDragLeave(groupId: string, event: DragEvent): void {
    const current = event.currentTarget as HTMLElement | null;
    const next = event.relatedTarget as Node | null;
    if (current && next && current.contains(next)) return;
    if (dragOverGroupId === groupId) dragOverGroupId = '';
  }
</script>

<section class="card runs-explorer-card">
  <div class="runs-action-bar">
    <h2>Runs</h2>
    <div class="section-actions">
      <button class="ghost small" title="Clear selection" aria-label="Clear selection" on:click={onClearSelection}>Clear</button>
      <button class="ghost small" title="Compare selected runs" aria-label="Compare selected runs" on:click={onCompareSelectedInControl}>Compare</button>
      <button class="ghost small" title="Open advanced comparison" aria-label="Open advanced comparison" on:click={onAdvancedCompare}>Advanced</button>
      <button class="ghost small" title="Create group from selected runs" aria-label="Create group from selected runs" on:click={onCreateGroupFromSelected} disabled={!selectedRunIds.size}>Group</button>
      <button class="ghost small" title="Move selected runs to a group" aria-label="Move selected runs to a group" on:click={onMoveSelectedToGroup} disabled={!selectedRunIds.size}>Move</button>
      <button class="ghost small" title="Remove selected runs from their groups" aria-label="Remove selected runs from their groups" on:click={onRemoveSelectedFromGroups} disabled={!selectedRunIds.size}>Ungroup</button>
      <button class="ghost small" title="Delete selected runs" aria-label="Delete selected runs" on:click={onDeleteSelected}>Delete</button>
    </div>
  </div>

  <div class="runs-compact-toolbar">
    <label class="compact-search">Search
      <input placeholder="Run id, alias, status, ABR…" bind:value={runSearch} />
    </label>
    <label>Status
      <select bind:value={statusFilter}>
        <option value="all">All</option>
        <option value="queued">Queued</option>
        <option value="running">Running</option>
        <option value="paused">Paused</option>
        <option value="finished">Finished</option>
        <option value="failed">Failed</option>
        <option value="canceled">Canceled</option>
      </select>
    </label>
    <label>Show
      <select bind:value={groupView}>
        <option value="all">All runs</option>
        <option value="batches">Batches only</option>
        <option value="ungrouped">Ungrouped only</option>
      </select>
    </label>
    <label>Groups
      <select bind:value={groupSort}>
        <option value="newest">Newest first</option>
        <option value="oldest">Oldest first</option>
        <option value="name">Name</option>
        <option value="size">Smallest first</option>
      </select>
    </label>
    <label>Rows
      <select bind:value={rowSort}>
        <option value="newest">Newest first</option>
        <option value="oldest">Chronological</option>
        <option value="name">Name</option>
      </select>
    </label>
    <p class="subtle run-count-summary">{filteredRunRows.length} / {runs.length} loaded{totalRuns ? ` (${totalRuns} matching)` : ''} • {selectedRunIds.size} selected</p>
  </div>
  {#if queueRuns.length}
  <div class="run-queue-panel">
    <div class="run-queue-head">
      <div>
        <strong>Run queue</strong>
        <span class="subtle">{queueRuns.length} active task{queueRuns.length === 1 ? '' : 's'}</span>
      </div>
    </div>
      <div class="run-queue-window" style:height={`${queueHeightPx}px`}>
        <div class="run-queue-list">
          {#each progressRows as row}
            <div class="run-progress-row" title={`${row.current} / ${row.total} segments · ${row.etaText} · ${row.phase}${row.detailText ? ` · ${row.detailText}` : ''}`}>
              <div class="run-progress-head">
                <span>{row.label}</span>
                <span class="run-progress-count">{row.current}/{row.total} <span>{row.etaText}</span></span>
              </div>
              <div class="run-progress-detail">
                <span class="run-progress-chip phase">{row.phase}</span>
                <span class="run-progress-chip">{row.heartbeatText}</span>
              </div>
              {#if row.detailItems.length}
                <div class="run-progress-detail run-progress-detail-grid">
                  {#each row.detailItems as item}
                    <span class="run-progress-chip">{item}</span>
                  {/each}
                </div>
              {/if}
              <div class="run-progress-meta">
                <span>{Math.round(progressPercent(row.current, row.total))}% complete</span>
                <span>{row.size} grouped runs</span>
              </div>
              <div class="run-progress-track">
                <div class="run-progress-fill" style:width={`${progressPercent(row.current, row.total)}%`}></div>
              </div>
            </div>
          {/each}
          {#each queueRuns as run}
            <div class={`run-queue-item ${run.status}`} title={`${queueRunTitle(run)} · ${queueTraceText(run)}`}>
              <span class={statusDotClass(run.status)}></span>
              <div class="run-queue-main">
                <span>{queueCompactText(run)}</span>
              </div>
              {#if ['queued', 'paused'].includes(String(run.status).toLowerCase())}
                <button class="ghost small queue-cancel" type="button" on:click={() => onRemoveRun(run.run_id)}>Cancel</button>
              {/if}
            </div>
          {/each}
        </div>
        <button
          type="button"
          class="runs-resize-handle queue-resize-handle"
          aria-label="Resize run queue"
          on:pointerdown={onQueueResizePointerDown}
          on:keydown={onQueueResizeHandleKeydown}
          title="Drag to resize queue"
        >
          <span class="runs-resize-grip" aria-hidden="true"></span>
        </button>
      </div>
  </div>
  {/if}

  <div class="runs-compact-window" style:height={`${listHeightPx}px`}>
    <div class="runs-compact-list" bind:this={listEl} on:scroll={onListScroll}>
      {#if !groupedRunSections.length}
        <p class="subtle">No runs match current filters.</p>
      {:else}
        {#each groupedRunSections as section (section.id)}
          {#if section.id !== ungroupedSectionId}
            <div
              class={`run-group-header ${section.archived ? 'archived' : ''} ${dragOverGroupId === section.id ? 'drop-target' : ''} ${section.stats.running ? 'has-active-runs' : ''}`}
              style={groupStyle(section)}
              role="group"
              aria-label={`Run group ${section.name}`}
              draggable="true"
              on:dragstart={(e) => onGroupDragStart(section.id, e)}
              on:dragend={() => { draggedGroupId = ''; dragOverGroupId = ''; }}
              on:dragenter={() => onGroupDragEnter(section.id)}
              on:dragover|preventDefault
              on:dragleave={(e) => onGroupDragLeave(section.id, e)}
              on:drop={(e) => onGroupDrop(section.id, e)}
            >
              <button class="ghost small icon-btn" type="button" title="Collapse group" aria-label="Collapse group" on:click={() => onToggleRunGroup(section.id)}>
                {collapsedRunGroupIds.has(section.id) ? '▸' : '▾'}
              </button>
              <div class="run-group-title">
                <div class="run-group-name-row">
                  <strong>{section.name}</strong>
                  <span class="run-group-count">{groupRunCount(section)}</span>
                </div>
                <span class="subtle">{#if loadingRunGroupIds.has(section.id)}<span class="loading-inline"><i class="loading-spinner" aria-hidden="true"></i>Loading runs…</span>{:else}{groupStatusText(section.stats)}{/if}{section.source ? ` · ${section.source}` : ''}{section.archived ? ' · archived' : ''}</span>
                {#if section.stats.progressTotal}
                  <div class="run-group-progress" aria-label="Group progress">
                    <div class="run-group-progress-fill" style:width={`${groupProgressPercent(section)}%`}></div>
                  </div>
                {/if}
              </div>
              <div class="run-group-actions">
                <button class="ghost small" type="button" on:click={() => onCompareRunGroup(groupRunIds(section), false)}>Compare</button>
                <button class="ghost small" type="button" on:click={() => onCompareRunGroup(groupRunIds(section), true)}>Advanced</button>
                <details class="run-group-more-menu">
                  <summary class="ghost small">More <span aria-hidden="true">▾</span></summary>
                  <div class="run-group-more-panel">
                    {#if groupHasActive(section.stats)}
                      <button class="ghost small" type="button" on:click={() => onControlRunGroup(section.id, 'pause')} disabled={!section.stats.running && !section.stats.queued}>Pause</button>
                      <button class="ghost small" type="button" on:click={() => onControlRunGroup(section.id, 'resume')} disabled={!section.stats.paused}>Resume</button>
                      <button class="ghost small danger" type="button" on:click={() => onControlRunGroup(section.id, 'cancel')}>Cancel</button>
                      <button class="ghost small danger" type="button" on:click={() => onControlRunGroup(section.id, 'force_cancel')}>Force cancel</button>
                    {/if}
                    <button class="ghost small" type="button" on:click={() => onRenameRunGroup(section.id)}>Rename</button>
                    <button class="ghost small" type="button" on:click={() => onDeleteRunGroup(section.id)}>Delete group</button>
                    <button class="ghost small danger" type="button" on:click={() => onDeleteRunGroupWithRuns(section.id)}>Delete group + runs</button>
                  </div>
                </details>
              </div>
            </div>
          {/if}
          {#if section.id === ungroupedSectionId || !collapsedRunGroupIds.has(section.id)}
            <div class="run-group-body" transition:slide={{ duration: 130 }}>
            {#if section.id !== ungroupedSectionId && loadingRunGroupIds.has(section.id)}
              <p class="subtle loading-inline"><i class="loading-spinner" aria-hidden="true"></i>Loading this batch…</p>
            {:else if section.id !== ungroupedSectionId && !section.rows.length}
              <p class="subtle">No runs in this batch match the current filters.</p>
            {/if}
            {#each section.rows as entry (entry.row.run.run_id)}
              {@const row = entry.row}
              {@const idx = entry.idx}
          <div
            class={`run-card-compact ${selectedRunIds.has(row.run.run_id) ? 'selected' : ''} ${selectedRunId === row.run.run_id ? 'active' : ''} ${row.run.run_group_id ? 'grouped' : ''}`}
            style={section.id !== ungroupedSectionId ? groupStyle(section) : ''}
            in:fade={{ duration: 90 }}
          >
            <div class="run-card-head">
              <input
                class="row-check"
                type="checkbox"
                checked={selectedRunIds.has(row.run.run_id)}
                on:click={(e) => onRowCheckboxClick(row.run.run_id, idx, e)}
              />
              <button
                class="run-row-hit"
                type="button"
                on:click={(e) => onRowSelectClick(row.run.run_id, idx, e)}
                on:keydown={(e) => onRowSelectKeydown(row.run.run_id, idx, e)}
                title="Select run (shift-click for range)"
                draggable="true"
                on:dragstart={(e) => onRunDragStart(row.run.run_id, e)}
                on:dragend={() => { draggedRunId = ''; }}
              >
                <span class={statusDotClass(row.run.status)}></span>
                <span class="run-primary" title={row.alias ? `${row.alias} (${row.run.run_id})` : row.run.run_id}>
                  <strong>{row.alias || row.run.run_id}</strong>
                </span>
                <span class="run-meta-chip">{row.run.status}</span>
                <span class="run-meta-chip">{formatRuntime(row.runtimeSec)}</span>
              </button>
              <div class="run-actions-icons">
                {#if canOpenInNavis(row.run)}
                  <button
                    class="ghost small icon-btn navis-icon-btn"
                    type="button"
                    title="Send to Navis and open"
                    aria-label="Open in Navis"
                    on:click|stopPropagation={() => onOpenInNavis(row.run.run_id)}
                    disabled={navisExportingRunIds.has(row.run.run_id)}
                  >
                    {navisExportingRunIds.has(row.run.run_id) ? '...' : 'NV'}
                  </button>
                {/if}
                <button
                  class="ghost small icon-btn fov-video-icon-btn"
                  type="button"
                  title={fovComparisonTitle(row.run)}
                  aria-label="Generate FoV comparison video"
                  on:click|stopPropagation={() => onGenerateFovComparison(row.run.run_id)}
                  disabled={!canGenerateFovComparison(row.run) || fovComparisonGeneratingRunIds.has(row.run.run_id)}
                >
                  {fovComparisonGeneratingRunIds.has(row.run.run_id) ? '...' : 'FV'}
                </button>
                <button class="ghost small icon-btn" type="button" title="Inspect run" aria-label="Inspect run" on:click|stopPropagation={() => onInspectRun(row.run.run_id)}>◫</button>
                <button class="ghost small icon-btn" type="button" title="Rename" aria-label="Rename run" on:click|stopPropagation={() => onQuickRenameRun(row.run.run_id)}>✎</button>
                <button class="ghost small icon-btn" type="button" title="Expand" aria-label="Expand" on:click|stopPropagation={() => onToggleRunDetails(row.run.run_id)}>
                  {expandedRunIds.has(row.run.run_id) ? '▾' : '▸'}
                </button>
                <button class="ghost small icon-btn" type="button" title="Delete" aria-label="Delete" on:click|stopPropagation={() => onRemoveRun(row.run.run_id)}>✕</button>
              </div>
            </div>

            {#if expandedRunIds.has(row.run.run_id)}
              {@const cfg = runConfigCache[row.run.run_id]}
              <div class="run-card-extra">
                <label>Alias
                  <input
                    class="run-alias"
                    placeholder="Alias"
                    maxlength={aliasMaxLength}
                    value={runAliases[row.run.run_id] || ''}
                    on:change={(e) => onAliasChange(row.run.run_id, e)}
                  />
                </label>
                {#if loadingDetailIds.has(row.run.run_id)}
                  <p class="subtle">Loading details…</p>
                {:else}
                  <div class="detail-grid">
                    <div><span class="micro subtle">Started</span><span>{formatDateTime(row.run.started)}</span></div>
                    <div><span class="micro subtle">Finished</span><span>{formatDateTime(row.run.finished)}</span></div>
                    <div><span class="micro subtle">Runtime</span><span>{formatRuntime(row.runtimeSec)}</span></div>
                    <div><span class="micro subtle">ABR</span><span>{cfg?.abr || '—'}</span></div>
                    <div><span class="micro subtle">Predictor</span><span>{fovPredictorLabel(cfg?.predictor || 'None')}</span></div>
                    <div>
                      <span class="micro subtle">Network</span>
                      <span class="ellipsis" title={cfg?.network_profile ? `profile:${cfg.network_profile}` : cfg?.network_trace || ''}>
                        {cfg?.network_profile ? `profile:${cfg.network_profile}` : basenamePath(cfg?.network_trace)}
                      </span>
                    </div>
                    <div>
                      <span class="micro subtle">FoV trace</span>
                      <span class="ellipsis" title={cfg?.fov_trace || ''}>{basenamePath(cfg?.fov_trace)}</span>
                    </div>
                  </div>
                {/if}
                <div class="detail-links">
                  <button
                    class="ghost small"
                    type="button"
                    on:click|stopPropagation={() => openReadableSummary(row.run.run_id, 'run')}
                    disabled={readableLoading}
                  >
                    {readableLoading && readableRunId === row.run.run_id && readableMode === 'run' ? 'Loading...' : 'View run summary'}
                  </button>
                </div>
                <div class="navis-export-panel">
                  <div class="navis-export-header">
                    <span class="micro subtle">Predictor Diagnostics</span>
                    <span class="navis-export-title">
                      {#if row.run.predictor_diagnostics_path}
                        Predictor timing and detailed diagnostics were recorded for this run.
                      {:else}
                        No predictor diagnostics were recorded for this run.
                      {/if}
                    </span>
                  </div>
                  {#if row.run.predictor_diagnostics_path}
                    <div class="detail-links">
                      <button
                        class="ghost small"
                        type="button"
                        on:click|stopPropagation={() => openReadableSummary(row.run.run_id, 'predictor')}
                        disabled={readableLoading}
                      >
                        {readableLoading && readableRunId === row.run.run_id && readableMode === 'predictor' ? 'Loading...' : 'View predictor diagnostics'}
                      </button>
                    </div>
                  {/if}
                </div>
                <div class={`navis-export-panel ${fovComparisonState(row.run)}`}>
                  <div class="navis-export-header">
                    <span class="micro subtle">FoV comparison</span>
                    <span class="navis-export-title">{fovComparisonTitle(row.run)}</span>
                  </div>
                  {#if row.run.fov_comparison?.frame_count}
                    <div class="navis-export-grid">
                      <div><span class="micro subtle">Frames</span><span>{row.run.fov_comparison.frame_count}</span></div>
                      <div><span class="micro subtle">Generated</span><span>{formatDateTime(row.run.fov_comparison.generated_at)}</span></div>
                    </div>
                  {/if}
                  <div class="detail-links">
                    <button
                      class="ghost small"
                      type="button"
                      on:click|stopPropagation={() => onGenerateFovComparison(row.run.run_id)}
                      disabled={!canGenerateFovComparison(row.run) || fovComparisonGeneratingRunIds.has(row.run.run_id)}
                    >
                      {fovComparisonGeneratingRunIds.has(row.run.run_id) ? 'Generating...' : 'Generate video'}
                    </button>
                    {#if row.run.fov_comparison?.video_path}
                      <button class="ghost small" type="button" on:click|stopPropagation={() => onPreviewFovComparison(row.run.run_id)}>View video</button>
                      <a href={row.run.fov_comparison.video_path} target="_blank">Download video</a>
                    {/if}
                    {#if row.run.fov_comparison?.data_path}
                      <button class="ghost small" type="button" on:click|stopPropagation={() => onPreviewFovComparison(row.run.run_id)}>Debug</button>
                    {/if}
                    {#if row.run.fov_comparison?.data_path}<a href={row.run.fov_comparison.data_path} target="_blank">Download data</a>{/if}
                  </div>
                  {#if row.run.fov_comparison?.error}
                    <p class="navis-export-error">{row.run.fov_comparison.error}</p>
                  {/if}
                </div>
                <details class="raw-artifacts">
                  <summary>Raw artifacts</summary>
                  <div class="detail-links">
                    {#if row.run.events_path}<a href={row.run.events_path} target="_blank">events.json</a>{/if}
                    {#if row.run.log_path}<a href={row.run.log_path} target="_blank">run.log</a>{/if}
                    {#if row.run.predictor_diagnostics_path}<a href={row.run.predictor_diagnostics_path} target="_blank">predictor_diagnostics.json</a>{/if}
                    {#if row.run.fov_comparison?.data_path}<a href={row.run.fov_comparison.data_path} target="_blank">fov_prediction_comparison.json</a>{/if}
                    {#if row.run.fov_comparison?.video_path}<a href={row.run.fov_comparison.video_path} target="_blank">fov_prediction_comparison.mp4</a>{/if}
                  </div>
                </details>
                <div class={`navis-export-panel ${navisExportState(row.run.navis_export)}`}>
                  <div class="navis-export-header">
                    <span class="micro subtle">Navis</span>
                    <span class="navis-export-title">{navisExportTitle(row.run.navis_export)}</span>
                  </div>
                  {#if row.run.navis_export}
                    <div class="navis-export-grid">
                      <div><span class="micro subtle">POST</span><span class="ellipsis" title={row.run.navis_export.url || ''}>{row.run.navis_export.url || '—'}</span></div>
                      <div><span class="micro subtle">HTTP</span><span>{row.run.navis_export.status_code ?? '—'}</span></div>
                      <div><span class="micro subtle">Latency</span><span>{row.run.navis_export.latency_ms ?? '—'} ms</span></div>
                      <div><span class="micro subtle">Payload</span><span>{row.run.navis_export.request_bytes ?? '—'} B</span></div>
                      <div><span class="micro subtle">Schema</span><span>{row.run.navis_export.payload_summary?.schema || '—'}</span></div>
                      <div><span class="micro subtle">Segments</span><span>{row.run.navis_export.payload_summary?.segment_count ?? '—'}</span></div>
                    </div>
                    <div class="detail-links">
                      <button
                        class="ghost small"
                        type="button"
                        on:click|stopPropagation={() => openNavisLog(row.run.run_id, row.run.navis_export)}
                      >
                        View Navis log
                      </button>
                    </div>
                    {#if row.run.navis_export.error}
                      <p class="navis-export-error">{row.run.navis_export.error}</p>
                    {/if}
                    {#if row.run.navis_export.response_text}
                      <p class="navis-export-response">{row.run.navis_export.response_text}</p>
                    {/if}
                  {:else}
                    <p class="subtle">No manual or automatic Navis export has been recorded for this run yet.</p>
                  {/if}
                </div>
              </div>
            {/if}
          </div>
            {/each}
            </div>
          {/if}
        {/each}
      {/if}
      {#if loadingMoreRuns}
        <p class="subtle loading-inline list-loading-indicator"><i class="loading-spinner" aria-hidden="true"></i>Loading more runs…</p>
      {/if}
    </div>
    <button
      type="button"
      class="runs-resize-handle"
      aria-label="Resize run rows"
      on:pointerdown={onResizePointerDown}
      on:keydown={onResizeHandleKeydown}
      title="Drag to resize rows area"
    >
      <span class="runs-resize-grip" aria-hidden="true"></span>
    </button>
  </div>
</section>

{#if navisLogResult}
  <div class="navis-log-backdrop" on:pointerdown|self={closeNavisLog}>
    <section class="card navis-log-modal">
      <div class="section-header">
        <div>
          <h3>Navis Export Log</h3>
          <p class="subtle">{navisLogRunId}</p>
        </div>
        <div class="section-actions">
          <button type="button" class="ghost small" on:click={closeNavisLog}>Close</button>
        </div>
      </div>
      <pre class="navis-log-pre">{navisLogText(navisLogResult)}</pre>
    </section>
  </div>
{/if}

{#if pendingDrop}
  <div class="navis-log-backdrop" role="presentation" on:pointerdown|self={() => { pendingDrop = null; }}>
    <section class="card navis-log-modal drop-confirm-modal" role="dialog" aria-modal="true" aria-labelledby="drop-confirm-title">
      <div class="section-header">
        <div>
          <h3 id="drop-confirm-title">Move into batch?</h3>
          <p class="subtle">{pendingDropText()}</p>
        </div>
        <div class="section-actions">
          <button type="button" class="ghost small" on:click={() => { pendingDrop = null; }}>Cancel</button>
          <button type="button" class="small" on:click={confirmPendingDrop}>Move</button>
        </div>
      </div>
    </section>
  </div>
{/if}

{#if readableResult || readableError || readableLoading}
  <div class="navis-log-backdrop" on:pointerdown|self={closeReadableSummary}>
    <section class="card navis-log-modal">
      <div class="section-header">
        <div>
          <h3>{readableMode === 'predictor' ? 'Predictor Diagnostics' : 'Run Summary'}</h3>
          <p class="subtle">{readableRunId}</p>
        </div>
        <div class="section-actions">
          <button type="button" class="ghost small" on:click={closeReadableSummary}>Close</button>
        </div>
      </div>
      {#if readableLoading}
        <p class="subtle">Loading run summary...</p>
      {:else if readableError}
        <p class="navis-export-error">{readableError}</p>
      {:else if readableResult}
        {#if readableMode === 'predictor'}
          <section class="summary-block">
            <div class="summary-card-grid">
              <div><span class="micro subtle">Predictor</span><strong>{fovPredictorLabel(readableResult.predictor_story.predictor_name || readableResult.overview.predictor)}</strong></div>
              <div><span class="micro subtle">Predictions</span><strong>{readableResult.predictor_story.prediction_count ?? 0}</strong></div>
              <div><span class="micro subtle">Median</span><strong>{fixed(readableResult.predictor_story.total?.median_ms, 1)} ms</strong></div>
              <div><span class="micro subtle">P95</span><strong>{fixed(readableResult.predictor_story.total?.p95_ms, 1)} ms</strong></div>
            </div>
            {#if readableResult.predictor_story.story.length}
              <ol class="story-list">
                {#each readableResult.predictor_story.story as item}
                  <li>{item}</li>
                {/each}
              </ol>
            {:else}
              <p class="subtle">{readableResult.predictor_story.summary}</p>
            {/if}
            {#if readableResult.predictor_story.slowest_stages.length}
              <h4>Slowest Stages</h4>
              <div class="summary-table-wrap">
                <table class="summary-table">
                  <thead><tr><th>Stage</th><th>Mean</th><th>P95</th><th>Max</th><th>Count</th></tr></thead>
                  <tbody>
                    {#each readableResult.predictor_story.slowest_stages as stage}
                      <tr>
                        <td>{stage.name}</td>
                        <td>{fixed(stage.mean_ms, 1)} ms</td>
                        <td>{fixed(stage.p95_ms, 1)} ms</td>
                        <td>{fixed(stage.max_ms, 1)} ms</td>
                        <td>{stage.count}</td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            {/if}
            {#if readableResult.predictor_story.metadata.length}
              <h4>Useful Metadata</h4>
              <div class="summary-table-wrap">
                <table class="summary-table">
                  <thead><tr><th>Field</th><th>Median</th><th>Min</th><th>Max</th><th>Samples</th></tr></thead>
                  <tbody>
                    {#each readableResult.predictor_story.metadata as field}
                      <tr>
                        <td>{field.name}</td>
                        <td>{fixed(field.median, 3)}</td>
                        <td>{fixed(field.min, 3)}</td>
                        <td>{fixed(field.max, 3)}</td>
                        <td>{field.count}</td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            {/if}
          </section>
        {:else}
          <section class="summary-block">
            <div class="summary-card-grid">
              <div><span class="micro subtle">Status</span><strong>{readableResult.overview.status}</strong></div>
              <div><span class="micro subtle">ABR</span><strong>{readableResult.overview.abr}</strong></div>
              <div><span class="micro subtle">Predictor</span><strong>{fovPredictorLabel(readableResult.overview.predictor)}</strong></div>
              <div><span class="micro subtle">Segments</span><strong>{readableResult.overview.segment_count}</strong></div>
              <div><span class="micro subtle">Playback</span><strong>{seconds(readableResult.overview.playback_time_s)}</strong></div>
              <div><span class="micro subtle">Rebuffer</span><strong>{seconds(readableResult.overview.rebuffer_time_s)} / {readableResult.overview.rebuffer_count}</strong></div>
              <div><span class="micro subtle">Avg bitrate</span><strong>{fixed(readableResult.overview.average_bitrate_mbps, 2)} Mbps</strong></div>
            </div>
            {#if readableResult.overview.error}
              <p class="navis-export-error">{readableResult.overview.error}</p>
            {/if}

            <h4>Segment Timeline</h4>
            <div class="summary-table-wrap">
              <table class="summary-table timeline-table">
                <thead>
                  <tr><th>Seg</th><th>Decision</th><th>Download</th><th>Size</th><th>Buffer</th><th>FoV</th><th>Selection</th><th>Notes</th></tr>
                </thead>
                <tbody>
                  {#each readableResult.timeline as row}
                    <tr>
                      <td>{row.segment_idx}</td>
                      <td>{seconds(row.decision_time_s)}</td>
                      <td>{seconds(row.download_time_s)}</td>
                      <td>{fixed(row.size_mbit, 2)} Mb</td>
                      <td>{seconds(row.buffer_after_s)}</td>
                      <td>{fovTileText(row.top_fov_tiles)}</td>
                      <td>{row.selection_summary}</td>
                      <td>{row.notes.join(' ') || '—'}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>

            <h4>Run Log</h4>
            <p class="subtle">{readableResult.log_story.summary}</p>
            {#if readableResult.log_story.events.length}
              <ol class="story-list">
                {#each readableResult.log_story.events as event}
                  <li>{event.segment_idx !== null && event.segment_idx !== undefined ? `S${event.segment_idx}: ` : ''}{event.message}</li>
                {/each}
              </ol>
            {/if}
          </section>
        {/if}
        <details class="raw-artifacts modal-raw-artifacts">
          <summary>Raw artifacts</summary>
          <div class="detail-links">
            {#if readableResult.raw_artifacts.events_path}<a href={readableResult.raw_artifacts.events_path} target="_blank">events.json</a>{/if}
            {#if readableResult.raw_artifacts.log_path}<a href={readableResult.raw_artifacts.log_path} target="_blank">run.log</a>{/if}
            {#if readableResult.raw_artifacts.predictor_diagnostics_path}<a href={readableResult.raw_artifacts.predictor_diagnostics_path} target="_blank">predictor_diagnostics.json</a>{/if}
          </div>
        </details>
      {:else}
        <p class="subtle">No summary was returned.</p>
      {/if}
    </section>
  </div>
{/if}

<style>
  .runs-explorer-card {
    padding: 0;
    overflow: hidden;
  }

  .runs-action-bar {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    align-items: center;
    gap: 0.65rem;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--card-border);
    background: rgba(148, 163, 184, 0.06);
  }

  .runs-action-bar h2 {
    margin: 0;
    font-size: 1.02rem;
    white-space: nowrap;
  }

  .runs-action-bar .section-actions {
    display: flex;
    min-width: 0;
    justify-content: flex-start;
    gap: 0.35rem;
    flex-wrap: wrap;
  }

  .runs-action-bar button {
    border-radius: 8px;
    min-height: 2rem;
    padding: 0.34rem 0.5rem;
    white-space: nowrap;
  }

  .runs-compact-toolbar {
    display: grid;
    grid-template-columns: minmax(180px, 1fr) minmax(118px, 0.36fr) minmax(128px, 0.4fr) max-content max-content;
    gap: 0.42rem;
    align-items: end;
    margin: 0;
    padding: 0.55rem 0.75rem 0.48rem;
  }

  .runs-compact-toolbar label {
    gap: 0.18rem;
    font-size: 0.82rem;
  }

  .runs-compact-toolbar input,
  .runs-compact-toolbar select {
    min-height: 2.05rem;
    padding-block: 0.34rem;
  }

  .compact-search {
    min-width: 0;
  }

  .archived-groups-toggle {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    min-height: 2.05rem;
    white-space: nowrap;
  }

  .archived-groups-toggle input {
    min-height: auto;
    margin: 0;
  }

  .run-count-summary {
    margin: 0;
    align-self: center;
    white-space: nowrap;
    font-size: 0.88rem;
  }

  .run-queue-panel {
    display: grid;
    gap: 0.28rem;
    padding: 0.38rem 0.75rem 0.42rem;
    border-top: 1px solid var(--card-border);
    background: rgba(15, 23, 42, 0.18);
  }

  .run-queue-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    min-height: 1.2rem;
  }

  .run-queue-head > div {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    min-width: 0;
  }

  .run-queue-head strong {
    font-size: 0.82rem;
  }

  .run-queue-head .subtle {
    font-size: 0.72rem;
  }

  .run-queue-window {
    display: grid;
    grid-template-rows: minmax(0, 1fr) 10px;
    min-height: 42px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.13);
    border-radius: 7px;
    background: rgba(255, 255, 255, 0.012);
  }

  .run-queue-list {
    display: flex;
    flex-direction: column;
    gap: 0.12rem;
    min-height: 0;
    overflow-y: auto;
    padding: 0.16rem;
  }

  .run-queue-item {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center;
    gap: 0.28rem;
    min-height: 1.34rem;
    padding: 0.1rem 0.18rem;
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 5px;
    background: rgba(255, 255, 255, 0.03);
    font-size: 0.68rem;
    line-height: 1.1;
  }

  .run-queue-item.running {
    border-color: rgba(245, 158, 11, 0.42);
    background: rgba(245, 158, 11, 0.06);
  }

  .run-queue-main {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .run-queue-main span {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .run-progress-row {
    display: grid;
    gap: 0.22rem;
    padding: 0.28rem 0.34rem;
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-radius: 5px;
    background: rgba(56, 189, 248, 0.055);
    font-size: 0.7rem;
    line-height: 1.18;
  }

  .run-progress-head {
    display: flex;
    justify-content: space-between;
    gap: 0.45rem;
    min-width: 0;
    color: var(--text-muted);
  }

  .run-progress-head span:first-child {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .run-progress-count {
    display: inline-flex;
    align-items: baseline;
    gap: 0.28rem;
    flex: 0 0 auto;
    white-space: nowrap;
    font-size: 0.66rem;
  }

  .run-progress-count span {
    color: var(--text-muted);
    font-size: 0.64rem;
  }

  .run-progress-detail {
    display: flex;
    flex-wrap: wrap;
    gap: 0.22rem;
    min-width: 0;
    color: var(--text-muted);
    opacity: 0.9;
  }

  .run-progress-detail-grid {
    max-height: 3.8rem;
    overflow: auto;
  }

  .run-progress-chip {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
    padding: 0.08rem 0.28rem;
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 4px;
    background: rgba(15, 23, 42, 0.2);
    white-space: nowrap;
  }

  .run-progress-chip.phase {
    color: var(--text-main);
    border-color: rgba(56, 189, 248, 0.3);
    background: rgba(56, 189, 248, 0.1);
  }

  .run-progress-meta {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    color: var(--text-muted);
    font-size: 0.63rem;
    white-space: nowrap;
  }

  .run-progress-track {
    height: 0.32rem;
    overflow: hidden;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.18);
  }

  .run-progress-fill {
    height: 100%;
    border-radius: inherit;
    background: #38bdf8;
    transition: width 160ms ease;
  }

  .run-queue-item.running .status-dot,
  .run-card-compact:has(.status-dot.running) .status-dot {
    animation: statusPulse 1.4s ease-in-out infinite;
  }

  .run-queue-empty {
    margin: 0;
    font-size: 0.72rem;
  }

  .queue-cancel {
    min-height: 1.15rem;
    padding: 0.04rem 0.28rem;
    border-radius: 5px;
    font-size: 0.64rem;
    line-height: 1;
  }

  .queue-resize-handle {
    min-height: 10px;
    height: 10px;
    border-top-color: rgba(255, 255, 255, 0.06);
  }

  .runs-compact-window {
    display: grid;
    grid-template-rows: minmax(0, 1fr) 12px;
    border: 0;
    border-top: 1px solid var(--card-border);
    border-radius: 0;
    background: rgba(255, 255, 255, 0.015);
    overflow: hidden;
  }

  .runs-compact-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    min-height: 0;
    overflow-y: auto;
    padding: 0.35rem 0.45rem;
  }

  .run-group-body {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    min-width: 0;
  }

  .runs-resize-handle {
    border-left: 0;
    border-right: 0;
    border-bottom: 0;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    cursor: ns-resize;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.02);
    outline: none;
    width: 100%;
    margin: 0;
    box-shadow: none;
  }

  .runs-resize-handle:focus-visible {
    box-shadow: inset 0 0 0 1px rgba(56, 189, 248, 0.6);
  }

  .runs-resize-handle:hover {
    background: rgba(255, 255, 255, 0.05);
  }

  .runs-resize-grip {
    width: 3.2rem;
    height: 3px;
    border-radius: 999px;
    background: rgba(203, 213, 225, 0.45);
  }

  :global(body.runs-resizing) {
    user-select: none;
    cursor: ns-resize;
  }

  .run-card-compact {
    flex: 0 0 auto;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.03);
    overflow: hidden;
    transition:
      border-color 150ms ease,
      background 150ms ease,
      box-shadow 150ms ease,
      transform 150ms ease;
    /* Browser-level row virtualization: off-viewport run cards skip paint and
       layout work while their intrinsic height keeps scroll position stable. */
    content-visibility: auto;
    contain-intrinsic-size: auto 58px;
  }

  .run-card-compact:hover {
    background: rgba(255, 255, 255, 0.045);
    transform: translateY(-1px);
  }

  .run-card-compact.grouped {
    border-left: 3px solid hsl(var(--group-accent));
  }

  .run-group-header {
    flex: 0 0 auto;
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center;
    gap: 0.45rem;
    padding: 0.34rem 0.42rem;
    border: 1px solid hsl(var(--group-accent) / 0.32);
    border-left: 3px solid hsl(var(--group-accent));
    border-radius: 8px;
    background:
      linear-gradient(90deg, hsl(var(--group-accent) / 0.12), rgba(255, 255, 255, 0.025) 48%),
      rgba(255, 255, 255, 0.025);
    box-shadow: inset 0 0 0 1px hsl(var(--group-accent) / 0.06);
    transition:
      border-color 150ms ease,
      background 150ms ease,
      box-shadow 150ms ease,
      transform 150ms ease;
  }

  .run-group-header.has-active-runs {
    box-shadow:
      inset 0 0 0 1px hsl(var(--group-accent) / 0.1),
      0 0 0 1px hsl(var(--group-accent) / 0.08);
  }

  .run-group-header.drop-target {
    border-color: hsl(var(--group-accent) / 0.85);
    background:
      linear-gradient(90deg, hsl(var(--group-accent) / 0.2), hsl(var(--group-accent) / 0.06) 56%),
      rgba(255, 255, 255, 0.04);
    box-shadow:
      inset 0 0 0 1px hsl(var(--group-accent) / 0.28),
      0 0 0 2px hsl(var(--group-accent) / 0.18);
    transform: translateY(-1px);
  }

  .run-group-header.archived {
    border-color: rgba(148, 163, 184, 0.18);
    background: rgba(148, 163, 184, 0.045);
  }

  .run-group-title {
    min-width: 0;
    display: grid;
    gap: 0.05rem;
  }

  .run-group-title span {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .run-group-name-row {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    min-width: 0;
  }

  .run-group-name-row strong {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .run-group-count {
    flex: 0 0 auto;
    min-width: 1.25rem;
    padding: 0.04rem 0.34rem;
    border: 1px solid hsl(var(--group-accent) / 0.35);
    border-radius: 999px;
    color: hsl(var(--group-accent));
    background: hsl(var(--group-accent) / 0.08);
    font-size: 0.68rem;
    font-weight: 800;
    text-align: center;
  }

  .run-group-progress {
    height: 0.26rem;
    width: min(16rem, 100%);
    overflow: hidden;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.16);
  }

  .run-group-progress-fill {
    height: 100%;
    border-radius: inherit;
    background: hsl(var(--group-accent));
    transition: width 180ms ease;
  }

  .run-group-actions {
    display: inline-flex;
    justify-content: flex-end;
    gap: 0.28rem;
    flex-wrap: nowrap;
    align-items: center;
  }

  .run-group-more-menu {
    position: relative;
    flex: 0 0 auto;
  }

  .run-group-more-menu summary {
    list-style: none;
    cursor: pointer;
    user-select: none;
  }

  .run-group-more-menu summary::-webkit-details-marker {
    display: none;
  }

  .run-group-more-menu[open] summary {
    border-color: hsl(var(--group-accent) / 0.55);
    background: hsl(var(--group-accent) / 0.1);
  }

  .run-group-more-panel {
    position: absolute;
    z-index: 20;
    top: calc(100% + 0.35rem);
    right: 0;
    display: grid;
    min-width: 11.5rem;
    gap: 0.28rem;
    padding: 0.38rem;
    border: 1px solid var(--modal-border);
    border-radius: 8px;
    background: var(--modal-surface);
    color: var(--text-main);
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.22);
  }

  .run-group-more-panel button {
    width: 100%;
    justify-content: flex-start;
    text-align: left;
  }

  .run-card-compact.selected {
    border-color: rgba(56, 189, 248, 0.7);
    box-shadow: inset 0 0 0 1px rgba(56, 189, 248, 0.45);
  }

  .run-card-compact.active {
    border-color: rgba(245, 158, 11, 0.65);
  }

  .run-card-head {
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 0.45rem;
    align-items: center;
    padding: 0.4rem 0.5rem;
  }

  .row-check {
    margin: 0;
  }

  .run-row-hit {
    border: 0;
    background: transparent;
    color: inherit;
    padding: 0.1rem;
    margin: 0;
    width: 100%;
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto auto;
    gap: 0.4rem;
    align-items: center;
    text-align: left;
    box-shadow: none;
  }

  .run-row-hit:hover {
    transform: none;
    background: rgba(255, 255, 255, 0.045);
    box-shadow: none;
  }

  .run-row-hit:active {
    transform: none;
    box-shadow: none;
  }

  .status-dot {
    width: 0.6rem;
    height: 0.6rem;
    border-radius: 999px;
    display: inline-block;
    border: 1px solid rgba(255, 255, 255, 0.3);
  }

  .status-dot.queued { background: #67e8f9; }
  .status-dot.running { background: #22d3ee; }
  .status-dot.paused { background: #facc15; }
  .status-dot.finished { background: #4ade80; }
  .status-dot.failed { background: #f87171; }
  .status-dot.canceled { background: #cbd5e1; }
  .status-dot.unknown { background: #94a3b8; }

  @keyframes statusPulse {
    0%, 100% {
      box-shadow: 0 0 0 0 rgba(34, 211, 238, 0);
      transform: scale(1);
    }
    50% {
      box-shadow: 0 0 0 4px rgba(34, 211, 238, 0.16);
      transform: scale(1.08);
    }
  }

  .run-primary {
    min-width: 0;
    display: inline-flex;
    flex-direction: column;
    gap: 0.05rem;
  }

  .run-primary strong {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .run-meta-chip {
    display: inline-flex;
    align-items: center;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 999px;
    padding: 0.12rem 0.45rem;
    font-size: 0.72rem;
    color: #cbd5e1;
    white-space: nowrap;
  }

  .run-actions-icons {
    display: inline-flex;
    gap: 0.35rem;
    justify-content: flex-end;
  }

  .icon-btn {
    min-width: 2rem;
    padding-inline: 0.4rem;
  }

  .navis-icon-btn {
    font-weight: 800;
    color: #67e8f9;
    min-width: 2.4rem;
  }

  .run-card-extra {
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding: 0.55rem 0.6rem 0.65rem;
    display: grid;
    gap: 0.45rem;
  }

  .detail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.4rem 0.55rem;
  }

  .detail-grid > div {
    display: grid;
    gap: 0.15rem;
    min-width: 0;
  }

  .ellipsis {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .detail-links {
    display: flex;
    gap: 0.65rem;
    flex-wrap: wrap;
  }

  .raw-artifacts {
    color: rgba(226, 232, 240, 0.86);
    font-size: 0.88rem;
  }

  .raw-artifacts summary {
    cursor: pointer;
    color: #cbd5e1;
    font-weight: 700;
  }

  .raw-artifacts .detail-links {
    margin-top: 0.45rem;
  }

  .navis-export-panel {
    display: grid;
    gap: 0.45rem;
    padding: 0.7rem 0.8rem;
    border-radius: 10px;
    border: 1px solid rgba(148, 163, 184, 0.22);
    background: rgba(15, 23, 42, 0.28);
  }

  .navis-export-panel.success {
    border-color: rgba(74, 222, 128, 0.35);
  }

  .navis-export-panel.failed {
    border-color: rgba(248, 113, 113, 0.35);
  }

  .navis-export-header {
    display: flex;
    gap: 0.5rem;
    align-items: baseline;
    justify-content: space-between;
  }

  .navis-export-title {
    font-size: 0.92rem;
    font-weight: 600;
  }

  .navis-export-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.6rem 0.8rem;
  }

  .navis-export-grid > div {
    display: grid;
    gap: 0.15rem;
    min-width: 0;
  }

  .navis-export-error {
    margin: 0;
    color: #fda4af;
    font-size: 0.9rem;
  }

  .navis-export-response {
    margin: 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.82rem;
    line-height: 1.45;
    color: rgba(226, 232, 240, 0.86);
    white-space: pre-wrap;
    word-break: break-word;
  }

  .navis-log-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(2, 6, 23, 0.72);
    display: grid;
    place-items: center;
    padding: 1.2rem;
    z-index: 1000;
  }

  .navis-log-modal {
    width: min(980px, 100%);
    max-height: min(84vh, 920px);
    overflow: auto;
    display: grid;
    gap: 0.75rem;
  }

  .navis-log-pre {
    margin: 0;
    padding: 1rem;
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.78);
    color: rgba(226, 232, 240, 0.94);
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-word;
    font: 0.85rem/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
  }

  .summary-block {
    display: grid;
    gap: 0.8rem;
  }

  .summary-block h4 {
    margin: 0.2rem 0 0;
    font-size: 0.96rem;
  }

  .summary-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
    gap: 0.55rem;
  }

  .loading-inline {
    display: inline-flex;
    align-items: center;
    gap: 0.38rem;
  }

  .drop-confirm-modal {
    max-width: 34rem;
  }

  .loading-spinner {
    width: 0.75rem;
    height: 0.75rem;
    flex: 0 0 auto;
    border: 2px solid rgba(125, 211, 252, 0.28);
    border-top-color: #38bdf8;
    border-radius: 50%;
    animation: runsSpin 0.7s linear infinite;
  }

  .list-loading-indicator {
    justify-content: center;
    margin: 0.7rem 0 0.15rem;
  }

  @keyframes runsSpin {
    to { transform: rotate(360deg); }
  }

  .summary-card-grid > div {
    display: grid;
    gap: 0.18rem;
    padding: 0.65rem 0.75rem;
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 8px;
    background: rgba(15, 23, 42, 0.34);
    min-width: 0;
  }

  .summary-card-grid strong {
    min-width: 0;
    overflow-wrap: anywhere;
  }

  .story-list {
    margin: 0;
    padding-left: 1.25rem;
    display: grid;
    gap: 0.38rem;
    color: rgba(226, 232, 240, 0.9);
  }

  .summary-table-wrap {
    overflow: auto;
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 8px;
  }

  .summary-table {
    width: 100%;
    border-collapse: collapse;
    min-width: 620px;
    font-size: 0.84rem;
  }

  .summary-table th,
  .summary-table td {
    padding: 0.48rem 0.55rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.16);
    text-align: left;
    vertical-align: top;
  }

  .summary-table th {
    color: #cbd5e1;
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    background: rgba(15, 23, 42, 0.38);
  }

  .summary-table tbody tr:last-child td {
    border-bottom: 0;
  }

  .timeline-table {
    min-width: 980px;
  }

  .modal-raw-artifacts {
    border-top: 1px solid rgba(148, 163, 184, 0.18);
    padding-top: 0.65rem;
  }

  @media (max-width: 980px) {
    .runs-compact-toolbar {
      grid-template-columns: minmax(180px, 1fr) minmax(118px, 0.44fr) minmax(128px, 0.48fr);
    }

    .runs-compact-toolbar .subtle {
      grid-column: 1 / -1;
      justify-self: end;
    }

    .navis-export-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 860px) {
    .runs-action-bar {
      grid-template-columns: 1fr;
      gap: 0.4rem;
      padding: 0.5rem 0.65rem;
    }

    .runs-action-bar h2 {
      display: none;
    }

    .runs-action-bar .section-actions {
      width: 100%;
    }

    .runs-action-bar button {
      flex: 1 1 5.8rem;
      min-width: 0;
    }

    .run-card-head {
      grid-template-columns: auto minmax(0, 1fr);
      align-items: start;
    }

    .run-actions-icons {
      grid-column: 1 / -1;
      flex-wrap: wrap;
    }
  }

  @media (max-width: 720px) {
    .navis-export-header {
      align-items: flex-start;
      flex-direction: column;
    }

    .navis-export-grid {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 620px) {
    .runs-compact-toolbar {
      grid-template-columns: minmax(0, 1fr) minmax(8rem, 0.72fr);
      align-items: stretch;
      padding-inline: 0.65rem;
    }

    .compact-search {
      grid-column: 1 / -1;
    }

    .runs-compact-toolbar .subtle {
      justify-self: start;
    }

    .runs-action-bar button {
      flex-basis: calc(33.333% - 0.35rem);
      padding: 0.34rem 0.42rem;
      font-size: 0.84rem;
    }

    .run-row-hit {
      grid-template-columns: auto minmax(0, 1fr);
    }

    .run-meta-chip {
      justify-self: start;
      font-size: 0.68rem;
      padding-inline: 0.36rem;
    }

    .icon-btn {
      min-width: 1.9rem;
      padding-inline: 0.35rem;
    }

    .navis-icon-btn {
      min-width: 2.1rem;
    }
  }

  @media (max-width: 460px) {
    .runs-compact-toolbar {
      grid-template-columns: 1fr;
    }

    .runs-action-bar button {
      flex-basis: calc(50% - 0.35rem);
    }
  }
</style>
