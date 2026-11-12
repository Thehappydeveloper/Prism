<script lang="ts">
  import { onMount } from 'svelte';
  import type { ChartConfiguration } from 'chart.js';
  import { gotoRoute } from '../core/router';
  import { fovPredictorLabel } from '../core/predictorLabels';
  import { getDiagnostics, getOptions, getRunFovComparisonSegment } from '../core/api';
  import { scheduleGraphWork, timed } from '../core/graphCache';
  import { loadRunAliases, loadSelectedCompareRuns, saveSelectedCompareRuns } from '../core/storage';
  import { loadQoeState, normalizeQoeModels, sanitizeQoeParams, saveQoeState, type QoEState } from '../core/qoe';
  import type { QoEModel, QoEParams, RunMetrics } from '../core/types';
  import { aggregateMetrics, type GroupingOptions } from '../features/advanced-compare/metrics';
  import {
    applyPublicationStyle, DEFAULT_EXPORT_SETTINGS, DEFAULT_PUBLICATION_SETTINGS, FIGURE_PRESETS,
    sortCategoricalChart, stableStyleMap, type FigureExportSettings, type PublicationSettings,
  } from '../features/advanced-compare/publication';
  import { avgEfficiencyPlugin } from '../features/advanced-compare/plugins/avgEfficiency';
  import { bitrateQualityPlugin } from '../features/advanced-compare/plugins/bitrateQuality';
  import { bufferStabilityPlugin } from '../features/advanced-compare/plugins/bufferStability';
  import { componentBitratePlugin } from '../features/advanced-compare/plugins/componentBitrate';
  import { fovBitrateWasteAbsolutePlugin } from '../features/advanced-compare/plugins/fovBitrateWasteAbsolute';
  import { fovBitrateWasteRatiosPlugin } from '../features/advanced-compare/plugins/fovBitrateWasteRatios';
  import { fovCatastrophicMissPlugin } from '../features/advanced-compare/plugins/fovCatastrophicMiss';
  import { fovOverlapIouPlugin } from '../features/advanced-compare/plugins/fovOverlapIou';
  import { fovPairwiseWinProbabilityPlugin } from '../features/advanced-compare/plugins/fovPairwiseWinProbability';
  import { fovPredictionBitrateWastePlugin } from '../features/advanced-compare/plugins/fovPredictionBitrateWaste';
  import { fovPredictionCalibrationPlugin } from '../features/advanced-compare/plugins/fovPredictionCalibration';
  import { fovPredictionErrorPlugin } from '../features/advanced-compare/plugins/fovPredictionError';
  import { fovReliabilityGuaranteePlugin } from '../features/advanced-compare/plugins/fovReliabilityGuarantee';
  import { fovQualityWastePlugin } from '../features/advanced-compare/plugins/fovQualityWaste';
  import { fovTopKRecallPlugin } from '../features/advanced-compare/plugins/fovTopKRecall';
  import { fovUsefulBitrateEfficiencyPlugin } from '../features/advanced-compare/plugins/fovUsefulBitrateEfficiency';
  import { qoeCdfPlugin } from '../features/advanced-compare/plugins/qoeCdf';
  import { qoeStallPlugin } from '../features/advanced-compare/plugins/qoeStall';
  import { predictorLatencyPlugin } from '../features/advanced-compare/plugins/predictorLatency';
  import { stallSummaryPlugin } from '../features/advanced-compare/plugins/stallSummary';
  import { temporalDeltaPlugin } from '../features/advanced-compare/plugins/temporalDelta';
  import { chartPluginRegistry } from '../features/advanced-compare/plugins/registry';
  import type { ChartPlugin, MetricDataset, PluginCategory, PluginControl } from '../features/advanced-compare/plugins/types';
  import PluginChart from '../features/advanced-compare/components/PluginChart.svelte';

  const QOE_STORAGE_KEY = 'compare_qoe_state_v4';
  const PLUGIN_STORAGE_KEY = 'compare_enabled_plugins_v1';
  const PLUGIN_KNOWN_STORAGE_KEY = 'compare_known_plugins_v1';
  const PLUGIN_OPTIONS_STORAGE_KEY = 'compare_plugin_options_v1';
  const FIGURE_SETTINGS_STORAGE_KEY = 'compare_publication_figure_v1';
  const GROUPING_STORAGE_KEY = 'compare_grouping_v1';

  const aliases = loadRunAliases();
  const query = new URLSearchParams(window.location.search);
  const runIdsFromUrl = (query.get('runs') || '').split(',').map((x) => x.trim()).filter(Boolean);
  let selectedRuns = runIdsFromUrl.length ? runIdsFromUrl : loadSelectedCompareRuns();

  let loading = true;
  let error = '';
  let infoMessage = '';

  let qoeModels: QoEModel[] = [];
  let qoeState: QoEState = { selectedId: '', paramsById: {} };
  let showQoeControls = false;
  let showFigureControls = false;

  let grouping: GroupingOptions = {
    byAbr: true,
    byPredictor: false,
    byPredictorPreset: false,
    byBandwidthPredictor: false,
    byNetwork: false,
    mode: 'mean',
  };

  let baseMetrics: RunMetrics | null = null;
  let renderedMetrics: RunMetrics | null = null;
  let loadedDatasets = new Set<MetricDataset>();

  let galleryMode: 'slideshow' | 'grid' = 'slideshow';
  let galleryLayout: 'horizontal' | 'vertical' = 'horizontal';
  let collapseQoeComparison = false;
  let slideIndex = 0;
  let totalSlides = 1;
  let buildingCharts = false;
  let chartBuildError = '';
  let chartBuildSerial = 0;

  type SegmentComparisonEntry = { runId: string; frame?: any; error?: string };
  let showSegmentComparison = false;
  let segmentComparisonSegment = 0;
  let segmentComparisonLoading = false;
  let segmentComparisonEntries: SegmentComparisonEntry[] = [];
  let segmentComparisonMessage = '';
  let segmentComparisonSelectedTile = '';

  function scoreColor(value: unknown, alpha = 0.8): string {
    const score = Math.max(0, Math.min(1, Number(value) || 0));
    const hue = 220 - score * 205;
    return `hsla(${hue}, 78%, ${31 + score * 22}%, ${alpha})`;
  }

  function frameBounds(frame: any): [number, number, number, number] {
    const values = Object.values(frame?.tiles || {}) as any[];
    const bounds = values.map((tile) => tile?.bounds_xz).filter((box) => Array.isArray(box) && box.length === 4);
    if (!bounds.length) return [0, 0, 1, 1];
    const xs = bounds.flatMap((box: number[]) => [Number(box[0]), Number(box[2])]);
    const zs = bounds.flatMap((box: number[]) => [Number(box[1]), Number(box[3])]);
    const minX = Math.min(...xs); const maxX = Math.max(...xs); const minZ = Math.min(...zs); const maxZ = Math.max(...zs);
    return [minX, minZ, Math.max(maxX - minX, 1e-6), Math.max(maxZ - minZ, 1e-6)];
  }

  function frameViewBox(frame: any): string {
    return frameBounds(frame).join(' ');
  }

  function tileGeometrySignature(frame: any): string {
    return (frame?.tile_ids || []).map((tileId: string) => `${tileId}:${(frame.tiles?.[tileId]?.bounds_xz || []).join(',')}`).join('|');
  }

  function predictorLabel(runId: string): string {
    const meta = baseMetrics?.meta?.[runId] as any;
    const predictor = meta?.predictor || 'Predictor';
    const preset = meta?.predictor_preset_name ? ` [${meta.predictor_preset_name}]` : '';
    return `${fovPredictorLabel(predictor)}${preset} · ${aliases[runId] || runId}`;
  }

  function selectComparisonTile(event: KeyboardEvent, tileId: unknown): void {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    segmentComparisonSelectedTile = String(tileId);
  }

  async function loadSegmentComparison(): Promise<void> {
    if (!selectedRuns.length) return;
    segmentComparisonLoading = true;
    segmentComparisonMessage = '';
    const segmentIdx = Math.max(0, Math.floor(Number(segmentComparisonSegment) || 0));
    segmentComparisonSegment = segmentIdx;
    const responses = await Promise.all(selectedRuns.map(async (runId): Promise<SegmentComparisonEntry> => {
      try { return { runId, frame: (await getRunFovComparisonSegment(runId, segmentIdx)).frame }; }
      catch (err) { return { runId, error: err instanceof Error ? err.message : String(err) }; }
    }));
    segmentComparisonEntries = responses;
    const available = responses.filter((entry) => entry.frame);
    if (!available.length) segmentComparisonMessage = 'No selected run has prediction and ground-truth tile data for this segment.';
    else if (available.length !== selectedRuns.length) segmentComparisonMessage = 'Some selected runs do not have comparison data for this segment.';
    const referenceFrame = available[0]?.frame;
    if (referenceFrame && available.length === selectedRuns.length && available.some((entry) => tileGeometrySignature(entry.frame) !== tileGeometrySignature(referenceFrame))) {
      segmentComparisonMessage = 'Warning: selected runs use different tile layouts for this segment; maps remain visible but are not directly comparable.';
    }
    const firstTile = available[0]?.frame?.tile_ids?.[0];
    if (firstTile && !available.some((entry) => entry.frame?.tile_ids?.includes(segmentComparisonSelectedTile))) segmentComparisonSelectedTile = String(firstTile);
    segmentComparisonLoading = false;
  }

  type BuiltChart = {
    id: string;
    title: string;
    description: string;
    config: ChartConfiguration | null;
    controls: PluginControl[];
    options: Record<string, string | number | boolean>;
    error?: string;
  };

  let builtCharts: BuiltChart[] = [];
  let pluginCount = 0;
  let enabledPluginIds = new Set<string>();
  let pluginOptions: Record<string, Record<string, string | number | boolean>> = {};
  let publicationSettings: PublicationSettings = { ...DEFAULT_PUBLICATION_SETTINGS };
  let exportSettings: FigureExportSettings = { ...DEFAULT_EXPORT_SETTINGS };
  let showPluginModal = false;
  let pluginCategoryFilter: PluginCategory | 'all' = 'all';
  const CHART_DESCRIPTIONS: Record<string, string> = {
    'qoe-cdf': 'Shows the share of segments at or above each QoE score. Curves farther up and right are better.',
    'qoe-stall': 'Each point compares average QoE with total rebuffering time. Higher QoE and less stall is better.',
    'avg-efficiency': 'Each point compares average delivered bitrate with average QoE. It shows the QoE gained for the bitrate used.',
    'fov-reliability-guarantee': 'Trace-level probability of satisfying a selected accuracy requirement, with a configurable one-sided confidence lower bound.',
    'predictor-latency': 'Compares median, mean, and p95 end-to-end prediction latency. Lower values and a small median-to-p95 gap are better.',
    'fov-prediction-error': 'MAE and RMSE compare predicted tile visibility with the actual target. Lower is better.',
    'fov-catastrophic-miss': 'The share of segments where the predicted top-K tiles capture under 10% of the actual FoV visibility mass. Lower is better.',
    'fov-overlap-iou': 'Binary IoU compares visible tile sets; weighted IoU also accounts for partial visibility. Higher is better.',
    'fov-pairwise-win-probability': 'Paired trace-bootstrap probability that each row predictor beats each column predictor on the selected metric.',
    'fov-topk-recall': 'Checks whether the predictor ranks actually visible tiles near the top. Higher recall and hit rate are better; lower peak miss rate is better.',
    'fov-prediction-calibration': 'For each predicted-visibility range, shows the mean actual visibility. A well-calibrated predictor follows the diagonal.',
    'fov-quality-waste': 'Estimates quality allocated outside the actual FoV and quality missed inside it. Lower is better.',
    'fov-bitrate-waste-absolute': 'Shows Mbps spent on tiles that are partly or fully outside the actual FoV. Lower is better.',
    'fov-bitrate-waste-ratios': 'Normalizes FoV bitrate waste by delivered bitrate, making runs with different total rates comparable. Lower is better.',
    'fov-prediction-bitrate-waste': 'Measures waste caused specifically by tiles the predictor prioritized that were not actually visible. Lower is better.',
    'fov-useful-bitrate-efficiency': 'Shows the fraction of delivered bits that land in the actual FoV versus high-quality invisible bits. More useful bits is better.',
    'temporal-delta': 'Distribution of quality changes between consecutive FoV-tile observations. More mass near zero means steadier quality.',
    'stall-summary': 'Compares total rebuffering time and number of stall events. Lower is better for both.',
    'buffer-stability': 'Summarizes buffer variation. Lower standard deviation indicates a steadier buffer.',
    'component-bitrate': 'Breaks delivered bitrate into V-PCC components, showing how each run allocates its budget.',
    'bitrate-quality': 'Plots segment bitrate against resulting FoV quality. It reveals the quality return from additional bitrate.',
    'combined-qoe': 'QoE score over playback time for the selected runs or groups. Higher and steadier is better.',
  };

  function chartDescription(id: string): string {
    return CHART_DESCRIPTIONS[id] || 'Summary of the selected run metrics.';
  }
  const pluginCategoryLabels: Record<PluginCategory | 'all', string> = {
    all: 'All',
    global: 'Global',
    fov: 'FoV',
    bandwidth: 'Bandwidth',
    abr: 'ABR',
    buffer: 'Buffer',
    qoe: 'QoE',
  };

  const fallbackPluginRegistry: ChartPlugin[] = [
    qoeCdfPlugin,
    qoeStallPlugin,
    avgEfficiencyPlugin,
    fovReliabilityGuaranteePlugin,
    fovPairwiseWinProbabilityPlugin,
    predictorLatencyPlugin,
    fovPredictionErrorPlugin,
    fovCatastrophicMissPlugin,
    fovOverlapIouPlugin,
    fovTopKRecallPlugin,
    fovPredictionCalibrationPlugin,
    fovQualityWastePlugin,
    fovBitrateWasteAbsolutePlugin,
    fovBitrateWasteRatiosPlugin,
    fovPredictionBitrateWastePlugin,
    fovUsefulBitrateEfficiencyPlugin,
    temporalDeltaPlugin,
    stallSummaryPlugin,
    bufferStabilityPlugin,
    componentBitratePlugin,
    bitrateQualityPlugin,
  ];

  function resolvedPluginRegistry(): ChartPlugin[] {
    if (chartPluginRegistry.length >= fallbackPluginRegistry.length) {
      return chartPluginRegistry;
    }
    const overrides = new Map(chartPluginRegistry.map((plugin) => [plugin.id, plugin]));
    return fallbackPluginRegistry.map((plugin) => overrides.get(plugin.id) || plugin);
  }

  function loadEnabledPluginIds(): Set<string> {
    const available = new Set(resolvedPluginRegistry().map((plugin) => plugin.id));
    try {
      const parsed = JSON.parse(localStorage.getItem(PLUGIN_STORAGE_KEY) || '[]');
      if (!Array.isArray(parsed)) return new Set(available);
      const filtered = parsed.filter((id) => typeof id === 'string' && available.has(id));
      const knownParsed = JSON.parse(localStorage.getItem(PLUGIN_KNOWN_STORAGE_KEY) || '[]');
      const fallbackKnown = new Set(fallbackPluginRegistry.map((plugin) => plugin.id));
      const known = Array.isArray(knownParsed) && knownParsed.length
        ? new Set(knownParsed.filter((id) => typeof id === 'string'))
        : fallbackKnown;
      const enabled = new Set(filtered.length ? filtered : Array.from(available));
      available.forEach((id) => {
        if (!known.has(id)) enabled.add(id);
      });
      localStorage.setItem(PLUGIN_KNOWN_STORAGE_KEY, JSON.stringify(Array.from(available)));
      return enabled;
    } catch {
      localStorage.setItem(PLUGIN_KNOWN_STORAGE_KEY, JSON.stringify(Array.from(available)));
      return new Set(available);
    }
  }

  function saveEnabledPluginIds(): void {
    localStorage.setItem(PLUGIN_STORAGE_KEY, JSON.stringify(Array.from(enabledPluginIds)));
  }

  function enabledPlugins(): ChartPlugin[] {
    const plugins = resolvedPluginRegistry();
    return plugins.filter((plugin) => enabledPluginIds.has(plugin.id));
  }

  function defaultPluginOptions(plugin: ChartPlugin): Record<string, string | number | boolean> {
    return Object.fromEntries((plugin.controls || []).map((control) => [control.key, control.defaultValue]));
  }

  function optionsFor(plugin: ChartPlugin): Record<string, string | number | boolean> {
    return { ...defaultPluginOptions(plugin), ...(pluginOptions[plugin.id] || {}) };
  }

  function loadPluginOptions(): void {
    try {
      const parsed = JSON.parse(localStorage.getItem(PLUGIN_OPTIONS_STORAGE_KEY) || '{}');
      pluginOptions = parsed && typeof parsed === 'object' ? parsed : {};
    } catch {
      pluginOptions = {};
    }
  }

  function updatePluginOption(pluginId: string, detail: { key: string; value: string | number | boolean }): void {
    pluginOptions = {
      ...pluginOptions,
      [pluginId]: { ...(pluginOptions[pluginId] || {}), [detail.key]: detail.value },
    };
    localStorage.setItem(PLUGIN_OPTIONS_STORAGE_KEY, JSON.stringify(pluginOptions));
    void buildCharts();
  }

  function saveFigureSettings(rebuild = false): void {
    localStorage.setItem(FIGURE_SETTINGS_STORAGE_KEY, JSON.stringify({ publicationSettings, exportSettings }));
    if (rebuild) void buildCharts();
  }

  function loadFigureSettings(): void {
    try {
      const saved = JSON.parse(localStorage.getItem(FIGURE_SETTINGS_STORAGE_KEY) || '{}');
      publicationSettings = { ...DEFAULT_PUBLICATION_SETTINGS, ...(saved.publicationSettings || {}) };
      exportSettings = { ...DEFAULT_EXPORT_SETTINGS, ...(saved.exportSettings || {}) };
    } catch {
      publicationSettings = { ...DEFAULT_PUBLICATION_SETTINGS };
      exportSettings = { ...DEFAULT_EXPORT_SETTINGS };
    }
    try {
      const savedGrouping = JSON.parse(localStorage.getItem(GROUPING_STORAGE_KEY) || 'null');
      if (savedGrouping) grouping = { ...grouping, ...savedGrouping };
    } catch { /* retain defaults */ }
  }

  function applyFigurePreset(presetId: string): void {
    const preset = FIGURE_PRESETS.find((candidate) => candidate.id === presetId);
    exportSettings = preset && preset.id !== 'custom'
      ? { ...exportSettings, preset: preset.id, width: preset.width, height: preset.height, unit: preset.unit }
      : { ...exportSettings, preset: 'custom' };
    saveFigureSettings();
  }

  function handleFigurePreset(event: Event): void {
    applyFigurePreset((event.currentTarget as HTMLSelectElement).value);
  }

  function updateFontFamily(event: Event): void {
    publicationSettings = { ...publicationSettings, fontFamily: (event.currentTarget as HTMLInputElement).value };
    saveFigureSettings(true);
  }

  function updateExportNumber(key: 'width' | 'height' | 'dpi', event: Event): void {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    if (!Number.isFinite(value) || value <= 0) return;
    exportSettings = { ...exportSettings, [key]: value, preset: key === 'dpi' ? exportSettings.preset : 'custom' };
    saveFigureSettings();
  }

  function updatePublicationNumber(event: Event): void {
    const value = Number((event.currentTarget as HTMLInputElement).value);
    if (!Number.isFinite(value) || value < 6 || value > 14) return;
    publicationSettings = { ...publicationSettings, baseFontPt: value };
    saveFigureSettings(true);
  }

  function downloadAdvancedCompareConfig(): void {
    const payload = {
      version: 1,
      createdAt: new Date().toISOString(),
      selectedRuns,
      grouping,
      enabledPluginIds: Array.from(enabledPluginIds),
      pluginOptions,
      qoeState,
      publicationSettings,
      exportSettings,
      predictorStyles: renderedMetrics ? stableStyleMap(renderedMetrics, aliases, publicationSettings.grayscale) : {},
      figures: builtCharts.map((chart) => ({ id: chart.id, title: chart.title, options: chart.options, axes: chart.config?.options?.scales })),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = 'advanced-compare-figure-config.json'; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function loadAdvancedCompareConfig(event: Event): Promise<void> {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    try {
      const payload = JSON.parse(await file.text());
      if (payload.version !== 1) throw new Error('Unsupported figure configuration version.');
      if (Array.isArray(payload.selectedRuns)) selectedRuns = payload.selectedRuns.map(String);
      grouping = { ...grouping, ...(payload.grouping || {}) };
      pluginOptions = payload.pluginOptions && typeof payload.pluginOptions === 'object' ? payload.pluginOptions : pluginOptions;
      publicationSettings = { ...DEFAULT_PUBLICATION_SETTINGS, ...(payload.publicationSettings || {}) };
      exportSettings = { ...DEFAULT_EXPORT_SETTINGS, ...(payload.exportSettings || {}) };
      if (Array.isArray(payload.enabledPluginIds)) enabledPluginIds = new Set(payload.enabledPluginIds.map(String));
      if (payload.qoeState) qoeState = payload.qoeState;
      saveSelectedCompareRuns(selectedRuns);
      localStorage.setItem(GROUPING_STORAGE_KEY, JSON.stringify(grouping));
      localStorage.setItem(PLUGIN_OPTIONS_STORAGE_KEY, JSON.stringify(pluginOptions));
      saveEnabledPluginIds(); saveFigureSettings(); saveQoeState(QOE_STORAGE_KEY, qoeState);
      await reloadDiagnostics();
    } catch (err) {
      error = `Unable to load figure configuration: ${err instanceof Error ? err.message : String(err)}`;
    } finally {
      input.value = '';
    }
  }

  function requestedDatasets(): MetricDataset[] {
    const datasets = new Set<MetricDataset>(['qoe_series']);
    enabledPlugins().forEach((plugin) => plugin.requiredDatasets.forEach((dataset) => datasets.add(dataset)));
    return Array.from(datasets).sort();
  }

  function pluginCategories(): Array<PluginCategory | 'all'> {
    const categories = new Set<PluginCategory>();
    resolvedPluginRegistry().forEach((plugin) => categories.add(plugin.category || 'global'));
    return ['all', ...Array.from(categories)];
  }

  function visiblePluginRegistry(): ChartPlugin[] {
    const plugins = resolvedPluginRegistry();
    if (pluginCategoryFilter === 'all') return plugins;
    return plugins.filter((plugin) => (plugin.category || 'global') === pluginCategoryFilter);
  }

  function pluginCategoryCount(category: PluginCategory | 'all'): number {
    if (category === 'all') return resolvedPluginRegistry().length;
    return resolvedPluginRegistry().filter((plugin) => (plugin.category || 'global') === category).length;
  }

  function activeModel(): QoEModel {
    const found = qoeModels.find((m) => m.id === qoeState.selectedId);
    return found || qoeModels[0]!;
  }

  function activeParams(): QoEParams {
    const model = activeModel();
    const cur = qoeState.paramsById[model.id] || {};
    const next = sanitizeQoeParams(model, cur);
    qoeState.paramsById[model.id] = next;
    return next;
  }

  function buildCombinedQoeChart(metrics: RunMetrics): ChartConfiguration | null {
    const series = metrics.qoe_series || {};
    const datasets = Object.entries(series)
      .map(([runId, values], idx) => {
        const data = (values || [])
          .map((point) => ({ x: Number(point.time_s), y: Number(point.score) }))
          .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
        if (!data.length) return null;
        return {
          label: metrics.labels?.[runId] ? `${aliases[runId] ? `${aliases[runId]} (${metrics.labels[runId]})` : metrics.labels[runId]}` : runId,
          data,
          borderColor: ['#22d3ee', '#f472b6', '#a3e635', '#f59e0b', '#38bdf8', '#c084fc', '#fb7185', '#14b8a6'][idx % 8],
          backgroundColor: ['#22d3ee', '#f472b6', '#a3e635', '#f59e0b', '#38bdf8', '#c084fc', '#fb7185', '#14b8a6'][idx % 8],
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

  async function reloadDiagnostics(): Promise<void> {
    if (!selectedRuns.length) {
      baseMetrics = null;
      renderedMetrics = null;
      builtCharts = [];
      totalSlides = 1;
      loading = false;
      infoMessage = 'No runs specified. Select runs on the main page and click Advanced compare.';
      return;
    }

    loading = true;
    error = '';
    try {
      const model = activeModel();
      const params = activeParams();
      const fetchStarted = performance.now();
      const metrics = await getDiagnostics({
        run_ids: selectedRuns,
        qoe_function: model.id,
        qoe_params: params,
        datasets: requestedDatasets(),
      });
      console.debug('[graph-timing] compare diagnostics fetch', { elapsedMs: Math.round(performance.now() - fetchStarted), runs: selectedRuns.length });
      baseMetrics = metrics;
      loadedDatasets = new Set(requestedDatasets());
      renderedMetrics = timed('compare aggregate metrics', () => aggregateMetrics(metrics!, grouping));
      await buildCharts();
      infoMessage = `Comparing ${selectedRuns.length} selected run(s).`;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      baseMetrics = null;
      renderedMetrics = null;
      builtCharts = [];
    } finally {
      loading = false;
    }
  }

  async function buildCharts(): Promise<void> {
    if (!renderedMetrics) {
      builtCharts = [];
      slideIndex = 0;
      totalSlides = 1;
      pluginCount = enabledPlugins().length;
      return;
    }
    const serial = ++chartBuildSerial;
    buildingCharts = true;
    chartBuildError = '';
    const plugins = enabledPlugins();
    pluginCount = plugins.length;
    try {
      const pluginsToBuild = galleryMode === 'slideshow' ? plugins.slice(slideIndex, slideIndex + 1) : plugins;
      const nextCharts = await scheduleGraphWork(() => timed('compare build charts', () => {
        if (collapseQoeComparison) {
          const combinedConfig = buildCombinedQoeChart(baseMetrics || renderedMetrics!);
          if (combinedConfig) applyPublicationStyle(combinedConfig, baseMetrics || renderedMetrics!, aliases, publicationSettings, 'QoE score over time');
          return [{
            id: 'combined-qoe',
            title: 'QoE score over time',
            description: chartDescription('combined-qoe'),
            controls: [],
            options: {},
            config: combinedConfig,
          }];
        }
        const builtById = new Map(builtCharts.map((chart) => [chart.id, chart]));
        for (const plugin of pluginsToBuild) {
          try {
            const config = plugin.build({
              metrics: renderedMetrics!,
              aliases,
              options: optionsFor(plugin),
            });
            if (config) {
              sortCategoricalChart(config, plugin.sortDirection || 'none');
              applyPublicationStyle(config, renderedMetrics!, aliases, publicationSettings, plugin.title);
            }
            builtById.set(plugin.id, {
              id: plugin.id,
              title: plugin.title,
              description: plugin.description || chartDescription(plugin.id),
              controls: plugin.controls || [],
              options: optionsFor(plugin),
              config,
            });
          } catch (err) {
            builtById.set(plugin.id, {
              id: plugin.id,
              title: plugin.title,
              description: plugin.description || chartDescription(plugin.id),
              controls: plugin.controls || [],
              options: optionsFor(plugin),
              config: null,
              error: err instanceof Error ? err.message : String(err),
            });
          }
        }
        return plugins.map((plugin) => builtById.get(plugin.id) || {
          id: plugin.id,
          title: plugin.title,
          description: plugin.description || chartDescription(plugin.id),
          controls: plugin.controls || [],
          options: optionsFor(plugin),
          config: null,
        });
      }));
      if (serial !== chartBuildSerial) return;
      builtCharts = nextCharts;
      totalSlides = Math.max(1, builtCharts.length);
      slideIndex = Math.min(slideIndex, totalSlides - 1);
    } catch (err) {
      if (serial === chartBuildSerial) {
        chartBuildError = err instanceof Error ? err.message : String(err);
        builtCharts = [];
      }
    } finally {
      if (serial === chartBuildSerial) buildingCharts = false;
    }
  }

  async function onGroupingChanged(): Promise<void> {
    localStorage.setItem(GROUPING_STORAGE_KEY, JSON.stringify(grouping));
    if (!baseMetrics) return;
    renderedMetrics = timed('compare aggregate metrics', () => aggregateMetrics(baseMetrics!, grouping));
    await buildCharts();
  }

  function onQoeModelChange(modelId: string): void {
    qoeState.selectedId = modelId;
    const model = activeModel();
    if (!qoeState.paramsById[model.id]) {
      qoeState.paramsById[model.id] = sanitizeQoeParams(model, {});
    }
    saveQoeState(QOE_STORAGE_KEY, qoeState);
  }

  function handleQoeModelSelect(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement | null;
    onQoeModelChange(target?.value || '');
  }

  function updateQoeField(name: string, value: string): void {
    const model = activeModel();
    const current = { ...(qoeState.paramsById[model.id] || {}) };
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      current[name] = numeric;
      qoeState.paramsById[model.id] = sanitizeQoeParams(model, current);
      saveQoeState(QOE_STORAGE_KEY, qoeState);
    }
  }

  function handleQoeFieldInput(name: string, event: Event): void {
    const target = event.currentTarget as HTMLInputElement | null;
    updateQoeField(name, target?.value || '');
  }

  function setGalleryMode(mode: 'slideshow' | 'grid'): void {
    galleryMode = mode;
    if (mode === 'slideshow') {
      slideIndex = 0;
    }
    void buildCharts();
  }

  function resetQoeParams(): void {
    const model = activeModel();
    qoeState.paramsById[model.id] = sanitizeQoeParams(model, {});
    saveQoeState(QOE_STORAGE_KEY, qoeState);
    reloadDiagnostics();
  }

  function applyQoeParams(): void {
    saveQoeState(QOE_STORAGE_KEY, qoeState);
    reloadDiagnostics();
  }

  function toggleQoeControls(): void {
    showQoeControls = !showQoeControls;
  }

  function openPluginModal(): void {
    showPluginModal = true;
  }

  function closePluginModal(): void {
    showPluginModal = false;
  }

  function pluginEnabled(pluginId: string): boolean {
    return enabledPluginIds.has(pluginId);
  }

  function togglePlugin(pluginId: string): void {
    const next = new Set(enabledPluginIds);
    if (next.has(pluginId)) next.delete(pluginId);
    else next.add(pluginId);
    enabledPluginIds = next;
    saveEnabledPluginIds();
    const missingDataset = requestedDatasets().some((dataset) => !loadedDatasets.has(dataset));
    if (missingDataset) void reloadDiagnostics();
    else void buildCharts();
  }

  function toggleCollapseQoeComparison(): void {
    collapseQoeComparison = !collapseQoeComparison;
    void buildCharts();
  }

  function moveSlide(dir: number): void {
    slideIndex = (slideIndex + dir + totalSlides) % totalSlides;
    void buildCharts();
  }

  onMount(async () => {
    saveSelectedCompareRuns(selectedRuns);
    try {
      const options = await getOptions();
      qoeModels = normalizeQoeModels(options.qoe_functions);
      qoeState = loadQoeState(QOE_STORAGE_KEY, qoeModels);
      enabledPluginIds = loadEnabledPluginIds();
      loadPluginOptions();
      loadFigureSettings();
      if (!qoeState.selectedId) qoeState.selectedId = qoeModels[0]?.id || '';
      await reloadDiagnostics();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      loading = false;
    }
  });
</script>

<header class="hero">
  <div class="hero-actions">
    <a class="button-link ghost" href="/" on:click|preventDefault={() => gotoRoute('home')}>Main menu</a>
    <a class="button-link ghost" href="/static/datasets.html" on:click|preventDefault={() => gotoRoute('datasets')}>Datasets</a>
    <a class="button-link ghost" href="/static/index.html" on:click|preventDefault={() => gotoRoute('control')}>Single Run</a>
    <a class="button-link ghost" href="/static/aggregate.html" on:click|preventDefault={() => gotoRoute('batch')}>Batch Planner</a>
    <a class="button-link active" aria-current="page" href="/static/compare.html" on:click|preventDefault={() => gotoRoute('compare')}>Compare Runs</a>
    <a class="button-link ghost" href="/static/reports.html" on:click|preventDefault={() => gotoRoute('reports')}>Reports</a>
    <a class="button-link ghost" href="/static/extra.html" on:click|preventDefault={() => gotoRoute('extra')}>Extra</a>
    <a class="button-link ghost" href="/static/settings.html" on:click|preventDefault={() => gotoRoute('settings')}>Settings</a>
  </div>
</header>

<main>
  <section class="card">
    <div class="section-header">
      <h2>Diagnostics</h2>
      <p class="subtle">{infoMessage}</p>
    </div>

    <div class="compare-controls">
      <div>
        <p class="eyebrow micro">Group by</p>
        <div class="chip-row">
          <label class="pill-toggle"><input type="checkbox" bind:checked={grouping.byAbr} on:change={onGroupingChanged} /> ABR</label>
          <label class="pill-toggle"><input type="checkbox" bind:checked={grouping.byPredictor} on:change={onGroupingChanged} /> FoV predictor</label>
          <label class="pill-toggle"><input type="checkbox" bind:checked={grouping.byPredictorPreset} disabled={!grouping.byPredictor} on:change={onGroupingChanged} /> Separate presets</label>
          <label class="pill-toggle"><input type="checkbox" bind:checked={grouping.byBandwidthPredictor} on:change={onGroupingChanged} /> Bandwidth predictor</label>
          <label class="pill-toggle"><input type="checkbox" bind:checked={grouping.byNetwork} on:change={onGroupingChanged} /> Network type</label>
        </div>
      </div>
      <div>
        <p class="eyebrow micro">Aggregation</p>
        <select bind:value={grouping.mode} on:change={onGroupingChanged}>
          <option value="mean">Mean</option>
          <option value="median">Median</option>
        </select>
      </div>
      <div>
        <p class="eyebrow micro">Layout</p>
        <div class="button-group mode-toggle">
          <button class={`ghost small ${galleryLayout === 'horizontal' ? 'active' : ''}`} on:click={() => (galleryLayout = 'horizontal')}>Horizontal</button>
          <button class={`ghost small ${galleryLayout === 'vertical' ? 'active' : ''}`} on:click={() => (galleryLayout = 'vertical')}>Vertical</button>
        </div>
      </div>
      <div class="section-actions compare-action-row">
        <button class={`ghost small ${collapseQoeComparison ? 'active' : ''}`} on:click={toggleCollapseQoeComparison}>
          {collapseQoeComparison ? 'Show detailed graphs' : 'Collapse into single QoE graph'}
        </button>
        <button class="ghost small" on:click={toggleQoeControls}>
          {showQoeControls ? 'Hide QoE controls' : 'Show QoE controls'}
        </button>
        <button class={`ghost small ${showFigureControls ? 'active' : ''}`} on:click={() => (showFigureControls = !showFigureControls)}>
          {showFigureControls ? 'Hide figure export' : 'Paper figure export'}
        </button>
      </div>
    </div>

    {#if showFigureControls}
      <div class="figure-controls">
        <div class="figure-control-grid">
          <label>Paper preset
            <select value={exportSettings.preset} on:change={handleFigurePreset}>
              {#each FIGURE_PRESETS as preset}<option value={preset.id}>{preset.label}</option>{/each}
            </select>
          </label>
          <label>Width
            <input type="number" min="0.1" step="0.01" value={exportSettings.width} on:change={(event) => updateExportNumber('width', event)} />
          </label>
          <label>Height
            <input type="number" min="0.1" step="0.01" value={exportSettings.height} on:change={(event) => updateExportNumber('height', event)} />
          </label>
          <label>Unit
            <select bind:value={exportSettings.unit} on:change={() => { exportSettings = { ...exportSettings, preset: 'custom' }; saveFigureSettings(); }}>
              <option value="in">inches</option><option value="cm">centimetres</option><option value="mm">millimetres</option><option value="px">pixels</option>
            </select>
          </label>
          <label>DPI
            <input type="number" min="72" max="1200" step="1" value={exportSettings.dpi} on:change={(event) => updateExportNumber('dpi', event)} />
          </label>
          <label>Format
            <select bind:value={exportSettings.format} on:change={() => saveFigureSettings()}>
              <option value="svg">SVG (vector)</option><option value="pdf">PDF (vector print)</option><option value="png">PNG</option>
            </select>
          </label>
          <label>Font size
            <input type="number" min="6" max="14" step="0.25" value={publicationSettings.baseFontPt} on:change={updatePublicationNumber} />
          </label>
          <label>Legend
            <select bind:value={publicationSettings.legendPosition} on:change={() => saveFigureSettings(true)}>
              <option value="bottom">Bottom</option><option value="top">Top</option><option value="left">Left</option><option value="right">Right</option>
            </select>
          </label>
          <label class="figure-check"><input type="checkbox" bind:checked={publicationSettings.showTitle} on:change={() => saveFigureSettings(true)} /> Include title</label>
          <label class="figure-check"><input type="checkbox" bind:checked={publicationSettings.grayscale} on:change={() => saveFigureSettings(true)} /> Grayscale preview/export</label>
        </div>
        <label class="font-family-control">Figure font family
          <input value={publicationSettings.fontFamily} on:change={updateFontFamily} />
        </label>
        <div class="figure-config-actions">
          <span class="micro subtle">SVG is preferred. PDF opens the browser’s vector print dialog; choose “Save as PDF”. PNG is rendered directly at the selected size and DPI.</span>
          <button class="ghost small" type="button" on:click={downloadAdvancedCompareConfig}>Save figure configuration</button>
          <label class="button-link ghost small config-upload">Load figure configuration<input type="file" accept="application/json,.json" on:change={loadAdvancedCompareConfig} /></label>
        </div>
      </div>
    {/if}

    {#if showQoeControls && qoeModels.length}
      {@const model = activeModel()}
      <div class="qoe-controls">
        <div class="qoe-head">
          <select aria-label="QoE model" value={qoeState.selectedId} on:change={handleQoeModelSelect}>
            {#each qoeModels as modelOption}
              <option value={modelOption.id}>{modelOption.label}</option>
            {/each}
          </select>
          <p class="micro subtle">{model.equation}{model.description ? ` - ${model.description}` : ''}</p>
        </div>
        <div class="qoe-grid">
          {#each model.fields || [] as field}
            <label>{field.label || field.name}
              <input
                type="number"
                step={field.step ?? 'any'}
                min={field.min}
                max={field.max}
                value={qoeState.paramsById[model.id]?.[field.name] ?? field.default ?? 0}
                on:change={(e) => handleQoeFieldInput(field.name, e)}
              />
            </label>
          {/each}
        </div>
        <div class="qoe-actions">
          <button class="ghost small" on:click={applyQoeParams}>Apply QoE params</button>
          <button class="ghost small" on:click={resetQoeParams}>Reset QoE params</button>
        </div>
      </div>
    {/if}
  </section>

  <section class="card segment-compare">
    <div class="section-header">
      <div>
        <h2>Segment tile-score comparison</h2>
        <p class="micro subtle">Compare each selected predictor’s tile scores against the same ground-truth target. Scores use a shared 0–1 color scale.</p>
      </div>
      <div class="section-actions">
        <label>Segment
          <input type="number" min="0" step="1" bind:value={segmentComparisonSegment} />
        </label>
        <button class={`ghost small ${showSegmentComparison ? 'active' : ''}`} on:click={() => { showSegmentComparison = !showSegmentComparison; if (showSegmentComparison && !segmentComparisonEntries.length) void loadSegmentComparison(); }}>
          {showSegmentComparison ? 'Hide tile comparison' : 'Compare tile scores'}
        </button>
        {#if showSegmentComparison}<button class="ghost small" on:click={() => void loadSegmentComparison()} disabled={segmentComparisonLoading}>Load segment</button>{/if}
      </div>
    </div>
    {#if showSegmentComparison}
      {#if segmentComparisonLoading}<p class="subtle">Loading selected segment…</p>{/if}
      {#if segmentComparisonMessage}<p class="subtle">{segmentComparisonMessage}</p>{/if}
      {#if segmentComparisonEntries.length}
        {@const reference = segmentComparisonEntries.find((entry) => entry.frame)?.frame}
        <div class="segment-score-grid">
          {#if reference}
            <article class="segment-score-card truth-card">
              <h3>Ground-truth target</h3><p class="micro subtle">Segment {reference.segment_idx} · {reference.ground_truth_mode || 'target'}</p>
              <svg viewBox={frameViewBox(reference)} aria-label="Ground truth tile scores">
                {#each reference.tile_ids as tileId}
                  {@const box = reference.tiles?.[tileId]?.bounds_xz || [0, 0, 0, 0]}
                  <rect role="button" tabindex="0" aria-label={`Select target tile ${tileId}`} x={Math.min(box[0], box[2])} y={Math.min(box[1], box[3])} width={Math.abs(box[2] - box[0])} height={Math.abs(box[3] - box[1])} fill={scoreColor(reference.actual?.[tileId], String(tileId) === segmentComparisonSelectedTile ? 0.96 : 0.78)} on:click={() => (segmentComparisonSelectedTile = String(tileId))} on:keydown={(event) => selectComparisonTile(event, tileId)}><title>T{tileId}: target {Number(reference.actual?.[tileId] || 0).toFixed(3)}</title></rect>
                {/each}
              </svg>
            </article>
          {/if}
          {#each segmentComparisonEntries as entry (entry.runId)}
            <article class="segment-score-card">
              <h3>{predictorLabel(entry.runId)}</h3>
              {#if entry.error}<p class="micro subtle">{entry.error}</p>
              {:else if entry.frame}
                <p class="micro subtle">MAE {Number(entry.frame.metrics?.mae || 0).toFixed(3)} · IoU {Number(entry.frame.metrics?.soft_iou || 0).toFixed(3)}</p>
                <svg viewBox={frameViewBox(entry.frame)} aria-label={`Predicted tile scores for ${predictorLabel(entry.runId)}`}>
                  {#each entry.frame.tile_ids as tileId}
                    {@const box = entry.frame.tiles?.[tileId]?.bounds_xz || [0, 0, 0, 0]}
                    <rect role="button" tabindex="0" aria-label={`Select predicted tile ${tileId}`} class:selected={String(tileId) === segmentComparisonSelectedTile} x={Math.min(box[0], box[2])} y={Math.min(box[1], box[3])} width={Math.abs(box[2] - box[0])} height={Math.abs(box[3] - box[1])} fill={scoreColor(entry.frame.predicted?.[tileId], String(tileId) === segmentComparisonSelectedTile ? 0.96 : 0.78)} on:click={() => (segmentComparisonSelectedTile = String(tileId))} on:keydown={(event) => selectComparisonTile(event, tileId)}><title>T{tileId}: prediction {Number(entry.frame.predicted?.[tileId] || 0).toFixed(3)} · target {Number(entry.frame.actual?.[tileId] || 0).toFixed(3)}</title></rect>
                  {/each}
                </svg>
                {#if segmentComparisonSelectedTile}<p class="micro subtle">T{segmentComparisonSelectedTile}: p {Number(entry.frame.predicted?.[segmentComparisonSelectedTile] || 0).toFixed(3)} · t {Number(entry.frame.actual?.[segmentComparisonSelectedTile] || 0).toFixed(3)}</p>{/if}
              {/if}
            </article>
          {/each}
        </div>
      {/if}
    {/if}
  </section>

  <section class="card compare-gallery">
    <div class="section-header">
      <div>
        <h2>Charts (Registry plugins)</h2>
        <p class="micro subtle">Plugins: {pluginCount} • Cards: {builtCharts.length}</p>
        {#if buildingCharts}
          <p class="micro subtle">Generating graphs…</p>
        {/if}
      </div>
      <div class="section-actions">
        <div class="button-group mode-toggle">
          <button class={`ghost small ${galleryMode === 'slideshow' ? 'active' : ''}`} on:click={() => setGalleryMode('slideshow')}>Slideshow</button>
          <button class={`ghost small ${galleryMode === 'grid' ? 'active' : ''}`} on:click={() => setGalleryMode('grid')}>Grid</button>
        </div>
        <button class="ghost small" on:click={openPluginModal} disabled={collapseQoeComparison}>Plugins</button>
        {#if galleryMode === 'slideshow'}
          <div class="button-group slide-nav active">
            <button class="ghost small" on:click={() => moveSlide(-1)}>Prev</button>
            <span class="micro subtle">{slideIndex + 1}/{totalSlides}</span>
            <button class="ghost small" on:click={() => moveSlide(1)}>Next</button>
          </div>
        {:else}
          <span class="micro subtle">All {builtCharts.length}</span>
        {/if}
      </div>
    </div>

    {#if loading}
      <p class="subtle">Loading diagnostics…</p>
    {:else if error}
      <p class="subtle">Failed to load diagnostics: {error}</p>
    {:else if chartBuildError}
      <p class="subtle">Failed to generate charts: {chartBuildError}</p>
    {:else}
      {#if galleryMode === 'grid'}
        <div class={`grid-gallery ${galleryLayout === 'vertical' ? 'vertical' : 'horizontal'}`}>
          {#each builtCharts as chart (chart.id)}
            <PluginChart chartId={chart.id} title={chart.title} description={chart.description} controls={chart.controls} options={chart.options} config={chart.config} error={chart.error || ''} {exportSettings} {publicationSettings} on:optionchange={(event) => updatePluginOption(chart.id, event.detail)} />
          {/each}
        </div>
      {:else}
        {#if builtCharts.length}
          {@const current = builtCharts[Math.max(0, Math.min(slideIndex, builtCharts.length - 1))]}
          <div class={`grid-gallery ${galleryLayout === 'vertical' ? 'vertical' : 'horizontal'}`}>
            {#key `${slideIndex}-${current?.id || 'none'}`}
              <PluginChart chartId={current?.id || 'unknown'} title={current?.title || ''} description={current?.description || ''} controls={current?.controls || []} options={current?.options || {}} config={current?.config || null} error={current?.error || ''} {exportSettings} {publicationSettings} on:optionchange={(event) => updatePluginOption(current?.id || 'unknown', event.detail)} />
            {/key}
          </div>
        {:else}
          <p class="subtle">No chart data yet.</p>
        {/if}
      {/if}
    {/if}
  </section>
</main>

{#if showPluginModal}
  <div class="plugin-modal-backdrop" on:pointerdown|self={closePluginModal}>
    <section class="card plugin-modal">
      <div class="section-header">
        <div>
          <h3>Select graph plugins</h3>
          <p class="subtle">{pluginCount} enabled of {resolvedPluginRegistry().length}</p>
        </div>
        <div class="section-actions">
          <button type="button" class="ghost small" on:click={closePluginModal}>Close</button>
        </div>
      </div>

      <div class="plugin-category-row">
        {#each pluginCategories() as category}
          <button
            type="button"
            class={`ghost small ${pluginCategoryFilter === category ? 'active' : ''}`}
            on:click={() => (pluginCategoryFilter = category)}
          >
            {pluginCategoryLabels[category]} ({pluginCategoryCount(category)})
          </button>
        {/each}
      </div>

      <div class="plugin-picker-list">
        {#each visiblePluginRegistry() as plugin}
          <label class="plugin-picker-row">
            <span class="plugin-picker-copy">
              <strong>{plugin.title}</strong>
              <span class="micro plugin-category-label">{pluginCategoryLabels[plugin.category || 'global']}</span>
              {#if plugin.description}
                <span class="micro subtle">{plugin.description}</span>
              {/if}
            </span>
            <input
              type="checkbox"
              checked={pluginEnabled(plugin.id)}
              on:change={() => togglePlugin(plugin.id)}
            />
          </label>
        {/each}
      </div>
    </section>
  </div>
{/if}

<style>
  .segment-compare .section-actions label { display: grid; gap: 0.2rem; font-size: 0.72rem; color: var(--text-muted); }
  .segment-compare input { width: 5rem; min-height: 2rem; }
  .segment-score-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: 0.75rem; margin-top: 0.8rem; }
  .segment-score-card { padding: 0.7rem; border: 1px solid var(--card-border); border-radius: 10px; background: var(--soft-bg-alt); min-width: 0; }
  .segment-score-card h3 { margin: 0; font-size: 0.88rem; overflow-wrap: anywhere; }
  .segment-score-card p { margin: 0.3rem 0 0.5rem; }
  .segment-score-card svg { width: 100%; aspect-ratio: 1.3; display: block; background: rgba(255,255,255,0.55); border: 1px solid rgba(100,116,139,0.25); }
  .segment-score-card rect { stroke: rgba(15,23,42,0.35); stroke-width: 0.02; cursor: pointer; }
  .segment-score-card rect.selected { stroke: #fff; stroke-width: 0.07; }
  .figure-controls { margin-top: 0.85rem; padding: 0.8rem; border: 1px solid var(--card-border); border-radius: 10px; background: var(--soft-bg-alt); display: grid; gap: 0.7rem; }
  .figure-control-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(8.5rem, 1fr)); gap: 0.55rem; align-items: end; }
  .figure-control-grid label, .font-family-control { display: grid; gap: 0.25rem; color: var(--text-muted); font-size: 0.72rem; font-weight: 600; }
  .figure-control-grid select, .figure-control-grid input[type='number'], .font-family-control input { min-height: 2rem; padding: 0.3rem 0.45rem; border-radius: 7px; font-size: 0.78rem; }
  .figure-control-grid .figure-check { display: flex; align-items: center; min-height: 2rem; gap: 0.4rem; }
  .figure-config-actions { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: 0.5rem; }
  .figure-config-actions > span { margin-right: auto; max-width: 48rem; }
  .config-upload { position: relative; overflow: hidden; cursor: pointer; }
  .config-upload input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .compare-action-row {
    align-self: end;
  }

  .qoe-head {
    display: grid;
    gap: 0.35rem;
    align-items: start;
  }

  .plugin-modal-backdrop {
    position: fixed;
    inset: 0;
    z-index: 1400;
    background: var(--overlay-backdrop);
    backdrop-filter: blur(4px);
    display: grid;
    place-items: center;
    padding: 1rem;
  }

  .plugin-modal {
    width: min(720px, 96vw);
    max-height: 85vh;
    overflow: auto;
    display: grid;
    gap: 0.9rem;
    background: var(--modal-surface);
    border: 1px solid var(--modal-border);
    box-shadow: 0 22px 64px rgba(2, 6, 23, 0.8);
  }

  .plugin-picker-list {
    display: grid;
    gap: 0.65rem;
  }

  .plugin-category-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
  }

  .plugin-category-row button {
    border-radius: 999px;
  }

  .plugin-picker-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 0.8rem;
    align-items: center;
    padding: 0.7rem 0.8rem;
    border-radius: 12px;
    border: 1px solid var(--card-border);
    background: var(--soft-bg-alt);
  }

  .plugin-picker-copy {
    display: grid;
    gap: 0.18rem;
    min-width: 0;
  }

  .plugin-picker-copy strong {
    color: var(--text-main);
  }

  .plugin-category-label {
    justify-self: start;
    border: 1px solid rgba(148, 163, 184, 0.24);
    border-radius: 999px;
    padding: 0.04rem 0.38rem;
    color: var(--text-muted);
  }

  .plugin-picker-row input {
    width: 1rem;
    height: 1rem;
  }

  @media (max-width: 720px) {
    .plugin-picker-row {
      grid-template-columns: 1fr;
      justify-items: start;
    }
  }
</style>
