<script lang="ts">
  import { onMount } from 'svelte';
  import type { ChartConfiguration } from 'chart.js';
  import { getDiagnostics, getOptions } from '../core/api';
  import { loadRunAliases, loadSelectedCompareRuns, saveSelectedCompareRuns } from '../core/storage';
  import { loadQoeState, normalizeQoeModels, sanitizeQoeParams, saveQoeState, type QoEState } from '../core/qoe';
  import type { QoEModel, QoEParams, RunMetrics } from '../core/types';
  import { aggregateMetrics, type GroupingOptions } from '../features/advanced-compare/metrics';
  import { avgEfficiencyPlugin } from '../features/advanced-compare/plugins/avgEfficiency';
  import { bitrateQualityPlugin } from '../features/advanced-compare/plugins/bitrateQuality';
  import { bufferStabilityPlugin } from '../features/advanced-compare/plugins/bufferStability';
  import { componentBitratePlugin } from '../features/advanced-compare/plugins/componentBitrate';
  import { fovQualityPlugin } from '../features/advanced-compare/plugins/fovQuality';
  import { qoeCdfPlugin } from '../features/advanced-compare/plugins/qoeCdf';
  import { qoeStallPlugin } from '../features/advanced-compare/plugins/qoeStall';
  import { stallSummaryPlugin } from '../features/advanced-compare/plugins/stallSummary';
  import { temporalDeltaPlugin } from '../features/advanced-compare/plugins/temporalDelta';
  import { chartPluginRegistry } from '../features/advanced-compare/plugins/registry';
  import type { ChartPlugin } from '../features/advanced-compare/plugins/types';
  import PluginChart from '../features/advanced-compare/components/PluginChart.svelte';

  const QOE_STORAGE_KEY = 'compare_qoe_state_v4';

  const aliases = loadRunAliases();
  const query = new URLSearchParams(window.location.search);
  const runIdsFromUrl = (query.get('runs') || '').split(',').map((x) => x.trim()).filter(Boolean);
  let selectedRuns = runIdsFromUrl.length ? runIdsFromUrl : loadSelectedCompareRuns();

  let loading = true;
  let error = '';
  let infoMessage = '';

  let qoeModels: QoEModel[] = [];
  let qoeState: QoEState = { selectedId: '', paramsById: {} };

  let grouping: GroupingOptions = {
    byAbr: true,
    byPredictor: false,
    byNetwork: false,
    mode: 'mean',
  };

  let baseMetrics: RunMetrics | null = null;
  let renderedMetrics: RunMetrics | null = null;

  let galleryMode: 'slideshow' | 'grid' = 'slideshow';
  let slideIndex = 0;
  let totalSlides = 1;

  type BuiltChart = {
    id: string;
    title: string;
    config: ChartConfiguration | null;
    error?: string;
  };

  let builtCharts: BuiltChart[] = [];
  let pluginCount = 0;

  const fallbackPluginRegistry: ChartPlugin[] = [
    qoeCdfPlugin,
    qoeStallPlugin,
    avgEfficiencyPlugin,
    fovQualityPlugin,
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
      const metrics = await getDiagnostics({
        run_ids: selectedRuns,
        qoe_function: model.id,
        qoe_params: params,
      });
      baseMetrics = metrics;
      renderedMetrics = aggregateMetrics(metrics, grouping);
      buildCharts();
      infoMessage = `Comparing ${selectedRuns.length} selected run(s): ${selectedRuns.join(', ')}`;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      baseMetrics = null;
      renderedMetrics = null;
      builtCharts = [];
    } finally {
      loading = false;
    }
  }

  function buildCharts(): void {
    if (!renderedMetrics) {
      builtCharts = [];
      slideIndex = 0;
      totalSlides = 1;
      pluginCount = resolvedPluginRegistry().length;
      return;
    }
    const plugins = resolvedPluginRegistry();
    pluginCount = plugins.length;
    builtCharts = plugins.map((plugin) => {
      try {
        return {
          id: plugin.id,
          title: plugin.title,
          config: plugin.build({
            metrics: renderedMetrics!,
            aliases,
          }),
        };
      } catch (err) {
        return {
          id: plugin.id,
          title: plugin.title,
          config: null,
          error: err instanceof Error ? err.message : String(err),
        };
      }
    });
    totalSlides = Math.max(1, builtCharts.length);
    slideIndex = Math.min(slideIndex, totalSlides - 1);
  }

  function onGroupingChanged(): void {
    if (!baseMetrics) return;
    renderedMetrics = aggregateMetrics(baseMetrics, grouping);
    buildCharts();
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

  function moveSlide(dir: number): void {
    slideIndex = (slideIndex + dir + totalSlides) % totalSlides;
  }

  onMount(async () => {
    saveSelectedCompareRuns(selectedRuns);
    try {
      const options = await getOptions();
      qoeModels = normalizeQoeModels(options.qoe_functions);
      qoeState = loadQoeState(QOE_STORAGE_KEY, qoeModels);
      if (!qoeState.selectedId) qoeState.selectedId = qoeModels[0]?.id || '';
      await reloadDiagnostics();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      loading = false;
    }
  });
</script>

<header class="hero">
  <div>
    <p class="eyebrow">Prism VPCC Lab</p>
    <h1>Advanced Compare</h1>
    <p class="lede">Plugin-based diagnostics graphs for ABR and FoV evaluation.</p>
  </div>
  <div class="hero-actions">
    <a class="button-link ghost" href="/">Main menu</a>
    <a class="button-link ghost" href="/static/index.html">Control Center</a>
    <a class="button-link ghost" href="/static/aggregate.html">Batch planner</a>
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
        <p class="eyebrow micro">QoE function</p>
        <select value={qoeState.selectedId} on:change={handleQoeModelSelect}>
          {#each qoeModels as model}
            <option value={model.id}>{model.label}</option>
          {/each}
        </select>
      </div>
    </div>

    {#if qoeModels.length}
      {@const model = activeModel()}
      <div class="qoe-controls">
        <p class="micro subtle">{model.equation}{model.description ? ` - ${model.description}` : ''}</p>
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

  <section class="card compare-gallery">
    <div class="section-header">
      <div>
        <h2>Charts (Registry plugins)</h2>
        <p class="micro subtle">Plugins: {pluginCount} • Cards: {builtCharts.length}</p>
      </div>
      <div class="section-actions">
        <div class="button-group mode-toggle">
          <button class={`ghost small ${galleryMode === 'slideshow' ? 'active' : ''}`} on:click={() => setGalleryMode('slideshow')}>Slideshow</button>
          <button class={`ghost small ${galleryMode === 'grid' ? 'active' : ''}`} on:click={() => setGalleryMode('grid')}>Grid</button>
        </div>
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
    {:else}
      {#if galleryMode === 'grid'}
        <div class="grid-gallery">
          {#each builtCharts as chart (chart.id)}
            <PluginChart title={chart.title} config={chart.config} error={chart.error || ''} />
          {/each}
        </div>
      {:else}
        {#if builtCharts.length}
          {@const current = builtCharts[Math.max(0, Math.min(slideIndex, builtCharts.length - 1))]}
          <div class="grid-gallery">
            {#key `${slideIndex}-${current?.id || 'none'}`}
              <PluginChart title={current?.title || ''} config={current?.config || null} error={current?.error || ''} />
            {/key}
          </div>
        {:else}
          <p class="subtle">No chart data yet.</p>
        {/if}
      {/if}
    {/if}
  </section>
</main>
