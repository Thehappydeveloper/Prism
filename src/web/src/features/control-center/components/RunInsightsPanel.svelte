<script lang="ts">
  import type { ChartConfiguration } from 'chart.js';
  import ChartCanvas from '../../../components/ChartCanvas.svelte';
  import type { QoEState } from '../../../core/qoe';
  import type { QoEModel } from '../../../core/types';

  interface CompareCard {
    runId: string;
    label: string;
    chart: ChartConfiguration | null;
    error?: string;
  }

  export let compareMode = false;
  export let comparing = false;
  export let compareCards: CompareCard[] = [];
  export let selectedRunId = '';
  export let error = '';

  export let showBandwidth = true;
  export let showBitrate = true;
  export let showBuffer = true;
  export let showQoe = true;
  export let tailMode: 'drain' | 'playback' = 'drain';
  export let showQoeControls = false;

  export let qoeModels: QoEModel[] = [];
  export let qoeState: QoEState = { selectedId: '', paramsById: {} };
  export let activeQoeModel: QoEModel | null = null;
  export let chartConfig: ChartConfiguration | null = null;

  export let onRefreshCharts: () => void = () => {};
  export let onToggleTailMode: () => void = () => {};
  export let onExitCompareMode: () => void = () => {};
  export let onToggleQoeControlsPanel: () => void = () => {};
  export let onQoeModelSelectChange: (event: Event) => void = () => {};
  export let onQoeFieldInputChange: (field: string, event: Event) => void = () => {};
  export let onApplyQoeToChart: () => void = () => {};
  export let onResetQoeForChart: () => void = () => {};
</script>

<section class="card">
  <div class="section-header">
    <div>
      <h2>Run insights</h2>
      <p class="subtle">
        {#if compareMode}
          {comparing ? 'Preparing comparison…' : `Comparing ${compareCards.length} run(s)`}
        {:else}
          {selectedRunId ? `Selected: ${selectedRunId}` : 'Choose a run to inspect.'}
        {/if}
      </p>
    </div>
    <div class="toggles">
      <label class="pill-toggle"><input type="checkbox" bind:checked={showBandwidth} on:change={onRefreshCharts} /> Bandwidth</label>
      <label class="pill-toggle"><input type="checkbox" bind:checked={showBitrate} on:change={onRefreshCharts} /> Bitrate</label>
      <label class="pill-toggle"><input type="checkbox" bind:checked={showBuffer} on:change={onRefreshCharts} /> Buffer</label>
      <label class="pill-toggle"><input type="checkbox" bind:checked={showQoe} on:change={onRefreshCharts} /> QoE</label>
      <button class="ghost small" on:click={onToggleTailMode}>
        X-axis: {tailMode === 'drain' ? 'buffer drains' : 'stop at last segment'}
      </button>
      {#if compareMode}
        <button class="ghost small" on:click={onExitCompareMode}>Exit compare</button>
      {/if}
    </div>
  </div>

  <div class="qoe-toggle-row">
    <button class="ghost small" on:click={onToggleQoeControlsPanel}>
      {showQoeControls ? 'Hide QoE controls' : 'Show QoE controls'}
    </button>
  </div>

  {#if showQoeControls && qoeModels.length}
    {@const model = activeQoeModel}
    {#if model}
      <div class="qoe-controls">
        <div class="qoe-head">
          <label>QoE function
            <select value={qoeState.selectedId} on:change={onQoeModelSelectChange}>
              {#each qoeModels as m}
                <option value={m.id}>{m.label}</option>
              {/each}
            </select>
          </label>
          <p class="micro subtle">{model.equation}{model.description ? ` - ${model.description}` : ''}</p>
        </div>
        <div class="qoe-grid">
          {#each model.fields || [] as field}
            <label>{field.label || field.name}
              <input
                type="number"
                min={field.min}
                max={field.max}
                step={field.step || 'any'}
                value={qoeState.paramsById[model.id]?.[field.name] ?? field.default ?? 0}
                on:change={(e) => onQoeFieldInputChange(field.name, e)}
              />
            </label>
          {/each}
        </div>
        <div class="qoe-actions">
          <button class="ghost small" on:click={onApplyQoeToChart}>Apply QoE params</button>
          <button class="ghost small" on:click={onResetQoeForChart}>Reset QoE params</button>
        </div>
      </div>
    {/if}
  {/if}

  {#if error}
    <p class="subtle">{error}</p>
  {/if}
  {#if compareMode}
    <div class="compare-grid">
      {#if comparing}
        <p class="subtle">Loading selected runs…</p>
      {:else if !compareCards.length}
        <p class="subtle">No runs selected for compare.</p>
      {:else}
        {#each compareCards as card (card.runId)}
          <div class="compare-card card">
            <div class="section-header">
              <h3>{card.label}</h3>
            </div>
            {#if card.error}
              <p class="subtle">{card.error}</p>
            {/if}
            <ChartCanvas config={card.chart} height={280} />
          </div>
        {/each}
      {/if}
    </div>
  {:else}
    <div class="chart-block">
      <ChartCanvas config={chartConfig} height={360} />
    </div>
  {/if}
</section>
