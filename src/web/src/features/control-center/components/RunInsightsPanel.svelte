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

  interface OracleCounterpart {
    runId: string;
    type: 'fov' | 'bandwidth' | 'both';
    label: string;
    reason: string;
  }

  export let compareMode = false;
  export let comparing = false;
  export let chartLoading = false;
  export let compareLayout: 'horizontal' | 'vertical' = 'horizontal';
  export let compareCollapseQoe = false;
  export let compareCards: CompareCard[] = [];
  export let selectedRunId = '';
  export let error = '';
  export let oracleCounterparts: OracleCounterpart[] = [];
  export let oracleCounterpartsLoading = false;
  export let selectedOracleCounterpartRunId = '';

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
  export let onCompareWithOracleCounterpart: (runId: string) => void = () => {};

  $: resolvedQoeModel =
    activeQoeModel || qoeModels.find((m) => m.id === qoeState.selectedId) || qoeModels[0] || null;
  $: selectedOracleCounterpart = oracleCounterparts.find((entry) => entry.runId === selectedOracleCounterpartRunId) || null;
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
  </div>

  {#if selectedRunId && !compareMode}
    <div class="oracle-counterpart-strip">
      <div>
        <strong>Oracle counterpart</strong>
        <p class="micro subtle">
          {#if oracleCounterpartsLoading}
            Searching same-group runs with matching scenario fields…
          {:else if selectedOracleCounterpart}
            {selectedOracleCounterpart.reason}
          {:else}
            No matching oracle run found in this run group.
          {/if}
        </p>
      </div>
      <div class="oracle-counterpart-actions">
        <select bind:value={selectedOracleCounterpartRunId} disabled={oracleCounterpartsLoading || !oracleCounterparts.length} aria-label="Oracle counterpart">
          {#if !oracleCounterparts.length}
            <option value="">No counterpart</option>
          {:else}
            {#each oracleCounterparts as counterpart}
              <option value={counterpart.runId}>{counterpart.label}</option>
            {/each}
          {/if}
        </select>
        <button
          class="ghost small"
          type="button"
          disabled={!selectedOracleCounterpartRunId || oracleCounterpartsLoading}
          on:click={() => onCompareWithOracleCounterpart(selectedOracleCounterpartRunId)}
        >
          Add to compare
        </button>
      </div>
    </div>
  {/if}

  <details class="insight-controls">
    <summary>Insight controls</summary>
    <div class="insight-controls-body">
      <div class="toggles">
        <label class="pill-toggle"><input type="checkbox" bind:checked={showBandwidth} on:change={onRefreshCharts} /> Bandwidth</label>
        <label class="pill-toggle"><input type="checkbox" bind:checked={showBitrate} on:change={onRefreshCharts} /> Bitrate</label>
        <label class="pill-toggle"><input type="checkbox" bind:checked={showBuffer} on:change={onRefreshCharts} /> Buffer</label>
        <label class="pill-toggle"><input type="checkbox" bind:checked={showQoe} on:change={onRefreshCharts} /> QoE</label>
      </div>

      <div class="qoe-toggle-row">
        <button class="ghost small" on:click={onToggleTailMode}>
          X-axis: {tailMode === 'drain' ? 'buffer drains' : 'stop at last segment'}
        </button>
        <button class="ghost small" on:click={onToggleQoeControlsPanel}>
          {showQoeControls ? 'Hide QoE controls' : 'Show QoE controls'}
        </button>
        {#if compareMode}
          <button class="ghost small" on:click={onExitCompareMode}>Exit compare</button>
        {/if}
      </div>
      {#if compareMode}
        <div class="qoe-toggle-row">
          <span class="micro subtle">Layout</span>
          <button class={`ghost small ${compareLayout === 'horizontal' ? 'active' : ''}`} on:click={() => (compareLayout = 'horizontal')}>
            Horizontal
          </button>
          <button class={`ghost small ${compareLayout === 'vertical' ? 'active' : ''}`} on:click={() => (compareLayout = 'vertical')}>
            Vertical
          </button>
          <button class={`ghost small ${compareCollapseQoe ? 'active' : ''}`} on:click={() => { compareCollapseQoe = !compareCollapseQoe; onRefreshCharts(); }}>
            {compareCollapseQoe ? 'Show detailed graphs' : 'Collapse into single QoE graph'}
          </button>
        </div>
      {/if}

      {#if showQoeControls}
        {#if !qoeModels.length}
          <p class="subtle">No QoE models are available from the backend options.</p>
        {:else if !resolvedQoeModel}
          <p class="subtle">QoE model selection is not initialized yet.</p>
        {:else}
          <div class="qoe-controls">
            <div class="qoe-head">
              <select aria-label="QoE model" value={qoeState.selectedId} on:change={onQoeModelSelectChange}>
                {#each qoeModels as m}
                  <option value={m.id}>{m.label}</option>
                {/each}
              </select>
              <p class="micro subtle">
                {resolvedQoeModel.equation}
                {resolvedQoeModel.description ? ` - ${resolvedQoeModel.description}` : ''}
              </p>
            </div>
            <div class="qoe-grid">
              {#each resolvedQoeModel.fields || [] as field}
                <label>{field.label || field.name}
                  <input
                    type="number"
                    min={field.min}
                    max={field.max}
                    step={field.step || 'any'}
                    value={qoeState.paramsById[resolvedQoeModel.id]?.[field.name] ?? field.default ?? 0}
                    on:input={(e) => onQoeFieldInputChange(field.name, e)}
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
    </div>
  </details>

  {#if error}
    <p class="subtle">{error}</p>
  {/if}
  {#if compareMode}
    <div class={`compare-grid ${compareLayout === 'vertical' ? 'vertical' : 'horizontal'}`}>
      {#if comparing}
        <p class="subtle">Generating comparison graphs…</p>
      {:else if !compareCards.length}
        <p class="subtle">No runs selected for compare.</p>
      {:else}
        {#each compareCards as card (card.runId)}
          <div class={`compare-card card ${card.runId === 'combined-qoe' ? 'combined-qoe' : ''}`}>
            <div class="section-header">
              <h3>{card.label}</h3>
            </div>
            {#if card.error}
              <p class="subtle">{card.error}</p>
            {/if}
            <ChartCanvas config={card.chart} height={card.runId === 'combined-qoe' ? 380 : 280} />
          </div>
        {/each}
      {/if}
    </div>
  {:else}
    <div class="chart-block">
      {#if chartLoading}
        <div class="chart-loading">Generating graph…</div>
      {/if}
      <ChartCanvas config={chartConfig} height={360} />
    </div>
  {/if}
</section>

<style>
  .insight-controls {
    margin: 0.2rem 0 0.75rem;
    border: 1px solid var(--field-border);
    border-radius: 12px;
    background: var(--soft-bg-alt);
    overflow: hidden;
  }

  .oracle-counterpart-strip {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: 0.75rem;
    margin: 0.35rem 0 0.75rem;
    padding: 0.55rem 0.65rem;
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-radius: 8px;
    background: rgba(56, 189, 248, 0.045);
  }

  .oracle-counterpart-strip strong {
    display: block;
    font-size: 0.88rem;
  }

  .oracle-counterpart-strip p {
    margin: 0.1rem 0 0;
  }

  .oracle-counterpart-actions {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    min-width: min(100%, 28rem);
  }

  .oracle-counterpart-actions select {
    min-width: 15rem;
  }

  .insight-controls > summary {
    cursor: pointer;
    padding: 0.58rem 0.75rem;
    font-weight: 650;
    color: var(--text-muted);
    list-style: none;
    user-select: none;
  }

  .insight-controls > summary::-webkit-details-marker {
    display: none;
  }

  .insight-controls > summary::after {
    content: '▾';
    float: right;
    opacity: 0.8;
  }

  .insight-controls:not([open]) > summary::after {
    content: '▸';
  }

  .insight-controls-body {
    border-top: 1px solid var(--field-border);
    padding: 0.7rem 0.75rem 0.75rem;
    display: grid;
    gap: 0.65rem;
  }

  .qoe-toggle-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    align-items: center;
  }

  .qoe-toggle-row button.active {
    background: var(--btn-main);
    color: var(--btn-text);
    border-color: transparent;
  }

  .compare-grid.vertical {
    grid-template-columns: 1fr;
  }

  .compare-grid.vertical :global(.chart-shell) {
    min-height: 340px;
  }

  .compare-card.combined-qoe {
    grid-column: 1 / -1;
  }

  .chart-block {
    position: relative;
  }

  .chart-loading {
    position: absolute;
    top: 0.75rem;
    right: 0.75rem;
    z-index: 2;
    padding: 0.35rem 0.55rem;
    border-radius: 999px;
    border: 1px solid var(--card-border);
    background: var(--soft-bg);
    color: var(--text-muted);
    font-size: 0.82rem;
  }
</style>
