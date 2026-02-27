<script lang="ts">
  import type { RunConfig, RunSummary } from '../../../core/types';
  import type { RunRow } from '../runExplorerUtils';

  export let runs: RunSummary[] = [];
  export let filteredRunRows: RunRow[] = [];
  export let selectedRunIds = new Set<string>();
  export let selectedRunId = '';
  export let expandedRunIds = new Set<string>();
  export let loadingDetailIds = new Set<string>();
  export let runConfigCache: Record<string, RunConfig | null> = {};
  export let runAliases: Record<string, string> = {};

  export let runSearch = '';
  export let statusFilter: 'all' | 'queued' | 'running' | 'finished' | 'failed' = 'all';
  export let runtimeFilter: 'all' | 'lt60' | '60to180' | 'gt180' | 'unfinished' = 'all';

  export let onSelectAllRuns: () => void = () => {};
  export let onSelectVisibleRuns: () => void = () => {};
  export let onInvertVisibleSelection: () => void = () => {};
  export let onClearSelection: () => void = () => {};
  export let onCompareSelectedInControl: () => void = () => {};
  export let onAdvancedCompare: () => void = () => {};
  export let onDeleteSelected: () => void = () => {};

  export let onRowCheckboxClick: (rowId: string, idx: number, event: MouseEvent) => void = () => {};
  export let onRowSelectClick: (rowId: string, idx: number, event: MouseEvent) => void = () => {};
  export let onRowSelectKeydown: (rowId: string, idx: number, event: KeyboardEvent) => void = () => {};
  export let onInspectRun: (runId: string) => void = () => {};
  export let onToggleRunDetails: (runId: string) => void = () => {};
  export let onRemoveRun: (runId: string) => void = () => {};
  export let onAliasChange: (runId: string, event: Event) => void = () => {};

  export let formatRuntime: (sec: number | null) => string = () => '—';
  export let formatDateTime: (value: string | undefined) => string = () => '—';
  export let statusDotClass: (status: string) => string = () => 'status-dot unknown';
  export let basenamePath: (path: string | null | undefined) => string = () => '—';
</script>

<section class="card">
  <div class="section-header">
    <div>
      <h2>Runs Explorer</h2>
      <p class="subtle">Compact list with quick multi-select and expandable details.</p>
    </div>
    <div class="section-actions">
      <button class="ghost small" on:click={onSelectAllRuns}>Select visible</button>
      <button class="ghost small" on:click={onSelectVisibleRuns}>Select filtered</button>
      <button class="ghost small" on:click={onInvertVisibleSelection}>Invert</button>
      <button class="ghost small" on:click={onClearSelection}>Clear</button>
      <button class="ghost small" on:click={onCompareSelectedInControl}>Compare selected</button>
      <button class="ghost small" on:click={onAdvancedCompare}>Advanced compare</button>
      <button class="ghost small" on:click={onDeleteSelected}>Delete selected</button>
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
        <option value="finished">Finished</option>
        <option value="failed">Failed</option>
      </select>
    </label>
    <label>Runtime
      <select bind:value={runtimeFilter}>
        <option value="all">All</option>
        <option value="lt60">&lt; 60s</option>
        <option value="60to180">60s to 3m</option>
        <option value="gt180">&gt; 3m</option>
        <option value="unfinished">No finish time</option>
      </select>
    </label>
    <p class="subtle">{filteredRunRows.length} / {runs.length} visible • {selectedRunIds.size} selected</p>
  </div>

  <div class="runs-compact-list">
    {#if !filteredRunRows.length}
      <p class="subtle">No runs match current filters.</p>
    {:else}
      {#each filteredRunRows as row, idx (row.run.run_id)}
        <div class={`run-card-compact ${selectedRunIds.has(row.run.run_id) ? 'selected' : ''} ${selectedRunId === row.run.run_id ? 'active' : ''}`}>
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
            >
              <span class={statusDotClass(row.run.status)}></span>
              <span class="run-primary">
                <strong>{row.alias || row.run.run_id}</strong>
                {#if row.alias}
                  <span class="micro subtle">{row.run.run_id}</span>
                {/if}
              </span>
              <span class="run-meta-chip">{row.run.status}</span>
              <span class="run-meta-chip">{formatRuntime(row.runtimeSec)}</span>
            </button>
            <div class="run-actions-icons">
              <button class="ghost small icon-btn" title="Inspect" aria-label="Inspect" on:click={() => onInspectRun(row.run.run_id)}>⌕</button>
              <button class="ghost small icon-btn" title="Expand" aria-label="Expand" on:click={() => onToggleRunDetails(row.run.run_id)}>
                {expandedRunIds.has(row.run.run_id) ? '▾' : '▸'}
              </button>
              <button class="ghost small icon-btn" title="Delete" aria-label="Delete" on:click={() => onRemoveRun(row.run.run_id)}>✕</button>
            </div>
          </div>

          {#if expandedRunIds.has(row.run.run_id)}
            {@const cfg = runConfigCache[row.run.run_id]}
            <div class="run-card-extra">
              <label>Alias
                <input
                  class="run-alias"
                  placeholder="Alias"
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
                  <div><span class="micro subtle">Predictor</span><span>{cfg?.predictor || 'None'}</span></div>
                  <div>
                    <span class="micro subtle">Network</span>
                    <span class="ellipsis" title={cfg?.network_trace || ''}>{basenamePath(cfg?.network_trace)}</span>
                  </div>
                  <div>
                    <span class="micro subtle">FoV trace</span>
                    <span class="ellipsis" title={cfg?.fov_trace || ''}>{basenamePath(cfg?.fov_trace)}</span>
                  </div>
                </div>
              {/if}
              <div class="detail-links">
                {#if row.run.events_path}<a href={row.run.events_path} target="_blank">events.json</a>{/if}
                {#if row.run.log_path}<a href={row.run.log_path} target="_blank">run.log</a>{/if}
              </div>
            </div>
          {/if}
        </div>
      {/each}
    {/if}
  </div>
</section>

<style>
  .runs-compact-toolbar {
    display: grid;
    grid-template-columns: minmax(220px, 1.2fr) repeat(2, minmax(160px, 0.7fr)) auto;
    gap: 0.5rem;
    align-items: end;
    margin-top: 0.75rem;
    margin-bottom: 0.6rem;
  }

  .compact-search {
    min-width: 0;
  }

  .runs-compact-list {
    display: grid;
    gap: 0.5rem;
    max-height: 520px;
    overflow-y: auto;
    padding-right: 0.15rem;
  }

  .run-card-compact {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.03);
    overflow: hidden;
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
    background: rgba(255, 255, 255, 0.03);
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
  .status-dot.finished { background: #4ade80; }
  .status-dot.failed { background: #f87171; }
  .status-dot.unknown { background: #94a3b8; }

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
  }

  .icon-btn {
    min-width: 2rem;
    padding-inline: 0.4rem;
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

  @media (max-width: 980px) {
    .runs-compact-toolbar {
      grid-template-columns: 1fr;
      align-items: stretch;
    }
  }
</style>
