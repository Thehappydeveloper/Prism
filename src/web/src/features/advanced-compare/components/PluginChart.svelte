<script lang="ts">
  import { Chart, registerables, type ChartConfiguration } from 'chart.js';
  import zoomPlugin from 'chartjs-plugin-zoom';
  import { createEventDispatcher, onDestroy, onMount } from 'svelte';
  import type { PluginControl } from '../plugins/types';
  import { exportFigure } from '../figureExport';
  import { DEFAULT_EXPORT_SETTINGS, DEFAULT_PUBLICATION_SETTINGS, type ExportFormat, type FigureExportSettings, type PublicationSettings } from '../publication';

  Chart.register(...registerables, zoomPlugin);

  export let title: string;
  export let chartId = title;
  export let description = '';
  export let controls: PluginControl[] = [];
  export let options: Record<string, string | number | boolean> = {};
  export let config: ChartConfiguration | null;
  export let error = '';
  export let exportSettings: FigureExportSettings = DEFAULT_EXPORT_SETTINGS;
  export let publicationSettings: PublicationSettings = DEFAULT_PUBLICATION_SETTINGS;

  let canvasEl: HTMLCanvasElement | null = null;
  let frameEl: HTMLDivElement | null = null;
  let chart: Chart | null = null;
  let renderedConfig: ChartConfiguration | null = null;
  let pendingFrame = 0;
  let renderError = '';
  let exportError = '';
  let exportingFormat: ExportFormat | null = null;
  const exportFormats: ExportFormat[] = ['svg', 'pdf', 'png'];
  let showDescription = false;
  let frameWidth: number | null = null;
  let frameHeight = 340;
  let resizeObserver: ResizeObserver | null = null;
  let stopActiveResize: (() => void) | null = null;
  const dispatch = createEventDispatcher<{ optionchange: { key: string; value: string | number | boolean } }>();

  const sizeStorageKey = () => `prism-chart-size-v1:${chartId}`;

  function saveSize(): void {
    localStorage.setItem(sizeStorageKey(), JSON.stringify({ width: frameWidth, height: frameHeight }));
  }

  function resetSize(): void {
    frameWidth = null;
    frameHeight = 340;
    localStorage.removeItem(sizeStorageKey());
    requestAnimationFrame(() => chart?.resize());
  }

  function startResize(event: PointerEvent, horizontal: boolean, vertical: boolean): void {
    if (!frameEl) return;
    event.preventDefault();
    const rect = frameEl.getBoundingClientRect();
    const startX = event.clientX;
    const startY = event.clientY;
    const startWidth = rect.width;
    const startHeight = rect.height;

    const move = (moveEvent: PointerEvent) => {
      if (horizontal) frameWidth = Math.max(420, Math.min(2200, startWidth + moveEvent.clientX - startX));
      if (vertical) frameHeight = Math.max(260, Math.min(1400, startHeight + moveEvent.clientY - startY));
    };
    const stop = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', stop);
      window.removeEventListener('pointercancel', stop);
      stopActiveResize = null;
      saveSize();
      chart?.resize();
    };
    stopActiveResize?.();
    stopActiveResize = stop;
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', stop);
    window.addEventListener('pointercancel', stop);
  }

  function changeControl(control: PluginControl, event: Event): void {
    const target = event.currentTarget as HTMLInputElement | HTMLSelectElement;
    let value: string | number | boolean = target.value;
    if (control.type === 'checkbox') value = (target as HTMLInputElement).checked;
    else if (control.type === 'number') value = Number(target.value);
    else {
      const selected = control.options?.find((candidate) => String(candidate.value) === target.value);
      if (selected) value = selected.value;
    }
    dispatch('optionchange', { key: control.key, value });
  }

  async function exportCurrentFigure(format: ExportFormat): Promise<void> {
    if (!config || exportingFormat) return;
    exportingFormat = format;
    exportError = '';
    try {
      await exportFigure(config, title, { ...exportSettings, format }, publicationSettings);
    } catch (err) {
      exportError = err instanceof Error ? err.message : String(err);
    } finally {
      exportingFormat = null;
    }
  }

  function renderChart() {
    if (!canvasEl || !config) return;
    try {
      const started = performance.now();
      chart?.destroy();
      chart = new Chart(canvasEl, {
        ...config,
        // Parsing is dataset-specific. Categorical bar charts use primitive arrays,
        // while scatter/line plugins opt into parsing:false themselves for {x,y} data.
        options: { ...config.options, animation: false },
      });
      renderedConfig = config;
      chart.setActiveElements([]);
      chart.update();
      console.debug('[graph-timing] plugin chart render', { title, elapsedMs: Math.round(performance.now() - started) });
      renderError = '';
    } catch (err) {
      chart = null;
      renderError = err instanceof Error ? err.message : String(err);
    }
  }

  $: if (!config && chart) {
    chart.destroy();
    chart = null;
    renderedConfig = null;
  }

  $: if (canvasEl && config) {
    if (renderedConfig !== config) {
      if (pendingFrame) cancelAnimationFrame(pendingFrame);
      pendingFrame = requestAnimationFrame(() => {
        if (canvasEl && config) renderChart();
        pendingFrame = 0;
      });
    }
  }

  onMount(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(sizeStorageKey()) || 'null');
      if (saved && Number(saved.width) >= 420) frameWidth = Number(saved.width);
      if (saved && Number(saved.height) >= 260) frameHeight = Number(saved.height);
    } catch {
      // Invalid saved dimensions fall back to the responsive defaults.
    }
    if (frameEl) {
      resizeObserver = new ResizeObserver(() => chart?.resize());
      resizeObserver.observe(frameEl);
    }
  });

  onDestroy(() => {
    if (pendingFrame) cancelAnimationFrame(pendingFrame);
    stopActiveResize?.();
    resizeObserver?.disconnect();
    chart?.destroy();
  });
</script>

<figure>
  <figcaption class="chart-heading">
    <span>{title}</span>
    <span class="chart-heading-actions">
      {#if config}
        <span class="chart-export-group" aria-label="Figure export formats">
          {#each exportFormats as format}
            <button
              class="chart-export"
              type="button"
              title={format === 'pdf'
                ? `Open vector PDF print dialog at ${exportSettings.width} × ${exportSettings.height} ${exportSettings.unit}`
                : `Export ${format.toUpperCase()} at ${exportSettings.width} × ${exportSettings.height} ${exportSettings.unit}`}
              on:click={() => exportCurrentFigure(format)}
              disabled={exportingFormat !== null}
            >
              {exportingFormat === format ? 'Exporting…' : `Export ${format.toUpperCase()}`}
            </button>
          {/each}
        </span>
      {/if}
      {#if frameWidth !== null || frameHeight !== 340}
        <button class="chart-size-reset" type="button" title="Reset chart dimensions" on:click={resetSize}>Reset size</button>
      {/if}
      {#if description}
        <span class="chart-help-wrap">
          <button class="chart-help" type="button" aria-label={`About ${title}`} aria-expanded={showDescription} on:click={() => (showDescription = !showDescription)}>?</button>
          <span class="chart-help-tooltip" role="tooltip">{description}</span>
        </span>
      {/if}
    </span>
  </figcaption>
  {#if description && showDescription}
    <div class="chart-description">{description}</div>
  {/if}
  {#if exportError}<div class="chart-export-error">Export failed: {exportError}</div>{/if}
  {#if controls.length}
    <div class="chart-controls" aria-label={`${title} controls`}>
      {#each controls as control (control.key)}
        <label class:checkbox-control={control.type === 'checkbox'} title={control.help || ''}>
          <span>{control.label}</span>
          {#if control.type === 'select'}
            <select value={String(options[control.key] ?? control.defaultValue)} on:change={(event) => changeControl(control, event)}>
              {#each control.options || [] as choice}
                <option value={String(choice.value)}>{choice.label}</option>
              {/each}
            </select>
          {:else if control.type === 'number'}
            <input type="number" value={Number(options[control.key] ?? control.defaultValue)} min={control.min} max={control.max} step={control.step} on:change={(event) => changeControl(control, event)} />
          {:else}
            <input type="checkbox" checked={Boolean(options[control.key] ?? control.defaultValue)} on:change={(event) => changeControl(control, event)} />
          {/if}
        </label>
      {/each}
    </div>
  {/if}
  <div class="chart-viewport">
    <div
      class="chart-frame"
      bind:this={frameEl}
      style={`width:${frameWidth === null ? '100%' : `${frameWidth}px`};height:${frameHeight}px`}
    >
      <div class="chart-shell">
        {#if error}
          <div class="empty-chart">Plugin error: {error}</div>
        {:else if renderError}
          <div class="empty-chart">Render error: {renderError}</div>
        {:else if config}
          <canvas bind:this={canvasEl}></canvas>
        {:else}
          <div class="empty-chart">No data for this chart.</div>
        {/if}
      </div>
      <button class="resize-handle resize-right" type="button" aria-label="Resize chart width" on:pointerdown={(event) => startResize(event, true, false)}></button>
      <button class="resize-handle resize-bottom" type="button" aria-label="Resize chart height" on:pointerdown={(event) => startResize(event, false, true)}></button>
      <button class="resize-handle resize-corner" type="button" aria-label="Resize chart width and height" on:pointerdown={(event) => startResize(event, true, true)}></button>
    </div>
  </div>
</figure>

<style>
  figure {
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .chart-heading { position: relative; display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; font-weight: 600; }
  .chart-heading-actions { display: inline-flex; align-items: center; gap: 0.35rem; }
  .chart-export-group { display: inline-flex; align-items: center; gap: 0.2rem; }
  .chart-size-reset { padding: 0.2rem 0.45rem; border-radius: 6px; border: 1px solid var(--ghost-border); background: transparent; color: var(--text-muted); font-size: 0.7rem; line-height: 1; }
  .chart-export { padding: 0.25rem 0.5rem; border-radius: 6px; border: 1px solid var(--ghost-border); background: var(--soft-bg-alt); color: var(--text-main); font-size: 0.7rem; line-height: 1; }
  .chart-export-error { padding: 0.4rem 0.55rem; border-left: 2px solid #dc2626; color: #b91c1c; font-size: 0.76rem; }
  .chart-help-wrap { position: relative; display: inline-grid; place-items: center; }
  .chart-help {
    display: inline-grid;
    place-items: center;
    inline-size: 1.35rem;
    block-size: 1.35rem;
    min-inline-size: 1.35rem;
    margin: 0;
    padding: 0;
    border-radius: 50%;
    border: 1px solid #64748b;
    color: var(--text-muted);
    background: transparent;
    cursor: help;
    font-size: 0.78rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: 0;
    box-shadow: none;
  }
  .chart-help:hover, .chart-help:focus-visible { transform: none; color: var(--text-main); border-color: var(--accent-main); background: var(--soft-bg-alt); box-shadow: none; }
  .chart-help-tooltip { position: absolute; z-index: 20; right: 0; top: calc(100% + 0.45rem); width: min(320px, 70vw); padding: 0.55rem 0.65rem; border: 1px solid var(--card-border); border-radius: 8px; background: var(--modal-surface); color: var(--text-main); font-size: 0.78rem; font-weight: 400; line-height: 1.35; box-shadow: 0 10px 26px rgba(2, 6, 23, 0.35); opacity: 0; visibility: hidden; pointer-events: none; }
  .chart-help-wrap:hover .chart-help-tooltip, .chart-help:focus-visible + .chart-help-tooltip { opacity: 1; visibility: visible; }
  .chart-description { margin: -0.15rem 0 0.4rem; padding: 0.5rem 0.65rem; max-width: 52rem; border-left: 2px solid var(--accent-main); font-size: 0.82rem; font-weight: 400; color: var(--text-muted); }
  .chart-controls { display: flex; flex-wrap: wrap; align-items: end; gap: 0.45rem 0.65rem; padding: 0.45rem 0.55rem; border: 1px solid var(--card-border); border-radius: 9px; background: var(--soft-bg-alt); }
  .chart-controls label { display: grid; gap: 0.2rem; min-width: 8rem; color: var(--text-muted); font-size: 0.7rem; font-weight: 600; }
  .chart-controls select, .chart-controls input[type='number'] { min-height: 1.85rem; padding: 0.25rem 0.4rem; border-radius: 6px; font-size: 0.76rem; }
  .chart-controls .checkbox-control { display: inline-flex; align-items: center; align-self: center; min-width: auto; gap: 0.35rem; }
  .chart-controls input[type='checkbox'] { width: 1rem; height: 1rem; }
  .chart-viewport { width: 100%; overflow: auto; padding: 0 0.4rem 0.4rem 0; }
  .chart-frame { position: relative; min-width: 420px; min-height: 260px; max-width: none; }
  .chart-shell {
    width: 100%;
    min-height: 260px;
    height: 100%;
    position: relative;
  }
  .resize-handle { position: absolute; z-index: 12; margin: 0; padding: 0; border: 0; border-radius: 0; background: transparent; box-shadow: none; touch-action: none; }
  .resize-handle:hover, .resize-handle:active { transform: none; background: color-mix(in srgb, var(--accent-main) 35%, transparent); box-shadow: none; }
  .resize-right { top: 0; right: -4px; width: 9px; height: calc(100% - 12px); cursor: ew-resize; }
  .resize-bottom { left: 0; bottom: -4px; width: calc(100% - 12px); height: 9px; cursor: ns-resize; }
  .resize-corner { right: -4px; bottom: -4px; width: 16px; height: 16px; cursor: nwse-resize; border-radius: 4px; background: linear-gradient(135deg, transparent 45%, var(--text-muted) 46%, var(--text-muted) 55%, transparent 56%, transparent 68%, var(--text-muted) 69%, var(--text-muted) 78%, transparent 79%); }
  .empty-chart {
    color: #94a3b8;
    font-size: 0.9rem;
    padding: 1rem;
    border: 1px dashed rgba(148, 163, 184, 0.35);
    border-radius: 0.5rem;
  }
</style>
