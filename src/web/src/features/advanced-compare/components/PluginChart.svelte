<script lang="ts">
  import { Chart, registerables, type ChartConfiguration } from 'chart.js';
  import zoomPlugin from 'chartjs-plugin-zoom';
  import { onDestroy } from 'svelte';

  Chart.register(...registerables, zoomPlugin);

  export let title: string;
  export let config: ChartConfiguration | null;
  export let error = '';

  let canvasEl: HTMLCanvasElement | null = null;
  let chart: Chart | null = null;
  let renderError = '';

  function renderChart() {
    if (!canvasEl || !config) return;
    try {
      chart?.destroy();
      chart = new Chart(canvasEl, config);
      chart.setActiveElements([]);
      chart.update();
      renderError = '';
    } catch (err) {
      chart = null;
      renderError = err instanceof Error ? err.message : String(err);
    }
  }

  $: if (!config && chart) {
    chart.destroy();
    chart = null;
  }

  $: if (canvasEl && config) {
    renderChart();
  }

  onDestroy(() => {
    chart?.destroy();
  });
</script>

<figure>
  <figcaption>{title}</figcaption>
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
</figure>

<style>
  figure {
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .chart-shell {
    min-height: 280px;
    height: 280px;
    position: relative;
  }
  .empty-chart {
    color: #94a3b8;
    font-size: 0.9rem;
    padding: 1rem;
    border: 1px dashed rgba(148, 163, 184, 0.35);
    border-radius: 0.5rem;
  }
</style>
