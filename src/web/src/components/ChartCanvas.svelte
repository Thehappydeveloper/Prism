<script lang="ts">
  import { Chart, registerables, type ChartConfiguration } from 'chart.js';
  import zoomPlugin from 'chartjs-plugin-zoom';
  import { onDestroy } from 'svelte';

  const stallOverlayPlugin = {
    id: 'stallOverlay',
    beforeDatasetsDraw(chart: any, _args: any, opts: any) {
      const stalls = opts?.stalls || [];
      if (!stalls.length) return;
      const { chartArea, ctx, scales } = chart;
      if (!chartArea || !scales?.x) return;
      const height = chartArea.bottom - chartArea.top;
      ctx.save();
      ctx.fillStyle = opts?.color || 'rgba(248,113,113,0.15)';
      ctx.strokeStyle = opts?.borderColor || 'rgba(248,113,113,0.55)';
      ctx.lineWidth = 1;
      stalls.forEach((stall: { start: number; end: number }) => {
        const xStart = scales.x.getPixelForValue(stall.start);
        const xEnd = scales.x.getPixelForValue(stall.end);
        const left = Math.min(xStart, xEnd);
        const width = Math.abs(xEnd - xStart);
        ctx.fillRect(left, chartArea.top, width, height);
        ctx.strokeRect(left, chartArea.top, width, height);
      });
      ctx.restore();
    },
  };

  Chart.register(...registerables, zoomPlugin, stallOverlayPlugin);

  export let config: ChartConfiguration | null = null;
  export let height = 320;

  let canvasEl: HTMLCanvasElement | null = null;
  let chart: Chart | null = null;

  $: if (!config && chart) {
    chart.destroy();
    chart = null;
  }

  $: if (canvasEl && config) {
    chart?.destroy();
    chart = new Chart(canvasEl, config);
    chart.setActiveElements([]);
    chart.update();
  }

  onDestroy(() => {
    chart?.destroy();
  });
</script>

<div class="chart-shell" style={`height:${height}px`}>
  {#if config}
    <canvas bind:this={canvasEl}></canvas>
  {:else}
    <div class="empty-chart">No chart data yet.</div>
  {/if}
</div>

<style>
  .chart-shell {
    position: relative;
    min-height: 240px;
  }
  .empty-chart {
    color: #94a3b8;
    border: 1px dashed rgba(148, 163, 184, 0.3);
    border-radius: 0.5rem;
    padding: 1rem;
  }
</style>
