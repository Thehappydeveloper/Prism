import type { ChartConfiguration } from 'chart.js';
import { buildDatasets, type AxisBounds, type Layer, type LayerVisibility, type StallSpan } from './runChart';

function ceil1(v: number): number {
  if (!Number.isFinite(v) || v <= 0) return 1;
  return Math.ceil(v * 10) / 10;
}

function ceilInt(v: number): number {
  if (!Number.isFinite(v) || v <= 0) return 1;
  return Math.ceil(v);
}

function formatTick(v: number | string): string {
  const num = Number(v);
  if (!Number.isFinite(num)) return '';
  const rounded = Math.round(num * 10) / 10;
  return Number.isInteger(rounded) ? `${rounded}` : rounded.toFixed(1);
}

function formatIntTick(v: number | string): string {
  const num = Number(v);
  if (!Number.isFinite(num)) return '';
  return Math.round(num).toString();
}

export function toChartConfig(
  layers: Layer[],
  stalls: StallSpan[],
  bounds: AxisBounds,
  visibility: LayerVisibility,
  runTitle = '',
  showLegend = false,
): ChartConfiguration {
  const datasets = buildDatasets(layers, visibility, runTitle);
  return {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      resizeDelay: 120,
      interaction: { mode: 'nearest', intersect: false },
      plugins: {
        legend: { display: showLegend, position: 'bottom' },
        tooltip: {
          callbacks: {
            label: (ctx: any) => `${ctx.dataset.label}: ${Number(ctx.raw?.y || 0).toFixed(2)} @ ${Number(ctx.raw?.x || 0).toFixed(2)}s`,
          },
        },
        stallOverlay: {
          stalls,
          color: 'rgba(248,113,113,0.18)',
          borderColor: 'rgba(248,113,113,0.45)',
        },
      } as any,
      scales: {
        x: {
          type: 'linear',
          min: 0,
          max: bounds.maxX,
          ticks: { callback: (v) => formatTick(v as number | string) },
          title: { display: true, text: 'Time (s)' },
        },
        y: {
          position: 'left',
          min: 0,
          max: bounds.leftMax > 0 ? Math.max(1, ceilInt(bounds.leftMax * 1.1)) : 1,
          ticks: { maxTicksLimit: 6, callback: (v) => formatIntTick(v as number | string) },
          title: { display: true, text: 'Mbps' },
        },
        y1: {
          position: 'right',
          min: 0,
          max: bounds.rightMax > 0 ? ceil1(bounds.rightMax * 1.1) : 1,
          grid: { drawOnChartArea: false },
          ticks: { maxTicksLimit: 6, callback: (v) => formatTick(v as number | string) },
          title: { display: true, text: 'Buffer (s)' },
        },
        ...(bounds.hasQoe
          ? {
              y2: {
                position: 'right',
                min: 0,
                max: bounds.qoeMax > 0 ? Math.min(1, bounds.qoeMax * 1.05) : 1,
                grid: { drawOnChartArea: false },
                ticks: { maxTicksLimit: 6, callback: (v: number | string) => formatTick(v) },
                title: { display: true, text: 'QoE (0-1)' },
                offset: true,
              },
            }
          : {}),
      },
    },
  };
}
