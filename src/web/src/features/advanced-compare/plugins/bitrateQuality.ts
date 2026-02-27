import type { ChartPlugin } from './types';
import { labelForRun, palette } from '../metrics';

function groupPoints(points: Array<[number, number]>) {
  const buckets = new Map<string, { x: number; y: number; count: number }>();
  points.forEach(([x, y]) => {
    const key = `${x.toFixed(4)}|${y.toFixed(4)}`;
    const cur = buckets.get(key) || { x, y, count: 0 };
    cur.count += 1;
    buckets.set(key, cur);
  });
  return Array.from(buckets.values());
}

export const bitrateQualityPlugin: ChartPlugin = {
  id: 'bitrate-quality',
  title: 'Bitrate vs FoV quality',
  build: ({ metrics, aliases }) => {
    const datasets: any[] = [];

    Object.entries(metrics.bitrate_vs_quality || {}).forEach(([runId, points], idx) => {
      if (!points.length) return;
      const grouped = groupPoints(points);
      datasets.push({
        label: labelForRun(metrics, runId, aliases),
        data: grouped,
        showLine: false,
        pointRadius: (ctx: any) => 3 + Math.min(4, (ctx.raw?.count || 1) - 1),
        pointHoverRadius: (ctx: any) => 5 + Math.min(4, (ctx.raw?.count || 1) - 1),
        backgroundColor: palette(idx),
      });
    });

    if (!datasets.length) return null;

    return {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          x: { title: { display: true, text: 'Bitrate (Mbps)' }, min: 0 },
          y: { title: { display: true, text: 'FoV quality' }, min: 0, max: 1 },
        },
      },
    };
  },
};
