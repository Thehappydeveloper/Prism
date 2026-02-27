import type { ChartPlugin } from './types';
import { bucket, labelForRun, palette } from '../metrics';

export const temporalDeltaPlugin: ChartPlugin = {
  id: 'temporal-delta',
  title: 'FoV quality change histogram (ΔQ)',
  build: ({ metrics, aliases }) => {
    const runs = Object.entries(metrics.delta_fov || {});
    if (!runs.length) return null;

    const bins = 12;
    const labels = Array.from({ length: bins }, (_, i) => `${(i / bins).toFixed(2)}-${((i + 1) / bins).toFixed(2)}`);

    const datasets = runs.map(([runId, vals], idx) => {
      const hist = bucket(vals, bins, 0, 1);
      const total = hist.reduce((acc, x) => acc + x.count, 0) || 1;
      return {
        label: labelForRun(metrics, runId, aliases),
        data: hist.map((h) => Number(((h.count / total) * 100).toFixed(1))),
        backgroundColor: palette(idx),
      };
    });

    return {
      type: 'bar',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          x: { title: { display: true, text: 'ΔQ bin (FoV tiles)' } },
          y: {
            beginAtZero: true,
            title: { display: true, text: 'Share of observations (%)' },
            ticks: { callback: (v: number | string) => `${v}%` },
          },
        },
      },
    };
  },
};
