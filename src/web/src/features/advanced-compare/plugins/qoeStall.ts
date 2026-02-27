import type { ChartPlugin } from './types';
import { average, labelForRun, palette } from '../metrics';

export const qoeStallPlugin: ChartPlugin = {
  id: 'qoe-stall',
  title: 'QoE vs stall burden',
  build: ({ metrics, aliases }) => {
    const datasets: any[] = [];

    Object.keys(metrics.stall || {}).forEach((runId, idx) => {
      const stall = metrics.stall[runId] || { duration: 0, count: 0 };
      const qoe = metrics.qoe?.[runId] || [];
      if (!qoe.length) return;
      datasets.push({
        label: labelForRun(metrics, runId, aliases),
        data: [{ x: stall.duration || 0, y: average(qoe), count: stall.count || 0 }],
        pointRadius: 6,
        pointHoverRadius: 8,
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
          x: { title: { display: true, text: 'Total stall (s)' }, min: 0 },
          y: { title: { display: true, text: 'Avg QoE' }, min: 0, max: 1 },
        },
      },
    };
  },
};
