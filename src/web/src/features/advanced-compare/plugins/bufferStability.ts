import type { ChartPlugin } from './types';
import { labelForRun } from '../metrics';

export const bufferStabilityPlugin: ChartPlugin = {
  id: 'buffer-stability',
  title: 'Buffer stability (lower is smoother)',
  category: 'buffer',
  requiredDatasets: ['buffer'],
  sortDirection: 'asc',
  build: ({ metrics, aliases }) => {
    const keys = Object.keys(metrics.buffer || {});
    if (!keys.length) return null;

    return {
      type: 'bar',
      data: {
        labels: keys.map((k) => labelForRun(metrics, k, aliases)),
        datasets: [
          {
            label: 'Mean buffer (s)',
            data: keys.map((k) => metrics.buffer[k]?.mean || 0),
            backgroundColor: 'rgba(34,211,238,0.75)',
          },
          {
            label: 'Std dev (s)',
            data: keys.map((k) => metrics.buffer[k]?.std || 0),
            backgroundColor: 'rgba(245,158,11,0.8)',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'Seconds' } },
        },
      },
    };
  },
};
