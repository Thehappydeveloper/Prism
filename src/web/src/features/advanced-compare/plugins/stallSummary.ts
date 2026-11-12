import type { ChartPlugin } from './types';
import { labelForRun } from '../metrics';

export const stallSummaryPlugin: ChartPlugin = {
  id: 'stall-summary',
  title: 'Stall duration & count',
  category: 'qoe',
  requiredDatasets: ['stall'],
  sortDirection: 'asc',
  build: ({ metrics, aliases }) => {
    const keys = Object.keys(metrics.stall || {});
    if (!keys.length) return null;

    return {
      type: 'bar',
      data: {
        labels: keys.map((k) => labelForRun(metrics, k, aliases)),
        datasets: [
          {
            label: 'Stall time (s)',
            data: keys.map((k) => metrics.stall[k]?.duration || 0),
            backgroundColor: '#f472b6',
            yAxisID: 'y',
          },
          {
            label: 'Stall count',
            data: keys.map((k) => metrics.stall[k]?.count || 0),
            backgroundColor: 'rgba(148,163,184,0.7)',
            yAxisID: 'y1',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'Seconds stalled' } },
          y1: { beginAtZero: true, position: 'right', grid: { drawOnChartArea: false }, title: { display: true, text: 'Count' } },
        },
      },
    };
  },
};
