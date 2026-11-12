import type { ChartPlugin } from './types';
import { labelForRun } from '../metrics';

export const predictorLatencyPlugin: ChartPlugin = {
  id: 'predictor-latency',
  title: 'FoV predictor latency and tail risk',
  category: 'fov',
  requiredDatasets: ['latency'],
  sortDirection: 'asc',
  description: 'Compares typical prediction time with mean and p95 tail latency. Lower and tighter bars indicate a faster, more predictable predictor.',
  build: ({ metrics, aliases }) => {
    const entries = Object.entries(metrics.latency || {})
      .filter(([, latency]) => Number(latency?.count || 0) > 0 && Number.isFinite(Number(latency?.p95_ms)));
    if (!entries.length) return null;

    return {
      type: 'bar',
      data: {
        labels: entries.map(([runId]) => labelForRun(metrics, runId, aliases)),
        datasets: [
          {
            label: 'Median',
            data: entries.map(([, latency]) => Number(latency.median_ms || 0)),
            backgroundColor: '#22d3ee',
          },
          {
            label: 'Mean',
            data: entries.map(([, latency]) => Number(latency.mean_ms || 0)),
            backgroundColor: '#a78bfa',
          },
          {
            label: 'P95',
            data: entries.map(([, latency]) => Number(latency.p95_ms || 0)),
            backgroundColor: '#fb7185',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'bottom' },
          tooltip: {
            callbacks: {
              afterBody: (items: any[]) => {
                const latency = entries[items[0]?.dataIndex]?.[1];
                if (!latency) return '';
                return `Max: ${Number(latency.max_ms || 0).toFixed(1)} ms\nPredictions: ${Number(latency.count || 0)}`;
              },
            },
          },
        },
        scales: {
          x: { beginAtZero: true, title: { display: true, text: 'End-to-end prediction latency (ms)' } },
          y: { ticks: { autoSkip: false } },
        },
      },
    };
  },
};
