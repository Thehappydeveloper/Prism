import type { ChartPlugin } from './types';
import { average, labelForRun } from '../metrics';
import { fovPredictionStatsForRun } from './fovMetrics';

export const fovCatastrophicMissPlugin: ChartPlugin = {
  id: 'fov-catastrophic-miss',
  title: 'Catastrophic FoV miss rate',
  category: 'fov',
  requiredDatasets: ['fov_prediction'],
  sortDirection: 'asc',
  description: 'Share of segments where the predicted top-K tiles capture under 10% of the soft ground-truth FoV mass. Lower is better.',
  build: ({ metrics, aliases }) => {
    const entries = Object.keys(metrics.fov_prediction || {})
      .map((runId) => [runId, fovPredictionStatsForRun(metrics, runId)] as const)
      .filter(([, stats]) => Boolean(stats?.catastrophicTopKMissRate?.length));
    if (!entries.length) return null;
    return {
      type: 'bar',
      data: {
        labels: entries.map(([runId]) => labelForRun(metrics, runId, aliases)),
        datasets: [{
          label: 'Catastrophic miss rate',
          data: entries.map(([, stats]) => average(stats?.catastrophicTopKMissRate || [])),
          backgroundColor: '#ef4444',
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { min: 0, max: 1, title: { display: true, text: 'Segments' } } },
      },
    };
  },
};
