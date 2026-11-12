import type { ChartPlugin } from './types';
import { average, labelForRun } from '../metrics';
import { fovPredictionStatsForRun } from './fovMetrics';

export const fovOverlapIouPlugin: ChartPlugin = {
  id: 'fov-overlap-iou',
  title: 'FoV prediction overlap',
  category: 'fov',
  requiredDatasets: ['fov_prediction'],
  sortDirection: 'desc',
  description: 'Tile-set IoU and weighted IoU between predicted and actual visible tiles.',
  build: ({ metrics, aliases }) => {
    const entries = Object.keys(metrics.fov_prediction || {})
      .map((runId) => [runId, fovPredictionStatsForRun(metrics, runId)] as const)
      .filter(([, vals]) => vals && ((vals.weightedIou || []).length || (vals.iou || []).length));
    if (!entries.length) return null;

    return {
      type: 'bar',
      data: {
        labels: entries.map(([runId]) => labelForRun(metrics, runId, aliases)),
        datasets: [
          {
            label: 'Weighted IoU',
            data: entries.map(([, vals]) => average(vals?.weightedIou || [])),
            backgroundColor: '#22d3ee',
          },
          {
            label: 'Binary IoU',
            data: entries.map(([, vals]) => average(vals?.iou || [])),
            backgroundColor: '#94a3b8',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          y: { min: 0, max: 1, title: { display: true, text: 'IoU' } },
        },
      },
    };
  },
};
