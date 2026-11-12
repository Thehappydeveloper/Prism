import type { ChartPlugin } from './types';
import { average, labelForRun } from '../metrics';
import { fovPredictionStatsForRun } from './fovMetrics';

export const fovPredictionErrorPlugin: ChartPlugin = {
  id: 'fov-prediction-error',
  title: 'FoV prediction error',
  category: 'fov',
  requiredDatasets: ['fov_prediction'],
  sortDirection: 'asc',
  description: 'Compares predicted tile visibility against ground truth using MAE and RMSE.',
  build: ({ metrics, aliases }) => {
    const entries = Object.keys(metrics.fov_prediction || {})
      .map((runId) => [runId, fovPredictionStatsForRun(metrics, runId)] as const)
      .filter(([, vals]) => vals && ((vals.mae || []).length || (vals.rmse || []).length || (vals.playbackMae || []).length || (vals.playbackRmse || []).length));
    if (!entries.length) return null;

    return {
      type: 'bar',
      data: {
        labels: entries.map(([runId]) => labelForRun(metrics, runId, aliases)),
        datasets: [
          {
            label: 'Segment MAE',
            data: entries.map(([, vals]) => average(vals?.mae || [])),
            backgroundColor: '#fb7185',
          },
          {
            label: 'Segment RMSE',
            data: entries.map(([, vals]) => average(vals?.rmse || [])),
            backgroundColor: '#f59e0b',
          },
          {
            label: 'Playback MAE',
            data: entries.map(([, vals]) => average(vals?.playbackMae || [])),
            backgroundColor: '#38bdf8',
          },
          {
            label: 'Playback RMSE',
            data: entries.map(([, vals]) => average(vals?.playbackRmse || [])),
            backgroundColor: '#a78bfa',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          y: { min: 0, max: 1, title: { display: true, text: 'Visibility error' } },
        },
      },
    };
  },
};
