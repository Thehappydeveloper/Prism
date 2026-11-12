import type { ChartPlugin } from './types';
import { average, labelForRun } from '../metrics';
import { fovPredictionStatsForRun } from './fovMetrics';

export const fovPredictionBitrateWastePlugin: ChartPlugin = {
  id: 'fov-prediction-bitrate-waste',
  title: 'Prediction-targeted bitrate waste',
  category: 'fov',
  requiredDatasets: ['fov_prediction'],
  sortDirection: 'asc',
  description: 'Bitrate waste associated with predicted-visible tiles that were less visible in ground truth.',
  build: ({ metrics, aliases }) => {
    const entries = Object.keys(metrics.fov_prediction || {})
      .map((runId) => [runId, fovPredictionStatsForRun(metrics, runId)] as const)
      .filter(([, vals]) => vals && ((vals.predictionTargetedWasteFraction || []).length || (vals.overPredictedBitWasteFraction || []).length));
    if (!entries.length) return null;

    return {
      type: 'bar',
      data: {
        labels: entries.map(([runId]) => labelForRun(metrics, runId, aliases)),
        datasets: [
          {
            label: 'Wasted predicted-visible bits',
            data: entries.map(([, vals]) => average(vals?.predictionTargetedWasteFraction || [])),
            backgroundColor: '#f97316',
          },
          {
            label: 'Overprediction waste / total',
            data: entries.map(([, vals]) => average(vals?.overPredictedBitWasteFraction || [])),
            backgroundColor: '#fb7185',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          y: { min: 0, max: 1, title: { display: true, text: 'Fraction' } },
        },
      },
    };
  },
};
