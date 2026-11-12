import type { ChartPlugin } from './types';
import { average, labelForRun } from '../metrics';
import { fovPredictionStatsForRun } from './fovMetrics';

export const fovUsefulBitrateEfficiencyPlugin: ChartPlugin = {
  id: 'fov-useful-bitrate-efficiency',
  title: 'FoV useful bitrate efficiency',
  category: 'fov',
  requiredDatasets: ['fov_prediction'],
  sortDirection: 'desc',
  description: 'How much delivered bitrate landed inside actual FoV, compared with high-quality invisible bitrate.',
  build: ({ metrics, aliases }) => {
    const entries = Object.keys(metrics.fov_prediction || {})
      .map((runId) => [runId, fovPredictionStatsForRun(metrics, runId)] as const)
      .filter(([, vals]) => vals && ((vals.usefulVisibleBitFraction || []).length || (vals.highQualityInvisibleBitFraction || []).length));
    if (!entries.length) return null;

    return {
      type: 'bar',
      data: {
        labels: entries.map(([runId]) => labelForRun(metrics, runId, aliases)),
        datasets: [
          {
            label: 'Useful visible bits / total',
            data: entries.map(([, vals]) => average(vals?.usefulVisibleBitFraction || [])),
            backgroundColor: '#22c55e',
          },
          {
            label: 'High-quality invisible bits / total',
            data: entries.map(([, vals]) => average(vals?.highQualityInvisibleBitFraction || [])),
            backgroundColor: '#f97316',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          y: { min: 0, max: 1, title: { display: true, text: 'Fraction of delivered bits' } },
        },
      },
    };
  },
};
