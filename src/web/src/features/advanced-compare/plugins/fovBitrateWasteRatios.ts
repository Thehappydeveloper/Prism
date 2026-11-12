import type { ChartPlugin } from './types';
import { average, labelForRun } from '../metrics';
import { fovPredictionStatsForRun } from './fovMetrics';

export const fovBitrateWasteRatiosPlugin: ChartPlugin = {
  id: 'fov-bitrate-waste-ratios',
  title: 'FoV bitrate waste ratios',
  category: 'fov',
  requiredDatasets: ['fov_prediction'],
  sortDirection: 'asc',
  description: 'Ratio metrics for bandwidth spent outside actual FoV. Higher values mean more bitrate went to less visible tiles.',
  build: ({ metrics, aliases }) => {
    const entries = Object.keys(metrics.fov_prediction || {})
      .map((runId) => [runId, fovPredictionStatsForRun(metrics, runId)] as const)
      .filter(([, vals]) => vals && ((vals.visibilityWeightedWasteFraction || []).length || (vals.nonVisibleToVisibleBitRatio || []).length));
    if (!entries.length) return null;

    return {
      type: 'bar',
      data: {
        labels: entries.map(([runId]) => labelForRun(metrics, runId, aliases)),
        datasets: [
          {
            label: 'Waste / total bits',
            data: entries.map(([, vals]) => average(vals?.visibilityWeightedWasteFraction || [])),
            backgroundColor: '#f97316',
          },
          {
            label: 'Unseen / visible bits',
            data: entries.map(([, vals]) => average(vals?.nonVisibleToVisibleBitRatio || [])),
            backgroundColor: '#a855f7',
          },
          {
            label: 'Fully unseen / total bits',
            data: entries.map(([, vals]) => average(vals?.wastedUnseenBitFraction || [])),
            backgroundColor: '#94a3b8',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'Ratio' } },
        },
      },
    };
  },
};
