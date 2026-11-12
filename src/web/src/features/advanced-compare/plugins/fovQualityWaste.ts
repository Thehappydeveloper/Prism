import type { ChartPlugin } from './types';
import { average, labelForRun } from '../metrics';
import { fovPredictionStatsForRun } from './fovMetrics';

export const fovQualityWastePlugin: ChartPlugin = {
  id: 'fov-quality-waste',
  title: 'FoV quality waste',
  category: 'fov',
  requiredDatasets: ['fov_prediction'],
  sortDirection: 'asc',
  description: 'Estimates quality spent outside actual FoV and quality missed inside actual FoV.',
  build: ({ metrics, aliases }) => {
    const entries = Object.keys(metrics.fov_prediction || {})
      .map((runId) => [runId, fovPredictionStatsForRun(metrics, runId)] as const)
      .filter(([, vals]) => vals && ((vals.wastedVisibleQuality || []).length || (vals.missedVisibleQuality || []).length));
    if (!entries.length) return null;

    return {
      type: 'bar',
      data: {
        labels: entries.map(([runId]) => labelForRun(metrics, runId, aliases)),
        datasets: [
          {
            label: 'Wasted non-FoV quality',
            data: entries.map(([, vals]) => average(vals?.wastedVisibleQuality || [])),
            backgroundColor: '#f97316',
          },
          {
            label: 'Missed FoV quality',
            data: entries.map(([, vals]) => average(vals?.missedVisibleQuality || [])),
            backgroundColor: '#fb7185',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'Quality-weighted visibility mass' } },
        },
      },
    };
  },
};
