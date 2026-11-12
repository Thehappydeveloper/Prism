import type { ChartPlugin } from './types';
import { average, labelForRun } from '../metrics';
import { fovPredictionStatsForRun } from './fovMetrics';

export const fovBitrateWasteAbsolutePlugin: ChartPlugin = {
  id: 'fov-bitrate-waste-absolute',
  title: 'FoV bitrate waste',
  category: 'fov',
  requiredDatasets: ['fov_prediction'],
  sortDirection: 'asc',
  description: 'Mean bitrate sent outside the actual FoV; lower is better. Visibility-weighted waste scales each tile by how invisible it was, while fully unseen waste counts every tile below the visibility threshold.',
  build: ({ metrics, aliases }) => {
    const entries = Object.keys(metrics.fov_prediction || {})
      .map((runId) => [runId, fovPredictionStatsForRun(metrics, runId)] as const)
      .filter(([, vals]) => vals && ((vals.visibilityWeightedWasteBitrateMbps || []).length || (vals.wastedUnseenBitrateMbps || []).length));
    if (!entries.length) return null;

    return {
      type: 'bar',
      data: {
        labels: entries.map(([runId]) => labelForRun(metrics, runId, aliases)),
        datasets: [
          {
            label: 'Visibility-weighted waste',
            data: entries.map(([, vals]) => average(vals?.visibilityWeightedWasteBitrateMbps || [])),
            backgroundColor: '#f97316',
          },
          {
            label: 'Fully unseen waste',
            data: entries.map(([, vals]) => average(vals?.wastedUnseenBitrateMbps || [])),
            backgroundColor: '#fb7185',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'Mbps' } },
        },
      },
    };
  },
};
