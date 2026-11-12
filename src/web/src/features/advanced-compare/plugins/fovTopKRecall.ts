import type { ChartPlugin } from './types';
import { average, labelForRun } from '../metrics';
import { fovPredictionStatsForRun } from './fovMetrics';

export const fovTopKRecallPlugin: ChartPlugin = {
  id: 'fov-topk-recall',
  title: 'FoV top-K recall',
  category: 'fov',
  requiredDatasets: ['fov_prediction'],
  sortDirection: 'desc',
  description: 'Measures whether the predictor ranks the truly visible tiles near the top.',
  build: ({ metrics, aliases }) => {
    const entries = Object.keys(metrics.fov_prediction || {})
      .map((runId) => [runId, fovPredictionStatsForRun(metrics, runId)] as const)
      .filter(([, vals]) => vals && ((vals.topkRecall || []).length || (vals.top1Hit || []).length || (vals.visibleFrameRecall || []).length || (vals.topkRecallPerFrame || []).length));
    if (!entries.length) return null;

    return {
      type: 'bar',
      data: {
        labels: entries.map(([runId]) => labelForRun(metrics, runId, aliases)),
        datasets: [
          {
            label: 'Segment top-K recall',
            data: entries.map(([, vals]) => average(vals?.topkRecall || [])),
            backgroundColor: '#a3e635',
          },
          {
            label: 'Per-frame top-K recall',
            data: entries.map(([, vals]) => average(vals?.topkRecallPerFrame || [])),
            backgroundColor: '#22c55e',
          },
          {
            label: 'Visible-frame recall',
            data: entries.map(([, vals]) => average(vals?.visibleFrameRecall || [])),
            backgroundColor: '#facc15',
          },
          {
            label: 'Top-1 hit rate',
            data: entries.map(([, vals]) => average(vals?.top1Hit || [])),
            backgroundColor: '#38bdf8',
          },
          {
            label: 'Peak miss rate',
            data: entries.map(([, vals]) => average(vals?.peakMissRate || [])),
            backgroundColor: '#fb7185',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          y: { min: 0, max: 1, title: { display: true, text: 'Hit / recall rate' } },
        },
      },
    };
  },
};
