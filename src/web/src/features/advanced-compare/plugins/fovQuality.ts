import type { ChartPlugin } from './types';
import { average, labelForRun, runHasFov } from '../metrics';

export const fovQualityPlugin: ChartPlugin = {
  id: 'fov-quality',
  title: 'Average normalized FoV tile quality',
  build: ({ metrics, aliases }) => {
    const keys = Object.keys(metrics.fov_quality || {});
    if (!keys.length) return null;

    const fov = keys.map((k) => average(metrics.fov_quality[k]?.fov || []));
    const non = keys.map((k) => {
      if (!runHasFov(metrics, k)) return null;
      const vals = metrics.fov_quality[k]?.non || [];
      return vals.length ? average(vals) : 0;
    });

    const hasNon = non.some((v) => v !== null);
    const datasets: any[] = [{
      label: hasNon ? 'FoV' : 'Tile quality',
      data: fov,
      backgroundColor: '#22d3ee',
    }];

    if (hasNon) {
      datasets.push({ label: 'Non-FoV', data: non.map((v) => v ?? 0), backgroundColor: '#94a3b8' });
    }

    return {
      type: 'bar',
      data: {
        labels: keys.map((k) => labelForRun(metrics, k, aliases)),
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: { y: { min: 0, max: 1 } },
      },
    };
  },
};
