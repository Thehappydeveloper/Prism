import type { ChartPlugin } from './types';
import { average, labelForRun, palette } from '../metrics';

export const avgEfficiencyPlugin: ChartPlugin = {
  id: 'avg-efficiency',
  title: 'Avg bitrate vs Avg QoE',
  build: ({ metrics, aliases }) => {
    const datasets: any[] = [];

    Object.entries(metrics.bitrate_vs_quality || {}).forEach(([runId, points], idx) => {
      if (!points.length) return;
      const avgBitrate = average(points.map((p) => Number(p[0]) || 0));
      const avgQoE = average(metrics.qoe?.[runId] || []);
      datasets.push({
        label: labelForRun(metrics, runId, aliases),
        data: [{ x: avgBitrate, y: avgQoE }],
        pointRadius: 6,
        pointHoverRadius: 8,
        backgroundColor: palette(idx),
      });
    });

    if (!datasets.length) return null;

    return {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          x: { title: { display: true, text: 'Avg bitrate (Mbps)' }, min: 0 },
          y: { title: { display: true, text: 'Avg QoE' }, min: 0, max: 1 },
        },
      },
    };
  },
};
