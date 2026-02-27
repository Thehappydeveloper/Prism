import type { ChartPlugin } from './types';
import { labelForRun, palette } from '../metrics';

export const qoeCdfPlugin: ChartPlugin = {
  id: 'qoe-cdf',
  title: 'QoE CDF (per-segment)',
  build: ({ metrics, aliases }) => {
    const datasets: any[] = [];
    Object.entries(metrics.qoe || {}).forEach(([runId, values], idx) => {
      const sorted = [...values].filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
      if (!sorted.length) return;
      const maxV = sorted[sorted.length - 1] || 1;
      const epsilon = maxV * 0.01 || 0.01;
      const cdf = [{ x: 0, y: 1 }];
      sorted.forEach((v, j) => cdf.push({ x: v, y: 1 - (j + 1) / sorted.length }));
      cdf.push({ x: maxV + epsilon, y: 0 });
      datasets.push({
        label: labelForRun(metrics, runId, aliases),
        data: cdf,
        borderWidth: 2,
        stepped: true,
        fill: false,
        tension: 0,
        pointRadius: 0,
        borderColor: palette(idx),
      });
    });

    if (!datasets.length) return null;

    return {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          x: { type: 'linear', title: { display: true, text: 'QoE' }, min: 0, max: 1 },
          y: { title: { display: true, text: 'Fraction ≥ QoE' }, min: 0, max: 1 },
        },
      },
    };
  },
};
