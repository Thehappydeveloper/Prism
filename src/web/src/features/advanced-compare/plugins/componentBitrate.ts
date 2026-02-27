import type { ChartPlugin } from './types';
import { labelForRun, palette, runDuration } from '../metrics';

export const componentBitratePlugin: ChartPlugin = {
  id: 'component-bitrate',
  title: 'Component bitrate allocation',
  build: ({ metrics, aliases }) => {
    const keys = Object.keys(metrics.component_bits || {});
    if (!keys.length) return null;

    const comps = new Set<string>();
    Object.values(metrics.component_bits || {}).forEach((m) => Object.keys(m || {}).forEach((c) => comps.add(c)));

    const datasets = Array.from(comps).map((comp, idx) => ({
      label: comp,
      data: keys.map((k) => {
        const bits = metrics.component_bits[k]?.[comp] || 0;
        const dur = runDuration(metrics, k);
        return dur > 0 ? (bits / dur) / 1e6 : 0;
      }),
      backgroundColor: palette(idx),
      stack: 'bits',
    }));

    if (!datasets.length) return null;

    return {
      type: 'bar',
      data: { labels: keys.map((k) => labelForRun(metrics, k, aliases)), datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          x: { stacked: true },
          y: { stacked: true, beginAtZero: true, title: { display: true, text: 'Mbps' } },
        },
      },
    };
  },
};
