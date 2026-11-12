import type { ChartPlugin } from './types';
import { labelForRun, palette } from '../metrics';
import { ACCURACY_METRICS, metricDefinition, traceScoreMap, wilsonLower, type AccuracyMetricKey, type TraceReducer } from './reliabilityStats';

export const fovReliabilityGuaranteePlugin: ChartPlugin = {
  id: 'fov-reliability-guarantee',
  title: 'FoV accuracy reliability guarantee',
  category: 'fov',
  requiredDatasets: ['fov_prediction'],
  description: 'Each trace is one independent sample. Curves show the probability of meeting an accuracy requirement and/or its one-sided Wilson confidence lower bound.',
  controls: [
    { key: 'metric', label: 'Accuracy metric', type: 'select', defaultValue: 'weightedIou', options: ACCURACY_METRICS },
    { key: 'target', label: 'Selected threshold', type: 'number', defaultValue: 0.7, min: 0, max: 1, step: 0.05 },
    { key: 'confidence', label: 'Confidence', type: 'select', defaultValue: 0.95, options: [{ label: '90%', value: 0.9 }, { label: '95%', value: 0.95 }, { label: '99%', value: 0.99 }] },
    { key: 'reducer', label: 'Within-trace score', type: 'select', defaultValue: 'mean', options: [{ label: 'Mean', value: 'mean' }, { label: 'Median', value: 'median' }, { label: 'Worst segment', value: 'worst' }] },
    { key: 'warmup', label: 'Exclude first segments', type: 'number', defaultValue: 2, min: 0, max: 10, step: 1 },
    { key: 'display', label: 'Display', type: 'select', defaultValue: 'both', options: [{ label: 'Observed + lower bound', value: 'both' }, { label: 'Lower bound only', value: 'lower' }, { label: 'Observed only', value: 'observed' }] },
  ],
  build: ({ metrics, aliases, options }) => {
    const metric = String(options.metric || 'weightedIou') as AccuracyMetricKey;
    const definition = metricDefinition(metric);
    const confidence = Number(options.confidence || 0.95);
    const reducer = String(options.reducer || 'mean') as TraceReducer;
    const warmup = Math.max(0, Math.floor(Number(options.warmup || 0)));
    const target = Math.max(0, Math.min(1, Number(options.target ?? 0.7)));
    const display = String(options.display || 'both');
    const thresholds = Array.from({ length: 41 }, (_, index) => index / 40);
    const datasets: any[] = [];

    Object.keys(metrics.fov_prediction || {}).forEach((groupId, index) => {
      const scores = Array.from(traceScoreMap(metrics, groupId, metric, reducer, warmup).values());
      if (!scores.length) return;
      const points = thresholds.map((threshold) => {
        const successes = scores.filter((score) => definition.higherIsBetter ? score >= threshold : score <= threshold).length;
        return {
          x: threshold,
          y: successes / scores.length,
          observed: successes / scores.length,
          lower: wilsonLower(successes, scores.length, confidence),
          successes,
          total: scores.length,
        };
      });
      const color = palette(index);
      const label = labelForRun(metrics, groupId, aliases);
      if (display !== 'lower') {
        datasets.push({
          label: `${label} observed`,
          data: points.map((point) => ({ ...point, y: point.observed })),
          borderColor: color,
          backgroundColor: color,
          borderDash: display === 'both' ? [5, 4] : undefined,
          borderWidth: 1.5,
          pointRadius: points.map((point) => Math.abs(point.x - target) < 0.013 ? 4 : 0),
          tension: 0,
          parsing: false,
        });
      }
      if (display !== 'observed') {
        datasets.push({
          label: `${label} ${(confidence * 100).toFixed(0)}% lower bound`,
          data: points.map((point) => ({ ...point, y: point.lower })),
          borderColor: color,
          backgroundColor: color,
          borderWidth: 2.5,
          pointRadius: points.map((point) => Math.abs(point.x - target) < 0.013 ? 5 : 0),
          tension: 0,
          parsing: false,
        });
      }
    });
    if (!datasets.length) return null;
    datasets.push({
      label: `Selected threshold ${target.toFixed(2)}`,
      data: [{ x: target, y: 0 }, { x: target, y: 1 }],
      borderColor: '#64748b',
      borderDash: [3, 3],
      borderWidth: 1,
      pointRadius: 0,
      parsing: false,
    });

    return {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
          legend: { position: 'bottom' },
          tooltip: {
            callbacks: {
              afterBody: (items: any[]) => {
                const raw = items[0]?.raw;
                return raw?.total ? `${raw.successes}/${raw.total} independent traces met the requirement` : '';
              },
            },
          },
        },
        scales: {
          x: { type: 'linear', min: 0, max: 1, title: { display: true, text: `${definition.label} requirement (${definition.higherIsBetter ? '≥' : '≤'})` } },
          y: { min: 0, max: 1, title: { display: true, text: 'Probability of meeting requirement' } },
        },
      },
    };
  },
};
