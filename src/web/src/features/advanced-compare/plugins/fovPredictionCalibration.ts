import type { ChartPlugin } from './types';
import { labelForRun, palette } from '../metrics';
import { quantile, seededRandom, traceRunsForGroup } from './reliabilityStats';

interface CalibrationBucket {
  actualSum: number;
  count: number;
}

function bucketsForPoints(
  points: Array<{ predicted: number; actual: number; count?: number }>,
  bins: number,
): CalibrationBucket[] {
  const buckets = Array.from({ length: bins }, () => ({ actualSum: 0, count: 0 }));
  for (const point of points || []) {
    const predicted = Number(point.predicted);
    const actual = Number(point.actual);
    if (!Number.isFinite(predicted) || !Number.isFinite(actual)) continue;
    const index = Math.max(0, Math.min(bins - 1, Math.floor(predicted * bins)));
    const weight = Math.max(1, Number(point.count || 1));
    buckets[index]!.actualSum += Math.max(0, Math.min(1, actual)) * weight;
    buckets[index]!.count += weight;
  }
  return buckets;
}

function aggregate(traceBuckets: CalibrationBucket[][], bins: number): CalibrationBucket[] {
  const result = Array.from({ length: bins }, () => ({ actualSum: 0, count: 0 }));
  traceBuckets.forEach((trace) => trace.forEach((bucket, index) => {
    result[index]!.actualSum += bucket.actualSum;
    result[index]!.count += bucket.count;
  }));
  return result;
}

function rgba(hex: string, alpha: number): string {
  const clean = hex.replace('#', '');
  const value = Number.parseInt(clean.length === 3 ? clean.split('').map((part) => part + part).join('') : clean, 16);
  return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
}

export const fovPredictionCalibrationPlugin: ChartPlugin = {
  id: 'fov-prediction-calibration',
  title: 'FoV probability calibration',
  category: 'fov',
  requiredDatasets: ['fov_prediction'],
  description: 'Checks whether predicted tile probabilities match observed frequencies. Shaded confidence bands resample whole FoV traces, so repeated tiles within one trace are not treated as independent evidence.',
  controls: [
    { key: 'bins', label: 'Probability bins', type: 'select', defaultValue: 10, options: [5, 10, 15, 20].map((value) => ({ value, label: String(value) })) },
    { key: 'confidence', label: 'Confidence', type: 'select', defaultValue: 0.95, options: [{ value: 0.9, label: '90%' }, { value: 0.95, label: '95%' }, { value: 0.99, label: '99%' }] },
    { key: 'bootstrapSamples', label: 'Bootstrap samples', type: 'select', defaultValue: 2000, options: [500, 2000, 5000].map((value) => ({ value, label: String(value) })) },
    { key: 'minimumObservations', label: 'Min observations / bin', type: 'number', defaultValue: 25, min: 1, max: 100000, step: 1 },
    { key: 'showBands', label: 'Confidence bands', type: 'checkbox', defaultValue: true },
  ],
  build: ({ metrics, aliases, options }) => {
    const bins = Math.max(2, Number(options.bins || 10));
    const confidence = Number(options.confidence || 0.95);
    const bootstrapSamples = Math.max(100, Number(options.bootstrapSamples || 2000));
    const minimumObservations = Math.max(1, Number(options.minimumObservations || 1));
    const showBands = options.showBands !== false;
    const tail = (1 - confidence) / 2;
    const datasets: any[] = [];

    Object.keys(metrics.fov_prediction || {}).forEach((groupId, groupIndex) => {
      const traces = traceRunsForGroup(metrics, groupId);
      const byTrace = traces
        .map((trace) => bucketsForPoints(trace.stats.calibration || [], bins))
        .filter((trace) => trace.some((bucket) => bucket.count > 0));
      if (!byTrace.length) return;
      const observed = aggregate(byTrace, bins);
      const distributions = Array.from({ length: bins }, () => [] as number[]);
      const random = seededRandom(0x51f15e + groupIndex * 7919 + bins);
      if (showBands && byTrace.length > 1) {
        for (let sample = 0; sample < bootstrapSamples; sample += 1) {
          const selected = Array.from({ length: byTrace.length }, () => byTrace[Math.floor(random() * byTrace.length)]!);
          aggregate(selected, bins).forEach((bucket, index) => {
            if (bucket.count > 0) distributions[index]!.push(bucket.actualSum / bucket.count);
          });
        }
      }
      const color = palette(groupIndex);
      const point = (index: number, value: number) => ({
        x: (index + 0.5) / bins,
        y: value,
        count: observed[index]!.count,
        traces: byTrace.length,
      });
      const eligible = observed.map((bucket) => bucket.count >= minimumObservations);
      if (showBands && byTrace.length > 1) {
        datasets.push({
          label: `${labelForRun(metrics, groupId, aliases)} lower`,
          data: observed.map((bucket, index) => eligible[index] ? point(index, quantile(distributions[index]!, tail)) : null),
          borderColor: rgba(color, 0.18), backgroundColor: rgba(color, 0.12), borderWidth: 1,
          pointRadius: 0, tension: 0.12, spanGaps: false, parsing: false, _band: true,
        });
        datasets.push({
          label: `${labelForRun(metrics, groupId, aliases)} ${Math.round(confidence * 100)}% band`,
          data: observed.map((bucket, index) => eligible[index] ? point(index, quantile(distributions[index]!, 1 - tail)) : null),
          borderColor: rgba(color, 0.18), backgroundColor: rgba(color, 0.12), borderWidth: 1,
          pointRadius: 0, tension: 0.12, spanGaps: false, parsing: false, fill: '-1', _band: true,
        });
      }
      datasets.push({
        label: labelForRun(metrics, groupId, aliases),
        data: observed.map((bucket, index) => eligible[index] && bucket.count > 0 ? point(index, bucket.actualSum / bucket.count) : null),
        borderColor: color, backgroundColor: color, borderWidth: 2,
        pointRadius: 3, tension: 0.12, spanGaps: false, parsing: false,
      });
    });
    if (!datasets.length) return null;

    datasets.unshift({
      label: 'Perfect calibration',
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: '#64748b', backgroundColor: '#64748b', borderDash: [5, 5],
      borderWidth: 1, pointRadius: 0, parsing: false,
    });

    return {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
          legend: { position: 'bottom', labels: { filter: (item: any, data: any) => !data.datasets[item.datasetIndex]?._band } },
          tooltip: {
            filter: (item: any) => !item.dataset?._band,
            callbacks: {
              afterLabel: (context: any) => context.raw?.count
                ? `${Math.round(context.raw.count).toLocaleString()} observations from ${context.raw.traces} traces`
                : '',
            },
          },
        },
        scales: {
          x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Predicted tile visibility probability' } },
          y: { min: 0, max: 1, title: { display: true, text: 'Observed visibility frequency' } },
        },
      },
    };
  },
};
