import type { ChartPlugin } from './types';
import { labelForRun } from '../metrics';
import { ACCURACY_METRICS, metricDefinition, quantile, seededRandom, traceScoreMap, type AccuracyMetricKey, type TraceReducer } from './reliabilityStats';

function probabilityColor(probability: number): string {
  const distance = Math.abs(probability - 0.5) * 2;
  if (probability >= 0.5) return `rgba(20, 184, 166, ${0.18 + 0.72 * distance})`;
  return `rgba(244, 63, 94, ${0.18 + 0.72 * distance})`;
}

function winProbability(advantages: number[], margin: number): number {
  if (!advantages.length) return 0.5;
  return advantages.reduce((sum, value) => sum + (value > margin ? 1 : value < -margin ? 0 : 0.5), 0) / advantages.length;
}

export const fovPairwiseWinProbabilityPlugin: ChartPlugin = {
  id: 'fov-pairwise-win-probability',
  title: 'Paired FoV predictor win probability',
  category: 'fov',
  requiredDatasets: ['fov_prediction'],
  description: 'Every comparison is paired by FoV trace. A cell is the probability that its row predictor beats its column predictor; tooltips report a trace-bootstrap confidence interval.',
  controls: [
    { key: 'metric', label: 'Accuracy metric', type: 'select', defaultValue: 'weightedIou', options: ACCURACY_METRICS },
    { key: 'reducer', label: 'Within-trace score', type: 'select', defaultValue: 'mean', options: [{ label: 'Mean', value: 'mean' }, { label: 'Median', value: 'median' }, { label: 'Worst segment', value: 'worst' }] },
    { key: 'warmup', label: 'Exclude first segments', type: 'number', defaultValue: 2, min: 0, max: 10, step: 1 },
    { key: 'margin', label: 'Practical win margin', type: 'number', defaultValue: 0, min: 0, max: 1, step: 0.01, help: 'Differences inside ±margin count as a tie.' },
    { key: 'confidence', label: 'Bootstrap interval', type: 'select', defaultValue: 0.95, options: [{ label: '90%', value: 0.9 }, { label: '95%', value: 0.95 }, { label: '99%', value: 0.99 }] },
    { key: 'bootstrap', label: 'Bootstrap samples', type: 'select', defaultValue: 2000, options: [{ label: '500', value: 500 }, { label: '2,000', value: 2000 }, { label: '5,000', value: 5000 }] },
  ],
  build: ({ metrics, aliases, options }) => {
    const groupIds = Object.keys(metrics.fov_prediction || {});
    if (groupIds.length < 2) return null;
    const metric = String(options.metric || 'weightedIou') as AccuracyMetricKey;
    const definition = metricDefinition(metric);
    const reducer = String(options.reducer || 'mean') as TraceReducer;
    const warmup = Math.max(0, Math.floor(Number(options.warmup || 0)));
    const margin = Math.max(0, Number(options.margin || 0));
    const confidence = Number(options.confidence || 0.95);
    const bootstrapSamples = Math.max(100, Math.floor(Number(options.bootstrap || 2000)));
    const scoreMaps = groupIds.map((groupId) => traceScoreMap(metrics, groupId, metric, reducer, warmup));
    const labels = groupIds.map((groupId) => labelForRun(metrics, groupId, aliases));
    const points: any[] = [];

    groupIds.forEach((_rowId, row) => {
      groupIds.forEach((_columnId, column) => {
        const scenarios = Array.from(scoreMaps[row]!.keys()).filter((scenario) => scoreMaps[column]!.has(scenario));
        const advantages = scenarios.map((scenario) => {
          const rowScore = scoreMaps[row]!.get(scenario)!;
          const columnScore = scoreMaps[column]!.get(scenario)!;
          return definition.higherIsBetter ? rowScore - columnScore : columnScore - rowScore;
        });
        const probability = row === column ? 0.5 : winProbability(advantages, margin);
        let lower = probability;
        let upper = probability;
        if (row !== column && advantages.length) {
          const random = seededRandom(0x51f15e + row * 1009 + column * 9176);
          const bootstrap = Array.from({ length: bootstrapSamples }, () => {
            const sample = Array.from({ length: advantages.length }, () => advantages[Math.floor(random() * advantages.length)]!);
            return winProbability(sample, margin);
          });
          const alpha = (1 - confidence) / 2;
          lower = quantile(bootstrap, alpha);
          upper = quantile(bootstrap, 1 - alpha);
        }
        points.push({ x: column, y: row, probability, lower, upper, count: advantages.length, rowLabel: labels[row], columnLabel: labels[column] });
      });
    });

    const valueLabels = {
      id: 'pairwise-win-value-labels',
      afterDatasetsDraw(chart: any) {
        const context = chart.ctx;
        context.save();
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.font = '600 11px sans-serif';
        chart.getDatasetMeta(0).data.forEach((element: any, index: number) => {
          const point = points[index];
          context.fillStyle = point.probability < 0.25 || point.probability > 0.75 ? '#f8fafc' : '#0f172a';
          context.fillText(`${Math.round(point.probability * 100)}%`, element.x, element.y);
        });
        context.restore();
      },
    };

    return {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Row beats column',
          data: points,
          pointStyle: 'rectRounded',
          pointRadius: (context: any) => {
            const area = context.chart.chartArea;
            return area ? Math.max(10, Math.min(area.width, area.height) / groupIds.length * 0.38) : 20;
          },
          pointHoverRadius: (context: any) => {
            const area = context.chart.chartArea;
            return area ? Math.max(12, Math.min(area.width, area.height) / groupIds.length * 0.42) : 23;
          },
          backgroundColor: points.map((point) => probabilityColor(point.probability)),
          borderColor: '#64748b',
          borderWidth: 1,
          parsing: false,
        }],
      },
      plugins: [valueLabels],
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items: any[]) => `${items[0]?.raw?.rowLabel} vs ${items[0]?.raw?.columnLabel}`,
              label: (context: any) => `P(row wins): ${(context.raw.probability * 100).toFixed(1)}%`,
              afterLabel: (context: any) => `${(confidence * 100).toFixed(0)}% bootstrap CI: ${(context.raw.lower * 100).toFixed(1)}–${(context.raw.upper * 100).toFixed(1)}%\nPaired traces: ${context.raw.count}`,
            },
          },
        },
        scales: {
          x: { type: 'linear', min: -0.5, max: groupIds.length - 0.5, ticks: { stepSize: 1, callback: (value: number | string) => labels[Number(value)] || '' }, title: { display: true, text: 'Column predictor' } },
          y: { type: 'linear', reverse: true, min: -0.5, max: groupIds.length - 0.5, ticks: { stepSize: 1, callback: (value: number | string) => labels[Number(value)] || '' }, title: { display: true, text: 'Row predictor' } },
        },
      },
    };
  },
};
