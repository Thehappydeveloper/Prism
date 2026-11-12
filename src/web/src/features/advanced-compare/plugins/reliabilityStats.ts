import type { RunMetrics } from '../../../core/types';
import type { FovPredictionStats } from './fovMetrics';

export type AccuracyMetricKey = 'weightedIou' | 'iou' | 'topkRecall' | 'visibleFrameRecall' | 'mae' | 'rmse' | 'peakMissRate';
export type TraceReducer = 'mean' | 'median' | 'worst';

export const ACCURACY_METRICS: Array<{ value: AccuracyMetricKey; label: string; higherIsBetter: boolean }> = [
  { value: 'weightedIou', label: 'Weighted IoU', higherIsBetter: true },
  { value: 'iou', label: 'Binary IoU', higherIsBetter: true },
  { value: 'topkRecall', label: 'Top-k recall', higherIsBetter: true },
  { value: 'visibleFrameRecall', label: 'Visible-frame recall', higherIsBetter: true },
  { value: 'mae', label: 'MAE', higherIsBetter: false },
  { value: 'rmse', label: 'RMSE', higherIsBetter: false },
  { value: 'peakMissRate', label: 'Peak miss rate', higherIsBetter: false },
];

export interface TracePredictionStats {
  runId: string;
  scenarioKey: string;
  warmupSegments: number;
  stats: FovPredictionStats;
}

export function metricDefinition(metric: AccuracyMetricKey) {
  return ACCURACY_METRICS.find((candidate) => candidate.value === metric) || ACCURACY_METRICS[0]!;
}

export function traceRunsForGroup(metrics: RunMetrics, groupId: string): TracePredictionStats[] {
  const grouped = metrics.fov_prediction?.[groupId] as (FovPredictionStats & { _traceRuns?: TracePredictionStats[] }) | undefined;
  if (Array.isArray(grouped?._traceRuns)) return grouped._traceRuns;
  if (!grouped) return [];
  const meta = metrics.meta?.[groupId] || {};
  return [{
    runId: groupId,
    scenarioKey: String(meta.fov_trace || groupId),
    warmupSegments: Number(meta.buffer_window_segments || 0),
    stats: grouped,
  }];
}

function reduce(values: number[], reducer: TraceReducer, higherIsBetter: boolean): number | null {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!finite.length) return null;
  if (reducer === 'median') {
    const mid = Math.floor(finite.length / 2);
    return finite.length % 2 ? finite[mid]! : (finite[mid - 1]! + finite[mid]!) / 2;
  }
  if (reducer === 'worst') return higherIsBetter ? finite[0]! : finite[finite.length - 1]!;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

export function traceScore(
  trace: TracePredictionStats,
  metric: AccuracyMetricKey,
  reducer: TraceReducer,
  warmupOverride: number,
): number | null {
  const definition = metricDefinition(metric);
  const raw = ((((trace.stats as unknown) as Record<string, unknown>)[metric]) || []) as number[];
  const warmup = warmupOverride >= 0 ? warmupOverride : trace.warmupSegments;
  return reduce(raw.slice(Math.max(0, warmup)), reducer, definition.higherIsBetter);
}

export function traceScoreMap(
  metrics: RunMetrics,
  groupId: string,
  metric: AccuracyMetricKey,
  reducer: TraceReducer,
  warmup: number,
): Map<string, number> {
  const out = new Map<string, number>();
  traceRunsForGroup(metrics, groupId).forEach((trace) => {
    const value = traceScore(trace, metric, reducer, warmup);
    if (value !== null) out.set(trace.scenarioKey, value);
  });
  return out;
}

const ONE_SIDED_Z: Record<string, number> = { '0.9': 1.281551566, '0.95': 1.644853627, '0.99': 2.326347874 };

export function wilsonLower(successes: number, total: number, confidence: number): number {
  if (total <= 0) return 0;
  const z = ONE_SIDED_Z[String(confidence)] || ONE_SIDED_Z['0.95']!;
  const proportion = successes / total;
  const denominator = 1 + (z * z) / total;
  const center = proportion + (z * z) / (2 * total);
  const radius = z * Math.sqrt((proportion * (1 - proportion)) / total + (z * z) / (4 * total * total));
  return Math.max(0, (center - radius) / denominator);
}

export function quantile(values: number[], probability: number): number {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return 0;
  const index = Math.max(0, Math.min(sorted.length - 1, (sorted.length - 1) * probability));
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const fraction = index - lower;
  return sorted[lower]! * (1 - fraction) + sorted[upper]! * fraction;
}

export function seededRandom(seed = 0x9e3779b9): () => number {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
}
