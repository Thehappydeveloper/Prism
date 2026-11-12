import type { RunMetrics } from '../../core/types';

export interface GroupingOptions {
  byAbr: boolean;
  byPredictor: boolean;
  byPredictorPreset: boolean;
  byBandwidthPredictor: boolean;
  byNetwork: boolean;
  mode: 'mean' | 'median';
}

export function average(values: number[]): number {
  const vals = values.filter((v) => Number.isFinite(v));
  if (!vals.length) return 0;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

export function median(values: number[]): number {
  const vals = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (!vals.length) return 0;
  const mid = Math.floor(vals.length / 2);
  return vals.length % 2 ? vals[mid]! : (vals[mid - 1]! + vals[mid]!) / 2;
}

export function agg(values: number[], mode: 'mean' | 'median'): number {
  return mode === 'median' ? median(values) : average(values);
}

export function runDuration(metrics: RunMetrics, runId: string): number {
  return Number(metrics.meta?.[runId]?.duration_s ?? 0);
}

export function runHasFov(metrics: RunMetrics, runId: string): boolean {
  return Boolean(metrics.meta?.[runId]?.fov_present);
}

export function classifyNetwork(path = ''): string {
  const low = String(path).toLowerCase();
  if (low.includes('5g')) return '5G';
  if (low.includes('4g') || low.includes('lte')) return '4G';
  return 'Other';
}

function predictorGroupLabel(meta: NonNullable<RunMetrics['meta'][string]>, includePreset: boolean): string {
  const name = meta.predictor || 'Pred?';
  if (!includePreset) return name;
  const preset = String(meta.predictor_preset_name || '').trim();
  const params = meta.predictor_params || {};
  const entries = Object.entries(params).sort(([a], [b]) => a.localeCompare(b));
  const parameterLabel = entries.map(([key, value]) => `${key}=${typeof value === 'number' ? Number(value).toPrecision(4) : String(value)}`).join(', ');
  return `${name} [${preset || parameterLabel || 'default'}]`;
}

function groupKey(meta: NonNullable<RunMetrics['meta'][string]>, grouping: GroupingOptions): string {
  const parts: string[] = [];
  if (grouping.byAbr) parts.push(meta.abr || 'ABR?');
  if (grouping.byPredictor) parts.push(predictorGroupLabel(meta, grouping.byPredictorPreset));
  if (grouping.byBandwidthPredictor) parts.push(meta.bandwidth_predictor || 'BwPred?');
  if (grouping.byNetwork) parts.push(meta.network_class || classifyNetwork(meta.network_trace || ''));
  return parts.length ? parts.join(' / ') : (meta.run_ids?.[0] || 'run');
}

export function labelForRun(metrics: RunMetrics, runId: string, aliases: Record<string, string>): string {
  const base = metrics.labels?.[runId] || runId;
  const alias = aliases[runId];
  return alias ? `${alias} (${base})` : base;
}

function mergeMetricValue(target: unknown, source: unknown): unknown {
  if (Array.isArray(source)) return [...(Array.isArray(target) ? target : []), ...source];
  if (source && typeof source === 'object') {
    const merged: Record<string, unknown> = target && typeof target === 'object' && !Array.isArray(target)
      ? { ...(target as Record<string, unknown>) }
      : {};
    for (const [key, value] of Object.entries(source as Record<string, unknown>)) {
      merged[key] = mergeMetricValue(merged[key], value);
    }
    return merged;
  }
  return source;
}

export function aggregateMetrics(metrics: RunMetrics, grouping: GroupingOptions): RunMetrics {
  if (!grouping.byAbr && !grouping.byPredictor && !grouping.byBandwidthPredictor && !grouping.byNetwork) return metrics;

  const grouped: RunMetrics = {
    labels: {},
    qoe: {},
    qoe_series: {},
    fov_quality: {},
    fov_prediction: {},
    fov_segments: {},
    delta_fov: {},
    stall: {},
    component_bits: {},
    bitrate_vs_quality: {},
    buffer: {},
    latency: {},
    meta: {},
  };

  // Metadata is always returned, even when no active plugin requests QoE.
  const runIds = Object.keys(metrics.meta || {});
  const metaBuckets: Record<
    string,
    {
      run_ids: string[];
      durations: number[];
      fov_present: boolean;
      abr?: string;
      predictor?: string;
      bandwidth_predictor?: string;
      network_class?: string;
      stallDur: number[];
      stallCnt: number[];
      bufferMean: number[];
      bufferStd: number[];
      bufferMin: number[];
      bufferMax: number[];
      componentRates: Record<string, number[]>;
      latencyMedian: number[];
      latencyMean: number[];
      latencyP95: number[];
      latencyMax: number[];
      latencyCount: number;
    }
  > = {};

  for (const runId of runIds) {
    const meta = metrics.meta?.[runId] || {};
    const key = groupKey({ ...meta, run_ids: [runId] }, grouping);
    if (!metaBuckets[key]) {
      metaBuckets[key] = {
        run_ids: [],
        durations: [],
        fov_present: false,
        abr: meta.abr,
        predictor: meta.predictor,
        bandwidth_predictor: meta.bandwidth_predictor,
        network_class: meta.network_class,
        stallDur: [],
        stallCnt: [],
        bufferMean: [],
        bufferStd: [],
        bufferMin: [],
        bufferMax: [],
        componentRates: {},
        latencyMedian: [],
        latencyMean: [],
        latencyP95: [],
        latencyMax: [],
        latencyCount: 0,
      };
      grouped.qoe[key] = [];
      grouped.qoe_series![key] = [];
      grouped.fov_quality[key] = { fov: [], non: [] };
      grouped.fov_prediction![key] = {};
      grouped.fov_segments![key] = [];
      grouped.delta_fov[key] = [];
      grouped.bitrate_vs_quality[key] = [];
      grouped.component_bits[key] = {};
    }

    const bucket = metaBuckets[key];
    bucket.run_ids.push(runId);
    bucket.durations.push(runDuration(metrics, runId));
    bucket.fov_present = bucket.fov_present || runHasFov(metrics, runId);

    grouped.qoe[key]!.push(...(metrics.qoe?.[runId] || []));
    grouped.qoe_series![key]!.push(...(metrics.qoe_series?.[runId] || []));
    grouped.fov_quality[key]!.fov.push(...(metrics.fov_quality?.[runId]?.fov || []));
    grouped.fov_quality[key]!.non.push(...(metrics.fov_quality?.[runId]?.non || []));
    const fovPrediction = metrics.fov_prediction?.[runId];
    if (fovPrediction) {
      grouped.fov_prediction![key] = mergeMetricValue(grouped.fov_prediction![key], fovPrediction) as Record<string, unknown>;
      const groupedPrediction = grouped.fov_prediction![key] as Record<string, unknown>;
      const traceRuns = Array.isArray(groupedPrediction._traceRuns) ? groupedPrediction._traceRuns : [];
      traceRuns.push({
        runId,
        scenarioKey: String(meta.fov_trace || runId),
        warmupSegments: Number(meta.buffer_window_segments || 0),
        stats: fovPrediction,
      });
      groupedPrediction._traceRuns = traceRuns;
    }
    grouped.fov_segments![key]!.push(...(metrics.fov_segments?.[runId] || []));
    grouped.delta_fov[key]!.push(...(metrics.delta_fov?.[runId] || []));
    grouped.bitrate_vs_quality[key]!.push(...(metrics.bitrate_vs_quality?.[runId] || []));

    const stall = metrics.stall?.[runId] || { duration: 0, count: 0 };
    bucket.stallDur.push(Number(stall.duration || 0));
    bucket.stallCnt.push(Number(stall.count || 0));

    const buf = metrics.buffer?.[runId];
    if (buf) {
      bucket.bufferMean.push(Number(buf.mean || 0));
      bucket.bufferStd.push(Number(buf.std || 0));
      bucket.bufferMin.push(Number(buf.min || 0));
      bucket.bufferMax.push(Number(buf.max || 0));
    }

    const compBits = metrics.component_bits?.[runId] || {};
    const dur = Math.max(runDuration(metrics, runId), 1e-3);
    for (const [comp, bits] of Object.entries(compBits)) {
      if (!bucket.componentRates[comp]) bucket.componentRates[comp] = [];
      bucket.componentRates[comp].push(Number(bits || 0) / dur);
    }

    const latency = metrics.latency?.[runId];
    if (latency) {
      bucket.latencyMedian.push(Number(latency.median_ms || 0));
      bucket.latencyMean.push(Number(latency.mean_ms || 0));
      bucket.latencyP95.push(Number(latency.p95_ms || 0));
      bucket.latencyMax.push(Number(latency.max_ms || 0));
      bucket.latencyCount += Number(latency.count || 0);
    }
  }

  for (const [key, bucket] of Object.entries(metaBuckets)) {
    const duration = agg(bucket.durations, grouping.mode);
    const count = bucket.run_ids.length;
    grouped.labels[key] = count > 1 ? `${key} (${count} runs)` : key;
    grouped.meta[key] = {
      duration_s: duration,
      fov_present: bucket.fov_present,
      abr: bucket.abr,
      predictor: bucket.predictor,
      bandwidth_predictor: bucket.bandwidth_predictor,
      network_class: bucket.network_class,
      run_ids: bucket.run_ids,
    };
    grouped.stall[key] = {
      duration: agg(bucket.stallDur, grouping.mode),
      count: Math.round(agg(bucket.stallCnt, grouping.mode)),
    };

    if (bucket.bufferMean.length) {
      grouped.buffer[key] = {
        mean: agg(bucket.bufferMean, grouping.mode),
        std: agg(bucket.bufferStd, grouping.mode),
        min: agg(bucket.bufferMin, grouping.mode),
        max: agg(bucket.bufferMax, grouping.mode),
      };
    }

    if (bucket.latencyMean.length) {
      grouped.latency![key] = {
        count: bucket.latencyCount,
        median_ms: agg(bucket.latencyMedian, grouping.mode),
        mean_ms: agg(bucket.latencyMean, grouping.mode),
        p95_ms: agg(bucket.latencyP95, grouping.mode),
        max_ms: Math.max(...bucket.latencyMax),
      };
    }

    for (const [comp, rates] of Object.entries(bucket.componentRates)) {
      grouped.component_bits[key]![comp] = agg(rates, grouping.mode) * Math.max(duration, 1e-3);
    }
  }

  return grouped;
}

export function bucket(values: number[], bins: number, min: number, max: number): Array<{ bin: number; count: number }> {
  const width = (max - min) / bins;
  const out = Array.from({ length: bins }, (_, i) => ({ bin: min + i * width, count: 0 }));
  for (const raw of values) {
    const v = Number(raw);
    if (!Number.isFinite(v) || v < min || v > max) continue;
    const idx = Math.min(bins - 1, Math.floor((v - min) / width));
    out[idx]!.count += 1;
  }
  return out;
}

export function palette(index: number): string {
  const colors = ['#22d3ee', '#f472b6', '#a3e635', '#f59e0b', '#38bdf8', '#c084fc', '#fb7185', '#14b8a6'];
  return colors[index % colors.length]!;
}
