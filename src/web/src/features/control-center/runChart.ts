import type { EventLog, RunConfig } from '../../core/types';

export type TailMode = 'drain' | 'playback';

export interface Point {
  x: number;
  y: number;
}

export interface Layer {
  id: 'bandwidth' | 'bitrate' | 'buffer' | 'qoe';
  label: string;
  color: string;
  axis: 'left' | 'right' | 'qoe';
  data: Point[];
}

export interface StallSpan {
  start: number;
  end: number;
}

export interface LayerVisibility {
  bandwidth: boolean;
  bitrate: boolean;
  buffer: boolean;
  qoe: boolean;
}

export interface AxisBounds {
  maxX: number;
  leftMax: number;
  rightMax: number;
  qoeMax: number;
  hasQoe: boolean;
}

interface SegmentLike {
  time_s?: number;
  duration_s?: number;
  size_bits?: number;
}

const LAYERS_META: Record<Layer['id'], { label: string; color: string; axis: Layer['axis'] }> = {
  bandwidth: { label: 'Network bandwidth (trace)', color: '#22d3ee', axis: 'left' },
  bitrate: { label: 'Buffered avg bitrate', color: '#f472b6', axis: 'left' },
  buffer: { label: 'Buffer occupancy', color: '#a3e635', axis: 'right' },
  qoe: { label: 'QoE score', color: '#f59e0b', axis: 'qoe' },
};

function n(v: unknown, fallback = 0): number {
  const parsed = Number(v);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function safeMax(values: number[]): number {
  const finite = values.filter(Number.isFinite);
  return finite.length ? Math.max(...finite) : 0;
}

export function dedupeSeries(series: Point[]): Point[] {
  if (!series.length) return [];
  const out: Point[] = [series[0]!];
  for (let i = 1; i < series.length; i += 1) {
    const prev = out[out.length - 1]!;
    const cur = series[i]!;
    if (Math.abs(cur.x - prev.x) < 1e-9 && Math.abs(cur.y - prev.y) < 1e-9) continue;
    out.push(cur);
  }
  return out;
}

export function clampSeries(series: Point[], cutoff: number): Point[] {
  if (!Number.isFinite(cutoff) || !series.length) return series;
  const before = series.filter((p) => p.x <= cutoff);
  const lastBefore = [...series].reverse().find((p) => p.x <= cutoff);
  const firstAfter = series.find((p) => p.x > cutoff);
  const out = [...before];
  if (firstAfter && lastBefore && Math.abs(lastBefore.x - cutoff) > 1e-9) {
    const ratio = (cutoff - lastBefore.x) / (firstAfter.x - lastBefore.x);
    out.push({ x: cutoff, y: lastBefore.y + (firstAfter.y - lastBefore.y) * ratio });
  } else if (!firstAfter && lastBefore && lastBefore.x < cutoff) {
    out.push({ x: cutoff, y: lastBefore.y });
  }
  out.sort((a, b) => a.x - b.x);
  return dedupeSeries(out);
}

export function alignBandwidth(
  samples: Array<{ time_s: number; bandwidth_mbps: number }>,
  endTime: number,
  maxGap = 0.25,
): Point[] {
  if (!samples.length) return [];
  const sorted = [...samples]
    .filter((s) => Number.isFinite(n(s.time_s)) && Number.isFinite(n(s.bandwidth_mbps)))
    .sort((a, b) => n(a.time_s) - n(b.time_s));
  if (!sorted.length) return [];

  if (n(sorted[0]!.time_s) > 0) {
    sorted.unshift({ time_s: 0, bandwidth_mbps: n(sorted[0]!.bandwidth_mbps) });
  }
  if (Number.isFinite(endTime)) {
    const last = sorted[sorted.length - 1]!;
    if (n(last.time_s) < endTime) {
      sorted.push({ time_s: endTime, bandwidth_mbps: n(last.bandwidth_mbps) });
    }
  }

  const gapLimit = Math.max(0.01, maxGap);
  const aligned: Array<{ time_s: number; bandwidth_mbps: number }> = [];
  for (let i = 0; i < sorted.length; i += 1) {
    const cur = sorted[i]!;
    aligned.push(cur);
    const next = sorted[i + 1];
    if (!next) continue;
    const curT = n(cur.time_s);
    const nextT = n(next.time_s);
    const gap = nextT - curT;
    if (gap > gapLimit + 1e-9) {
      for (let t = curT + gapLimit; t < nextT - 1e-9; t += gapLimit) {
        const ratio = (t - curT) / gap;
        aligned.push({
          time_s: t,
          bandwidth_mbps: n(cur.bandwidth_mbps) + (n(next.bandwidth_mbps) - n(cur.bandwidth_mbps)) * ratio,
        });
      }
    }
  }
  aligned.sort((a, b) => n(a.time_s) - n(b.time_s));
  return dedupeSeries(aligned.map((s) => ({ x: n(s.time_s), y: n(s.bandwidth_mbps) })));
}

function buildBufferAndBitrateSeries(events: EventLog, playbackEnd: number): { buffer: Point[]; bitrate: Point[] } {
  const enqueues = (events.enqueued_segments || [])
    .map((s) => ({
      t: n(s.time_s),
      duration: n(s.duration_s),
      bitrate: n(s.duration_s) > 0 ? (n(s.size_bits) / n(s.duration_s)) / 1e6 : 0,
    }))
    .filter((s) => s.duration > 0)
    .sort((a, b) => a.t - b.t);

  const consumed = ((events.consumed_segments || []) as SegmentLike[]).map((s) => n(s.time_s) + n(s.duration_s));
  const rebuf = (events.rebuffering_events || []).map((r) => n(r.time_s) + n(r.duration_s));
  const lastEvent = safeMax([0, ...enqueues.map((e) => e.t), ...consumed, ...rebuf, ...((events.time_s || []).map((t) => n(t)))]);
  const totalEnqueued = enqueues.reduce((acc, e) => acc + e.duration, 0);
  const horizon = Math.max(Number.isFinite(playbackEnd) ? playbackEnd : 0, lastEvent + totalEnqueued);
  if (!enqueues.length && !horizon) return { buffer: [], bitrate: [] };

  const step = Math.max(0.02, Math.max(horizon, 1) / 1000);
  const queue: Array<{ remaining: number; bitrate: number }> = [];
  const buffer: Point[] = [];
  const bitrate: Point[] = [];
  let idx = 0;
  let buf = 0;
  let weighted = 0;

  for (let t = 0; t <= horizon + step; t += step) {
    while (idx < enqueues.length && enqueues[idx]!.t <= t + 1e-9) {
      const seg = enqueues[idx]!;
      queue.push({ remaining: seg.duration, bitrate: seg.bitrate });
      buf += seg.duration;
      weighted += seg.duration * seg.bitrate;
      idx += 1;
    }

    if (buf > 0) {
      let consume = Math.min(step, buf);
      buf -= consume;
      while (consume > 1e-9 && queue.length) {
        const cur = queue[0]!;
        const take = Math.min(cur.remaining, consume);
        cur.remaining -= take;
        consume -= take;
        weighted -= take * cur.bitrate;
        if (cur.remaining <= 1e-9) {
          queue.shift();
        }
      }
      if (weighted < 1e-9) weighted = 0;
    } else {
      queue.length = 0;
      weighted = 0;
    }
    buffer.push({ x: t, y: Math.max(0, buf) });
    bitrate.push({ x: t, y: buf > 1e-9 ? weighted / buf : 0 });
  }
  return { buffer, bitrate };
}

function trimBufferTail(series: Point[], playbackEnd: number, mode: TailMode): Point[] {
  if (!series.length) return [];

  let lastPositive = -1;
  for (let i = series.length - 1; i >= 0; i -= 1) {
    if (series[i]!.y > 0) {
      lastPositive = i;
      break;
    }
  }
  if (lastPositive < 0) {
    return [{ x: Number.isFinite(playbackEnd) ? playbackEnd : series[series.length - 1]!.x, y: 0 }];
  }

  let zeroIndex = series.findIndex((p, idx) => idx > lastPositive && p.y <= 0);
  if (zeroIndex < 0) zeroIndex = series.length - 1;

  let trimmed = series.slice(0, zeroIndex + 1);
  if (Number.isFinite(playbackEnd)) {
    const hasPlayback = trimmed.some((p) => Math.abs(p.x - playbackEnd) < 1e-9);
    if (!hasPlayback) {
      const before = [...trimmed].reverse().find((p) => p.x <= playbackEnd);
      const after = trimmed.find((p) => p.x > playbackEnd);
      let interpY = before ? before.y : 0;
      if (before && after && Math.abs(after.x - before.x) > 1e-9) {
        const ratio = (playbackEnd - before.x) / (after.x - before.x);
        interpY = before.y + (after.y - before.y) * ratio;
      }
      trimmed.push({ x: playbackEnd, y: interpY });
      trimmed.sort((a, b) => a.x - b.x);
    }
  }
  if (mode === 'playback' && Number.isFinite(playbackEnd)) {
    trimmed = trimmed.filter((p) => p.x <= playbackEnd);
  }
  return dedupeSeries(trimmed);
}

export function buildRunLayers(
  events: EventLog,
  config: RunConfig,
  traceSamples: Array<{ time_s: number; bandwidth_mbps: number }>,
  qoeValues: number[],
  tailMode: TailMode,
): { layers: Layer[]; stalls: StallSpan[]; horizon: number } {
  const consumed = ((events.consumed_segments || []) as SegmentLike[]).map((s) => n(s.time_s) + n(s.duration_s));
  const enqueuedEnd = (events.enqueued_segments || []).map((s) => n(s.time_s) + n(s.duration_s));
  const playbackEnd = Math.max(
    n(events.total_playback_time),
    safeMax(events.time_s || []),
    safeMax(consumed),
    safeMax(enqueuedEnd),
  );

  const raw = buildBufferAndBitrateSeries(events, playbackEnd);
  const bufferSeries = trimBufferTail(raw.buffer, playbackEnd, tailMode);
  const bufferEnd = bufferSeries.length ? bufferSeries[bufferSeries.length - 1]!.x : playbackEnd;
  const horizon = tailMode === 'playback' ? playbackEnd : bufferEnd;

  const bitrateSeries = clampSeries(raw.bitrate, horizon);
  const bandwidthSeries = traceSamples.length ? alignBandwidth(traceSamples, horizon) : [{ x: 0, y: 0 }, { x: horizon, y: 0 }];

  const segmentTimes = (events.enqueued_segments || []).map((seg) => n(seg.time_s));
  const qoeSeriesRaw = qoeValues.map((q, idx) => ({ x: segmentTimes[idx] ?? idx, y: n(q) }));
  const qoeSeries = dedupeSeries(clampSeries(qoeSeriesRaw, horizon));

  const stalls = (events.rebuffering_events || [])
    .map((ev) => ({ start: n(ev.time_s), end: Math.min(horizon, n(ev.time_s) + n(ev.duration_s)) }))
    .filter((s) => s.end > s.start);

  const layers: Layer[] = [
    { id: 'bandwidth', ...LAYERS_META.bandwidth, data: bandwidthSeries.filter((s) => s.x <= horizon) },
    { id: 'bitrate', ...LAYERS_META.bitrate, data: bitrateSeries },
    { id: 'buffer', ...LAYERS_META.buffer, data: clampSeries(bufferSeries, horizon) },
    { id: 'qoe', ...LAYERS_META.qoe, data: qoeSeries },
  ];

  return { layers, stalls, horizon: Math.max(1, horizon) };
}

export function buildDatasets(layers: Layer[], visibility: LayerVisibility, runLabel = ''): any[] {
  const visibleById: Record<Layer['id'], boolean> = {
    bandwidth: visibility.bandwidth,
    bitrate: visibility.bitrate,
    buffer: visibility.buffer,
    qoe: visibility.qoe,
  };
  return layers
    .filter((layer) => visibleById[layer.id])
    .map((layer) => ({
      label: runLabel ? `${runLabel} - ${layer.label}` : layer.label,
      data: layer.data,
      borderColor: layer.color,
      backgroundColor: layer.color,
      borderWidth: 2,
      tension: 0,
      parsing: false,
      pointRadius: 0,
      yAxisID: layer.axis === 'right' ? 'y1' : layer.axis === 'qoe' ? 'y2' : 'y',
    }));
}

export function computeBounds(layers: Layer[], visibility: LayerVisibility): AxisBounds {
  const visible = layers.filter(
    (layer) =>
      ((layer.id === 'bandwidth' && visibility.bandwidth) ||
        (layer.id === 'bitrate' && visibility.bitrate) ||
        (layer.id === 'buffer' && visibility.buffer) ||
        (layer.id === 'qoe' && visibility.qoe)) &&
      layer.data.length,
  );

  const safeFiniteMax = (values: number[]) => {
    const finite = values.filter(Number.isFinite);
    return finite.length ? Math.max(...finite) : 1;
  };

  const allX = visible.flatMap((l) => l.data.map((p) => p.x));
  const left = visible.filter((l) => l.axis === 'left').flatMap((l) => l.data.map((p) => p.y));
  const right = visible.filter((l) => l.axis === 'right').flatMap((l) => l.data.map((p) => p.y));
  const qoe = visible.filter((l) => l.axis === 'qoe').flatMap((l) => l.data.map((p) => p.y));

  return {
    maxX: Math.max(1, safeFiniteMax(allX)),
    leftMax: safeFiniteMax(left),
    rightMax: safeFiniteMax(right),
    qoeMax: safeFiniteMax(qoe),
    hasQoe: qoe.length > 0,
  };
}

export function aggregateBounds(bounds: AxisBounds[]): AxisBounds {
  return bounds.reduce(
    (acc, cur) => ({
      maxX: Math.max(acc.maxX, cur.maxX),
      leftMax: Math.max(acc.leftMax, cur.leftMax),
      rightMax: Math.max(acc.rightMax, cur.rightMax),
      qoeMax: Math.max(acc.qoeMax, cur.qoeMax),
      hasQoe: acc.hasQoe || cur.hasQoe,
    }),
    { maxX: 1, leftMax: 1, rightMax: 1, qoeMax: 1, hasQoe: false },
  );
}
