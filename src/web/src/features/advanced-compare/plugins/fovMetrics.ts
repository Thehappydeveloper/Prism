import type { RunMetrics } from '../../../core/types';

export interface FovPredictionStats {
  [key: string]: unknown;
  mae: number[];
  rmse: number[];
  playbackMae: number[];
  playbackRmse: number[];
  iou: number[];
  weightedIou: number[];
  distributionIntersection: number[];
  histogramSimilarity: number[];
  jsDivergence: number[];
  ndcg: number[];
  precisionAt: Record<number, number[]>;
  recallAt: Record<number, number[]>;
  ndcgAt: Record<number, number[]>;
  top1Hit: number[];
  topkRecall: number[];
  peakMissRate: number[];
  catastrophicTopKMissRate: number[];
  visibleFrameRecall: number[];
  topkRecallPerFrame: number[];
  falsePositiveMass: number[];
  falseNegativeMass: number[];
  wastedUnseenBits: number[];
  wastedUnseenBitFraction: number[];
  wastedUnseenBitrateMbps: number[];
  visibilityWeightedWastedBits: number[];
  visibilityWeightedWasteFraction: number[];
  visibilityWeightedWasteBitrateMbps: number[];
  usefulVisibleBits: number[];
  usefulVisibleBitFraction: number[];
  usefulVisibleBitrateMbps: number[];
  nonVisibleToVisibleBitRatio: number[];
  predictionTargetedWasteFraction: number[];
  overPredictedBitWasteFraction: number[];
  highQualityInvisibleBitFraction: number[];
  wastedVisibleQuality: number[];
  missedVisibleQuality: number[];
  calibration: Array<{ predicted: number; actual: number; count?: number }>;
}

function clamp01(value: unknown): number {
  const n = Number(value ?? 0);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

function rankedTiles(values: Record<string, number>, allTiles: Set<string>): string[] {
  return Array.from(allTiles).sort((a, b) => clamp01(values[b]) - clamp01(values[a]) || a.localeCompare(b));
}

function topTiles(values: Record<string, number>, allTiles: Set<string>, k: number): Set<string> {
  return new Set(rankedTiles(values, allTiles).filter((tileId) => clamp01(values[tileId]) > 0).slice(0, Math.max(1, k)));
}

function safeMeanSquare(values: number[]): number {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value * value, 0) / values.length;
}

function normalDistribution(values: Record<string, number>, allTiles: Set<string>): Record<string, number> {
  const clipped: Record<string, number> = {};
  allTiles.forEach((tileId) => {
    clipped[tileId] = Math.max(0, Number(values[tileId] ?? 0));
  });
  const total = Object.values(clipped).reduce((sum, value) => sum + value, 0);
  if (total <= 1e-12) {
    const uniform = allTiles.size ? 1 / allTiles.size : 0;
    allTiles.forEach((tileId) => {
      clipped[tileId] = uniform;
    });
    return clipped;
  }
  allTiles.forEach((tileId) => {
    clipped[tileId] = (clipped[tileId] || 0) / total;
  });
  return clipped;
}

function jsDivergence(predicted: Record<string, number>, actual: Record<string, number>, allTiles: Set<string>): number {
  const p = normalDistribution(predicted, allTiles);
  const q = normalDistribution(actual, allTiles);
  const m: Record<string, number> = {};
  allTiles.forEach((tileId) => {
    m[tileId] = 0.5 * ((p[tileId] || 0) + (q[tileId] || 0));
  });
  const kl = (a: Record<string, number>, b: Record<string, number>) => {
    let total = 0;
    allTiles.forEach((tileId) => {
      const av = a[tileId] || 0;
      const bv = b[tileId] || 0;
      if (av > 0 && bv > 0) total += av * Math.log2(av / bv);
    });
    return total;
  };
  return 0.5 * kl(p, m) + 0.5 * kl(q, m);
}

function ndcg(predicted: Record<string, number>, actual: Record<string, number>, allTiles: Set<string>, k?: number): number {
  const ranked = rankedTiles(predicted, allTiles).slice(0, k ? Math.max(1, k) : undefined);
  const ideal = rankedTiles(actual, allTiles).slice(0, k ? Math.max(1, k) : undefined);
  const dcg = (order: string[]) => order.reduce((sum, tileId, idx) => sum + clamp01(actual[tileId]) / Math.log2(idx + 2), 0);
  const idealScore = dcg(ideal);
  return idealScore > 1e-12 ? dcg(ranked) / idealScore : 1;
}

export function fovPredictionStatsForRun(metrics: RunMetrics, runId: string, visibleThreshold = 0.01, kValues = [1, 3, 5, 10]): FovPredictionStats | null {
  const previouslyComputed = metrics.fov_prediction?.[runId] as FovPredictionStats | undefined;
  if (previouslyComputed?.mae && previouslyComputed.mae.length) return previouslyComputed;
  const segments = metrics.fov_segments?.[runId] || [];
  if (!segments.length) return null;

  const out: FovPredictionStats = {
    mae: [],
    rmse: [],
    playbackMae: [],
    playbackRmse: [],
    iou: [],
    weightedIou: [],
    distributionIntersection: [],
    histogramSimilarity: [],
    jsDivergence: [],
    ndcg: [],
    precisionAt: {},
    recallAt: {},
    ndcgAt: {},
    top1Hit: [],
    topkRecall: [],
    peakMissRate: [],
    catastrophicTopKMissRate: [],
    visibleFrameRecall: [],
    topkRecallPerFrame: [],
    falsePositiveMass: [],
    falseNegativeMass: [],
    wastedUnseenBits: [],
    wastedUnseenBitFraction: [],
    wastedUnseenBitrateMbps: [],
    visibilityWeightedWastedBits: [],
    visibilityWeightedWasteFraction: [],
    visibilityWeightedWasteBitrateMbps: [],
    usefulVisibleBits: [],
    usefulVisibleBitFraction: [],
    usefulVisibleBitrateMbps: [],
    nonVisibleToVisibleBitRatio: [],
    predictionTargetedWasteFraction: [],
    overPredictedBitWasteFraction: [],
    highQualityInvisibleBitFraction: [],
    wastedVisibleQuality: [],
    missedVisibleQuality: [],
    calibration: [],
  };

  for (const segment of segments) {
    const predicted = segment.predicted || {};
    const actual = segment.actual || {};
    const allTiles = new Set([...Object.keys(predicted), ...Object.keys(actual)]);
    if (!allTiles.size) continue;

    const errors = Array.from(allTiles).map((tileId) => Math.abs(clamp01(predicted[tileId]) - clamp01(actual[tileId])));
    out.mae.push(errors.reduce((sum, value) => sum + value, 0) / errors.length);
    out.rmse.push(Math.sqrt(safeMeanSquare(errors)));

    const actualFrames = Array.isArray(segment.actual_frames) ? segment.actual_frames : [];
    if (actualFrames.length) {
      const frameErrors: number[] = [];
      let visibleFramePairs = 0;
      let visibleFrameHits = 0;
      const peakVisibleTiles = new Set<string>();
      let topkFrameRecallSum = 0;
      let topkFrameRecallCount = 0;

      actualFrames.forEach((frame) => {
        const frameActual = frame?.tile_visibility || {};
        const frameTiles = new Set([...Array.from(allTiles), ...Object.keys(frameActual)]);
        const frameActualSet = new Set(Array.from(frameTiles).filter((tileId) => clamp01(frameActual[tileId]) >= visibleThreshold));
        frameTiles.forEach((tileId) => {
          const pred = clamp01(predicted[tileId]);
          const act = clamp01(frameActual[tileId]);
          frameErrors.push(Math.abs(pred - act));
          if (act >= visibleThreshold) {
            visibleFramePairs += 1;
            peakVisibleTiles.add(tileId);
            if (pred >= visibleThreshold) visibleFrameHits += 1;
          }
        });

        const frameK = Math.max(1, frameActualSet.size);
        const predTopK = topTiles(predicted, frameTiles, frameK);
        topkFrameRecallSum += frameActualSet.size
          ? Array.from(predTopK).filter((tileId) => frameActualSet.has(tileId)).length / frameActualSet.size
          : 1;
        topkFrameRecallCount += 1;
      });

      out.playbackMae.push(frameErrors.length ? frameErrors.reduce((sum, value) => sum + value, 0) / frameErrors.length : 0);
      out.playbackRmse.push(Math.sqrt(safeMeanSquare(frameErrors)));
      out.visibleFrameRecall.push(visibleFramePairs > 0 ? visibleFrameHits / visibleFramePairs : 1);
      const peakMisses = Array.from(peakVisibleTiles).filter((tileId) => clamp01(predicted[tileId]) < visibleThreshold).length;
      out.peakMissRate.push(peakVisibleTiles.size ? peakMisses / peakVisibleTiles.size : 0);
      out.topkRecallPerFrame.push(topkFrameRecallCount ? topkFrameRecallSum / topkFrameRecallCount : 1);
    }

    const predSet = new Set(Array.from(allTiles).filter((tileId) => clamp01(predicted[tileId]) >= visibleThreshold));
    const actualSet = new Set(Array.from(allTiles).filter((tileId) => clamp01(actual[tileId]) >= visibleThreshold));
    const union = new Set([...predSet, ...actualSet]);
    out.iou.push(union.size ? Array.from(predSet).filter((tileId) => actualSet.has(tileId)).length / union.size : 1);

    let intersectionMass = 0;
    let unionMass = 0;
    allTiles.forEach((tileId) => {
      const pred = clamp01(predicted[tileId]);
      const act = clamp01(actual[tileId]);
      intersectionMass += Math.min(pred, act);
      unionMass += Math.max(pred, act);
      out.calibration.push({ predicted: pred, actual: act });
    });
    out.weightedIou.push(unionMass > 0 ? intersectionMass / unionMass : 1);

    const predDist = normalDistribution(predicted, allTiles);
    const actualDist = normalDistribution(actual, allTiles);
    const distributionIntersection = Array.from(allTiles).reduce(
      (sum, tileId) => sum + Math.min(predDist[tileId] || 0, actualDist[tileId] || 0),
      0,
    );
    out.distributionIntersection.push(distributionIntersection);
    out.histogramSimilarity.push(distributionIntersection);
    out.jsDivergence.push(jsDivergence(predicted, actual, allTiles));
    out.ndcg.push(ndcg(predicted, actual, allTiles));

    const actualPositive = Array.from(actualSet);
    const k = Math.max(1, actualPositive.length);
    const predTop1 = topTiles(predicted, allTiles, 1);
    const actualTop1 = topTiles(actual, allTiles, 1);
    out.top1Hit.push(predTop1.size && actualTop1.size && Array.from(predTop1).some((tileId) => actualTop1.has(tileId)) ? 1 : 0);
    const predTopK = topTiles(predicted, allTiles, k);
    const actualTopK = topTiles(actual, allTiles, k);
    out.topkRecall.push(actualTopK.size ? Array.from(predTopK).filter((tileId) => actualTopK.has(tileId)).length / actualTopK.size : 1);
    const actualMass = Array.from(allTiles).reduce((sum, tileId) => sum + clamp01(actual[tileId]), 0);
    const topKActualMass = Array.from(predTopK).reduce((sum, tileId) => sum + clamp01(actual[tileId]), 0);
    out.catastrophicTopKMissRate.push(actualMass > 1e-12 && topKActualMass / actualMass < 0.10 ? 1 : 0);
    const actualPositiveSet = new Set(actualPositive.map(String));
    const predictedRanking = rankedTiles(predicted, allTiles);
    kValues.forEach((rawK) => {
      const resolvedK = Math.max(1, Math.floor(Number(rawK) || 1));
      const predAtK = new Set(predictedRanking.slice(0, resolvedK));
      if (!out.precisionAt[resolvedK]) out.precisionAt[resolvedK] = [];
      if (!out.recallAt[resolvedK]) out.recallAt[resolvedK] = [];
      if (!out.ndcgAt[resolvedK]) out.ndcgAt[resolvedK] = [];
      out.precisionAt[resolvedK]!.push(predAtK.size ? Array.from(predAtK).filter((tileId) => actualPositiveSet.has(tileId)).length / predAtK.size : 1);
      out.recallAt[resolvedK]!.push(actualPositiveSet.size ? Array.from(predAtK).filter((tileId) => actualPositiveSet.has(tileId)).length / actualPositiveSet.size : 1);
      out.ndcgAt[resolvedK]!.push(ndcg(predicted, actual, allTiles, resolvedK));
    });

    out.falsePositiveMass.push(Array.from(predSet).filter((tileId) => !actualSet.has(tileId)).reduce((sum, tileId) => sum + clamp01(predicted[tileId]), 0));
    out.falseNegativeMass.push(Array.from(actualSet).filter((tileId) => !predSet.has(tileId)).reduce((sum, tileId) => sum + clamp01(actual[tileId]), 0));

    const wastedBits = Array.from(allTiles)
      .filter((tileId) => clamp01(actual[tileId]) < visibleThreshold)
      .reduce((sum, tileId) => sum + Math.max(0, Number(segment.tile_bits?.[tileId] ?? 0)), 0);
    const totalBits = Array.from(allTiles).reduce((sum, tileId) => sum + Math.max(0, Number(segment.tile_bits?.[tileId] ?? 0)), 0);
    const duration = Math.max(0, Number(segment.duration_s || 0));
    out.wastedUnseenBits.push(wastedBits);
    out.wastedUnseenBitFraction.push(totalBits > 0 ? wastedBits / totalBits : 0);
    out.wastedUnseenBitrateMbps.push(duration > 0 ? (wastedBits / duration) / 1e6 : 0);

    let weightedWastedBits = 0;
    let usefulVisibleBits = 0;
    let predictionTargetedBits = 0;
    let predictionTargetedWasteBits = 0;
    let overPredictedWasteBits = 0;
    let highQualityInvisibleBits = 0;
    let wastedQuality = 0;
    let missedQuality = 0;
    allTiles.forEach((tileId) => {
      const pred = clamp01(predicted[tileId]);
      const act = clamp01(actual[tileId]);
      const quality = clamp01(segment.tile_quality?.[tileId]);
      const bits = Math.max(0, Number(segment.tile_bits?.[tileId] ?? 0));
      const invisible = 1 - act;
      weightedWastedBits += bits * invisible;
      usefulVisibleBits += bits * act;
      predictionTargetedBits += bits * pred;
      predictionTargetedWasteBits += bits * pred * invisible;
      overPredictedWasteBits += bits * Math.max(0, pred - act);
      highQualityInvisibleBits += bits * quality * invisible;
      if (pred >= visibleThreshold && act < visibleThreshold) wastedQuality += quality * pred;
      if (act >= visibleThreshold && pred < visibleThreshold) missedQuality += (1 - quality) * act;
    });
    out.visibilityWeightedWastedBits.push(weightedWastedBits);
    out.visibilityWeightedWasteFraction.push(totalBits > 0 ? weightedWastedBits / totalBits : 0);
    out.visibilityWeightedWasteBitrateMbps.push(duration > 0 ? (weightedWastedBits / duration) / 1e6 : 0);
    out.usefulVisibleBits.push(usefulVisibleBits);
    out.usefulVisibleBitFraction.push(totalBits > 0 ? usefulVisibleBits / totalBits : 0);
    out.usefulVisibleBitrateMbps.push(duration > 0 ? (usefulVisibleBits / duration) / 1e6 : 0);
    out.nonVisibleToVisibleBitRatio.push(usefulVisibleBits > 0 ? weightedWastedBits / usefulVisibleBits : (weightedWastedBits > 0 ? Number.POSITIVE_INFINITY : 0));
    out.predictionTargetedWasteFraction.push(predictionTargetedBits > 0 ? predictionTargetedWasteBits / predictionTargetedBits : 0);
    out.overPredictedBitWasteFraction.push(totalBits > 0 ? overPredictedWasteBits / totalBits : 0);
    out.highQualityInvisibleBitFraction.push(totalBits > 0 ? highQualityInvisibleBits / totalBits : 0);
    out.wastedVisibleQuality.push(wastedQuality);
    out.missedVisibleQuality.push(missedQuality);
  }

  if (!out.mae.length && !out.calibration.length) return null;
  if (!metrics.fov_prediction) metrics.fov_prediction = {};
  metrics.fov_prediction[runId] = out;
  return out;
}
