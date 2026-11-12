from __future__ import annotations

import math
from typing import Any, Dict, Iterable


VISIBLE_THRESHOLD = 0.01
TOP_K_VALUES = (1, 3, 5, 10)
CALIBRATION_BINS = 20


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric)) if math.isfinite(numeric) else 0.0


def _ranked(values: Dict[str, Any], tiles: Iterable[str]) -> list[str]:
    return sorted(tiles, key=lambda tile_id: (-_clamp01(values.get(tile_id)), tile_id))


def _top(values: Dict[str, Any], tiles: set[str], count: int) -> set[str]:
    positive = [tile_id for tile_id in _ranked(values, tiles) if _clamp01(values.get(tile_id)) > 0]
    return set(positive[: max(1, count)])


def _distribution(values: Dict[str, Any], tiles: set[str]) -> Dict[str, float]:
    clipped = {tile_id: max(0.0, float(values.get(tile_id, 0.0) or 0.0)) for tile_id in tiles}
    total = sum(clipped.values())
    if total <= 1e-12:
        uniform = 1.0 / len(tiles) if tiles else 0.0
        return {tile_id: uniform for tile_id in tiles}
    return {tile_id: value / total for tile_id, value in clipped.items()}


def _js_divergence(predicted: Dict[str, Any], actual: Dict[str, Any], tiles: set[str]) -> float:
    p = _distribution(predicted, tiles)
    q = _distribution(actual, tiles)
    middle = {tile_id: 0.5 * (p[tile_id] + q[tile_id]) for tile_id in tiles}

    def kl(left: Dict[str, float], right: Dict[str, float]) -> float:
        return sum(
            value * math.log2(value / right[tile_id])
            for tile_id, value in left.items()
            if value > 0 and right[tile_id] > 0
        )

    return 0.5 * kl(p, middle) + 0.5 * kl(q, middle)


def _ndcg(predicted: Dict[str, Any], actual: Dict[str, Any], tiles: set[str], count: int | None = None) -> float:
    ranked = _ranked(predicted, tiles)
    ideal = _ranked(actual, tiles)
    if count is not None:
        ranked = ranked[: max(1, count)]
        ideal = ideal[: max(1, count)]

    def dcg(order: list[str]) -> float:
        return sum(_clamp01(actual.get(tile_id)) / math.log2(index + 2) for index, tile_id in enumerate(order))

    ideal_score = dcg(ideal)
    return dcg(ranked) / ideal_score if ideal_score > 1e-12 else 1.0


def _empty_stats() -> Dict[str, Any]:
    array_names = (
        "mae", "rmse", "playbackMae", "playbackRmse", "iou", "weightedIou",
        "distributionIntersection", "histogramSimilarity", "jsDivergence", "ndcg",
        "top1Hit", "topkRecall", "peakMissRate", "catastrophicTopKMissRate", "visibleFrameRecall", "topkRecallPerFrame",
        "falsePositiveMass", "falseNegativeMass", "wastedUnseenBits", "wastedUnseenBitFraction",
        "wastedUnseenBitrateMbps", "visibilityWeightedWastedBits", "visibilityWeightedWasteFraction",
        "visibilityWeightedWasteBitrateMbps", "usefulVisibleBits", "usefulVisibleBitFraction",
        "usefulVisibleBitrateMbps", "nonVisibleToVisibleBitRatio", "predictionTargetedWasteFraction",
        "overPredictedBitWasteFraction", "highQualityInvisibleBitFraction", "wastedVisibleQuality",
        "missedVisibleQuality",
    )
    out: Dict[str, Any] = {name: [] for name in array_names}
    out.update({"precisionAt": {}, "recallAt": {}, "ndcgAt": {}, "calibration": []})
    return out


def summarize_fov_segments(segments: list[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Reduce raw per-tile/per-frame maps to the sufficient statistics used by compare plugins."""
    out = _empty_stats()
    calibration = [{"sum": 0.0, "count": 0} for _ in range(CALIBRATION_BINS)]

    for segment in segments:
        predicted = segment.get("predicted") or {}
        actual = segment.get("actual") or {}
        tiles = set(map(str, predicted)) | set(map(str, actual))
        if not tiles:
            continue

        errors = [abs(_clamp01(predicted.get(tile_id)) - _clamp01(actual.get(tile_id))) for tile_id in tiles]
        out["mae"].append(sum(errors) / len(errors))
        out["rmse"].append(math.sqrt(sum(value * value for value in errors) / len(errors)))

        actual_frames = segment.get("actual_frames") or []
        if actual_frames:
            frame_errors: list[float] = []
            visible_pairs = visible_hits = 0
            peak_visible: set[str] = set()
            frame_recalls: list[float] = []
            for frame in actual_frames:
                frame_actual = frame.get("tile_visibility") or {}
                frame_tiles = tiles | set(map(str, frame_actual))
                visible = {tile_id for tile_id in frame_tiles if _clamp01(frame_actual.get(tile_id)) >= VISIBLE_THRESHOLD}
                for tile_id in frame_tiles:
                    pred = _clamp01(predicted.get(tile_id))
                    act = _clamp01(frame_actual.get(tile_id))
                    frame_errors.append(abs(pred - act))
                    if act >= VISIBLE_THRESHOLD:
                        visible_pairs += 1
                        peak_visible.add(tile_id)
                        visible_hits += int(pred >= VISIBLE_THRESHOLD)
                predicted_top = _top(predicted, frame_tiles, max(1, len(visible)))
                frame_recalls.append(len(predicted_top & visible) / len(visible) if visible else 1.0)
            out["playbackMae"].append(sum(frame_errors) / len(frame_errors) if frame_errors else 0.0)
            out["playbackRmse"].append(math.sqrt(sum(v * v for v in frame_errors) / len(frame_errors)) if frame_errors else 0.0)
            out["visibleFrameRecall"].append(visible_hits / visible_pairs if visible_pairs else 1.0)
            misses = sum(_clamp01(predicted.get(tile_id)) < VISIBLE_THRESHOLD for tile_id in peak_visible)
            out["peakMissRate"].append(misses / len(peak_visible) if peak_visible else 0.0)
            out["topkRecallPerFrame"].append(sum(frame_recalls) / len(frame_recalls) if frame_recalls else 1.0)

        pred_set = {tile_id for tile_id in tiles if _clamp01(predicted.get(tile_id)) >= VISIBLE_THRESHOLD}
        actual_set = {tile_id for tile_id in tiles if _clamp01(actual.get(tile_id)) >= VISIBLE_THRESHOLD}
        union = pred_set | actual_set
        out["iou"].append(len(pred_set & actual_set) / len(union) if union else 1.0)

        intersection_mass = union_mass = 0.0
        for tile_id in tiles:
            pred = _clamp01(predicted.get(tile_id))
            act = _clamp01(actual.get(tile_id))
            intersection_mass += min(pred, act)
            union_mass += max(pred, act)
            index = min(CALIBRATION_BINS - 1, int(pred * CALIBRATION_BINS))
            calibration[index]["sum"] += act
            calibration[index]["count"] += 1
        out["weightedIou"].append(intersection_mass / union_mass if union_mass else 1.0)

        pred_dist = _distribution(predicted, tiles)
        actual_dist = _distribution(actual, tiles)
        intersection = sum(min(pred_dist[tile_id], actual_dist[tile_id]) for tile_id in tiles)
        out["distributionIntersection"].append(intersection)
        out["histogramSimilarity"].append(intersection)
        out["jsDivergence"].append(_js_divergence(predicted, actual, tiles))
        out["ndcg"].append(_ndcg(predicted, actual, tiles))

        visible_count = max(1, len(actual_set))
        pred_top1, actual_top1 = _top(predicted, tiles, 1), _top(actual, tiles, 1)
        out["top1Hit"].append(1.0 if pred_top1 and actual_top1 and pred_top1 & actual_top1 else 0.0)
        pred_topk, actual_topk = _top(predicted, tiles, visible_count), _top(actual, tiles, visible_count)
        out["topkRecall"].append(len(pred_topk & actual_topk) / len(actual_topk) if actual_topk else 1.0)
        actual_mass = sum(_clamp01(actual.get(tile_id)) for tile_id in tiles)
        topk_actual_mass = sum(_clamp01(actual.get(tile_id)) for tile_id in pred_topk)
        out["catastrophicTopKMissRate"].append(1.0 if actual_mass > 1e-12 and topk_actual_mass / actual_mass < 0.10 else 0.0)
        ranking = _ranked(predicted, tiles)
        for count in TOP_K_VALUES:
            at_k = set(ranking[:count])
            out["precisionAt"].setdefault(str(count), []).append(len(at_k & actual_set) / len(at_k) if at_k else 1.0)
            out["recallAt"].setdefault(str(count), []).append(len(at_k & actual_set) / len(actual_set) if actual_set else 1.0)
            out["ndcgAt"].setdefault(str(count), []).append(_ndcg(predicted, actual, tiles, count))

        out["falsePositiveMass"].append(sum(_clamp01(predicted.get(t)) for t in pred_set - actual_set))
        out["falseNegativeMass"].append(sum(_clamp01(actual.get(t)) for t in actual_set - pred_set))

        tile_bits = segment.get("tile_bits") or {}
        tile_quality = segment.get("tile_quality") or {}
        total_bits = sum(max(0.0, float(tile_bits.get(t, 0.0) or 0.0)) for t in tiles)
        duration = max(0.0, float(segment.get("duration_s", 0.0) or 0.0))
        unseen_bits = sum(max(0.0, float(tile_bits.get(t, 0.0) or 0.0)) for t in tiles if _clamp01(actual.get(t)) < VISIBLE_THRESHOLD)
        weighted_waste = useful_bits = targeted_bits = targeted_waste = over_waste = high_quality_invisible = 0.0
        wasted_quality = missed_quality = 0.0
        for tile_id in tiles:
            pred, act = _clamp01(predicted.get(tile_id)), _clamp01(actual.get(tile_id))
            quality = _clamp01(tile_quality.get(tile_id))
            bits = max(0.0, float(tile_bits.get(tile_id, 0.0) or 0.0))
            invisible = 1.0 - act
            weighted_waste += bits * invisible
            useful_bits += bits * act
            targeted_bits += bits * pred
            targeted_waste += bits * pred * invisible
            over_waste += bits * max(0.0, pred - act)
            high_quality_invisible += bits * quality * invisible
            if pred >= VISIBLE_THRESHOLD and act < VISIBLE_THRESHOLD:
                wasted_quality += quality * pred
            if act >= VISIBLE_THRESHOLD and pred < VISIBLE_THRESHOLD:
                missed_quality += (1.0 - quality) * act
        append = out.__getitem__
        append("wastedUnseenBits").append(unseen_bits)
        append("wastedUnseenBitFraction").append(unseen_bits / total_bits if total_bits else 0.0)
        append("wastedUnseenBitrateMbps").append(unseen_bits / duration / 1e6 if duration else 0.0)
        append("visibilityWeightedWastedBits").append(weighted_waste)
        append("visibilityWeightedWasteFraction").append(weighted_waste / total_bits if total_bits else 0.0)
        append("visibilityWeightedWasteBitrateMbps").append(weighted_waste / duration / 1e6 if duration else 0.0)
        append("usefulVisibleBits").append(useful_bits)
        append("usefulVisibleBitFraction").append(useful_bits / total_bits if total_bits else 0.0)
        append("usefulVisibleBitrateMbps").append(useful_bits / duration / 1e6 if duration else 0.0)
        append("nonVisibleToVisibleBitRatio").append(weighted_waste / useful_bits if useful_bits else (1e12 if weighted_waste else 0.0))
        append("predictionTargetedWasteFraction").append(targeted_waste / targeted_bits if targeted_bits else 0.0)
        append("overPredictedBitWasteFraction").append(over_waste / total_bits if total_bits else 0.0)
        append("highQualityInvisibleBitFraction").append(high_quality_invisible / total_bits if total_bits else 0.0)
        append("wastedVisibleQuality").append(wasted_quality)
        append("missedVisibleQuality").append(missed_quality)

    out["calibration"] = [
        {
            "predicted": (index + 0.5) / CALIBRATION_BINS,
            "actual": bucket["sum"] / bucket["count"],
            "count": bucket["count"],
        }
        for index, bucket in enumerate(calibration)
        if bucket["count"]
    ]
    return out if out["mae"] or out["calibration"] else None
