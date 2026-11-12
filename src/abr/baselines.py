import math
import random
from typing import Dict

from abr.base import ABR


DEFAULT_COMPONENT_WEIGHTS = {"geom": 0.35, "attr": 0.35, "occp": 0.2, "atlas": 0.0}

def _segment_duration(tile_groups) -> float:
    if not tile_groups:
        return 1.0
    first_tile = next(iter(tile_groups.values()))
    if not first_tile.component_options:
        return 1.0
    first_component = next(iter(first_tile.component_options.values()))
    return first_component[0].duration_s if first_component else 1.0


def _decision_bandwidth(network, clock, segment_duration_s: float) -> float:
    if hasattr(network, "predict_bandwidth_for_decision"):
        return network.predict_bandwidth_for_decision(clock.get_time(), segment_duration_s=segment_duration_s)
    return network.get_instantaneous_bandwidth(clock.get_time())


class RandomABR(ABR):
    def decide(self, tiles, segment_frames_path, network, fov, buffer, clock):
        decision = {}
        tile_groups = self.group_tiles(tiles)
        for tile_id, tile in tile_groups.items():
            decision[tile_id] = {
                component: random.randrange(len(reps))
                for component, reps in tile.component_options.items()
            }
        return decision

class MinQualityABR(ABR):
    def decide(self, tiles, segment_frames_path, network, fov, buffer, clock):
        decisions = {}
        for tile_id, tile in self.group_tiles(tiles).items():
            decisions[tile_id] = {
                comp: self.lowest_rep_idx(tile, comp) for comp in tile.component_options
            }
        return decisions

class MaxQualityABR(ABR):
    def decide(self, tiles, segment_frames_path, network, fov, buffer, clock):
        decisions = {}
        for tile_id, tile in self.group_tiles(tiles).items():
            decisions[tile_id] = {
                comp: self.highest_rep_idx(tile, comp) for comp in tile.component_options
            }
        return decisions

class BandwidthAwareABR(ABR):
    def __init__(self, safety_factor=0.8):
        self.safety_factor = safety_factor

    def decide(self, tiles, segment_frames_path, network, fov, buffer, clock):
        tile_groups = self.group_tiles(tiles)
        bw_available = _decision_bandwidth(network, clock, _segment_duration(tile_groups)) * self.safety_factor
        bw_per_tile = bw_available / len(tile_groups) if tile_groups else 0

        decision = {}
        for tile_id, tile in tile_groups.items():
            num_components = len(tile.component_options)
            bw_per_component = bw_per_tile / max(1, num_components)
            tile_decision = {}
            for comp, reps in tile.component_options.items():
                reps_sorted = sorted(reps, key=lambda t: t.rep_idx)
                chosen = reps_sorted[0]
                for rep in reps_sorted:
                    if rep.size_bits / max(rep.duration_s, 1e-9) <= bw_per_component:
                        chosen = rep
                tile_decision[comp] = chosen.rep_idx
            decision[tile_id] = tile_decision
        return decision

class FoVRoundRobinABR(ABR):
    abr_requires_predictor = True
    abr_supports_predictor = True
    def __init__(self, safety_factor=0.8):
        self.safety_factor = safety_factor    

    @staticmethod
    def _rep_norm(rank: int, count: int) -> float:
        if count <= 1:
            return 1.0
        return max(0.0, min(1.0, rank / float(count - 1)))

    def _upgrade_lanes_by_weighted_utility(self, decision, lanes, remaining_bits):
        while remaining_bits > 0 and lanes:
            best_lane = None
            best_extra_bits = 0.0
            best_utility = 0.0
            for lane in lanes:
                reps_sorted = lane["reps_sorted"]
                current_idx = int(lane["current_idx"])
                next_idx = current_idx + 1
                if next_idx >= len(reps_sorted):
                    continue

                current_rep = reps_sorted[current_idx]
                next_rep = reps_sorted[next_idx]
                extra_bits = float(next_rep.size_bits - current_rep.size_bits)
                if extra_bits < 0 or extra_bits > remaining_bits:
                    continue

                quality_gain = self._rep_norm(next_idx, len(reps_sorted)) - self._rep_norm(current_idx, len(reps_sorted))
                visibility = max(float(lane.get("score", 0.0) or 0.0), 0.0)
                if visibility <= 0.0:
                    visibility = float(lane.get("fallback_score", 0.0) or 0.0)
                if extra_bits <= 0:
                    utility = float("inf") if quality_gain > 0 else 0.0
                else:
                    utility = visibility * quality_gain / extra_bits
                if utility > best_utility:
                    best_lane = lane
                    best_extra_bits = max(extra_bits, 0.0)
                    best_utility = utility

            if best_lane is None:
                break

            best_lane["current_idx"] = int(best_lane["current_idx"]) + 1
            next_rep = best_lane["reps_sorted"][best_lane["current_idx"]]
            decision[best_lane["tile_id"]][best_lane["component"]] = next_rep.rep_idx
            remaining_bits -= best_extra_bits
        return remaining_bits

    def decide(self, tiles, segment_frames_path, network, fov, buffer, clock):
        tile_groups = self.group_tiles(tiles)
        proportions = self.predict_fov(fov, tile_groups, segment_frames_path)
        segment_duration = _segment_duration(tile_groups)
        bw_bps = _decision_bandwidth(network, clock, segment_duration) * self.safety_factor
        budget_bits = bw_bps * segment_duration

        decision = {}
        baseline_cost_bits = 0
        for tile_id, tile in tile_groups.items():
            tile_decision = {}
            for comp, reps in tile.component_options.items():
                reps_sorted = sorted(reps, key=lambda t: t.size_bits)
                lowest = reps_sorted[0]
                tile_decision[comp] = lowest.rep_idx
                baseline_cost_bits += lowest.size_bits
            decision[tile_id] = tile_decision

        remaining_bits = max(budget_bits - baseline_cost_bits, 0)

        lanes = []
        for tile_id, tile in tile_groups.items():
            score = float(proportions.get(tile_id, 0) or 0)
            for comp, reps in tile.component_options.items():
                sorted_reps = sorted(reps, key=lambda t: t.size_bits)
                lane = {
                    "tile_id": tile_id,
                    "component": comp,
                    "reps_sorted": sorted_reps,
                    "current_idx": 0,
                    "score": score,
                }
                lanes.append(lane)

        has_positive_fov = any(float(lane.get("score", 0.0) or 0.0) > 0.0 for lane in lanes)
        if lanes and not has_positive_fov:
            for lane in lanes:
                lane["fallback_score"] = 1.0

        remaining_bits = self._upgrade_lanes_by_weighted_utility(decision, lanes, remaining_bits)
        if remaining_bits > 0 and has_positive_fov:
            for lane in lanes:
                lane["score"] = 0.0
                lane["fallback_score"] = 1.0
            remaining_bits = self._upgrade_lanes_by_weighted_utility(decision, lanes, remaining_bits)

        return decision


class ScoreThresholdABR(ABR):
    """Directly map each predicted tile score to an equally spaced quality band.

    This intentionally ignores bandwidth, buffer state, and the relative scores
    of other tiles.  It is an evaluation control rather than a deployable ABR:
    a predictor score of ``s`` selects band ``floor(s * N)`` among ``N``
    representations (clamped so that a score of 1 selects the highest band).
    Consequently, every component of a tile receives the representation band
    implied by that tile's own score, with no fallback or budget redistribution.
    """

    abr_requires_predictor = True
    abr_supports_predictor = True

    @staticmethod
    def _representation_for_score(reps, score: float) -> int:
        """Return the size-ranked representation selected by an equal score band."""
        reps_sorted = sorted(reps, key=lambda rep: rep.size_bits)
        if not reps_sorted:
            return 0
        clipped_score = max(0.0, min(1.0, float(score or 0.0)))
        band = min(len(reps_sorted) - 1, int(clipped_score * len(reps_sorted)))
        return reps_sorted[band].rep_idx

    def decide(self, tiles, segment_frames_path, network, fov, buffer, clock):
        tile_groups = self.group_tiles(tiles)
        predicted_scores = self.predict_fov(fov, tile_groups, segment_frames_path) or {}
        decision = {}
        for tile_id, tile in tile_groups.items():
            score = predicted_scores.get(tile_id, 0.0)
            decision[tile_id] = {
                component: self._representation_for_score(reps, score)
                for component, reps in tile.component_options.items()
            }
        return decision


class ScoreThresholdBudgetABR(ScoreThresholdABR):
    """Score-threshold evaluation ABR with a strict bandwidth cap.

    The controller first applies :class:`ScoreThresholdABR`'s independent,
    equally spaced score bands.  If that direct mapping exceeds the predicted
    segment budget, it steps down the lowest-score tiles first.  It never
    upgrades a tile beyond its score-selected band and never uses a fallback
    allocation, so the score ordering remains directly inspectable.
    """

    def __init__(self, safety_factor: float = 0.7):
        self.safety_factor = max(0.0, float(safety_factor))

    @staticmethod
    def _sorted_representations(reps):
        return sorted(reps, key=lambda rep: rep.size_bits)

    @classmethod
    def _selected_band(cls, reps, rep_idx: int) -> int:
        """Find a representation's size-ranked band, defaulting safely to zero."""
        for band, rep in enumerate(cls._sorted_representations(reps)):
            if rep.rep_idx == rep_idx:
                return band
        return 0

    @classmethod
    def _decision_size_bits(cls, tile_groups, bands) -> float:
        total = 0.0
        for tile_id, tile in tile_groups.items():
            for component, reps in tile.component_options.items():
                ordered = cls._sorted_representations(reps)
                if not ordered:
                    continue
                band = max(0, min(int(bands[tile_id][component]), len(ordered) - 1))
                total += float(ordered[band].size_bits)
        return total

    def decide(self, tiles, segment_frames_path, network, fov, buffer, clock):
        tile_groups = self.group_tiles(tiles)
        predicted_scores = self.predict_fov(fov, tile_groups, segment_frames_path) or {}
        bands = {}
        for tile_id, tile in tile_groups.items():
            score = predicted_scores.get(tile_id, 0.0)
            bands[tile_id] = {
                component: self._selected_band(reps, self._representation_for_score(reps, score))
                for component, reps in tile.component_options.items()
            }

        segment_duration = _segment_duration(tile_groups)
        budget_bits = max(0.0, _decision_bandwidth(network, clock, segment_duration) * self.safety_factor * segment_duration)

        # At minimum quality the segment may still exceed the network budget.
        # The defined behavior is to return the all-minimum decision.
        minimum_bands = {
            tile_id: {component: 0 for component in tile.component_options}
            for tile_id, tile in tile_groups.items()
        }
        if self._decision_size_bits(tile_groups, minimum_bands) > budget_bits:
            bands = minimum_bands
        else:
            # Fully exhaust one lower-score tile before touching a higher-score
            # tile.  Within that tile, choose the one-step downgrade with the
            # largest immediate bit saving; ties are deterministic.
            order = sorted(tile_groups, key=lambda tile_id: (float(predicted_scores.get(tile_id, 0.0) or 0.0), str(tile_id)))
            current_bits = self._decision_size_bits(tile_groups, bands)
            for tile_id in order:
                tile = tile_groups[tile_id]
                while current_bits > budget_bits:
                    candidates = []
                    for component, reps in tile.component_options.items():
                        ordered = self._sorted_representations(reps)
                        band = int(bands[tile_id][component])
                        if band <= 0 or band >= len(ordered):
                            continue
                        saving = float(ordered[band].size_bits - ordered[band - 1].size_bits)
                        candidates.append((saving, str(component)))
                    if not candidates:
                        break
                    saving, component = max(candidates, key=lambda item: (item[0], item[1]))
                    bands[tile_id][component] -= 1
                    current_bits -= max(0.0, saving)
                if current_bits <= budget_bits:
                    break

        return {
            tile_id: {
                component: self._sorted_representations(reps)[int(bands[tile_id][component])].rep_idx
                for component, reps in tile.component_options.items()
                if reps
            }
            for tile_id, tile in tile_groups.items()
        }


class RankBudgetABR(ScoreThresholdBudgetABR):
    """Allocate a bandwidth budget from predictor ordering rather than score scale.

    This is an evaluation ABR for comparing tile-importance predictors whose
    numeric score scales need not be calibrated alike.  Positive scores are
    converted into tie-preserving percentile ranks; a strictly monotonic score
    transform therefore produces the same allocation.  Every tile, including
    zero-score tiles, belongs to a tie-preserving rank tier; zero scores form
    the lowest tier when positive scores exist.

    The controller starts every tile at minimum quality and spends the feasible
    segment budget through incremental representation upgrades.  The predictor
    determines only their order: higher tiers are exhausted before lower tiers,
    while equal-score tiles are upgraded in balanced passes.  The zero-score
    fallback prevents a sparse predictor from forfeiting usable budget merely
    because it emits fewer positive scores than a broad predictor.
    """

    @staticmethod
    def _finite_nonnegative_score(value) -> float:
        try:
            score = float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return score if math.isfinite(score) and score > 0.0 else 0.0

    @staticmethod
    def _rep_norm(rank: int, count: int) -> float:
        if count <= 1:
            return 1.0
        return max(0.0, min(1.0, rank / float(count - 1)))

    @classmethod
    def _rank_priorities(cls, tile_groups, predicted_scores) -> tuple[Dict[str, float], list[list[str]]]:
        """Return tie-preserving rank percentiles and high-to-low score tiers."""
        scores = {
            str(tile_id): cls._finite_nonnegative_score((predicted_scores or {}).get(tile_id, 0.0))
            for tile_id in tile_groups
        }
        ranked_ids = list(tile_groups)
        if not ranked_ids:
            return {}, []

        # Score groups, rather than individual tiles, preserve predictor ties.
        buckets: Dict[float, list[str]] = {}
        for tile_id in ranked_ids:
            buckets.setdefault(scores[tile_id], []).append(tile_id)
        tiers = [sorted(buckets[score], key=str) for score in sorted(buckets, reverse=True)]

        total = len(ranked_ids)
        priorities: Dict[str, float] = {str(tile_id): 0.0 for tile_id in tile_groups}
        next_rank = 1
        for tier in tiers:
            # Mid-rank percentile: highest singleton is near 1, lowest near 0.
            mid_rank = next_rank + ((len(tier) - 1) / 2.0)
            priority = max(0.0, min(1.0, (total - mid_rank + 0.5) / float(total)))
            for tile_id in tier:
                priorities[tile_id] = priority
            next_rank += len(tier)
        return priorities, tiers

    @classmethod
    def _fill_remaining_budget(cls, tile_groups, bands, tiers, current_bits: float, budget_bits: float) -> None:
        """Spend feasible budget in strict rank order and balanced tie passes."""
        for tier in tiers:
            # Exhaust a higher tier before considering a lower one.  Within a
            # tied tier, one upgrade per tile per pass avoids tile-ID bias.
            while True:
                upgraded = False
                for tile_id in tier:
                    tile = tile_groups[tile_id]
                    candidates = []
                    for component, reps in tile.component_options.items():
                        ordered = cls._sorted_representations(reps)
                        band = int(bands[tile_id][component])
                        if band < 0 or band + 1 >= len(ordered):
                            continue
                        extra_bits = float(ordered[band + 1].size_bits - ordered[band].size_bits)
                        if extra_bits < 0.0 or current_bits + extra_bits > budget_bits:
                            continue
                        # Prefer the least-developed component, then the least costly step.
                        quality_level = (band / float(max(1, len(ordered) - 1)))
                        candidates.append((quality_level, extra_bits, str(component)))
                    if not candidates:
                        continue
                    _level, extra_bits, component = min(candidates, key=lambda item: (item[0], item[1], item[2]))
                    bands[tile_id][component] += 1
                    current_bits += max(0.0, extra_bits)
                    upgraded = True
                if not upgraded:
                    break

    def decide(self, tiles, segment_frames_path, network, fov, buffer, clock):
        tile_groups = self.group_tiles(tiles)
        predicted_scores = self.predict_fov(fov, tile_groups, segment_frames_path) or {}
        _priorities, tiers = self._rank_priorities(tile_groups, predicted_scores)

        segment_duration = _segment_duration(tile_groups)
        budget_bits = max(
            0.0,
            _decision_bandwidth(network, clock, segment_duration) * self.safety_factor * segment_duration,
        )
        minimum_bands = {
            str(tile_id): {component: 0 for component in tile.component_options}
            for tile_id, tile in tile_groups.items()
        }
        bands = minimum_bands
        current_bits = self._decision_size_bits(tile_groups, bands)
        if current_bits <= budget_bits:
            self._fill_remaining_budget(tile_groups, bands, tiers, current_bits, budget_bits)

        return {
            str(tile_id): {
                component: self._sorted_representations(reps)[int(bands[str(tile_id)][component])].rep_idx
                for component, reps in tile.component_options.items()
                if reps
            }
            for tile_id, tile in tile_groups.items()
        }


class CoverageAwareUtilityABR(ScoreThresholdBudgetABR):
    """Score-aware allocation with diminishing returns across predicted FoV tiles.

    Unlike :class:`RankBudgetABR`, this controller retains score magnitude.
    It normalizes the positive score map within a decision, then greedily
    selects the affordable representation step with the greatest predicted
    visibility-weighted quality gain per bit.  A tile-level diminishing-return
    factor encourages first useful upgrades across the predicted FoV before
    exhausting a small number of top-ranked tiles.

    This is deliberately an exploratory ABR, intended to test whether a
    predictor with a spatially structured score map benefits from score-aware
    allocation.  It must be evaluated on held-out traces before paper use.
    """

    abr_requires_predictor = True
    abr_supports_predictor = True

    def __init__(
        self,
        safety_factor: float = 1.0,
        score_power: float = 0.75,
        coverage_strength: float = 2.5,
        fallback_score: float = 0.025,
    ):
        self.safety_factor = max(0.0, float(safety_factor))
        self.score_power = max(0.05, float(score_power))
        self.coverage_strength = max(0.0, float(coverage_strength))
        # A sparse predictor must not strand usable bandwidth after its
        # positive support is fully upgraded.  This score is deliberately
        # far below every positive normalized score, so it is only a fallback.
        self.fallback_score = max(0.0, float(fallback_score))

    @staticmethod
    def _finite_nonnegative_score(value) -> float:
        try:
            score = float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return score if math.isfinite(score) and score > 0.0 else 0.0

    @staticmethod
    def _rep_norm(rank: int, count: int) -> float:
        if count <= 1:
            return 1.0
        return max(0.0, min(1.0, rank / float(count - 1)))

    def _normalized_scores(self, tile_groups, predicted_scores) -> Dict[str, float]:
        raw = {
            str(tile_id): self._finite_nonnegative_score((predicted_scores or {}).get(tile_id, 0.0))
            for tile_id in tile_groups
        }
        ceiling = max(raw.values(), default=0.0)
        if ceiling <= 0.0:
            return {tile_id: 1.0 for tile_id in raw}
        return {
            tile_id: (score / ceiling) ** self.score_power if score > 0.0 else 0.0
            for tile_id, score in raw.items()
        }

    @staticmethod
    def _tile_progress(tile, bands) -> float:
        progress = []
        for component, reps in tile.component_options.items():
            count = len(reps)
            if count <= 1:
                continue
            progress.append(float(bands[component]) / float(count - 1))
        return sum(progress) / len(progress) if progress else 0.0

    @staticmethod
    def _component_weight(component: str) -> float:
        """Use the same component saliency as the playback-quality metric."""
        return max(0.0, float(DEFAULT_COMPONENT_WEIGHTS.get(component, 0.1)))

    def decide(self, tiles, segment_frames_path, network, fov, buffer, clock):
        tile_groups = self.group_tiles(tiles)
        predicted_scores = self.predict_fov(fov, tile_groups, segment_frames_path) or {}
        scores = self._normalized_scores(tile_groups, predicted_scores)
        segment_duration = _segment_duration(tile_groups)
        budget_bits = max(
            0.0,
            _decision_bandwidth(network, clock, segment_duration) * self.safety_factor * segment_duration,
        )
        bands = {
            str(tile_id): {component: 0 for component in tile.component_options}
            for tile_id, tile in tile_groups.items()
        }
        current_bits = self._decision_size_bits(tile_groups, bands)
        if current_bits > budget_bits:
            return {
                str(tile_id): {
                    component: self._sorted_representations(reps)[0].rep_idx
                    for component, reps in tile.component_options.items()
                    if reps
                }
                for tile_id, tile in tile_groups.items()
            }

        # First spend on predicted-visible support.  If it saturates or its
        # next steps are unaffordable, use a small equal fallback score for
        # zero-score tiles rather than discarding feasible segment budget.
        for use_fallback in (False, True):
            best = None
            while True:
                best = None
                for tile_id, tile in tile_groups.items():
                    raw_score = scores.get(str(tile_id), 0.0)
                    score = self.fallback_score if use_fallback and raw_score <= 0.0 else raw_score
                    if score <= 0.0:
                        continue
                    progress = self._tile_progress(tile, bands[str(tile_id)])
                    coverage_discount = 1.0 + self.coverage_strength * progress
                    for component, reps in tile.component_options.items():
                        component_weight = self._component_weight(str(component))
                        if component_weight <= 0.0:
                            continue
                        ordered = self._sorted_representations(reps)
                        band = int(bands[str(tile_id)][component])
                        if band + 1 >= len(ordered):
                            continue
                        extra_bits = float(ordered[band + 1].size_bits - ordered[band].size_bits)
                        if extra_bits < 0.0 or current_bits + extra_bits > budget_bits:
                            continue
                        gain = self._rep_norm(band + 1, len(ordered)) - self._rep_norm(band, len(ordered))
                        utility = score * component_weight * gain / max(extra_bits, 1e-9) / coverage_discount
                        candidate = (utility, -extra_bits, str(tile_id), str(component), extra_bits)
                        if best is None or candidate > best:
                            best = candidate
                if best is None:
                    break
                _utility, _negative_bits, tile_id, component, extra_bits = best
                bands[tile_id][component] += 1
                current_bits += extra_bits
            # Positive support is retained as the only eligible set in the
            # first pass; the second pass considers only zero-score tiles.
            # If all scores were zero, normalization already made them equal.
            if not any(score <= 0.0 for score in scores.values()):
                break

        return {
            str(tile_id): {
                component: self._sorted_representations(reps)[int(bands[str(tile_id)][component])].rep_idx
                for component, reps in tile.component_options.items()
                if reps
            }
            for tile_id, tile in tile_groups.items()
        }


class FoVAwareUniformABR(ABR):
    """Uniform quality for all tiles in FoV, respecting available bandwidth."""
    abr_requires_predictor = True
    abr_supports_predictor = True
    def __init__(self, safety_factor: float = 0.8):
        self.safety_factor = safety_factor

    def decide(self, tiles, segment_frames_path, network, fov, buffer, clock) -> Dict[str, Dict[str, int]]:
        decision = {}
        tile_groups = self.group_tiles(tiles)

        # Predict FoV fractions for one representative per tile
        proportions = self.predict_fov(fov, tile_groups, segment_frames_path)

        visible_tiles = [tile_id for tile_id in tile_groups if proportions.get(tile_id, 0.0) > 0]

        bw = _decision_bandwidth(network, clock, _segment_duration(tile_groups)) * self.safety_factor
        bw_per_tile = bw / len(visible_tiles) if visible_tiles else 0

        for tile_id, tile in tile_groups.items():
            tile_decision = {}
            for comp, reps in tile.component_options.items():
                reps_sorted = sorted(reps, key=lambda r: r.size_bits)
                max_level = 0
                if tile_id in visible_tiles:
                    for rep in reps_sorted:
                        if rep.size_bits / max(rep.duration_s, 1e-9) <= bw_per_tile:
                            max_level = max(max_level, rep.rep_idx)
                tile_decision[comp] = max_level if tile_id in visible_tiles else reps_sorted[0].rep_idx
            decision[tile_id] = tile_decision

        return decision


class OfflineGreedyOptimalABR(ABR):
    """
    Level-1 offline oracle optimizer.

    This is a segment-greedy upper-bound approximation: it expects oracle
    bandwidth and FoV predictors, starts from minimum quality, then spends the
    exact segment budget on the highest FoV-weighted quality gain per bit.
    """

    abr_requires_predictor = True
    abr_supports_predictor = True
    PARAM_METADATA = {
        "safety_factor": {
            "label": "Safety factor",
            "category": "Oracle",
            "help": "Multiplier on the oracle bandwidth budget. Use 1.0 for an upper-bound run.",
        },
        "invisible_weight": {
            "label": "Invisible weight",
            "category": "Oracle",
            "help": "Residual utility for tiles with zero future FoV score.",
        },
    }

    def __init__(self, safety_factor: float = 1.0, invisible_weight: float = 0.0):
        self.safety_factor = max(0.0, float(safety_factor))
        self.invisible_weight = max(0.0, float(invisible_weight))

    @staticmethod
    def _rep_norm(rank: int, count: int) -> float:
        if count <= 1:
            return 1.0
        return max(0.0, min(1.0, rank / float(count - 1)))

    def _component_weight(self, component: str) -> float:
        return float(DEFAULT_COMPONENT_WEIGHTS.get(component, 0.1))

    def decide(self, tiles, segment_frames_path, network, fov, buffer, clock) -> Dict[str, Dict[str, int]]:
        del buffer
        tile_groups = self.group_tiles(tiles)
        proportions = self.predict_fov(fov, tile_groups, segment_frames_path)
        segment_duration = _segment_duration(tile_groups)
        budget_bits = _decision_bandwidth(network, clock, segment_duration) * self.safety_factor * segment_duration

        decision: Dict[str, Dict[str, int]] = {}
        baseline_cost_bits = 0.0
        lanes = []
        for tile_id, tile in tile_groups.items():
            decision[tile_id] = {}
            visibility = max(float(proportions.get(tile_id, 0.0) or 0.0), self.invisible_weight)
            for component, reps in tile.component_options.items():
                reps_sorted = sorted(reps, key=lambda rep: (rep.size_bits, rep.rep_idx))
                if not reps_sorted:
                    continue
                current_idx = 0
                decision[tile_id][component] = reps_sorted[current_idx].rep_idx
                baseline_cost_bits += float(reps_sorted[current_idx].size_bits)
                lanes.append(
                    {
                        "tile_id": tile_id,
                        "component": component,
                        "reps_sorted": reps_sorted,
                        "current_idx": current_idx,
                        "visibility": visibility,
                    }
                )

        remaining_bits = max(float(budget_bits) - baseline_cost_bits, 0.0)
        while remaining_bits > 0.0:
            best_lane = None
            best_score = 0.0
            best_extra_bits = 0.0
            for lane in lanes:
                reps_sorted = lane["reps_sorted"]
                current_idx = int(lane["current_idx"])
                next_idx = current_idx + 1
                if next_idx >= len(reps_sorted):
                    continue
                current_rep = reps_sorted[current_idx]
                next_rep = reps_sorted[next_idx]
                extra_bits = float(next_rep.size_bits - current_rep.size_bits)
                if extra_bits < 0:
                    continue
                if extra_bits > remaining_bits:
                    continue
                if extra_bits <= 0:
                    score = float("inf")
                else:
                    quality_gain = self._rep_norm(next_idx, len(reps_sorted)) - self._rep_norm(current_idx, len(reps_sorted))
                    utility_gain = float(lane["visibility"]) * self._component_weight(str(lane["component"])) * quality_gain
                    score = utility_gain / extra_bits
                if score > best_score:
                    best_lane = lane
                    best_score = score
                    best_extra_bits = extra_bits
            if best_lane is None:
                break
            best_lane["current_idx"] = int(best_lane["current_idx"]) + 1
            next_rep = best_lane["reps_sorted"][best_lane["current_idx"]]
            decision[str(best_lane["tile_id"])][str(best_lane["component"])] = next_rep.rep_idx
            remaining_bits -= best_extra_bits

        return decision
