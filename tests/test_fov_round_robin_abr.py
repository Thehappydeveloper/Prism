from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for module_name, class_name in (
    ("core.network", "Network"),
    ("core.fov", "Fov"),
    ("core.buffer", "Buffer"),
    ("core.clock", "Clock"),
    ("core.media_segment", "TileComponentOptions"),
):
    module = types.ModuleType(module_name)
    setattr(module, class_name, object)
    sys.modules.setdefault(module_name, module)

from abr.baselines import CoverageAwareUtilityABR, FoVRoundRobinABR, RankBudgetABR, ScoreThresholdABR, ScoreThresholdBudgetABR


@dataclass
class Rep:
    rep_idx: int
    size_bits: int
    duration_s: float = 1.0


@dataclass
class Tile:
    unique_id: str
    component_options: dict[str, list[Rep]]


class FixedBandwidthNetwork:
    def __init__(self, bandwidth_bps: float):
        self.bandwidth_bps = bandwidth_bps

    def predict_bandwidth_for_decision(self, *_args, **_kwargs) -> float:
        return self.bandwidth_bps


class ZeroClock:
    def get_time(self) -> float:
        return 0.0


class FixedFovAbr(FoVRoundRobinABR):
    def __init__(self, scores: dict[str, float]):
        super().__init__(safety_factor=1.0)
        self.scores = scores

    def predict_fov(self, _fov, _tile_groups, _frame_paths):
        return dict(self.scores)


class FixedScoreThresholdAbr(ScoreThresholdABR):
    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def predict_fov(self, _fov, _tile_groups, _frame_paths):
        return dict(self.scores)


class FixedScoreThresholdBudgetAbr(ScoreThresholdBudgetABR):
    def __init__(self, scores: dict[str, float], safety_factor: float = 1.0):
        super().__init__(safety_factor=safety_factor)
        self.scores = scores

    def predict_fov(self, _fov, _tile_groups, _frame_paths):
        return dict(self.scores)


class FixedRankBudgetAbr(RankBudgetABR):
    def __init__(self, scores: dict[str, float], safety_factor: float = 1.0):
        super().__init__(safety_factor=safety_factor)
        self.scores = scores

    def predict_fov(self, _fov, _tile_groups, _frame_paths):
        return dict(self.scores)


class FixedCoverageAwareAbr(CoverageAwareUtilityABR):
    def __init__(self, scores: dict[str, float], safety_factor: float = 1.0, coverage_strength: float = 1.5):
        super().__init__(safety_factor=safety_factor, coverage_strength=coverage_strength)
        self.scores = scores

    def predict_fov(self, _fov, _tile_groups, _frame_paths):
        return dict(self.scores)


class FoVRoundRobinABRTests(unittest.TestCase):
    def test_weighted_utility_keeps_upgrading_high_visibility_tile(self) -> None:
        reps = [Rep(0, 10), Rep(1, 20), Rep(2, 30)]
        tiles = [
            Tile("high", {"geom": reps}),
            Tile("low", {"geom": reps}),
        ]
        abr = FixedFovAbr({"high": 1.0, "low": 0.1})

        decision = abr.decide(
            tiles,
            segment_frames_path=[],
            network=FixedBandwidthNetwork(40),
            fov=None,
            buffer=None,
            clock=ZeroClock(),
        )

        self.assertEqual(decision["high"]["geom"], 2)
        self.assertEqual(decision["low"]["geom"], 0)


class ScoreThresholdABRTests(unittest.TestCase):
    def test_maps_scores_to_equal_representation_bands_without_redistribution(self) -> None:
        reps = [Rep(0, 10), Rep(1, 20), Rep(2, 30), Rep(3, 40)]
        tiles = [
            Tile("zero", {"geom": reps}),
            Tile("low", {"geom": reps}),
            Tile("mid", {"geom": reps}),
            Tile("high", {"geom": reps}),
            Tile("one", {"geom": reps}),
        ]
        abr = FixedScoreThresholdAbr({"zero": 0.0, "low": 0.249, "mid": 0.5, "high": 0.75, "one": 1.0})

        decision = abr.decide(
            tiles,
            segment_frames_path=[],
            network=FixedBandwidthNetwork(1),
            fov=None,
            buffer=None,
            clock=ZeroClock(),
        )

        self.assertEqual(
            {tile_id: components["geom"] for tile_id, components in decision.items()},
            {"zero": 0, "low": 0, "mid": 2, "high": 3, "one": 3},
        )

    def test_uses_each_component_size_rank_and_clamps_invalid_scores(self) -> None:
        tiles = [
            Tile("tile", {
                "geom": [Rep(2, 30), Rep(0, 10), Rep(1, 20)],
                "attr": [Rep(7, 5), Rep(9, 15)],
            }),
        ]
        abr = FixedScoreThresholdAbr({"tile": 5.0})

        decision = abr.decide(tiles, [], FixedBandwidthNetwork(1), None, None, ZeroClock())

        self.assertEqual(decision, {"tile": {"geom": 2, "attr": 9}})


class ScoreThresholdBudgetABRTests(unittest.TestCase):
    def test_downgrades_lowest_score_tile_before_higher_score_tile(self) -> None:
        reps = [Rep(0, 10), Rep(1, 20), Rep(2, 30)]
        tiles = [Tile("low", {"geom": reps}), Tile("high", {"geom": reps})]
        abr = FixedScoreThresholdBudgetAbr({"low": 0.5, "high": 1.0})

        decision = abr.decide(tiles, [], FixedBandwidthNetwork(40), None, None, ZeroClock())

        self.assertEqual(decision, {"low": {"geom": 0}, "high": {"geom": 2}})

    def test_returns_all_minimum_when_minimum_representation_cost_exceeds_budget(self) -> None:
        reps = [Rep(0, 10), Rep(1, 20)]
        tiles = [Tile("a", {"geom": reps}), Tile("b", {"geom": reps})]
        abr = FixedScoreThresholdBudgetAbr({"a": 1.0, "b": 1.0})

        decision = abr.decide(tiles, [], FixedBandwidthNetwork(15), None, None, ZeroClock())

        self.assertEqual(decision, {"a": {"geom": 0}, "b": {"geom": 0}})


class CoverageAwareUtilityABRTests(unittest.TestCase):
    def test_coverage_factor_spreads_first_upgrades_across_high_score_tiles(self) -> None:
        reps = [Rep(0, 10), Rep(1, 20), Rep(2, 30)]
        tiles = [Tile("high", {"geom": reps}), Tile("nearby", {"geom": reps})]
        abr = FixedCoverageAwareAbr({"high": 1.0, "nearby": 0.8}, coverage_strength=1.5)

        decision = abr.decide(tiles, [], FixedBandwidthNetwork(40), None, None, ZeroClock())

        self.assertEqual(decision["high"]["geom"], 1)
        self.assertEqual(decision["nearby"]["geom"], 1)

    def test_score_magnitude_is_retained_after_per_decision_normalization(self) -> None:
        reps = [Rep(0, 10), Rep(1, 20), Rep(2, 30)]
        tiles = [Tile("high", {"geom": reps}), Tile("low", {"geom": reps})]
        abr = FixedCoverageAwareAbr({"high": 1.0, "low": 0.1}, coverage_strength=0.0)

        decision = abr.decide(tiles, [], FixedBandwidthNetwork(40), None, None, ZeroClock())

        self.assertEqual(decision["high"]["geom"], 2)
        self.assertEqual(decision["low"]["geom"], 0)

    def test_uses_zero_score_fallback_after_positive_support_saturates(self) -> None:
        reps = [Rep(0, 10), Rep(1, 20)]
        tiles = [Tile("high", {"geom": reps}), Tile("zero", {"geom": reps})]
        abr = FixedCoverageAwareAbr({"high": 1.0, "zero": 0.0}, coverage_strength=1.5)

        decision = abr.decide(tiles, [], FixedBandwidthNetwork(40), None, None, ZeroClock())

        self.assertEqual(decision["high"]["geom"], 1)
        self.assertEqual(decision["zero"]["geom"], 1)

    def test_does_not_spend_budget_on_zero_saliency_atlas_before_visible_components(self) -> None:
        reps = [Rep(0, 10), Rep(1, 20)]
        tiles = [Tile("tile", {"atlas": reps, "geom": reps})]
        abr = FixedCoverageAwareAbr({"tile": 1.0}, coverage_strength=0.0)

        decision = abr.decide(tiles, [], FixedBandwidthNetwork(50), None, None, ZeroClock())

        self.assertEqual(decision["tile"]["geom"], 1)
        self.assertEqual(decision["tile"]["atlas"], 0)


class RankBudgetABRTests(unittest.TestCase):
    def test_is_invariant_to_a_strictly_monotonic_score_scale(self) -> None:
        reps = [Rep(0, 10), Rep(1, 20), Rep(2, 30)]
        tiles = [Tile("low", {"geom": reps}), Tile("mid", {"geom": reps}), Tile("high", {"geom": reps})]
        small_scale = FixedRankBudgetAbr({"low": 0.01, "mid": 0.02, "high": 0.03})
        large_scale = FixedRankBudgetAbr({"low": 0.2, "mid": 0.6, "high": 0.9})
        network = FixedBandwidthNetwork(60)

        self.assertEqual(
            small_scale.decide(tiles, [], network, None, None, ZeroClock()),
            large_scale.decide(tiles, [], network, None, None, ZeroClock()),
        )

    def test_preserves_rank_order_when_budget_is_tight(self) -> None:
        reps = [Rep(0, 10), Rep(1, 20), Rep(2, 30)]
        tiles = [Tile("low", {"geom": reps}), Tile("mid", {"geom": reps}), Tile("high", {"geom": reps})]
        abr = FixedRankBudgetAbr({"low": 0.1, "mid": 0.2, "high": 0.3})

        decision = abr.decide(tiles, [], FixedBandwidthNetwork(50), None, None, ZeroClock())

        self.assertGreaterEqual(decision["high"]["geom"], decision["mid"]["geom"])
        self.assertGreaterEqual(decision["mid"]["geom"], decision["low"]["geom"])
        self.assertEqual(decision["high"]["geom"], 2)
        self.assertEqual(decision["low"]["geom"], 0)

    def test_all_zero_scores_form_one_tied_group(self) -> None:
        reps = [Rep(0, 10), Rep(1, 20), Rep(2, 30)]
        tiles = [Tile("a", {"geom": reps}), Tile("b", {"geom": reps})]
        abr = FixedRankBudgetAbr({"a": 0.0, "b": 0.0})

        decision = abr.decide(tiles, [], FixedBandwidthNetwork(40), None, None, ZeroClock())

        self.assertEqual(decision["a"]["geom"], decision["b"]["geom"])

    def test_uses_zero_score_tier_after_higher_tier_saturates(self) -> None:
        reps = [Rep(0, 10), Rep(1, 20)]
        tiles = [Tile("high", {"geom": reps}), Tile("zero_a", {"geom": reps}), Tile("zero_b", {"geom": reps})]
        abr = FixedRankBudgetAbr({"high": 1.0, "zero_a": 0.0, "zero_b": 0.0})

        decision = abr.decide(tiles, [], FixedBandwidthNetwork(50), None, None, ZeroClock())

        self.assertEqual(decision["high"]["geom"], 1)
        self.assertEqual(sum(item["geom"] for item in decision.values()), 2)


if __name__ == "__main__":
    unittest.main()
