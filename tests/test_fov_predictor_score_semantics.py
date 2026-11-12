from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fov_predictors.baselines import DistanceBasedTileImportancePredictor, UniformTileImportancePredictor
from fov_predictors.raycast import raycast_visibility, reachability_pipeline
from fov_predictors.raycast.path_planning import dijkstra_connected
from fov_predictors.raycast.reachability_pipeline import ReachabilityParams, resolve_auto_pruning
from fov_predictors.raycast.voxel_grid import VoxelGrid


@dataclass
class TileStub:
    unique_id: str
    min_corner: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    max_corner: tuple[float, float, float] = (1.0, 1.0, 1.0)


class FovStub:
    latency_recorder = None


class VoxelGridStub:
    min_corner = np.asarray((-1.0, -1.0, -1.0), dtype=float)
    max_index = np.asarray((2, 2, 2), dtype=np.int64)
    voxel_size = 1.0
    dims = (3, 3, 3)

    def voxel_to_world_center(self, _voxel):
        return np.asarray((0.0, 0.0, 0.0), dtype=float)


class LinearVoxelGridStub:
    min_corner = np.asarray((0.0, 0.0, 0.0), dtype=float)
    max_index = np.asarray((4, 0, 0), dtype=np.int64)
    voxel_size = 1.0
    dims = (5, 1, 1)

    def world_to_voxel(self, point, clamp=True):
        del clamp
        return tuple(np.floor(np.asarray(point, dtype=float)).astype(int).tolist())

    def voxel_to_world_center(self, voxel):
        return np.asarray((float(voxel[0]) + 0.5, 0.5, 0.5), dtype=float)


class SpatialVoxelGridStub:
    min_corner = np.asarray((0.0, 0.0, 0.0), dtype=float)
    max_index = np.asarray((4, 0, 0), dtype=np.int64)
    voxel_size = 1.0
    dims = (5, 1, 1)

    def voxel_to_world_center(self, voxel):
        return np.asarray((float(voxel[0]) + 0.5, 0.5, 0.5), dtype=float)


class FovPredictorScoreSemanticsTests(unittest.TestCase):
    def test_uniform_predictor_returns_configured_independent_score(self) -> None:
        predictor = UniformTileImportancePredictor(score=0.75)

        scores = predictor.predict([TileStub("a"), TileStub("b")], [], FovStub())

        self.assertEqual(scores, {"a": 0.75, "b": 0.75})
        self.assertGreater(sum(scores.values()), 1.0)

    def test_distance_predictor_does_not_normalize_across_tiles(self) -> None:
        predictor = DistanceBasedTileImportancePredictor(sigma=1.0)

        scores = predictor.predict([TileStub("a"), TileStub("b")], [], FovStub())

        self.assertEqual(scores, {"a": 1.0, "b": 1.0})
        self.assertGreater(sum(scores.values()), 1.0)

    def test_raycast_aggregation_does_not_normalize_across_tiles(self) -> None:
        tiles = [TileStub("a"), TileStub("b")]
        tile_samples = {
            "a": np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            "b": np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        }

        with patch.object(raycast_visibility, "_ray_visible_from_world", return_value=True):
            scores, _stats = raycast_visibility.aggregate_tile_importance(
                tiles=tiles,
                voxel_probabilities={(0, 0, 0): 0.5, (1, 0, 0): 0.5},
                tile_samples=tile_samples,
                occupied_set=set(),
                occupied_lookup=None,
                voxel_grid=VoxelGridStub(),
                user_position=(0.0, 0.0, 0.0),
                r_vis=10.0,
                sigma_g=1.0,
            )

        self.assertEqual(scores, {"a": 1.0, "b": 1.0})
        self.assertGreater(sum(scores.values()), 1.0)

    def test_spawned_raycast_workers_match_single_process_visibility(self) -> None:
        """Parallel raycasting must not depend on fork-inherited server state."""
        tiles = [
            TileStub("near", (1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            TileStub("far", (3.0, 0.0, 0.0), (4.0, 1.0, 1.0)),
        ]
        grid = VoxelGrid(
            min_corner=np.asarray((0.0, 0.0, 0.0), dtype=float),
            max_corner=np.asarray((5.0, 1.0, 1.0), dtype=float),
            voxel_size=1.0,
        )
        common_kwargs = dict(
            tiles=tiles,
            voxel_probabilities={(0, 0, 0): 0.6, (1, 0, 0): 0.4},
            tile_samples={
                "near": np.asarray([[1.5, 0.5, 0.5]], dtype=float),
                "far": np.asarray([[3.5, 0.5, 0.5]], dtype=float),
            },
            occupied_set=set(),
            occupied_lookup=None,
            voxel_grid=grid,
            user_position=(0.5, 0.5, 0.5),
            r_vis=10.0,
            sigma_g=10.0,
            aggregation_mode="expected_visibility",
        )

        serial_scores, serial_stats = raycast_visibility.aggregate_tile_importance(
            **common_kwargs,
            raycast_processes=1,
        )
        parallel_scores, parallel_stats = raycast_visibility.aggregate_tile_importance(
            **common_kwargs,
            raycast_processes=2,
        )

        self.assertEqual(parallel_scores, serial_scores)
        self.assertEqual(parallel_stats["raycast_processes_used"], 2.0)
        self.assertEqual(parallel_stats["raycast_parallel_spawn"], 1.0)
        self.assertEqual(serial_stats["raycast_parallel_spawn"], 0.0)

    def test_source_propagation_uses_direct_visibility_when_no_source_tile_is_near_an_origin(self) -> None:
        """Uniform reachable origins must not turn into an all-one fallback."""
        grid = LinearVoxelGridStub()
        tiles = [TileStub("far", (3.0, 0.0, 0.0), (4.0, 1.0, 1.0))]
        tile_samples = {"far": np.asarray([[3.5, 0.5, 0.5]], dtype=float)}

        with patch.object(raycast_visibility, "_ray_visible_from_world", return_value=True):
            scores, stats = raycast_visibility.aggregate_tile_importance(
                tiles=tiles,
                voxel_probabilities={(0, 0, 0): 1.0},
                tile_samples=tile_samples,
                occupied_set=set(),
                occupied_lookup=None,
                voxel_grid=grid,
                user_position=(0.5, 0.5, 0.5),
                r_vis=10.0,
                sigma_g=10.0,
                aggregation_mode="source_propagation",
                source_presence_sigma_l=0.1,
            )

        self.assertEqual(stats["source_tiles_with_score"], 0.0)
        self.assertEqual(stats["source_propagation_direct_visibility_fallback"], 1.0)
        self.assertGreater(scores["far"], 0.0)
        self.assertLess(scores["far"], 1.0)

    def test_probability_diverse_selector_adds_global_boundary_coverage(self) -> None:
        voxel_items = [
            ((0, 0, 0), 0.60, np.asarray((0.5, 0.5, 0.5), dtype=float)),
            ((1, 0, 0), 0.30, np.asarray((1.5, 0.5, 0.5), dtype=float)),
            ((4, 0, 0), 0.10, np.asarray((4.5, 0.5, 0.5), dtype=float)),
        ]

        topk = raycast_visibility.select_candidate_voxels(
            voxel_items=voxel_items,
            max_candidate_voxels=2,
            voxel_selection_mode="probability_topk",
        )
        diverse = raycast_visibility.select_candidate_voxels(
            voxel_items=voxel_items,
            max_candidate_voxels=2,
            voxel_selection_mode="probability_diverse",
            voxel_selection_core_fraction=0.5,
        )

        self.assertEqual([vox for vox, _prob, _center in topk], [(0, 0, 0), (1, 0, 0)])
        self.assertEqual([vox for vox, _prob, _center in diverse], [(0, 0, 0), (4, 0, 0)])

    def test_dijkstra_computes_multiple_path_direction_horizons_in_one_pass(self) -> None:
        free = {(0, 0, 0), (1, 0, 0), (1, 1, 0)}

        def center(voxel):
            return np.asarray(voxel, dtype=float)

        dist, _prev, path_dir_maps = dijkstra_connected(
            free_voxels=free,
            start_voxel=(0, 0, 0),
            voxel_size=1.0,
            offsets=((1, 0, 0), (0, 1, 0)),
            k_steps=(1, 2),
            voxel_to_center=center,
        )

        self.assertIn((1, 1, 0), dist)
        np.testing.assert_allclose(path_dir_maps[1][(1, 1, 0)], np.asarray((1.0, 0.0, 0.0)))
        np.testing.assert_allclose(path_dir_maps[2][(1, 1, 0)], np.asarray((1.0, 1.0, 0.0)) / np.sqrt(2.0))

    def test_path_direction_steps_are_derived_from_user_horizons(self) -> None:
        params = ReachabilityParams(
            v_max=2.0,
            navigation_voxel_size_l=0.5,
            history_horizons_s=(0.5, 1.0, 2.0),
        )

        self.assertEqual(reachability_pipeline._path_dir_steps(params), (2, 4, 8))

    def test_default_voxel_cap_is_not_tile_distance_biased(self) -> None:
        tiles = [
            TileStub("near", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            TileStub("far", (3.0, 0.0, 0.0), (4.0, 1.0, 1.0)),
        ]
        tile_samples = {
            "near": np.asarray([[0.5, 0.5, 0.5]], dtype=float),
            "far": np.asarray([[3.5, 0.5, 0.5]], dtype=float),
        }

        def visible_from_matching_x(**kwargs):
            return int(kwargs["start_voxel"][0]) == int(np.floor(kwargs["target_point_world"][0]))

        common_kwargs = dict(
            tiles=tiles,
            voxel_probabilities={(0, 0, 0): 0.9, (3, 0, 0): 0.1},
            tile_samples=tile_samples,
            occupied_set=set(),
            occupied_lookup=None,
            voxel_grid=SpatialVoxelGridStub(),
            user_position=(0.5, 0.5, 0.5),
            r_vis=10.0,
            sigma_g=0.5,
            max_candidate_voxels=1,
            voxel_selection_core_fraction=1.0,
        )

        with patch.object(raycast_visibility, "_ray_visible_from_world", side_effect=visible_from_matching_x):
            default_scores, default_stats = raycast_visibility.aggregate_tile_importance(
                **common_kwargs,
                voxel_selection_mode="probability_diverse",
            )
            legacy_scores, legacy_stats = raycast_visibility.aggregate_tile_importance(
                **common_kwargs,
                voxel_selection_mode="tile_distance_biased_legacy",
            )

        self.assertEqual(default_stats["candidate_voxel_selection_mode"], "probability_diverse")
        self.assertEqual(default_stats["legacy_distance_biased_enabled"], 0.0)
        self.assertEqual(legacy_stats["legacy_distance_biased_enabled"], 1.0)
        self.assertEqual(default_scores["far"], 0.0)
        self.assertGreater(legacy_scores["far"], default_scores["far"])

    def test_raycast_counts_first_target_tile_hit_as_visible(self) -> None:
        grid = LinearVoxelGridStub()
        target_bounds = (
            np.asarray((2.0, 0.0, 0.0), dtype=float),
            np.asarray((4.0, 1.0, 1.0), dtype=float),
        )

        visible = raycast_visibility._ray_visible_from_world(
            start_world=(0.5, 0.5, 0.5),
            start_voxel=(0, 0, 0),
            target_point_world=(3.5, 0.5, 0.5),
            occupied_set={(2, 0, 0)},
            voxel_grid=grid,
            target_voxel=(3, 0, 0),
            target_tile_bounds=target_bounds,
        )
        blocked = raycast_visibility._ray_visible_from_world(
            start_world=(0.5, 0.5, 0.5),
            start_voxel=(0, 0, 0),
            target_point_world=(3.5, 0.5, 0.5),
            occupied_set={(1, 0, 0)},
            voxel_grid=grid,
            target_voxel=(3, 0, 0),
            target_tile_bounds=target_bounds,
        )

        self.assertTrue(visible)
        self.assertFalse(blocked)

    def test_navigation_probabilities_are_mapped_to_visibility_grid(self) -> None:
        class CoarseGrid:
            voxel_size = 2.0
            min_corner = np.asarray((0.0, 0.0, 0.0), dtype=float)

            def world_to_voxel(self, point, clamp=True):
                del clamp
                return tuple(np.floor(np.asarray(point, dtype=float) / self.voxel_size).astype(int).tolist())

        class FineGrid:
            def voxel_to_world_center(self, voxel):
                return np.asarray((float(voxel[0]) + 0.5, 0.5, 0.5), dtype=float)

        mapped, stats = reachability_pipeline.map_navigation_probs_to_visibility_grid(
            nav_probs={(0, 0, 0): 0.25, (1, 0, 0): 0.75},
            navigation_grid=FineGrid(),
            visibility_grid=CoarseGrid(),
        )

        self.assertEqual(mapped, {(0, 0, 0): 1.0})
        self.assertEqual(stats["navigation_to_visibility_collisions"], 1.0)

    def test_auto_pruning_keeps_frontier_when_tile_budget_is_tight(self) -> None:
        voxel_probs = {(idx, 0, 0): 1.0 / 100.0 for idx in range(100)}
        params = ReachabilityParams(
            max_voxels_eval=1,
            auto_max_voxels_eval=True,
            auto_min_voxel_prob=True,
            auto_min_samples_per_tile=4,
            auto_voxel_mass_target=0.95,
        )

        resolved_cap, _min_prob, stats = resolve_auto_pruning(
            voxel_probs=voxel_probs,
            active_tile_count=10_000,
            visibility_ray_budget=20,
            params=params,
        )

        self.assertGreaterEqual(resolved_cap, 8)
        self.assertEqual(stats["resolved_max_voxels_eval"], float(resolved_cap))


if __name__ == "__main__":
    unittest.main()
