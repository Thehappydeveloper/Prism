import random
from typing import Dict

from abr.base import ABR

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
        bw_available = network.get_instantaneous_bandwidth(clock.get_time()) * self.safety_factor
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
    def __init__(self, safety_factor=0.8):
        self.safety_factor = safety_factor    

    def decide(self, tiles, segment_frames_path, network, fov, buffer, clock):
        tile_groups = self.group_tiles(tiles)
        proportions = self.predict_fov(fov, tile_groups, segment_frames_path)
        bw_bps = network.get_instantaneous_bandwidth(clock.get_time()) * self.safety_factor
        if tile_groups:
            first_tile = next(iter(tile_groups.values()))
            first_component = next(iter(first_tile.component_options.values()))
            segment_duration = first_component[0].duration_s
        else:
            segment_duration = 1.0
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

        upgradable = []
        for tile_id, tile in tile_groups.items():
            if proportions.get(tile_id, 0) <= 0:
                continue
            for comp, reps in tile.component_options.items():
                sorted_reps = sorted(reps, key=lambda t: t.size_bits)
                upgradable.append(
                    {
                        "tile_id": tile_id,
                        "component": comp,
                        "reps_sorted": sorted_reps,
                        "current_idx": 0,
                    }
                )

        idx = 0
        while remaining_bits > 0 and upgradable:
            upgrade = upgradable[idx % len(upgradable)]
            if upgrade["current_idx"] + 1 < len(upgrade["reps_sorted"]):
                next_rep = upgrade["reps_sorted"][upgrade["current_idx"] + 1]
                if next_rep.size_bits <= remaining_bits:
                    decision[upgrade["tile_id"]][upgrade["component"]] = next_rep.rep_idx
                    upgrade["current_idx"] += 1
                    remaining_bits -= next_rep.size_bits
            idx += 1
            if idx > len(upgradable) * 5:
                break

        return decision

class FoVAwareUniformABR(ABR):
    """Uniform quality for all tiles in FoV, respecting available bandwidth."""
    abr_requires_predictor = True
    def __init__(self, safety_factor: float = 0.8):
        self.safety_factor = safety_factor

    def decide(self, tiles, segment_frames_path, network, fov, buffer, clock) -> Dict[str, Dict[str, int]]:
        decision = {}
        tile_groups = self.group_tiles(tiles)

        # Predict FoV fractions for one representative per tile
        proportions = self.predict_fov(fov, tile_groups, segment_frames_path)

        visible_tiles = [tile_id for tile_id in tile_groups if proportions.get(tile_id, 0.0) > 0]

        bw = network.get_instantaneous_bandwidth(clock.get_time()) * self.safety_factor
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
