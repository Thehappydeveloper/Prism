from __future__ import annotations

import logging
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.media_segment import TileBounds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentTemplateInfo:
    initialization: Optional[str]
    media: str
    start_number: int
    timescale: int
    duration: int


@dataclass(frozen=True)
class RepresentationInfo:
    id: str
    bandwidth: Optional[int]
    quality_id: str
    template: SegmentTemplateInfo


class V3CMpd:
    """Parser for V3C DASH MPDs with per-component tiling."""

    CORE_COMPONENTS = {"geom", "occp", "attr"}

    def __init__(self, mpd_path: Path, media_root: Optional[Path] = None):
        self.mpd_path = mpd_path
        self.media_root = media_root or mpd_path.parent

        tree = ET.parse(mpd_path)
        self.root = tree.getroot()
        self.ns = {
            "mpd": self.root.tag.split("}")[0].strip("{"),
            "v3c": "urn:mpeg:mpegI:v3c:2020",
        }
        self.base_url = self._parse_base_url()

        self.presentation_duration_s = self._parse_iso_duration(
            self.root.attrib.get("mediaPresentationDuration", "PT0S")
        )
        (
            self.segment_duration_s,
            self.segment_count,
            self.tile_components,
        ) = self._parse_adaptation_sets()

        self.bounds_by_segment = self._parse_tile_boundaries()

    # --------- Public helpers ---------

    def iter_tile_ids(self) -> List[str]:
        return sorted(
            self.tile_components.keys(),
            key=lambda tid: int(tid) if str(tid).isdigit() else tid,
        )

    # --------- Parsing internals ---------

    def _parse_base_url(self) -> Path:
        base_el = self.root.find("mpd:BaseURL", self.ns)
        text = base_el.text.strip() if base_el is not None and base_el.text else "."
        return (self.media_root / text).resolve()

    def _parse_iso_duration(self, duration: str) -> float:
        """
        Minimal ISO-8601 duration parser supporting PT#H#M#S.
        """
        pattern = r"PT(?:(?P<hours>\d+(?:\.\d+)?)H)?(?:(?P<minutes>\d+(?:\.\d+)?)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?"
        match = re.match(pattern, duration)
        if not match:
            raise ValueError(f"Unsupported duration format: {duration}")
        hours = float(match.group("hours") or 0)
        minutes = float(match.group("minutes") or 0)
        seconds = float(match.group("seconds") or 0)
        return hours * 3600 + minutes * 60 + seconds

    def _parse_quality_id(self, rep_id: str, fallback_idx: int) -> str:
        match = re.search(r"_q(?P<q>\d+)", rep_id)
        if match:
            return f"q{match.group('q')}"
        return f"q{fallback_idx}"

    def _parse_component_type(self, adaptation_el) -> str:
        vc_prop = adaptation_el.find(
            'mpd:EssentialProperty[@schemeIdUri="urn:mpeg:mpegI:v3c:2020:videoComponent"]',
            self.ns,
        )
        if vc_prop is not None:
            vc = vc_prop.find("v3c:videoComponent", self.ns)
            if vc is not None and "type" in vc.attrib:
                return vc.attrib["type"]

        adaptation_id = adaptation_el.attrib.get("id", "").lower()
        if "atlas" in adaptation_id:
            return "atlas"
        return "unknown"

    def _parse_tile_id(self, adaptation_id: str) -> Optional[str]:
        match = re.match(r"tile(?P<tile_id>\d+)", adaptation_id)
        if match:
            return match.group("tile_id")
        return None

    def _parse_adaptation_sets(
        self,
    ) -> Tuple[float, int, Dict[str, Dict[str, Dict[str, RepresentationInfo]]]]:
        period = self.root.find("mpd:Period", self.ns)
        if period is None:
            raise ValueError("MPD Period not found")

        tile_components: Dict[str, Dict[str, Dict[str, RepresentationInfo]]] = {}
        segment_duration_s: Optional[float] = None

        for adaptation_el in period.findall("mpd:AdaptationSet", self.ns):
            adaptation_id = adaptation_el.attrib.get("id", "")
            tile_id = self._parse_tile_id(adaptation_id)
            if tile_id is None:
                continue

            component_type = self._parse_component_type(adaptation_el)
            representations: Dict[str, RepresentationInfo] = {}

            adaptation_template = adaptation_el.find("mpd:SegmentTemplate", self.ns)

            for idx, rep_el in enumerate(adaptation_el.findall("mpd:Representation", self.ns)):
                rep_id = rep_el.attrib.get("id", f"{adaptation_id}_rep{idx}")
                quality_id = self._parse_quality_id(rep_id, idx)
                bandwidth = (
                    int(rep_el.attrib["bandwidth"])
                    if "bandwidth" in rep_el.attrib
                    else None
                )

                template_el = rep_el.find("mpd:SegmentTemplate", self.ns)
                if template_el is None:
                    template_el = adaptation_template
                if template_el is None:
                    raise ValueError(f"SegmentTemplate missing for {rep_id}")

                template = SegmentTemplateInfo(
                    initialization=template_el.attrib.get("initialization"),
                    media=template_el.attrib["media"],
                    start_number=int(template_el.attrib.get("startNumber", "1")),
                    timescale=int(template_el.attrib.get("timescale", "1")),
                    duration=int(template_el.attrib["duration"]),
                )

                current_seg_duration = template.duration / template.timescale
                if segment_duration_s is None:
                    segment_duration_s = current_seg_duration
                elif not math.isclose(segment_duration_s, current_seg_duration, rel_tol=1e-3):
                    logger.warning(
                        "Segment duration mismatch detected (existing %.3fs, found %.3fs). Using the first one.",
                        segment_duration_s,
                        current_seg_duration,
                    )

                representations[quality_id] = RepresentationInfo(
                    id=rep_id,
                    bandwidth=bandwidth,
                    quality_id=quality_id,
                    template=template,
                )

            if not representations:
                continue

            tile_components.setdefault(tile_id, {})[component_type] = representations

        if segment_duration_s is None:
            raise ValueError("Could not infer segment duration from MPD")

        segment_count = math.ceil(self.presentation_duration_s / segment_duration_s)
        return segment_duration_s, segment_count, tile_components

    def _parse_tile_boundaries(self) -> Dict[int, Dict[str, TileBounds]]:
        period = self.root.find("mpd:Period", self.ns)
        if period is None:
            return {}

        event_stream = None
        for candidate in period.findall("mpd:EventStream", self.ns):
            scheme = candidate.attrib.get("schemeIdUri")
            if scheme in ("urn:mpeg:dash:tile-boundaries", "urn:v3ctk:tile-boundaries"):
                event_stream = candidate
                break
        if event_stream is None:
            logger.warning(
                "No tile-boundaries EventStream found in MPD (expected schemeIdUri urn:mpeg:dash:tile-boundaries or urn:v3ctk:tile-boundaries)."
            )
            return {}

        timescale = int(event_stream.attrib.get("timescale", "1"))
        events = []
        for event_el in event_stream.findall("mpd:Event", self.ns):
            start_s = int(event_el.attrib.get("presentationTime", "0")) / timescale
            duration_s = int(event_el.attrib.get("duration", "0")) / timescale
            end_s = start_s + duration_s if duration_s > 0 else start_s

            bounds: Dict[str, TileBounds] = {}
            for bounds_el in event_el.findall(".//{*}TileBoundaries"):
                for tile_el in bounds_el.findall(".//{*}Tile"):
                    tid = tile_el.attrib.get("id")
                    if tid is None:
                        continue
                    xmin = float(tile_el.attrib["xmin"])
                    xmax = float(tile_el.attrib["xmax"])
                    ymin = float(tile_el.attrib["ymin"])
                    ymax = float(tile_el.attrib["ymax"])
                    zmin = float(tile_el.attrib["zmin"])
                    zmax = float(tile_el.attrib["zmax"])
                    bounds[tid] = TileBounds(
                        min_corner=(xmin, ymin, zmin),
                        max_corner=(xmax, ymax, zmax),
                    )

            events.append((start_s, end_s, bounds))

        events.sort(key=lambda e: e[0])
        bounds_by_segment: Dict[int, Dict[str, TileBounds]] = {}
        latest: Dict[str, TileBounds] = {}

        for seg_idx in range(self.segment_count):
            seg_start = seg_idx * self.segment_duration_s
            matching_event = next(
                (ev for ev in events if ev[0] <= seg_start < ev[1]), None
            )
            if matching_event:
                _, _, bounds = matching_event
                latest.update(bounds)
            bounds_by_segment[seg_idx] = dict(latest)

        if not bounds_by_segment:
            logger.warning("No tile boundaries could be mapped to segments.")
        return bounds_by_segment

    # --------- Query helpers used by Video ---------

    def component_quality_order(self, tile_id: str, component: str) -> List[Tuple[str, int]]:
        reps = self.tile_components.get(tile_id, {}).get(component, {})
        if not reps:
            return []

        ordered = sorted(
            reps.values(),
            key=lambda r: r.bandwidth if r.bandwidth is not None else float("inf"),
        )
        return [(rep.quality_id, idx) for idx, rep in enumerate(ordered)]

    def get_bounds(self, tile_id: str, segment_idx: int) -> Optional[TileBounds]:
        bounds = self.bounds_by_segment.get(segment_idx, {})
        return bounds.get(tile_id)

    def get_component_representation(
        self, tile_id: str, component: str, quality_id: str
    ) -> Optional[RepresentationInfo]:
        return (
            self.tile_components.get(tile_id, {})
            .get(component, {})
            .get(quality_id)
        )

    def get_default_atlas(self, tile_id: str) -> Optional[RepresentationInfo]:
        atlas_reps = self.tile_components.get(tile_id, {}).get("atlas", {})
        if not atlas_reps:
            return None
        return min(atlas_reps.values(), key=lambda r: r.bandwidth or 0)
