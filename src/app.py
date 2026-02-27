from __future__ import annotations

import argparse
import datetime
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from uuid import uuid4

from abr import baselines as abr_baselines
from core.event_logger import EventLogger
from core.fov import Fov
from core.network import Network
from core.video import Video
from fov_predictors import baselines as fov_baselines
from player.player import Player


@dataclass
class PlaybackConfig:
    mpd: Path
    content_dir: Path
    network_trace: Path
    fov_trace: Path
    abr: str = "BandwidthAwareABR"
    predictor: Optional[str] = None
    rtt: float = 0.06
    block_size_bits: int = 250_000  # small default block size
    enable_fov_upgrades: bool = False
    enable_mid_download_cancel: bool = False
    buffer_window_segments: int = 10
    buffer_unbounded: bool = False
    show_progress: bool = True
    output_dir: Path = Path("results")
    run_id: Optional[str] = None


def generate_run_id() -> str:
    now = datetime.datetime.now()
    suffix = uuid4().hex[:4]
    return f"run_{now.strftime('%y%m%d_%H%M%S')}_{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI runner for the VPCC player.")
    parser.add_argument("--mpd", type=Path, required=True, help="Path to the V3C MPD manifest.")
    parser.add_argument("--content-dir", type=Path, required=True, help="Directory with original PLY frames.")
    parser.add_argument("--network-trace", type=Path, required=True, help="CSV network trace path.")
    parser.add_argument("--fov-trace", type=Path, required=True, help="CSV FoV trace path.")
    parser.add_argument("--abr", default="BandwidthAwareABR", help="ABR class name from abr.baselines.")
    parser.add_argument("--predictor", default=None, help="FoV predictor class name from fov_predictors.baselines.")
    parser.add_argument("--rtt", type=float, default=0.06, help="Network RTT in seconds.")
    parser.add_argument("--block-size-bits", type=int, default=1_000_000, help="Network block size in bits.")
    parser.add_argument("--enable-fov-upgrades", action="store_true", help="Enable FoV-centric upgrade planning.")
    parser.add_argument(
        "--enable-mid-download-cancel",
        action="store_true",
        help="Enable mid-download cancellation/downswitch when bandwidth drops.",
    )
    parser.add_argument(
        "--buffer-window-segments",
        type=int,
        default=10,
        help="Max segments to buffer ahead before waiting for playback to drain (0 = fetch to end).",
    )
    parser.add_argument(
        "--buffer-unbounded",
        action="store_true",
        help="Disable the buffer limit and fetch until the end of the video.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Base directory to store run artifacts (events, logs).")
    parser.add_argument("--run-id", default=None, help="Optional run id; defaults to timestamp.")
    return parser.parse_args()


def instantiate_abr(name: str, predictor_name: Optional[str] = None):
    abr_cls = getattr(abr_baselines, name, None)
    if not abr_cls:
        raise ValueError(f"Unknown ABR class '{name}'.")
    if getattr(abr_cls, "abr_requires_predictor", False) and (not predictor_name or predictor_name == "None"):
        raise ValueError(f"ABR '{name}' requires a FoV predictor; please select one.")
    return abr_cls()


def instantiate_predictor(name: str | None):
    if not name or name == "None":
        return None
    predictor_cls = getattr(fov_baselines, name, None)
    if not predictor_cls:
        raise ValueError(f"Unknown FoV predictor class '{name}'.")
    return predictor_cls()


def save_config(run_dir: Path, config: PlaybackConfig):
    """Write the effective playback configuration to run_dir/config.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, default=str)


def run_playback(config: PlaybackConfig) -> Path:
    run_id = config.run_id or generate_run_id()
    run_dir = config.output_dir.resolve() / run_id
    config.run_id = run_id
    save_config(run_dir, config)

    network = Network(config.network_trace, rtt=config.rtt, block_size_bits=config.block_size_bits)
    predictor = instantiate_predictor(config.predictor)
    video = Video(
        manifest_file=config.mpd,
        original_content_dir=config.content_dir,
        media_root=None,
    )
    if config.buffer_unbounded or config.buffer_window_segments == 0:
        buffer_window = video.num_segments()
    else:
        buffer_window = int(config.buffer_window_segments or 0)
        if buffer_window < 1:
            raise ValueError("Buffer window must be 0 (unbounded) or at least 1 segment.")
        if buffer_window > video.num_segments():
            buffer_window = video.num_segments()
    fov = Fov(csv_path=config.fov_trace, predictor=predictor, start_frame_index=video.start_frame_idx)
    abr = instantiate_abr(config.abr, config.predictor)

    event_logger = EventLogger()
    player = Player(
        abr=abr,
        network=network,
        fov=fov,
        video=video,
        event_logger=event_logger,
        show_progress=config.show_progress,
        run_dir=run_dir,
        run_id=run_id,
        enable_fov_upgrades=config.enable_fov_upgrades,
        enable_mid_download_cancel=config.enable_mid_download_cancel,
        max_buffer_segments=buffer_window,
    )

    player.play()

    event_logger.save(run_dir / "events.json")

    return run_dir


def main():
    args = parse_args()
    config = PlaybackConfig(
        mpd=args.mpd,
        content_dir=args.content_dir,
        network_trace=args.network_trace,
        fov_trace=args.fov_trace,
        abr=args.abr,
        predictor=args.predictor,
        rtt=args.rtt,
        block_size_bits=args.block_size_bits,
        enable_fov_upgrades=args.enable_fov_upgrades,
        enable_mid_download_cancel=args.enable_mid_download_cancel,
        buffer_window_segments=args.buffer_window_segments,
        buffer_unbounded=args.buffer_unbounded,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    run_dir = run_playback(config)
    print(f"Run completed. Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
