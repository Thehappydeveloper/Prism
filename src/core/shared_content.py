from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Sequence

import pandas as pd

from core.compute_processes import register_compute_pool, unregister_compute_pool
from core.timing import TimingRecorder, active_timing


_ACTIVE_CONTENT: ContextVar["SegmentContentWindow | None"] = ContextVar("active_segment_content", default=None)


def _resolve_ply_path(path: str | Path) -> Path:
    ply_path = Path(path)
    if ply_path.suffix.lower() != ".ply":
        ply_path = ply_path.with_suffix(ply_path.suffix + ".ply")
    return ply_path.resolve()


def active_segment_content() -> "SegmentContentWindow | None":
    return _ACTIVE_CONTENT.get()


def _read_ply_frame_worker(path_text: str) -> tuple[str, pd.DataFrame]:
    from utils.ply_reader import PlyReader

    path = _resolve_ply_path(path_text)
    return str(path), PlyReader().read(path)


@dataclass
class SegmentContentWindow:
    """Short-lived PLY frame window shared by compatible runs for one segment."""

    segment_idx: int
    frame_paths: Sequence[Path]
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def get_frame(self, path: str | Path) -> pd.DataFrame | None:
        return self.frames.get(str(_resolve_ply_path(path)))

    @contextmanager
    def activate(self) -> Iterator["SegmentContentWindow"]:
        token = _ACTIVE_CONTENT.set(self)
        try:
            yield self
        finally:
            _ACTIVE_CONTENT.reset(token)


class SegmentContentLoader:
    """Load a segment's PLY frames once, then release them after grouped use."""

    def __init__(self, worker_count: int = 1, timing: TimingRecorder | None = None) -> None:
        self.worker_count = max(1, int(worker_count or 1))
        self.timing = timing
        self._executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> "SegmentContentLoader":
        if self.worker_count > 1:
            # The loader is owned by the threaded batch server.  Avoid the
            # platform-default ``fork`` context so it cannot pass inherited
            # locks into a child that later coexists with raycast workers.
            self._executor = ProcessPoolExecutor(
                max_workers=self.worker_count,
                mp_context=mp.get_context("spawn"),
            )
            register_compute_pool(self._executor)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._executor is not None:
            executor = self._executor
            try:
                executor.shutdown(wait=True, cancel_futures=False)
            finally:
                unregister_compute_pool(executor)
            self._executor = None

    def load_window(self, segment_idx: int, frame_paths: Sequence[Path], *, needs_ply: bool) -> SegmentContentWindow | None:
        if not needs_ply or not frame_paths:
            return None
        resolved = [str(_resolve_ply_path(path)) for path in frame_paths]
        unique_paths = list(dict.fromkeys(resolved))
        recorder = self.timing or active_timing()
        if recorder is not None:
            recorder.increment("shared_content.window_loads")
            recorder.increment("shared_content.frames_requested", len(unique_paths))

        stage = recorder.stage("shared_content.load_window", segment_idx=segment_idx, frame_count=len(unique_paths)) if recorder else nullcontext()
        with stage:
            if self.worker_count <= 1 or len(unique_paths) <= 1:
                frames = dict(_read_ply_frame_worker(path) for path in unique_paths)
            else:
                executor = self._executor
                if executor is None:
                    transient_executor = ProcessPoolExecutor(
                        max_workers=self.worker_count,
                        mp_context=mp.get_context("spawn"),
                    )
                    register_compute_pool(transient_executor)
                    try:
                        with transient_executor:
                            frames = dict(transient_executor.map(_read_ply_frame_worker, unique_paths))
                    finally:
                        unregister_compute_pool(transient_executor)
                else:
                    frames = dict(executor.map(_read_ply_frame_worker, unique_paths))
        return SegmentContentWindow(segment_idx=segment_idx, frame_paths=[Path(path) for path in unique_paths], frames=frames)
