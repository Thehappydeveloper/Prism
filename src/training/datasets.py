from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.rotations import QUAT_COLUMNS, ensure_fov_quaternions
from core.trace_columns import normalize_fov_dataframe, normalize_network_dataframe, read_trace_csv_subset

from .indexing import read_index_rows


class TraceFrameCache:
    def __init__(self, max_entries: int = 256):
        self.max_entries = max_entries
        self._cache: OrderedDict[tuple[str, str, str, bool], pd.DataFrame] = OrderedDict()

    def get(self, path: str, family: str, angle_convention: str | None = None, include_tile_columns: bool = False) -> pd.DataFrame:
        key = (path, family, angle_convention or "", bool(include_tile_columns))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        df = read_trace_csv_subset(path, family, include_tile_columns=include_tile_columns)
        if family == "fov":
            df = normalize_fov_dataframe(df, source=path)
            df = ensure_fov_quaternions(df, angle_convention=angle_convention)
            columns = ["FrameNumber", "HMDPX", "HMDPY", "HMDPZ", *QUAT_COLUMNS]
            if include_tile_columns:
                columns.extend(str(col) for col in df.columns if str(col).startswith("tile_"))
            df = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
            df["FrameNumber"] = df["FrameNumber"].astype(int)
            df = df.set_index("FrameNumber").sort_index()
        else:
            df = normalize_network_dataframe(df, source=path)
            df = df[["Time", "Bandwidth"]].apply(pd.to_numeric, errors="coerce").dropna().sort_values("Time")
        self._cache[key] = df
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return df


class IndexedTraceDataset:
    def __init__(self, index_file: str | Path, family: str, cache: TraceFrameCache | None = None):
        self.rows = read_index_rows(index_file)
        self.family = family
        self.cache = cache or TraceFrameCache()
        self.target_kind = str((self.rows[0] if self.rows else {}).get("target_kind") or ("pose" if family == "fov" else "bandwidth"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        trace = str(row["trace"])
        include_tile_columns = self.family == "fov" and row.get("target_kind") == "tile_visibility"
        df = self.cache.get(trace, self.family, angle_convention=row.get("angle_convention"), include_tile_columns=include_tile_columns)
        if self.family == "fov":
            columns = ["HMDPX", "HMDPY", "HMDPZ", *QUAT_COLUMNS]
            hist = df.loc[[int(v) for v in row["history"]], columns].to_numpy(dtype=np.float32)
            if row.get("target_kind") == "tile_visibility":
                tile_columns = list(row.get("tile_columns") or [])
                target = df.loc[int(row["target"]), tile_columns].to_numpy(dtype=np.float32)
            else:
                target = df.loc[int(row["target"]), columns].to_numpy(dtype=np.float32)
        else:
            times = df["Time"].to_numpy(dtype=float)
            bandwidth = df["Bandwidth"].to_numpy(dtype=float)
            hist_values = np.interp(np.asarray(row["history"], dtype=float), times, bandwidth).astype(np.float32)
            target = np.asarray([np.interp(float(row["target"]), times, bandwidth)], dtype=np.float32)
            hist = hist_values[:, None]
        return hist.astype(np.float32), target.astype(np.float32)


def compute_stats(dataset: IndexedTraceDataset, max_samples: int = 20000) -> dict[str, Any]:
    if len(dataset) == 0:
        raise ValueError("Cannot compute stats for an empty dataset.")
    step = max(1, len(dataset) // max_samples)
    xs = []
    ys = []
    for idx in range(0, len(dataset), step):
        x, y = dataset[idx]
        xs.append(x.reshape(-1))
        ys.append(y.reshape(-1))
    x_mat = np.stack(xs)
    y_mat = np.stack(ys)
    x_mean = x_mat.mean(axis=0)
    x_std = np.maximum(x_mat.std(axis=0), 1e-6)
    y_mean = y_mat.mean(axis=0)
    y_std = np.maximum(y_mat.std(axis=0), 1e-6)
    return {
        "x_mean": x_mean.astype(float).tolist(),
        "x_std": x_std.astype(float).tolist(),
        "y_mean": y_mean.astype(float).tolist(),
        "y_std": y_std.astype(float).tolist(),
        "input_shape": list(dataset[0][0].shape),
        "output_size": int(dataset[0][1].reshape(-1).shape[0]),
        "target_kind": dataset.target_kind,
        "output_tile_ids": [str(col).split("_", 1)[1] for col in (((dataset.rows[0] if dataset.rows else {}).get("tile_columns")) or [])],
    }
