from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .export import load_exported_model_metadata
from .models import build_model, require_torch, resolve_device


class TrainedModelRuntime:
    def __init__(self, model_id: str, device: str = "auto"):
        self.model_id = str(model_id or "").strip()
        if not self.model_id:
            raise ValueError("model_id is required.")
        self.metadata = load_exported_model_metadata(self.model_id)
        self.stats = self.metadata["stats"]
        self.plan_model = self.metadata.get("model") or {}
        self.history = self.metadata.get("history") or {}
        self.torch, _ = require_torch()
        self.device = resolve_device(device)
        checkpoint = Path(str(self.metadata.get("checkpoint") or ""))
        if not checkpoint.exists():
            raise FileNotFoundError(f"Trained model checkpoint not found: {checkpoint}")
        self.model = build_model(self.plan_model, self.stats["input_shape"], int(self.stats["output_size"])).to(self.device)
        try:
            payload = self.torch.load(checkpoint, map_location=self.device, weights_only=False)
        except TypeError:
            payload = self.torch.load(checkpoint, map_location=self.device)
        state = payload.get("model_state") if isinstance(payload, dict) else None
        if state is None:
            raise ValueError("Checkpoint does not contain model_state.")
        self.model.load_state_dict(state)
        self.model.eval()

    def predict_array(self, history_array: np.ndarray) -> np.ndarray:
        x = np.asarray(history_array, dtype=np.float32)
        expected = tuple(int(v) for v in self.stats["input_shape"])
        if x.shape != expected:
            raise ValueError(f"Expected history shape {expected}, got {x.shape}.")
        x_mean = np.asarray(self.stats["x_mean"], dtype=np.float32).reshape(expected)
        x_std = np.asarray(self.stats["x_std"], dtype=np.float32).reshape(expected)
        y_mean = np.asarray(self.stats["y_mean"], dtype=np.float32)
        y_std = np.asarray(self.stats["y_std"], dtype=np.float32)
        x_norm = (x - x_mean) / x_std
        with self.torch.no_grad():
            tensor = self.torch.tensor(x_norm[None, :, :], dtype=self.torch.float32, device=self.device)
            pred = self.model(tensor).detach().cpu().numpy()[0]
        return pred * y_std + y_mean


def frame_index_from_path(frame_path: Path) -> int:
    stem = Path(frame_path).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not extract frame index from {frame_path}")
    return int(digits)


def fov_history_array(runtime: TrainedModelRuntime, fov_data: Any, anchor_frame: int) -> np.ndarray:
    history = runtime.history
    fps = 30.0
    fixed = int(history.get("fixed_history_samples") or 0)
    step_s = max(float(history.get("history_step_s") or 0.5), 1e-6)
    duration_s = max(float(history.get("history_duration_s") or 0.0), step_s)
    count = fixed if fixed > 0 else int(np.floor(duration_s / step_s)) + 1
    step = max(1, int(round(step_s * fps)))
    frames = [int(anchor_frame - (count - idx - 1) * step) for idx in range(count)]
    rows = []
    for frame in frames:
        row = fov_data.get_frame(frame - int(getattr(fov_data, "start_frame_index", 0) or 0))
        if row is None:
            raise ValueError("Not enough FoV history for trained predictor.")
        rows.append([float(row[name]) for name in ["HMDPX", "HMDPY", "HMDPZ", "Quat_X", "Quat_Y", "Quat_Z", "Quat_W"]])
    return np.asarray(rows, dtype=np.float32)


def bandwidth_history_array(runtime: TrainedModelRuntime, network: Any, time_s: float, window_s: float | None = None) -> np.ndarray:
    history = runtime.history
    input_shape = runtime.stats["input_shape"]
    count = int(input_shape[0])
    step_s = max(float(history.get("history_step_s") or 0.5), 1e-6)
    start = float(time_s) - (count - 1) * step_s
    times = np.asarray([start + idx * step_s for idx in range(count)], dtype=float)
    if window_s is not None and window_s > 0:
        times = np.maximum(times, float(time_s) - float(window_s))
    values = [float(network.get_instantaneous_bandwidth(float(t))) / 1e6 for t in times]
    return np.asarray(values, dtype=np.float32).reshape(count, 1)
