from __future__ import annotations

from typing import Any


TRACE_ONLY_MODEL_TYPES = {"mlp", "gru", "lstm"}
POINT_CLOUD_MODEL_TYPES = {"pointnet", "pointnet2", "sparse_conv", "sparseconv"}


def model_type(model_config: dict[str, Any] | None) -> str:
    return str((model_config or {}).get("model_type") or "mlp").strip().lower()


def requires_point_cloud_features(model_config: dict[str, Any] | None) -> bool:
    return model_type(model_config) in POINT_CLOUD_MODEL_TYPES


def is_trace_only_model(model_config: dict[str, Any] | None) -> bool:
    return model_type(model_config) in TRACE_ONLY_MODEL_TYPES


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for training and trained predictor inference. Install torch or use Prism baseline predictors.") from exc
    return torch, nn


def build_model(model_config: dict[str, Any], input_shape: list[int], output_size: int):
    torch, nn = require_torch()
    selected_model_type = model_type(model_config)
    hidden_size = int(model_config.get("hidden_size") or 64)
    num_layers = int(model_config.get("num_layers") or 1)
    dropout = float(model_config.get("dropout") or 0.0)
    seq_len = int(input_shape[0])
    feature_size = int(input_shape[1])

    if selected_model_type in POINT_CLOUD_MODEL_TYPES:
        raise ValueError(
            f"Model type '{selected_model_type}' requires point cloud feature extraction, "
            "which is not implemented by the current trace-window trainer."
        )

    if selected_model_type == "gru":
        class GRURegressor(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.GRU(feature_size, hidden_size, num_layers=max(1, num_layers), dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
                self.head = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                out, _ = self.rnn(x)
                return self.head(out[:, -1, :])

        return GRURegressor()

    if selected_model_type == "lstm":
        class LSTMRegressor(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(feature_size, hidden_size, num_layers=max(1, num_layers), dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
                self.head = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                out, _ = self.rnn(x)
                return self.head(out[:, -1, :])

        return LSTMRegressor()

    class MLPRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(seq_len * feature_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )

        def forward(self, x):
            return self.net(x)

    return MLPRegressor()


def resolve_device(device: str):
    torch, _ = require_torch()
    requested = str(device or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)
