from __future__ import annotations

from typing import Any, Protocol


class TrainingGraphPlugin(Protocol):
    id: str
    title: str
    type: str

    def build(self, metrics: dict[str, Any]) -> dict[str, Any]:
        ...


class LossGraphPlugin:
    id = "loss"
    title = "Training / Validation"
    type = "line"

    def build(self, metrics: dict[str, Any]) -> dict[str, Any]:
        epochs = metrics.get("epochs") or []
        return {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "x": [item.get("epoch") for item in epochs],
            "series": [
                {"label": "train_loss", "values": [item.get("train_loss") for item in epochs]},
                {"label": "validation_mae", "values": [item.get("validation_mae") for item in epochs]},
            ],
        }


class EpochTimeGraphPlugin:
    id = "epoch_time"
    title = "Epoch Time"
    type = "bar"

    def build(self, metrics: dict[str, Any]) -> dict[str, Any]:
        epochs = metrics.get("epochs") or []
        return {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "x": [item.get("epoch") for item in epochs],
            "series": [{"label": "elapsed_s", "values": [item.get("elapsed_s") for item in epochs]}],
        }


class TileMetricGraphPlugin:
    id = "tile_metrics"
    title = "Tile Visibility Metrics"
    type = "line"

    def build(self, metrics: dict[str, Any]) -> dict[str, Any]:
        epochs = metrics.get("epochs") or []
        return {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "x": [item.get("epoch") for item in epochs],
            "series": [
                {"label": "tile_iou", "values": [((item.get("validation") or {}).get("tile_iou")) for item in epochs]},
                {"label": "tile_f1", "values": [((item.get("validation") or {}).get("tile_f1")) for item in epochs]},
            ],
        }


GRAPH_PLUGINS: list[TrainingGraphPlugin] = [
    LossGraphPlugin(),
    EpochTimeGraphPlugin(),
    TileMetricGraphPlugin(),
]


def build_training_graphs(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    graphs = []
    for plugin in GRAPH_PLUGINS:
        graph = plugin.build(metrics)
        values = [value for series in graph.get("series", []) for value in (series.get("values") or [])]
        if any(value is not None for value in values):
            graphs.append(graph)
    return graphs

