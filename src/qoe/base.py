from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence


@dataclass
class UiField:
    """Metadata describing a tunable QoE parameter that can be surfaced in the UI."""

    name: str
    label: str
    default: float
    min: float | None = None
    max: float | None = None
    step: float | None = None
    help: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "help": self.help,
        }


class QoEFunction(ABC):
    """Abstract QoE function with optional UI metadata."""

    id: str = "QoEFunction"
    label: str = "QoE function"
    equation: str = ""
    description: str = ""
    implementation: str | None = None  # Optional key to hint the client how to compute.

    @classmethod
    def ui_fields(cls) -> Sequence[UiField]:
        """Return tunable fields that should be exposed to the UI."""
        return []

    @classmethod
    def defaults(cls) -> Dict[str, float]:
        return {field.name: field.default for field in cls.ui_fields()}

    @classmethod
    def metadata(cls) -> Dict[str, Any]:
        """Return a JSON-serializable description for the client."""
        return {
            "id": getattr(cls, "id", None) or cls.__name__,
            "label": getattr(cls, "label", None) or cls.__name__,
            "equation": getattr(cls, "equation", "") or "",
            "description": getattr(cls, "description", "") or "",
            "implementation": getattr(cls, "implementation", None),
            "fields": [field.to_dict() for field in cls.ui_fields()],
            "defaults": cls.defaults(),
        }

    def apply_overrides(self, overrides: Mapping[str, Any] | None = None) -> None:
        """Update tunable parameters from a mapping (used by the UI)."""
        if not overrides:
            return
        for key, value in overrides.items():
            if not hasattr(self, key):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            setattr(self, key, numeric)

    @abstractmethod
    def compute_time_series(self, events: Mapping, viewport_tiles: Iterable[str] | None = None):
        """Return QoE timeline as ``[(time_s, score), ...]``."""
        raise NotImplementedError
