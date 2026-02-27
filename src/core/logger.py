from __future__ import annotations

import json
import logging
from pathlib import Path


_CURRENT_RUN_ROOT: str | None = None


class JsonLineFormatter(logging.Formatter):
    """Simple JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(log_dir: Path, run_id: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure a run-scoped logger. Adds a human-readable log file
    and a JSONL log file, plus console output. Does not touch the root logger.
    """
    global _CURRENT_RUN_ROOT
    _CURRENT_RUN_ROOT = run_id

    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(run_id)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    text_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_dir / f"{run_id}.log", mode="w")
    file_handler.setFormatter(text_formatter)
    logger.addHandler(file_handler)

    json_handler = logging.FileHandler(log_dir / f"{run_id}.ndjson", mode="w")
    json_handler.setFormatter(JsonLineFormatter())
    logger.addHandler(json_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(text_formatter)
    logger.addHandler(console_handler)

    return logger


def get_run_logger(name: str) -> logging.Logger:
    """Helper to get a logger with consistent naming."""
    if _CURRENT_RUN_ROOT:
        return logging.getLogger(_CURRENT_RUN_ROOT).getChild(name)
    return logging.getLogger(name)
