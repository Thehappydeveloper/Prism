from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time
from datetime import datetime
from uuid import uuid4
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, Iterable, List, Type
from urllib.parse import unquote
import inspect

from abr import baselines as abr_baselines
from abr.base import ABR
from app import PlaybackConfig, run_playback, save_config
from fov_predictors import baselines as fov_baselines
from fov_predictors.base import FovPredictor
from qoe import heuristics as qoe_functions
from qoe.base import QoEFunction
from qoe.heuristics import WeightedQoEHeuristic

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
WEB_DIST_ROOT = WEB_ROOT / "dist"
RESULTS_ROOT = ROOT / ".." / "results"
PROJECT_ROOT = ROOT.parent


def generate_run_id() -> str:
    now = datetime.now()
    suffix = uuid4().hex[:4]
    return f"run_{now.strftime('%y%m%d_%H%M%S')}_{suffix}"


class RunJob(threading.Thread):
    def __init__(self, config: PlaybackConfig):
        super().__init__(daemon=True)
        self.config = config
        self.run_id = config.run_id or generate_run_id()
        self.config.run_id = self.run_id
        self.run_dir = config.output_dir / self.run_id
        self.status = "queued"
        self.error: str | None = None
        self.started = datetime.now().isoformat()
        self.finished: str | None = None
        save_config(self.run_dir, self.config)

    def run(self):
        self.status = "running"
        try:
            run_playback(self.config)
            self.status = "finished"
        except Exception as exc:  # pylint: disable=broad-except
            self.status = "failed"
            self.error = str(exc)
        finally:
            self.finished = datetime.now().isoformat()
            status_file = self.run_dir / "status.json"
            status_file.parent.mkdir(parents=True, exist_ok=True)
            with status_file.open("w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "error": self.error,
            "started": self.started,
            "finished": self.finished,
            "events_path": f"/results/{self.run_id}/events.json",
            "log_path": f"/results/{self.run_id}/{self.run_id}.log",
        }


RUNS: Dict[str, RunJob] = {}
RUNS_LOCK = threading.Lock()


def _list_files(base: Path, patterns: Iterable[str]) -> List[str]:
    files: List[str] = []
    if not base.exists():
        return files
    for pattern in patterns:
        files.extend([str(p.relative_to(PROJECT_ROOT)) for p in base.rglob(pattern) if p.is_file()])
    return sorted(files)


def _collect_files(bases: Iterable[Path], patterns: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for base in bases:
        for path in _list_files(base, patterns):
            if path in seen:
                continue
            seen.add(path)
            out.append(path)
    return sorted(out)


def _list_subclasses(module, base_type: Type) -> List[str]:
    names = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if not issubclass(obj, base_type) or obj is base_type:
            continue
        if obj.__name__.startswith("_"):
            continue
        names.append(obj.__name__)
    return sorted(set(names))


def _abr_requirements() -> Dict[str, bool]:
    reqs: Dict[str, bool] = {}
    for _, obj in inspect.getmembers(abr_baselines, inspect.isclass):
        if obj.__module__ != abr_baselines.__name__:
            continue
        if not issubclass(obj, ABR) or obj is ABR:
            continue
        reqs[obj.__name__] = bool(getattr(obj, "abr_requires_predictor", False))
    return reqs


def _has_csv_under(paths: Iterable[Path]) -> bool:
    """Return True if any provided path contains at least one CSV file."""
    for base in paths:
        if not base.exists():
            continue
        if any(base.rglob("*.csv")):
            return True
    return False


def _ensure_traces_ready() -> Dict[str, bool]:
    """Download network and FoV traces if they are missing so the UI has data."""
    network_root = PROJECT_ROOT / "data" / "network_traces"
    fov_root = PROJECT_ROOT / "data" / "fov_traces"
    fov_dataset = fov_root / "6DoF-Video-FoV-Dataset"
    network_root.mkdir(parents=True, exist_ok=True)
    fov_root.mkdir(parents=True, exist_ok=True)

    status = {
        "network": _has_csv_under([network_root]),
        "fov": _has_csv_under([fov_root, fov_dataset]),
    }

    if not status["network"]:
        try:
            print("No network traces found; downloading via scripts/get_network_traces.py …")
            from scripts import get_network_traces

            get_network_traces.main()
            status["network"] = _has_csv_under([network_root])
        except Exception as exc:  # pylint: disable=broad-except
            print(f"⚠️ Failed to auto-fetch network traces: {exc}")

    if not status["fov"]:
        try:
            print("No FoV traces found; downloading via scripts/get_fov_traces.py …")
            from scripts import get_fov_traces

            get_fov_traces.main()
            status["fov"] = _has_csv_under([fov_root, fov_dataset])
        except Exception as exc:  # pylint: disable=broad-except
            print(f"⚠️ Failed to auto-fetch FoV traces: {exc}")

    return status


def _qoe_options() -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    for _, obj in inspect.getmembers(qoe_functions, inspect.isclass):
        if obj.__module__ != qoe_functions.__name__:
            continue
        if not issubclass(obj, QoEFunction) or obj is QoEFunction:
            continue
        try:
            meta = obj.metadata()
        except Exception as exc:  # pylint: disable=broad-except
            # Surface a minimal entry so the UI does not silently drop the model.
            meta = {"id": obj.__name__, "label": obj.__name__, "error": str(exc)}
        meta.setdefault("id", obj.__name__)
        meta.setdefault("label", obj.__name__)
        models.append(meta)
    return sorted(models, key=lambda m: m.get("label") or m.get("id"))


def _resolve_qoe_model(model_id: str | None, params: Dict | None) -> QoEFunction:
    """Instantiate a QoE model by id/name with optional parameter overrides."""
    fallback = WeightedQoEHeuristic
    target_cls = None
    if model_id:
        for _, obj in inspect.getmembers(qoe_functions, inspect.isclass):
            if obj.__module__ != qoe_functions.__name__:
                continue
            if not issubclass(obj, QoEFunction) or obj is QoEFunction:
                continue
            if getattr(obj, "id", None) == model_id or obj.__name__ == model_id:
                target_cls = obj
                break
    cls = target_cls or fallback
    try:
        return cls(overrides=params or {})
    except TypeError:
        # Older QoE classes may not accept overrides kwarg
        inst = cls()
        try:
            inst.apply_overrides(params or {})
        except Exception:  # pylint: disable=broad-except
            pass
        return inst
    except Exception:  # pylint: disable=broad-except
        return fallback(overrides=params or {})


def _discover_options() -> Dict[str, List[str] | Dict]:
    # Ensure canonical folders exist so newly downloaded traces have a home.
    network_root = PROJECT_ROOT / "data" / "network_traces"
    fov_root = PROJECT_ROOT / "data" / "fov_traces"
    network_root.mkdir(parents=True, exist_ok=True)
    fov_root.mkdir(parents=True, exist_ok=True)

    network_candidates = [
        network_root,
        PROJECT_ROOT / "data" / "network_traces",
    ]
    fov_candidates = [
        fov_root,
        PROJECT_ROOT / "data" / "6DoF-Video-FoV-Dataset",
        PROJECT_ROOT / "data" / "fov_traces",
    ]
    fov_traces = _collect_files(fov_candidates, ["*.csv"])
    network_traces = _collect_files(network_candidates, ["*.csv"])
    fov_root_abs = fov_root.resolve()
    # Avoid double-counting FoV CSVs as network traces when they live under fov_root.
    filtered_network = []
    fov_set = set(fov_traces)
    for p in network_traces:
        abs_path = (PROJECT_ROOT / p).resolve()
        if p in fov_set:
            continue
        if fov_root_abs in abs_path.parents:
            continue
        filtered_network.append(p)
    network_traces = filtered_network or network_traces

    return {
        "mpds": _list_files(PROJECT_ROOT, ["*.mpd"]),
        "network_traces": network_traces,
        "fov_traces": fov_traces,
        "network_root": str(network_root.relative_to(PROJECT_ROOT)),
        "fov_root": str(fov_root.relative_to(PROJECT_ROOT)),
        "abr_classes": _list_subclasses(abr_baselines, ABR),
        "fov_predictors": _list_subclasses(fov_baselines, FovPredictor),
        "abr_requires_predictor": _abr_requirements(),
        "qoe_functions": _qoe_options(),
    }


def _safe_path(path_str: str | Path) -> Path:
    path_obj = Path(path_str).expanduser()
    path = path_obj if path_obj.is_absolute() else (PROJECT_ROOT / path_obj)
    return path.resolve()


def _parse_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on", "y")
    return False


def _find_mpd_in_folder(folder: str | Path) -> Path | None:
    candidate = _safe_path(folder)
    if candidate.is_file() and candidate.suffix.lower() == ".mpd":
        return candidate
    if candidate.is_dir():
        matches = [p for p in candidate.iterdir() if p.is_file() and p.suffix.lower() == ".mpd"]
        if matches:
            return matches[0].resolve()
    return None


def load_existing_runs() -> List[Dict]:
    runs: List[Dict] = []
    if not RESULTS_ROOT.exists():
        return runs
    for run_dir in RESULTS_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        status_file = run_dir / "status.json"
        status = {}
        if status_file.exists():
            status = json.loads(status_file.read_text())
        runs.append(
            {
                "run_id": run_dir.name,
                "status": status.get("status", "unknown"),
                "started": status.get("started"),
                "finished": status.get("finished"),
                "events_path": f"/results/{run_dir.name}/events.json",
                "log_path": f"/results/{run_dir.name}/{run_dir.name}.log",
            }
        )
    return sorted(runs, key=lambda r: r["run_id"], reverse=True)


def _load_run(run_id: str) -> Dict | None:
    events_file = RESULTS_ROOT / run_id / "events.json"
    cfg_file = RESULTS_ROOT / run_id / "config.json"
    if not events_file.exists() or not cfg_file.exists():
        return None
    try:
        events = json.loads(events_file.read_text())
    except json.JSONDecodeError as exc:
        return {"run_id": run_id, "config": json.loads(cfg_file.read_text()), "events_error": str(exc)}
    config = json.loads(cfg_file.read_text())
    return {"run_id": run_id, "events": events, "config": config}


def _parse_quality_id(qid: str | None) -> int | None:
    if not qid:
        return None
    q = qid.lower().replace("quality_", "")
    if q.startswith("q"):
        q = q[1:]
    try:
        return int(q)
    except ValueError:
        return None


def _comp_max_levels(events: Dict) -> Dict[str, int]:
    comp_max: Dict[str, int] = {}
    for seg in events.get("enqueued_segments", []) or []:
        for tile in (seg.get("tiles") or {}).values():
            for comp_name, comp in (tile.get("components") or {}).items():
                lvl = _parse_quality_id(comp.get("quality_id")) or comp.get("rep_idx", 0)
                comp_max[comp_name] = max(comp_max.get(comp_name, 0), int(lvl))
    return comp_max


def _tile_quality(tile: Dict, comp_max: Dict[str, int]) -> float:
    total = 0.0
    w = 0.0
    for comp_name, comp in (tile.get("components") or {}).items():
        lvl = _parse_quality_id(comp.get("quality_id")) or comp.get("rep_idx", 0)
        max_lvl = comp_max.get(comp_name, max(lvl, 1))
        total += max(0.0, min(1.0, lvl / max_lvl))
        w += 1.0
    return total / w if w else 0.0


def _segment_fov_sets(events: Dict) -> tuple[list[set[str]], bool]:
    """Return (per-segment FoV tile sets, has_any_fov_info)."""
    fov_sets: list[set[str]] = []
    has_fov_info = False
    for seg in events.get("enqueued_segments", []) or []:
        hinted = seg.get("tiles_in_fov") or seg.get("viewport_tiles") or []
        seg_fov = {str(t) for t in hinted}
        if seg_fov:
            has_fov_info = True
        fov_sets.append(seg_fov)
    return fov_sets, has_fov_info


def _gather_diagnostics(runs: List[Dict], qoe_model: QoEFunction | None = None) -> Dict:
    qoe_model = qoe_model or WeightedQoEHeuristic()
    metrics = {
        "labels": {},  # run_id -> label
        "qoe": {},
        "fov_quality": {},
        "delta_fov": {},
        "stall": {},
        "component_bits": {},
        "bitrate_vs_quality": {},
        "buffer": {},
        "meta": {},  # run_id -> extra info (duration, fov availability, etc.)
    }
    def _network_class(path: str | None) -> str | None:
        if not path:
            return None
        low = str(path).lower()
        if "5g" in low:
            return "5G"
        if "4g" in low or "lte" in low:
            return "4G"
        return None

    for run in runs:
        if run.get("events_error"):
            # Skip runs with unreadable events; caller can inspect error if needed.
            metrics["meta"][run["run_id"]] = {"error": run["events_error"]}
            continue
        run_id = run.get("run_id", "unknown_run")
        abr = run["config"].get("abr", "unknown")
        predictor = run["config"].get("predictor") or "None"
        net_trace = run["config"].get("network_trace")
        net_class = _network_class(net_trace)
        short = run_id.replace("run_", "")
        short = short[-6:] if len(short) > 6 else short
        metrics["labels"][run_id] = f"{short} ({abr})"
        events = run["events"]
        comp_max = _comp_max_levels(events)
        fov_sets, has_fov_info = _segment_fov_sets(events)
        metrics["qoe"].setdefault(run_id, [])
        metrics["fov_quality"].setdefault(run_id, {"fov": [], "non": []})
        metrics["delta_fov"].setdefault(run_id, [])
        metrics["stall"].setdefault(run_id, {"duration": 0.0, "count": 0})
        metrics["component_bits"].setdefault(run_id, {})
        metrics["bitrate_vs_quality"].setdefault(run_id, [])
        metrics["buffer"].setdefault(run_id, {})
        metrics["meta"].setdefault(
            run_id,
            {
                "duration_s": 0.0,
                "fov_present": has_fov_info,
                "abr": abr,
                "predictor": predictor,
                "network_trace": net_trace,
                "network_class": net_class,
            },
        )

        qoe_series = qoe_model.compute_time_series(events)
        metrics["qoe"][run_id].extend([q for _, q in qoe_series])

        prev_tile_quality: Dict[str, float] = {}
        playback_seconds = 0.0

        for idx, seg in enumerate(events.get("enqueued_segments", []) or []):
            seg_start = seg.get("time_s", 0.0)
            seg_dur = seg.get("duration_s", 1.0)
            seg_bits = seg.get("size_bits", 0.0)
            playback_seconds += float(seg_dur or 0.0)
            tile_q = {}
            fov_sum = 0.0
            fov_count = 0
            seg_fov = fov_sets[idx] if idx < len(fov_sets) else set()
            seg_has_fov = bool(seg_fov)

            for tile_id, tile in (seg.get("tiles") or {}).items():
                tid = str(tile_id)
                tq = _tile_quality(tile, comp_max)
                tile_q[tid] = tq
                if has_fov_info and seg_has_fov:
                    in_fov = tid in seg_fov
                elif has_fov_info and not seg_has_fov:
                    # Trace has FoV info, but this segment is missing it; assume all tiles are FoV.
                    in_fov = True
                else:
                    # No FoV hints at all; treat everything as FoV for compatibility.
                    in_fov = True

                cls = "fov" if in_fov else "non"
                metrics["fov_quality"][run_id][cls].append(tq)
                if in_fov:
                    fov_sum += tq
                    fov_count += 1
                for comp_name, comp in (tile.get("components") or {}).items():
                    metrics["component_bits"][run_id][comp_name] = metrics["component_bits"][run_id].get(comp_name, 0.0) + comp.get("size_bits", 0.0)

            for tid, tq in tile_q.items():
                if has_fov_info and seg_has_fov and tid not in seg_fov:
                    prev_tile_quality.pop(tid, None)
                    continue
                if tid in prev_tile_quality:
                    metrics["delta_fov"][run_id].append(abs(tq - prev_tile_quality[tid]))
                prev_tile_quality[tid] = tq

            if fov_count == 0 and tile_q:
                # Missing FoV annotations for this segment; fall back to all tiles.
                fov_sum = sum(tile_q.values())
                fov_count = len(tile_q)
            avg_fov_q = (fov_sum / fov_count) if fov_count else 0.0
            bitrate_mbps = (seg_bits / seg_dur) / 1e6 if seg_dur else 0.0
            metrics["bitrate_vs_quality"][run_id].append((bitrate_mbps, avg_fov_q))

        total_stall = sum((s.get("duration_s", 0) for s in events.get("rebuffering_events") or []))
        metrics["stall"][run_id]["duration"] += total_stall
        metrics["stall"][run_id]["count"] += len(events.get("rebuffering_events") or [])
        metrics["meta"][run_id]["duration_s"] = playback_seconds

        buffer_samples = events.get("buffer_occupancy_s") or []
        if buffer_samples:
            vals = [float(b) for b in buffer_samples if isinstance(b, (float, int))]
            if vals:
                mean_buf = sum(vals) / len(vals)
                var_buf = sum((b - mean_buf) ** 2 for b in vals) / len(vals)
                metrics["buffer"][run_id] = {
                    "mean": mean_buf,
                    "std": var_buf ** 0.5,
                    "min": min(vals),
                    "max": max(vals),
                }

    return metrics


class RequestHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @staticmethod
    def _web_entrypoint() -> Path | None:
        """Return built SPA entrypoint when available."""
        dist_index = WEB_DIST_ROOT / "index.html"
        if dist_index.exists():
            return dist_index
        return None

    @staticmethod
    def _resolve_static_asset(path_only: str) -> Path:
        """
        Resolve /static/* path against built frontend assets only.
        """
        rel = path_only.replace("/static/", "", 1)
        dist_path = (WEB_DIST_ROOT / rel).resolve()
        try:
            dist_path.relative_to(WEB_DIST_ROOT.resolve())
        except ValueError:
            dist_path = WEB_DIST_ROOT / rel
        return dist_path

    @staticmethod
    def _is_web_asset(path: Path) -> bool:
        """Return True for files served from the built web app root."""
        try:
            path.resolve().relative_to(WEB_DIST_ROOT.resolve())
            return True
        except ValueError:
            return False

    def _serve_web_build_missing(self):
        message = (
            "Web UI build not found at src/web/dist.\n"
            "Build the frontend with:\n"
            "  cd src/web\n"
            "  npm install\n"
            "  npm run build\n"
        )
        data = message.encode("utf-8")
        self.send_response(503)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path_only = self.path.split("?", 1)[0]
        if path_only == "/":
            entrypoint = self._web_entrypoint()
            if entrypoint is None:
                return self._serve_web_build_missing()
            return self._serve_file(entrypoint)
        if path_only.startswith("/static/"):
            if path_only in (
                "/static/home.html",
                "/static/index.html",
                "/static/aggregate.html",
                "/static/compare.html",
            ):
                entrypoint = self._web_entrypoint()
                if entrypoint is None:
                    return self._serve_web_build_missing()
                return self._serve_file(entrypoint)
            return self._serve_file(self._resolve_static_asset(path_only))
        if path_only.startswith("/results/"):
            rel = path_only.replace("/results/", "")
            return self._serve_file(RESULTS_ROOT / rel)
        if path_only == "/api/options":
            return self._respond_json(_discover_options())
        if path_only == "/api/runs":
            return self._respond_json(self._list_runs())
        if path_only == "/api/runs/stream":
            return self._serve_runs_stream()
        if path_only.startswith("/api/runs/") and path_only.endswith("/events"):
            run_id = path_only.split("/")[3]
            events_file = RESULTS_ROOT / run_id / "events.json"
            if events_file.exists():
                return self._serve_file(events_file, content_type="application/json")
            return self._respond_json({"error": "not found"}, status=404)
        if path_only.startswith("/api/runs/") and path_only.endswith("/config"):
            run_id = path_only.split("/")[3]
            cfg_file = RESULTS_ROOT / run_id / "config.json"
            if cfg_file.exists():
                return self._serve_file(cfg_file, content_type="application/json")
            return self._respond_json({"error": "not found"}, status=404)
        if self.path.startswith("/api/trace?path="):
            try:
                return self._serve_trace(self.path.split("path=")[-1])
            except Exception as exc:  # pylint: disable=broad-except
                return self._respond_json({"error": str(exc)}, status=400)

        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/diagnostics":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length else b"{}"
            payload = json.loads(body.decode("utf-8") or "{}")
            run_ids = payload.get("run_ids") or []
            qoe_fn = payload.get("qoe_function")
            qoe_params = payload.get("qoe_params") or {}
            qoe_model = _resolve_qoe_model(qoe_fn, qoe_params)
            runs = []
            if run_ids:
                for rid in run_ids:
                    loaded = _load_run(rid)
                    if loaded:
                        runs.append(loaded)
            else:
                for rid in [p.name for p in RESULTS_ROOT.iterdir() if p.is_dir()]:
                    loaded = _load_run(rid)
                    if loaded:
                        runs.append(loaded)
            if not runs:
                return self._respond_json({"error": "no runs found"}, status=404)
            metrics = _gather_diagnostics(runs, qoe_model)
            return self._respond_json(metrics)
        if self.path == "/api/runs":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length else b"{}"
            payload = json.loads(body.decode("utf-8"))
            try:
                job = self._start_run(payload)
                return self._respond_json({"run_id": job.run_id, "status": job.status})
            except Exception as exc:  # pylint: disable=broad-except
                return self._respond_json({"error": str(exc)}, status=400)
        return self._respond_json({"error": "Unsupported endpoint"}, status=404)

    def do_DELETE(self):
        if self.path.startswith("/api/runs/"):
            run_id = self.path.split("/")[3]
            with RUNS_LOCK:
                RUNS.pop(run_id, None)
            run_dir = RESULTS_ROOT / run_id
            if run_dir.exists() and run_dir.is_dir():
                shutil.rmtree(run_dir)
            return self._respond_json({"status": "deleted", "run_id": run_id})
        return self._respond_json({"error": "Unsupported endpoint"}, status=404)

    # ---------- Helpers ----------

    def _start_run(self, payload: Dict) -> RunJob:
        run_id = payload.get("run_id") or generate_run_id()
        mpd_path = payload.get("mpd")
        if not mpd_path and payload.get("v3c_folder"):
            mpd_guess = _find_mpd_in_folder(payload["v3c_folder"])
            if mpd_guess:
                mpd_path = str(mpd_guess)
        if not mpd_path:
            raise ValueError("MPD path missing and no MPD found in provided V3C folder.")
        raw_buffer_window = payload.get("buffer_window_segments")
        buffer_window = 10 if raw_buffer_window is None else int(raw_buffer_window)
        config = PlaybackConfig(
            mpd=_safe_path(mpd_path),
            content_dir=_safe_path(payload["content_dir"]),
            network_trace=_safe_path(payload["network_trace"]),
            fov_trace=_safe_path(payload["fov_trace"]),
            abr=payload.get("abr") or "BandwidthAwareABR",
            predictor=payload.get("predictor"),
            rtt=float(payload.get("rtt") or 0.06),
            block_size_bits=int(payload.get("block_size_bits") or 250_000),
            enable_fov_upgrades=_parse_bool(payload.get("enable_fov_upgrades")),
            enable_mid_download_cancel=_parse_bool(payload.get("enable_mid_download_cancel")),
            buffer_window_segments=buffer_window,
            buffer_unbounded=_parse_bool(payload.get("buffer_unbounded")),
            show_progress=False,
            output_dir=RESULTS_ROOT,
            run_id=run_id,
        )
        job = RunJob(config)
        with RUNS_LOCK:
            RUNS[run_id] = job
        job.start()
        return job

    def _list_runs(self) -> List[Dict]:
        with RUNS_LOCK:
            in_memory = {job.run_id: job.to_dict() for job in RUNS.values()}
        disk_runs = {run["run_id"]: run for run in load_existing_runs()}
        combined = {**disk_runs, **in_memory}
        return sorted(combined.values(), key=lambda r: r["run_id"], reverse=True)

    def _serve_file(self, path: Path, content_type: str | None = None):
        if not path.exists():
            self.send_response(404)
            self.end_headers()
            return
        if content_type is None:
            content_type = "text/html" if path.suffix == ".html" else "text/plain"
            if path.suffix == ".css":
                content_type = "text/css"
            elif path.suffix == ".js":
                content_type = "application/javascript"
            elif path.suffix == ".json":
                content_type = "application/json"

        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        # Frontend assets change frequently during development; disable browser caching
        # to avoid serving stale JS/CSS bundles after a rebuild.
        if self._is_web_asset(path):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(data)

    def _respond_json(self, payload: Dict, status: int = 200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_trace(self, encoded_path: str):
        path_str = unquote(encoded_path.split("?")[0])
        trace_path = _safe_path(path_str)
        if not trace_path.exists():
            return self._respond_json({"error": "Trace not found"}, status=404)
        lines = trace_path.read_text().strip().splitlines()
        if not lines:
            return self._respond_json({"error": "Trace is empty"}, status=400)
        header = lines[0].split(",")
        if len(header) < 2:
            return self._respond_json({"error": "Unexpected trace format"}, status=400)
        samples = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                samples.append({"time_s": float(parts[0]), "bandwidth_mbps": float(parts[1])})
            except ValueError:
                continue
        return self._respond_json({"samples": samples})

    def _serve_runs_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        last_payload = None
        last_ping = time.monotonic()
        try:
            while True:
                runs = self._list_runs()
                payload = json.dumps(runs)
                if payload != last_payload:
                    self.wfile.write(f"event: runs\ndata: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    last_payload = payload
                    last_ping = time.monotonic()
                elif time.monotonic() - last_ping > 15:
                    self.wfile.write(b"event: ping\ndata: {}\n\n")
                    self.wfile.flush()
                    last_ping = time.monotonic()
                time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError):
            return


def run_server(host: str = "0.0.0.0", port: int = 8000):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    _ensure_traces_ready()
    server = ThreadingHTTPServer((host, port), RequestHandler)
    print(f"Serving UI at http://{host}:{port}")
    server.serve_forever()


def _snapshot_mtimes() -> Dict[str, float]:
    mtimes: Dict[str, float] = {}
    for path in PROJECT_ROOT.rglob("*.py"):
        try:
            mtimes[str(path)] = path.stat().st_mtime
        except FileNotFoundError:
            continue
    return mtimes


def _run_with_reload(host: str, port: int, interval: float = 1.0):
    print("Starting server with autoreload enabled (polling for code changes).")
    while True:
        baseline = _snapshot_mtimes()
        proc = Process(target=run_server, args=(host, port))
        proc.start()
        try:
            while proc.is_alive():
                time.sleep(interval)
                if _snapshot_mtimes() != baseline:
                    print("Code change detected; restarting server.")
                    proc.terminate()
                    proc.join(timeout=5)
                    break
        except KeyboardInterrupt:
            proc.terminate()
            proc.join(timeout=5)
            raise SystemExit(0)


def _ensure_web_dist() -> bool:
    """Build web frontend when missing so first-time startup stays simple."""
    dist_index = WEB_DIST_ROOT / "index.html"
    if dist_index.exists():
        return True

    npm = shutil.which("npm")
    if not npm:
        print("Warning: Web build missing (`src/web/dist/index.html`) and `npm` is not available.")
        print("   Install Node.js/npm, then run: cd src/web && npm install && npm run build")
        return False

    print("Web build missing; running frontend build automatically...")
    try:
        subprocess.run([npm, "install"], cwd=WEB_ROOT, check=True)
        subprocess.run([npm, "run", "build"], cwd=WEB_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Warning: Failed to auto-build frontend: {exc}")
        print("   You can retry manually with: cd src/web && npm install && npm run build")
        return False

    if not dist_index.exists():
        print("Warning: Frontend build command completed but `src/web/dist/index.html` was not produced.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Web UI server for the V3C player.")
    parser.add_argument("--host", default="0.0.0.0", help="Host/interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--reload", action="store_true", help="Watch code changes and auto-restart the server (dev only).")
    args = parser.parse_args()

    _ensure_web_dist()

    if args.reload:
        _run_with_reload(args.host, args.port)
    else:
        run_server(args.host, args.port)


if __name__ == "__main__":
    main()
