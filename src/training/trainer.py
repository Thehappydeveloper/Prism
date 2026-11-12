from __future__ import annotations

import json
import math
import random
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import TRAINING_MODELS_ROOT, TRAINING_RUNS_ROOT, new_id, now_iso, read_json, write_json
from .datasets import IndexedTraceDataset, TraceFrameCache, compute_stats
from .export import get_model_metadata, list_training_models
from .indexing import build_sample_index, read_index_rows
from .models import build_model, require_torch, resolve_device
from .plans import get_training_plan
from .graphs import build_training_graphs
from server_app.config import prism_settings

TRAINING_JOBS_LOCK = threading.Lock()
TRAINING_JOBS: dict[str, "TrainingJob"] = {}


def _status_path(run_dir: Path) -> Path:
    return run_dir / "status.json"


def _training_num_workers(plan_id: str | None) -> int:
    plan = get_training_plan(plan_id)
    training = (plan or {}).get("training") or {}
    return max(0, int(training.get("num_workers") or 0))


def _core_usage_for_plan(plan_id: str | None) -> int:
    return max(1, 1 + _training_num_workers(plan_id))


def _job_resource_info(plan_id: str | None) -> dict[str, Any]:
    num_workers = _training_num_workers(plan_id)
    core_usage = _core_usage_for_plan(plan_id)
    settings = prism_settings()
    require_single = bool(settings.get("training_require_single_worker_for_parallel", True))
    return {
        "num_workers": num_workers,
        "estimated_core_usage": core_usage,
        "parallel_eligible": (num_workers <= 1) if require_single else True,
    }


def _queued_wait_reason(job: "TrainingJob") -> str | None:
    if job.status != "queued":
        return None
    settings = prism_settings()
    running = [candidate for candidate in TRAINING_JOBS.values() if candidate.status in {"running", "stopping"}]
    max_parallel = max(1, int(settings.get("training_max_parallel_runs") or 1))
    core_limit = max(1, int(settings.get("training_cpu_core_limit") or 1))
    require_single = bool(settings.get("training_require_single_worker_for_parallel", True))
    resource = _job_resource_info(job.plan_id)
    if require_single and not resource["parallel_eligible"] and running:
        return "waiting because another run is active and this run uses multiple workers"
    if len(running) >= max_parallel:
        return "waiting for parallel run limit"
    used_cores = sum(_core_usage_for_plan(candidate.plan_id) for candidate in running)
    if used_cores + int(resource["estimated_core_usage"]) > core_limit:
        return "waiting for CPU core budget"
    return "queued"


def _dispatch_training_jobs_locked() -> None:
    settings = prism_settings()
    max_parallel = max(1, int(settings.get("training_max_parallel_runs") or 1))
    core_limit = max(1, int(settings.get("training_cpu_core_limit") or 1))
    require_single = bool(settings.get("training_require_single_worker_for_parallel", True))

    running = [job for job in TRAINING_JOBS.values() if job.status in {"running", "stopping"}]
    used_cores = sum(_core_usage_for_plan(job.plan_id) for job in running)
    for job in sorted(TRAINING_JOBS.values(), key=lambda item: str(item.run_id)):
        if job.status != "queued":
            continue
        resource = _job_resource_info(job.plan_id)
        if require_single and not resource["parallel_eligible"] and running:
            reason = "waiting because another run is active and this run uses multiple workers"
            job.progress["wait_reason"] = reason
            if not job.log_lines or reason not in job.log_lines[-1]:
                job._log(reason)
            job._persist_status()
            continue
        if len(running) >= max_parallel:
            reason = "waiting for parallel run limit"
            job.progress["wait_reason"] = reason
            if not job.log_lines or reason not in job.log_lines[-1]:
                job._log(reason)
            job._persist_status()
            continue
        if used_cores + int(resource["estimated_core_usage"]) > core_limit:
            reason = "waiting for CPU core budget"
            job.progress["wait_reason"] = reason
            if not job.log_lines or reason not in job.log_lines[-1]:
                job._log(reason)
            job._persist_status()
            continue
        job.progress["wait_reason"] = ""
        job._log("dispatched")
        job._persist_status()
        job.start()
        running.append(job)
        used_cores += int(resource["estimated_core_usage"])


def _normalize_batch(torch, x, y, stats: dict[str, Any], device):
    x = x.float().to(device)
    y = y.float().to(device)
    x_mean = torch.tensor(stats["x_mean"], dtype=torch.float32, device=device).view(1, *x.shape[1:])
    x_std = torch.tensor(stats["x_std"], dtype=torch.float32, device=device).view(1, *x.shape[1:])
    y_mean = torch.tensor(stats["y_mean"], dtype=torch.float32, device=device).view(1, -1)
    y_std = torch.tensor(stats["y_std"], dtype=torch.float32, device=device).view(1, -1)
    return (x - x_mean) / x_std, (y.view(y.shape[0], -1) - y_mean) / y_std


def _denormalize_y(torch, value, stats: dict[str, Any], device):
    y_mean = torch.tensor(stats["y_mean"], dtype=torch.float32, device=device).view(1, -1)
    y_std = torch.tensor(stats["y_std"], dtype=torch.float32, device=device).view(1, -1)
    return value * y_std + y_mean


def _resource_snapshot(torch) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    try:
        import resource

        snapshot["max_rss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        if torch.cuda.is_available():
            snapshot["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated())
            snapshot["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved())
    except Exception:  # pylint: disable=broad-except
        pass
    return snapshot


def _device_label(torch, device) -> str:
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if device_type == "cuda" and torch.cuda.is_available():
        index = getattr(device, "index", None)
        if index is None:
            index = torch.cuda.current_device()
        return f"cuda:{index} {torch.cuda.get_device_name(index)}"
    return str(device)


def _cuda_synchronize(torch, device) -> None:
    try:
        if getattr(device, "type", "") == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:  # pylint: disable=broad-except
        pass


def _benchmark_inference(torch, model, input_shape: list[int], batch_size: int, device) -> dict[str, float]:
    runs = 50 if batch_size <= 512 else 25
    warmup = 5
    model.eval()
    sample = torch.zeros((batch_size, *input_shape), dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        _cuda_synchronize(torch, device)
        started = time.perf_counter()
        for _ in range(runs):
            model(sample)
        _cuda_synchronize(torch, device)
    elapsed = max(time.perf_counter() - started, 1e-9)
    batch_ms = elapsed * 1000.0 / runs
    return {
        "inference_ms_per_batch": batch_ms,
        "inference_ms_per_sample": batch_ms / max(batch_size, 1),
    }


def _sampler_for_plan(torch, train_dataset: IndexedTraceDataset, plan: dict[str, Any], generator=None):
    strategy = str(plan.get("mixing_strategy") or "proportional").lower()
    if strategy == "proportional":
        return None
    rows = train_dataset.rows
    counts: dict[str, int] = {}
    for row in rows:
        dataset_id = str(row.get("dataset_id") or "")
        counts[dataset_id] = counts.get(dataset_id, 0) + 1
    manual = {str(k): float(v) for k, v in dict(plan.get("manual_dataset_weights") or {}).items()}
    weights = []
    for row in rows:
        dataset_id = str(row.get("dataset_id") or "")
        if strategy == "manual":
            base = max(manual.get(dataset_id, 1.0), 0.0)
            weights.append(base / max(counts.get(dataset_id, 1), 1))
        elif strategy == "equal_dataset":
            weights.append(1.0 / max(counts.get(dataset_id, 1), 1))
        else:
            weights.append(1.0)
    if not any(weight > 0 for weight in weights):
        return None
    from torch.utils.data import WeightedRandomSampler

    return WeightedRandomSampler(weights, num_samples=len(rows), replacement=True, generator=generator)


def _evaluate_loader_metrics(torch, model, loader, stats: dict[str, Any], device, family: str) -> dict[str, float] | None:
    if loader is None:
        return None
    model.eval()
    mae_values = []
    mse_values = []
    smape_values = []
    pos_values = []
    quat_angle_values = []
    tile_iou_values = []
    tile_precision_values = []
    tile_recall_values = []
    tile_f1_values = []
    with torch.no_grad():
        for x, y in loader:
            x_norm, y_norm = _normalize_batch(torch, x, y, stats, device)
            pred = model(x_norm)
            pred_raw = _denormalize_y(torch, pred, stats, device)
            target_raw = _denormalize_y(torch, y_norm, stats, device)
            diff = pred_raw - target_raw
            mae_values.append(torch.mean(torch.abs(diff)).item())
            mse_values.append(torch.mean(diff * diff).item())
            denom = torch.clamp((torch.abs(pred_raw) + torch.abs(target_raw)) / 2.0, min=1e-6)
            smape_values.append(torch.mean(torch.abs(diff) / denom).item())
            if family == "fov" and stats.get("target_kind") == "tile_visibility":
                pred_bin = pred_raw >= 0.5
                target_bin = target_raw >= 0.5
                intersection = torch.logical_and(pred_bin, target_bin).sum(dim=1).float()
                union = torch.logical_or(pred_bin, target_bin).sum(dim=1).float().clamp(min=1.0)
                precision_den = pred_bin.sum(dim=1).float().clamp(min=1.0)
                recall_den = target_bin.sum(dim=1).float().clamp(min=1.0)
                precision = intersection / precision_den
                recall = intersection / recall_den
                f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-6)
                tile_iou_values.append(torch.mean(intersection / union).item())
                tile_precision_values.append(torch.mean(precision).item())
                tile_recall_values.append(torch.mean(recall).item())
                tile_f1_values.append(torch.mean(f1).item())
            elif family == "fov" and pred_raw.shape[1] >= 7:
                pos_values.append(torch.mean(torch.abs(diff[:, :3])).item())
                pred_quat = torch.nn.functional.normalize(pred_raw[:, 3:7], dim=1)
                target_quat = torch.nn.functional.normalize(target_raw[:, 3:7], dim=1)
                dot = torch.abs(torch.sum(pred_quat * target_quat, dim=1)).clamp(max=1.0)
                quat_angle_values.append(torch.mean(2.0 * torch.acos(dot) * 180.0 / np.pi).item())
    if not mae_values:
        return None
    out = {
        "mae": float(np.mean(mae_values)),
        "mse": float(np.mean(mse_values)),
        "smape": float(np.mean(smape_values)),
    }
    if pos_values:
        out["position_mae"] = float(np.mean(pos_values))
        out["quaternion_angular_error_deg"] = float(np.mean(quat_angle_values))
    if tile_iou_values:
        out["tile_iou"] = float(np.mean(tile_iou_values))
        out["tile_precision"] = float(np.mean(tile_precision_values))
        out["tile_recall"] = float(np.mean(tile_recall_values))
        out["tile_f1"] = float(np.mean(tile_f1_values))
    return out


class TrainingJob(threading.Thread):
    def __init__(self, plan_id: str, resume_run_id: str | None = None):
        super().__init__(daemon=True)
        self.plan_id = plan_id
        self.run_id = resume_run_id or new_id("train")
        self.run_dir = TRAINING_RUNS_ROOT / self.run_id
        self.plan_name: str | None = None
        self.status = "queued"
        self.error: str | None = None
        self.traceback: str | None = None
        self.started: str | None = None
        self.finished: str | None = None
        resource = _job_resource_info(self.plan_id)
        self.progress = {
            "epoch": 0,
            "epochs": 0,
            "batch": 0,
            "batches": 0,
            "phase": "queued",
            "current_loss": None,
            "latest_train_loss": None,
            "latest_validation_mae": None,
            "learning_rate": None,
            "elapsed_s": 0.0,
            "wait_reason": "queued",
            "device": "",
            "device_name": "",
            "data_time_s": None,
            "compute_time_s": None,
            "batch_time_s": None,
            "avg_batch_time_s": None,
            "samples_per_s": None,
            "eta_s": None,
            "inference_ms_per_batch": None,
            "inference_ms_per_sample": None,
            **resource,
        }
        self.split_counts: dict[str, int] | None = None
        self.dataset_counts: list[dict[str, Any]] | None = None
        self.log_lines: list[str] = []
        self.stop_requested = False
        self.pause_requested = False
        self.resume_run_id = resume_run_id
        self._log("queued")
        self._persist_status()

    def _log(self, message: str) -> None:
        line = f"{now_iso()} {message}"
        self.log_lines.append(line)
        self.log_lines = self.log_lines[-200:]
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with (self.run_dir / "training.log").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass

    def _persist_status(self) -> None:
        write_json(
            _status_path(self.run_dir),
            {
                "run_id": self.run_id,
                "plan_id": self.plan_id,
                "plan_name": self.plan_name,
                "status": self.status,
                "error": self.error,
                "traceback": self.traceback,
                "started": self.started,
                "finished": self.finished,
                "progress": self.progress,
                "split_counts": self.split_counts,
                "dataset_counts": self.dataset_counts,
                "resources": _job_resource_info(self.plan_id),
                "log": list(self.log_lines[-80:]),
            },
        )

    def _index_progress(self, event: dict[str, Any]) -> None:
        current = int(event.get("current") or 0)
        total = int(event.get("total") or 0)
        self.progress.update(
            {
                "phase": "indexing",
                "index_current": current,
                "index_total": total,
                "index_dataset": event.get("dataset_name") or event.get("dataset_id") or "",
                "index_trace": Path(str(event.get("trace") or "")).name if event.get("trace") else "",
                "index_file_samples": int(event.get("file_samples") or 0),
                "index_dataset_samples": int(event.get("dataset_samples") or 0),
            }
        )
        message = str(event.get("message") or "").strip()
        if message and (current == 0 or current == total or current % 10 == 0):
            self._log(f"indexing {current}/{total}: {message}")
        self._persist_status()

    def request_stop(self) -> None:
        self.stop_requested = True
        if self.status in {"queued", "running"}:
            self._log("stop requested")
            self.status = "stopping"
            self.progress["phase"] = "stopping"
            self._persist_status()

    def request_pause(self) -> None:
        self.pause_requested = True
        self._log("pause requested")

    def run(self) -> None:
        if self.stop_requested:
            self.status = "stopped"
            self.finished = now_iso()
            self._log("stopped before start")
            self._persist_status()
            return
        self.status = "running"
        self.started = now_iso()
        self.progress["phase"] = "starting"
        self.progress["wait_reason"] = ""
        self._log("started")
        self._persist_status()
        try:
            self._run_training()
            if self.status == "running":
                self.status = "finished"
        except Exception as exc:  # pylint: disable=broad-except
            self.status = "failed"
            self.error = str(exc)
            self.traceback = traceback.format_exc()
            self._log(f"failed: {self.error}")
        finally:
            self.finished = now_iso()
            self._log(f"finished with status {self.status}")
            self._persist_status()
            with TRAINING_JOBS_LOCK:
                _dispatch_training_jobs_locked()

    def _run_training(self) -> None:
        torch, nn = require_torch()
        from torch.utils.data import DataLoader

        plan = get_training_plan(self.plan_id)
        if not plan:
            raise ValueError("Training plan was not found.")
        self.plan_name = str(plan.get("name") or self.plan_id)
        random.seed(int(plan.get("seed") or 0))
        np.random.seed(int(plan.get("seed") or 0))
        torch.manual_seed(int(plan.get("seed") or 0))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.run_dir / "plan.json", plan)

        self.progress["phase"] = "indexing"
        self._log("indexing samples")
        self._persist_status()
        index_manifest = build_sample_index(plan, progress_callback=self._index_progress)
        self.split_counts = dict(index_manifest.get("counts") or {})
        self.dataset_counts = list(index_manifest.get("per_dataset") or [])
        self.progress["split_counts"] = dict(self.split_counts)
        self._log(
            "index ready: "
            f"train={self.split_counts.get('train', 0)} "
            f"validation={self.split_counts.get('validation', 0)} "
            f"test={self.split_counts.get('test', 0)}"
        )
        write_json(self.run_dir / "sample_index.json", index_manifest)
        write_json(self.run_dir / "datasets.json", index_manifest.get("datasets") or [])
        self._persist_status()
        if self.stop_requested:
            self.status = "stopped"
            self.progress["phase"] = "stopped"
            self._log("stopped after indexing")
            return

        training = plan.get("training") or {}
        trace_cache_entries = max(1, int(training.get("trace_cache_entries") or 256))
        cache = TraceFrameCache(max_entries=trace_cache_entries)
        self.progress.update({"phase": "loading", "trace_cache_entries": trace_cache_entries})
        self._log("loading indexed datasets")
        self._persist_status()
        train_dataset = IndexedTraceDataset(index_manifest["split_files"]["train"], plan["family"], cache=cache)
        val_dataset = IndexedTraceDataset(index_manifest["split_files"]["validation"], plan["family"], cache=cache)
        test_dataset = IndexedTraceDataset(index_manifest["split_files"]["test"], plan["family"], cache=cache)
        self.progress["phase"] = "preprocessing"
        self._log("computing preprocessing statistics from up to 20000 training samples")
        self._persist_status()
        stats = compute_stats(train_dataset)
        write_json(self.run_dir / "preprocessing.json", stats)
        if self.stop_requested:
            self.status = "stopped"
            self.progress["phase"] = "stopped"
            self._log("stopped after preprocessing")
            return

        device = resolve_device(str(training.get("device") or "auto"))
        device_name = _device_label(torch, device)
        self.progress.update({"phase": "building_model", "device": str(device), "device_name": device_name})
        self._log(f"building model on device {device_name}")
        self._persist_status()
        model = build_model(plan.get("model") or {}, stats["input_shape"], int(stats["output_size"])).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(training.get("learning_rate") or 0.001))
        scheduler = None
        scheduler_name = str(training.get("scheduler") or "none").lower()
        if scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(training.get("scheduler_step_size") or 10)), gamma=float(training.get("scheduler_gamma") or 0.5))
        elif scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(1, int(training.get("scheduler_patience") or 3)), factor=float(training.get("scheduler_gamma") or 0.5))
        criterion = nn.MSELoss()
        batch_size = max(1, int(training.get("batch_size") or 64))
        num_workers = max(0, int(training.get("num_workers") or 0))
        pin_memory = bool(str(device).startswith("cuda"))
        seed = int(plan.get("seed") or 0)
        epoch_generator = torch.Generator()
        epoch_generator.manual_seed(seed)
        sampler = _sampler_for_plan(torch, train_dataset, plan, generator=epoch_generator)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, generator=epoch_generator if sampler is None else None, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory) if len(val_dataset) else None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory) if len(test_dataset) else None

        epochs = max(1, int(training.get("epochs") or 1))
        checkpoint_frequency = max(1, int(training.get("checkpoint_frequency") or 1))
        metrics = {"epochs": [], "test": None, "resources": []}
        best_val = math.inf
        best_checkpoint = None
        patience = max(0, int(training.get("early_stopping_patience") or 0))
        stale_epochs = 0
        start_epoch = 1
        resume_batch = 1
        paused_checkpoint = self.run_dir / "checkpoints" / "paused.pt"
        if self.resume_run_id and paused_checkpoint.exists():
            checkpoint = self._load_checkpoint(torch, paused_checkpoint, device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = max(1, int(checkpoint.get("epoch") or 1))
            resume_batch = max(1, int(checkpoint.get("batch") or 1))
            metrics = read_json(self.run_dir / "metrics.json", metrics)
            best_val = min(
                [float(item.get("validation_mae") if item.get("validation_mae") is not None else item.get("train_loss", math.inf)) for item in metrics.get("epochs", [])] or [math.inf]
            )
        inference_stats = _benchmark_inference(torch, model, list(stats["input_shape"]), batch_size, device)
        self.progress.update({"epochs": epochs, "batches": len(train_loader), **inference_stats})
        self._log(
            "training configured: "
            f"epochs={epochs} batches_per_epoch={len(train_loader)} batch_size={batch_size} "
            f"trace_cache_entries={trace_cache_entries} "
            f"inference_ms_per_batch={inference_stats['inference_ms_per_batch']:.3f}"
        )
        self._persist_status()
        use_amp = bool(training.get("mixed_precision")) and str(device).startswith("cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if hasattr(torch.cuda, "amp") else None

        for epoch in range(start_epoch, epochs + 1):
            if self.stop_requested:
                self.status = "stopped"
                return
            epoch_generator = torch.Generator()
            epoch_generator.manual_seed(seed + epoch)
            sampler = _sampler_for_plan(torch, train_dataset, plan, generator=epoch_generator)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=sampler is None,
                sampler=sampler,
                generator=epoch_generator if sampler is None else None,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            model.train()
            losses = []
            started = time.perf_counter()
            epoch_timer_started = started
            fetch_started = time.perf_counter()
            self.progress.update({"epoch": epoch, "phase": "train", "batch": 0, "current_loss": None, "epoch_started_at": datetime.now().isoformat()})
            self._log(f"epoch {epoch}/{epochs} train start")
            self._persist_status()
            for batch_idx, (x, y) in enumerate(train_loader, start=1):
                data_ready_at = time.perf_counter()
                data_time_s = data_ready_at - fetch_started
                if self.stop_requested:
                    self.status = "stopped"
                    self.progress["phase"] = "stopped"
                    self._log(f"stopped at epoch {epoch} batch {batch_idx}")
                    self._persist_status()
                    return
                if epoch == start_epoch and batch_idx < resume_batch:
                    continue
                if self.pause_requested:
                    self._save_checkpoint(model, optimizer, epoch, batch_idx, stats, plan, index_manifest, "paused.pt")
                    self.status = "paused"
                    return
                compute_started = time.perf_counter()
                x_norm, y_norm = _normalize_batch(torch, x, y, stats, device)
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    with torch.cuda.amp.autocast():
                        pred = model(x_norm)
                        loss = criterion(pred, y_norm)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(x_norm)
                    loss = criterion(pred, y_norm)
                    loss.backward()
                    optimizer.step()
                _cuda_synchronize(torch, device)
                compute_time_s = time.perf_counter() - compute_started
                batch_time_s = data_time_s + compute_time_s
                batch_loss = float(loss.item())
                losses.append(batch_loss)
                elapsed_s = time.perf_counter() - started
                completed_batches = max(0, (epoch - 1) * len(train_loader) + batch_idx)
                total_batches = max(1, epochs * len(train_loader))
                avg_batch_time_s = (time.perf_counter() - epoch_timer_started) / max(batch_idx, 1)
                eta_s = max(0.0, total_batches - completed_batches) * avg_batch_time_s
                self.progress.update(
                    {
                        "batch": batch_idx,
                        "current_loss": batch_loss,
                        "elapsed_s": elapsed_s,
                        "data_time_s": data_time_s,
                        "compute_time_s": compute_time_s,
                        "batch_time_s": batch_time_s,
                        "avg_batch_time_s": avg_batch_time_s,
                        "samples_per_s": batch_size / max(batch_time_s, 1e-9),
                        "eta_s": eta_s,
                    }
                )
                if batch_idx == 1 or batch_idx % 20 == 0:
                    self._log(
                        f"epoch {epoch}/{epochs} batch {batch_idx}/{len(train_loader)} "
                        f"loss={batch_loss:.6g} data_s={data_time_s:.3f} compute_s={compute_time_s:.3f} eta_s={eta_s:.0f}"
                    )
                    self._persist_status()
                fetch_started = time.perf_counter()
            if self.stop_requested:
                self.status = "stopped"
                self.progress["phase"] = "stopped"
                self._log(f"stopped after epoch {epoch} train")
                self._persist_status()
                return
            self.progress["phase"] = "validation"
            self._log(f"epoch {epoch}/{epochs} validation start")
            self._persist_status()
            val_metrics = self._evaluate_loader(torch, model, val_loader, stats, device, plan["family"]) if val_loader is not None else None
            val_loss = val_metrics.get("mae") if isinstance(val_metrics, dict) else None
            train_loss = float(np.mean(losses)) if losses else 0.0
            elapsed = time.perf_counter() - started
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_mae": val_loss,
                "validation": val_metrics,
                "learning_rate": float(optimizer.param_groups[0].get("lr", 0.0)),
                "elapsed_s": elapsed,
                "finished_at": datetime.now().isoformat(),
            }
            metrics["epochs"].append(epoch_metrics)
            metrics["resources"].append({"epoch": epoch, **_resource_snapshot(torch)})
            self._log(
                f"epoch {epoch}/{epochs} complete: "
                f"train_loss={train_loss:.6g} "
                f"validation_mae={val_loss if val_loss is not None else '-'} "
                f"elapsed_s={elapsed:.1f}"
            )
            self.progress.update(
                {
                    "phase": "train",
                    "latest_train_loss": train_loss,
                    "latest_validation_mae": val_loss,
                    "learning_rate": float(optimizer.param_groups[0].get("lr", 0.0)),
                    "elapsed_s": elapsed,
                }
            )
            write_json(self.run_dir / "metrics.json", metrics)
            if epoch % checkpoint_frequency == 0:
                self._log(f"saving checkpoint for epoch {epoch}")
                self._save_checkpoint(model, optimizer, epoch, 0, stats, plan, index_manifest, f"epoch_{epoch}.pt")
            score = val_loss if val_loss is not None else train_loss
            if score < best_val:
                best_val = score
                stale_epochs = 0
                self._log(f"new best checkpoint at epoch {epoch} score={score:.6g}")
                best_checkpoint = self._save_checkpoint(model, optimizer, epoch, 0, stats, plan, index_manifest, "best.pt")
            else:
                stale_epochs += 1
            if scheduler is not None:
                if scheduler_name == "plateau":
                    scheduler.step(score)
                else:
                    scheduler.step()
            self._persist_status()
            if patience > 0 and stale_epochs >= patience:
                break

        if self.stop_requested:
            self.status = "stopped"
            self.progress["phase"] = "stopped"
            self._log("stopped before test evaluation")
            self._persist_status()
            return
        self.progress["phase"] = "test"
        self._log("test evaluation start")
        self._persist_status()
        test_metrics = self._evaluate_loader(torch, model, test_loader, stats, device, plan["family"]) if test_loader is not None else None
        metrics["test"] = {**(test_metrics or {}), "evaluated_at": datetime.now().isoformat()}
        write_json(self.run_dir / "metrics.json", metrics)
        self._log(f"test evaluation complete: mae={metrics['test'].get('mae', '-')}")
        model_id = new_id("model")
        model_dir = TRAINING_MODELS_ROOT / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(best_checkpoint or self.run_dir / "checkpoints" / "best.pt")
        if checkpoint_path.exists():
            import shutil

            shutil.copy2(checkpoint_path, model_dir / "model.pt")
            self._log(f"copied best checkpoint to model {model_id}")
        metadata = {
            "id": model_id,
            "name": self.plan_name or model_id,
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "plan_name": self.plan_name,
            "family": plan["family"],
            "model": plan.get("model") or {},
            "history": plan.get("history") or {},
            "training": training,
            "stats": stats,
            "index": index_manifest,
            "metrics": metrics,
            "checkpoint": str(model_dir / "model.pt"),
            "created_at": now_iso(),
            "exported": False,
        }
        write_json(model_dir / "metadata.json", metadata)
        write_json(self.run_dir / "model.json", metadata)
        self._log(f"model metadata saved: {model_id}")

    def _save_checkpoint(self, model, optimizer, epoch: int, batch: int, stats: dict[str, Any], plan: dict[str, Any], index_manifest: dict[str, Any], name: str) -> str:
        torch, _ = require_torch()
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / name
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "batch": batch,
                "stats": stats,
                "plan": plan,
                "index": index_manifest,
                "rng": {
                    "torch": torch.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                },
            },
            path,
        )
        return str(path)

    def _load_checkpoint(self, torch, path: Path, device):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)

    def _evaluate_loader(self, torch, model, loader, stats: dict[str, Any], device, family: str) -> dict[str, float] | None:
        return _evaluate_loader_metrics(torch, model, loader, stats, device, family)


def start_training_run(plan_id: str, resume_run_id: str | None = None) -> dict[str, Any]:
    if resume_run_id:
        status = job_status(resume_run_id)
        if status.get("status") != "paused":
            raise ValueError("Only paused training runs can be resumed.")
    job = TrainingJob(plan_id, resume_run_id=resume_run_id)
    with TRAINING_JOBS_LOCK:
        TRAINING_JOBS[job.run_id] = job
        _dispatch_training_jobs_locked()
    return job_status(job.run_id)


def _job_status_from_job(job: TrainingJob) -> dict[str, Any]:
    return {
        "run_id": job.run_id,
        "plan_id": job.plan_id,
        "plan_name": job.plan_name,
        "status": job.status,
        "error": job.error,
        "traceback": job.traceback,
        "started": job.started,
        "finished": job.finished,
        "progress": job.progress,
        "split_counts": job.split_counts,
        "dataset_counts": job.dataset_counts,
        "resources": _job_resource_info(job.plan_id),
        "log": list(job.log_lines[-80:]),
    }


def _read_training_log_tail(run_dir: Path, limit: int = 80) -> list[str]:
    log_path = run_dir / "training.log"
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-limit:]


def _status_from_run_dir(run_dir: Path) -> dict[str, Any] | None:
    plan = read_json(run_dir / "plan.json", None)
    status = read_json(run_dir / "status.json", None)
    if isinstance(status, dict):
        status.setdefault("run_id", run_dir.name)
        if isinstance(plan, dict) and plan.get("name"):
            status.setdefault("plan_name", str(plan["name"]))
        status.setdefault("resources", _job_resource_info(status.get("plan_id")))
        if not status.get("log"):
            status["log"] = _read_training_log_tail(run_dir)
        return status

    metrics = read_json(run_dir / "metrics.json", None)
    model = read_json(run_dir / "model.json", None)
    if not any(isinstance(value, dict) for value in (plan, metrics, model)) and not (run_dir / "training.log").exists():
        return None

    plan_id = None
    if isinstance(plan, dict):
        plan_id = plan.get("id")
    if not plan_id and isinstance(model, dict):
        plan_id = model.get("plan_id")

    progress: dict[str, Any] = {"phase": "recovered"}
    split_counts = None
    dataset_counts = None
    index_manifest = None
    if isinstance(model, dict):
        index_manifest = model.get("index")
    if not isinstance(index_manifest, dict):
        index_manifest = read_json(run_dir / "sample_index.json", None)
    if isinstance(index_manifest, dict):
        split_counts = index_manifest.get("counts")
        dataset_counts = index_manifest.get("per_dataset")
        if isinstance(split_counts, dict):
            progress["split_counts"] = dict(split_counts)

    if isinstance(metrics, dict):
        epochs = metrics.get("epochs") or []
        if isinstance(epochs, list) and epochs:
            latest = epochs[-1]
            progress.update(
                {
                    "epoch": latest.get("epoch") if isinstance(latest, dict) else len(epochs),
                    "epochs": len(epochs),
                    "latest_train_loss": latest.get("train_loss") if isinstance(latest, dict) else None,
                    "latest_validation_mae": latest.get("validation_mae") if isinstance(latest, dict) else None,
                    "elapsed_s": latest.get("elapsed_s") if isinstance(latest, dict) else None,
                }
            )

    status_value = "finished" if isinstance(model, dict) else "recovered"
    if isinstance(metrics, dict) and metrics.get("test"):
        status_value = "finished"
    elif (run_dir / "checkpoints" / "paused.pt").exists():
        status_value = "paused"

    return {
        "run_id": run_dir.name,
        "plan_id": plan_id,
        "plan_name": str(plan.get("name")) if isinstance(plan, dict) and plan.get("name") else None,
        "status": status_value,
        "error": None,
        "traceback": None,
        "started": None,
        "finished": model.get("created_at") if isinstance(model, dict) else None,
        "progress": progress,
        "split_counts": split_counts if isinstance(split_counts, dict) else None,
        "dataset_counts": dataset_counts if isinstance(dataset_counts, list) else None,
        "resources": _job_resource_info(plan_id),
        "log": _read_training_log_tail(run_dir),
    }


def job_status(run_id: str) -> dict[str, Any]:
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(run_id)
    if job:
        return _job_status_from_job(job)
    status = _status_from_run_dir(TRAINING_RUNS_ROOT / run_id)
    if status:
        return status
    return {"run_id": run_id, "status": "unknown"}


def list_training_runs() -> list[dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    if TRAINING_RUNS_ROOT.exists():
        for path in TRAINING_RUNS_ROOT.iterdir():
            if path.is_dir():
                status = _status_from_run_dir(path)
                if status:
                    runs[str(status.get("run_id") or path.name)] = status
    with TRAINING_JOBS_LOCK:
        for run_id, job in TRAINING_JOBS.items():
            runs[run_id] = _job_status_from_job(job)
    return sorted(runs.values(), key=lambda item: str(item.get("run_id") or ""), reverse=True)


def control_training_run(run_id: str, action: str) -> dict[str, Any]:
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(run_id)
        if not job:
            raise ValueError("Training run is not active.")
        if action == "pause":
            job.request_pause()
        elif action == "stop":
            job.request_stop()
            if not job.is_alive():
                job.status = "stopped"
                job.finished = now_iso()
                job.progress["phase"] = "stopped"
                job._persist_status()
                _dispatch_training_jobs_locked()
        else:
            raise ValueError("Unsupported training run action.")
    return job_status(run_id)


def evaluate_model(model_id: str, split: str = "test") -> dict[str, Any]:
    torch, _ = require_torch()
    metadata = get_model_metadata(model_id)
    index = metadata.get("index") or {}
    split_files = index.get("split_files") or {}
    index_file = split_files.get(split) or split_files.get("test")
    if not index_file:
        raise ValueError("Requested split is not available for this model.")
    from torch.utils.data import DataLoader

    device = resolve_device(str((metadata.get("training") or {}).get("device") or "auto"))
    dataset = IndexedTraceDataset(index_file, str(metadata.get("family") or "bandwidth"))
    loader = DataLoader(dataset, batch_size=max(1, int((metadata.get("training") or {}).get("batch_size") or 64)), shuffle=False)
    model = build_model(metadata.get("model") or {}, metadata["stats"]["input_shape"], int(metadata["stats"]["output_size"])).to(device)
    checkpoint_path = Path(str(metadata.get("checkpoint") or ""))
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    result = _evaluate_loader_metrics(torch, model, loader, metadata["stats"], device, str(metadata.get("family") or "bandwidth"))
    result = result or {}
    result.update({"model_id": model_id, "split": split, "evaluated_at": now_iso()})
    model_dir = TRAINING_MODELS_ROOT / model_id
    evaluations = read_json(model_dir / "evaluations.json", [])
    if not isinstance(evaluations, list):
        evaluations = []
    evaluations.append(result)
    write_json(model_dir / "evaluations.json", evaluations)
    return result


def delete_training_run(run_id: str) -> dict[str, Any]:
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.pop(run_id, None)
    if job and job.status == "running":
        job.request_stop()
    run_dir = TRAINING_RUNS_ROOT / run_id
    deleted = False
    if run_dir.exists() and run_dir.is_dir():
        import shutil

        shutil.rmtree(run_dir)
        deleted = True
    return {"run_id": run_id, "deleted": deleted}


def training_results_payload(run_id: str) -> dict[str, Any]:
    run_dir = TRAINING_RUNS_ROOT / run_id
    metrics = read_json(run_dir / "metrics.json", {})
    graphs = build_training_graphs(metrics if isinstance(metrics, dict) else {})
    return {"run_id": run_id, "metrics": metrics, "graphs": graphs, "models": list_training_models()}
