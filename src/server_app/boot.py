from __future__ import annotations

import argparse
import time
from http.server import ThreadingHTTPServer
from multiprocessing import Process
from typing import Dict

from .config import PROJECT_ROOT, RESULTS_ROOT, SETTINGS_ROOT, WEB_DIST_ROOT, WEB_ROOT
from .discovery import ensure_traces_ready
from .http import RequestHandler
from .runs import recover_queued_runs


def run_server(host: str = "0.0.0.0", port: int = 8000):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    SETTINGS_ROOT.mkdir(parents=True, exist_ok=True)
    trace_status = ensure_traces_ready()
    if not all(trace_status.values()):
        print("Trace data is incomplete. Add data manually or run the explicit trace download helpers in src/scripts/.")
    recover_queued_runs()
    server = ThreadingHTTPServer((host, port), RequestHandler)
    print(f"Serving UI at http://{host}:{port}")
    server.serve_forever()


def snapshot_mtimes() -> Dict[str, float]:
    mtimes: Dict[str, float] = {}
    for path in PROJECT_ROOT.rglob("*.py"):
        try:
            mtimes[str(path)] = path.stat().st_mtime
        except FileNotFoundError:
            continue
    return mtimes


def run_with_reload(host: str, port: int, interval: float = 1.0):
    print("Starting server with autoreload enabled (polling for code changes).")
    while True:
        baseline = snapshot_mtimes()
        proc = Process(target=run_server, args=(host, port))
        proc.start()
        try:
            while proc.is_alive():
                time.sleep(interval)
                if snapshot_mtimes() != baseline:
                    print("Code change detected; restarting server.")
                    proc.terminate()
                    proc.join(timeout=5)
                    break
        except KeyboardInterrupt:
            proc.terminate()
            proc.join(timeout=5)
            raise SystemExit(0)


def ensure_web_dist() -> bool:
    dist_index = WEB_DIST_ROOT / "index.html"
    if dist_index.exists():
        return True
    print("Warning: Web build missing (`src/web/dist/index.html`).")
    print("   Build it explicitly: cd src/web && npm ci && npm run build")
    return True


def main():
    parser = argparse.ArgumentParser(description="Web UI server for the V3C player.")
    parser.add_argument("--host", default="0.0.0.0", help="Host/interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--reload", action="store_true", help="Watch code changes and auto-restart the server (dev only).")
    args = parser.parse_args()

    ensure_web_dist()
    if args.reload:
        run_with_reload(args.host, args.port)
    else:
        run_server(args.host, args.port)
