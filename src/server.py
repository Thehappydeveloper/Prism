from __future__ import annotations

from server_app.boot import ensure_web_dist, main, run_server, run_with_reload
from server_app.config import generate_run_id
from server_app.diagnostics import gather_diagnostics
from server_app.discovery import discover_options, resolve_qoe_model
from server_app.http import RequestHandler
from server_app.runs import RunJob, load_existing_runs, load_run

__all__ = [
    "RequestHandler",
    "RunJob",
    "discover_options",
    "ensure_web_dist",
    "gather_diagnostics",
    "generate_run_id",
    "load_existing_runs",
    "load_run",
    "main",
    "resolve_qoe_model",
    "run_server",
    "run_with_reload",
]


if __name__ == "__main__":
    main()
