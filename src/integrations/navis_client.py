from __future__ import annotations

import json
import time
from typing import Any, Dict
from urllib import error, request
from urllib.parse import urljoin


def _join_url(base_url: str, path: str) -> str:
    base = base_url.strip().rstrip("/") + "/"
    rel = (path or "/").lstrip("/")
    return urljoin(base, rel)


def _http_json(
    method: str,
    url: str,
    timeout_s: float,
    payload: Dict[str, Any] | None = None,
    headers: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    body = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    req = request.Request(url=url, data=body, headers=req_headers, method=method.upper())
    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout_s) as res:
            raw = res.read().decode("utf-8", errors="replace")
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            parsed = None
            if raw.strip():
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = None
            return {
                "ok": 200 <= int(res.status) < 300,
                "status_code": int(res.status),
                "body_text": raw,
                "body_json": parsed,
                "latency_ms": round(elapsed_ms, 2),
            }
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        parsed = None
        if raw.strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
        return {
            "ok": False,
            "status_code": int(exc.code),
            "body_text": raw,
            "body_json": parsed,
            "latency_ms": round(elapsed_ms, 2),
            "error": str(exc),
        }
    except error.URLError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "ok": False,
            "status_code": None,
            "body_text": "",
            "body_json": None,
            "latency_ms": round(elapsed_ms, 2),
            "error": str(exc.reason),
        }


def probe_navis(base_url: str | None, health_path: str = "/health", timeout_s: float = 0.35) -> Dict[str, Any]:
    if not base_url or not str(base_url).strip():
        return {
            "configured": False,
            "reachable": False,
            "base_url": "",
            "health_path": health_path,
            "checked_url": "",
            "status_code": None,
            "latency_ms": None,
            "error": "PRISM_NAVIS_URL is not set.",
        }

    base = str(base_url).strip()
    health_url = _join_url(base, health_path or "/health")
    health_result = _http_json("GET", health_url, timeout_s=timeout_s)

    if health_result.get("status_code") is not None:
        code = int(health_result["status_code"])
        # Treat 4xx as "server is up but endpoint may be different".
        if code < 500:
            return {
                "configured": True,
                "reachable": True,
                "base_url": base,
                "health_path": health_path,
                "checked_url": health_url,
                "status_code": code,
                "latency_ms": health_result.get("latency_ms"),
                "error": None,
            }

    # Fallback to root if health endpoint is unavailable.
    root_url = _join_url(base, "/")
    root_result = _http_json("GET", root_url, timeout_s=timeout_s)
    reachable = False
    code = root_result.get("status_code")
    if code is not None and int(code) < 500:
        reachable = True
    return {
        "configured": True,
        "reachable": reachable,
        "base_url": base,
        "health_path": health_path,
        "checked_url": root_url if reachable else health_url,
        "status_code": code if reachable else health_result.get("status_code"),
        "latency_ms": root_result.get("latency_ms") if reachable else health_result.get("latency_ms"),
        "error": None if reachable else health_result.get("error") or root_result.get("error") or "Navis is unreachable.",
    }


def post_run_to_navis(
    base_url: str,
    export_path: str,
    payload: Dict[str, Any],
    timeout_s: float = 2.0,
    auth_token: str | None = None,
) -> Dict[str, Any]:
    url = _join_url(base_url, export_path)
    headers: Dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    request_bytes = len(json.dumps(payload).encode("utf-8"))
    result = _http_json("POST", url, timeout_s=timeout_s, payload=payload, headers=headers)
    return {
        "ok": bool(result.get("ok")),
        "url": url,
        "status_code": result.get("status_code"),
        "latency_ms": result.get("latency_ms"),
        "request_bytes": request_bytes,
        "response": result.get("body_json"),
        "response_text": (result.get("body_text") or "")[:500],
        "error": result.get("error"),
    }
