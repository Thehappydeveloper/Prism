from __future__ import annotations

import argparse
import json
from pathlib import Path

from .service import build_report, list_reports, validate_report


def main() -> None:
    parser = argparse.ArgumentParser(prog="prism reports")
    parser.add_argument("command", choices=("list", "validate", "build"))
    parser.add_argument("report_id", nargs="?")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.command == "list":
        print(json.dumps(list_reports(), indent=2)); return
    if not args.report_id:
        parser.error("report_id is required for validate and build")
    result = validate_report(args.report_id, args.results_root) if args.command == "validate" else build_report(args.report_id, args.results_root)
    print(json.dumps(result, indent=2))
    if result["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
