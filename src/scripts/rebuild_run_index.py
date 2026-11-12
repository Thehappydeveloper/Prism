"""Rebuild Prism's disposable result-folder index."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.run_index import rebuild_run_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the local Prism run index from result folders.")
    parser.add_argument("--results", type=Path, default=Path("results"), help="Results root (default: results)")
    args = parser.parse_args()
    count = rebuild_run_index(args.results.resolve())
    print(f"Indexed {count} run folders in {args.results.resolve()}")


if __name__ == "__main__":
    main()
