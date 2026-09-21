#!/usr/bin/env python3
"""Run the preprocessing steps in their safe order."""

from __future__ import annotations

# 1. 경로 및 설정
import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Intentionally rebuild derived outputs.")
    return parser.parse_args()


# 2. 각 단계를 순서대로 실행
def run(script: str, force: bool = False) -> None:
    command = [sys.executable, str(SCRIPT_DIR / script)]
    if force:
        command.append("--force")
    print(f"[RUN] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    args = parse_args()
    run("01_build_manifest.py", args.force)
    run("02_similarity.py", args.force)
    run("03_make_splits.py", args.force)
    run("04_make_chunks.py", args.force)
    run("05_validate_preprocessed.py")
    print("[DONE] preprocessing completed; train/validation/inference data are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
