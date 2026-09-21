#!/usr/bin/env python3
"""Final QC gate: fail if IDs or identical 5 kb sequences occur across splits."""

from __future__ import annotations

# 1. 경로 및 설정
import argparse
import csv
import datetime as dt
import gzip
import json
import hashlib
import os
import resource
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_STARTS = {}


# 이 스크립트에서 사용하는 hash·로그 보조 함수
def timestamp(): return dt.datetime.now().strftime("%Y%m%d-%H%M%S")
def now_local(): return dt.datetime.now().astimezone().isoformat(timespec="seconds")
def now_utc(): return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
class Tee:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True); self.handle = path.open("a", encoding="utf-8", buffering=1)
        self.stdout, self.stderr = sys.stdout, sys.stderr
    def start(self): sys.stdout = sys.stderr = self
    def write(self, text): self.stdout.write(text); self.stdout.flush(); self.handle.write(text); self.handle.flush(); return len(text)
    def flush(self): self.stdout.flush(); self.handle.flush()
    def close(self): sys.stdout, sys.stderr = self.stdout, self.stderr; self.handle.close()
def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); temp.replace(path)
def print_run_header(stage, inputs: Iterable[Path], output):
    print("=" * 100); print(f"[START] stage={stage} | local={now_local()} | utc={now_utc()}")
    print(f"[PATH] project={PROJECT_ROOT}\n[PATH] input={'; '.join(map(str, inputs))}\n[PATH] output={output}")
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"); print("=" * 100)
    started = time.perf_counter(); RUN_STARTS[started] = {"started_local": now_local(), "started_utc": now_utc()}; return started
def finish_stats(stage, started, output_json, *, bases=0, genomes=0, contigs=0, chunks=0, input_bytes=0, extra=None):
    seconds = max(time.perf_counter() - started, 1e-9); mbp = bases / 1_000_000
    report = {**RUN_STARTS.get(started, {}), "stage": stage, "completed_local": now_local(), "completed_utc": now_utc(),
              "runtime_seconds": seconds, "runtime_hms": str(dt.timedelta(seconds=round(seconds))), "input_bytes": int(input_bytes),
              "bases": int(bases), "genomes": int(genomes), "contigs": int(contigs), "sequences": int(contigs), "chunks": int(chunks),
              "mbp_per_second": mbp / seconds, "seconds_per_mbp": seconds / mbp if mbp else None,
              "peak_cpu_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "peak_gpu_memory_bytes": 0, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if extra: report.update(extra)
    write_json(output_json, report); print(f"[DONE] stage={stage} | runtime={report['runtime_hms']} | chunks={chunks:,} | bases={bases:,}"); return report
RC_TRANS = str.maketrans("ACGTN", "TGCAN")
def sequence_hashes(sequence):
    forward = hashlib.sha256(sequence.encode()).hexdigest(); reverse = sequence.translate(RC_TRANS)[::-1]
    return forward, hashlib.sha256(min(sequence, reverse).encode()).hexdigest()
def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


DATA_ROOT = PROJECT_ROOT / "data" / "preprocessed"
REPORT = DATA_ROOT / "qc_report.json"
SPLITS = ("train", "validation", "inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("main", "sub", "all"), default="all")
    return parser.parse_args()


# 2. Disk-backed SQLite에 관측값 저장
def validate_task(task: str) -> dict:
    paths = {split: DATA_ROOT / task / f"{split}.csv.gz" for split in SPLITS}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    counts = Counter()
    with tempfile.NamedTemporaryFile(prefix=f"genomeocean_{task}_", suffix=".sqlite") as temporary:
        connection = sqlite3.connect(temporary.name)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("CREATE TABLE seen(kind TEXT, key TEXT, split TEXT, label TEXT)")

        buffer = []
        for split, path in paths.items():
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    _, canonical = sequence_hashes(row["sequence"])
                    observations = (
                        ("split_group", row["split_group"], split, row["labels"]),
                        ("genome", row["genome_id"], split, row["labels"]),
                        ("contig", row["genome_id"] + "|" + row["contig_id"], split, row["labels"]),
                        ("chunk_hash", canonical, split, row["labels"]),
                    )
                    buffer.extend(observations)
                    counts[f"{split}_chunks"] += 1
                    counts[f"{split}_bases"] += int(row["valid_length"])
                    if len(buffer) >= 40_000:
                        connection.executemany("INSERT INTO seen VALUES (?, ?, ?, ?)", buffer)
                        buffer.clear()
            print(f"[QC] task={task} | split={split} | chunks={counts[f'{split}_chunks']:,}")
        if buffer:
            connection.executemany("INSERT INTO seen VALUES (?, ?, ?, ?)", buffer)
        connection.commit()

        overlap_rows = connection.execute(
            """
            SELECT kind, COUNT(*) FROM (
                SELECT kind, key
                FROM seen
                GROUP BY kind, key
                HAVING COUNT(DISTINCT split) > 1
            ) GROUP BY kind
            """
        ).fetchall()
        conflict_count = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT key FROM seen WHERE kind='chunk_hash'
                GROUP BY key HAVING COUNT(DISTINCT label) > 1
            )
            """
        ).fetchone()[0]
        examples = connection.execute(
            """
            SELECT kind, key, GROUP_CONCAT(DISTINCT split)
            FROM seen GROUP BY kind, key HAVING COUNT(DISTINCT split) > 1 LIMIT 20
            """
        ).fetchall()
        connection.close()

    overlaps = {kind: number for kind, number in overlap_rows}
    return {
        "task": task,
        "counts": dict(counts),
        "cross_split_overlaps": overlaps,
        "chunk_hash_label_conflicts": conflict_count,
        "examples": [{"kind": kind, "key": key, "splits": splits} for kind, key, splits in examples],
        "passed": not overlaps and conflict_count == 0,
    }


def main() -> int:
    args = parse_args()
    tasks = ("main", "sub") if args.task == "all" else (args.task,)
    log_path = PROJECT_ROOT / "logs" / "preprocess" / f"05_validate_{timestamp()}.log"
    tee = Tee(log_path)
    tee.start()
    started = print_run_header("preprocess/validate", [DATA_ROOT / task for task in tasks], REPORT)
    try:
        reports = [validate_task(task) for task in tasks]
        passed = all(report["passed"] for report in reports)
        # Main Viral(class 1) 전체와 독립 sub 데이터 전체가 동일 chunk/split인지 확인한다.
        shared_ok = True
        for split in SPLITS:
            digests = []
            for task in ("main", "sub"):
                digest = hashlib.sha256()
                with gzip.open(DATA_ROOT / task / f"{split}.csv.gz", "rt", newline="") as handle:
                    for row in csv.DictReader(handle):
                        if task == "main" and int(row["labels"]) != 1:
                            continue
                        label = row["sub_labels"] if task == "main" else row["labels"]
                        digest.update((row["chunk_id"] + "|" + label + "|" + row["sequence"]).encode())
                digests.append(digest.hexdigest())
            shared_ok &= digests[0] == digests[1]
        passed &= shared_ok
        fingerprint = DATA_ROOT / "dataset_checksums.json"
        payload = {"passed": passed, "tasks": reports, "main_sub_shared_split": shared_ok,
                   "dataset_checksums_sha256": file_sha256(fingerprint)}
        write_json(REPORT, payload)
        total_bases = sum(sum(value for key, value in report["counts"].items() if key.endswith("_bases")) for report in reports)
        total_chunks = sum(sum(value for key, value in report["counts"].items() if key.endswith("_chunks")) for report in reports)
        finish_stats(
            "preprocess/validate",
            started,
            DATA_ROOT / "validate_runtime.json",
            bases=total_bases,
            chunks=total_chunks,
            extra={"passed": passed, "report": str(REPORT)},
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if passed else 1
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
