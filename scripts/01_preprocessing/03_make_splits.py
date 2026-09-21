#!/usr/bin/env python3
"""Union leakage groups and assign them to train/validation/inference."""

from __future__ import annotations

# 1. 경로 및 설정
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
RUN_STARTS = {}


# 이 스크립트에서 사용하는 설정·group·로그 보조 함수
def load_yaml(name):
    import yaml
    section = name.removesuffix(".yaml"); path = CONFIG_DIR / ("pipeline.yaml" if section in {"training", "preprocessing", "split", "inference"} else name)
    with path.open(encoding="utf-8") as handle: value = yaml.safe_load(handle)
    return value[section] if section in {"training", "preprocessing", "split", "inference"} else value
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
    write_json(output_json, report); print(f"[DONE] stage={stage} | runtime={report['runtime_hms']} | genomes={genomes:,} | contigs={contigs:,} | bases={bases:,}")
    return report
def open_text(path): return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(encoding="utf-8", newline="")
def read_tsv(path):
    with open_text(path) as handle: yield from csv.DictReader(handle, delimiter="\t")
def stable_id(*parts, length=20): return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:length]
def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


class UnionFind:
    def __init__(self): self.parent, self.size = {}, {}
    def add(self, item):
        if item not in self.parent: self.parent[item], self.size[item] = item, 1
    def find(self, item):
        self.add(item); root = item
        while self.parent[root] != root: root = self.parent[root]
        while self.parent[item] != item: parent = self.parent[item]; self.parent[item] = root; item = parent
        return root
    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a == b: return
        if self.size[a] < self.size[b]: a, b = b, a
        self.parent[b] = a; self.size[a] += self.size[b]


RAW_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "records.raw.tsv.gz"
FINAL_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "genomes.tsv.gz"
EDGE_FILE = PROJECT_ROOT / "data" / "manifests" / "similarity_edges.tsv"
GROUP_ASSIGNMENTS = PROJECT_ROOT / "data" / "splits" / "group_assignments.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--similarity-edges",
        type=Path,
        default=EDGE_FILE,
        help="Optional TSV with left_group and right_group columns from an external 95/85 similarity search.",
    )
    return parser.parse_args()


def deterministic_tie(group: str, split: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{group}|{split}".encode()).hexdigest()


# 2. Base group, exact duplicate, 선택적 near-duplicate edge를 하나로 합치기
def build_components(edge_file: Path) -> tuple[UnionFind, set[str], int]:
    union = UnionFind()
    first_by_hash: dict[str, tuple[str, str]] = {}
    hash_labels: dict[str, set[str]] = defaultdict(set)

    for row in read_tsv(RAW_MANIFEST):
        if row["split"] == "quarantine":
            continue
        group = row["base_group_id"]
        digest = row["canonical_sha256"]
        label = row["main_label"] + "/" + row["sub_label"]
        union.add(group)
        hash_labels[digest].add(label)
        previous = first_by_hash.get(digest)
        if previous and previous[1] == label:
            union.union(group, previous[0])
        elif previous is None:
            first_by_hash[digest] = (group, label)

    conflict_hashes = {digest for digest, labels in hash_labels.items() if len(labels) > 1}
    edge_count = 0
    if edge_file.exists():
        for row in read_tsv(edge_file):
            left = (row.get("left_group") or "").strip()
            right = (row.get("right_group") or "").strip()
            if left and right:
                union.union(left, right)
                edge_count += 1
    return union, conflict_hashes, edge_count


# 3. Component별 genome 수와 염기 수 계산
def component_names(union: UnionFind) -> dict[str, str]:
    minimum: dict[str, str] = {}
    for group in union.parent:
        root = union.find(group)
        minimum[root] = min(group, minimum.get(root, group))
    return {group: "SIM_" + stable_id(minimum[union.find(group)]) for group in union.parent}


def collect_stats(
    group_to_component: dict[str, str], conflict_hashes: set[str], excluded_record_ids: set[str]
) -> tuple[dict[str, dict], int, int, int]:
    stats: dict[str, dict] = {}
    bases = contigs = 0
    genomes_all: set[str] = set()
    for row in read_tsv(RAW_MANIFEST):
        bases += int(row["sequence_length"])
        contigs += 1
        genomes_all.add(row["genome_id"])
        if (row["split"] == "quarantine" or row["canonical_sha256"] in conflict_hashes
                or row["record_id"] in excluded_record_ids):
            continue
        component = group_to_component[row["base_group_id"]]
        item = stats.setdefault(
            component,
            {"bases": 0, "contigs": 0, "genomes": set(), "sources": Counter(), "labels": Counter()},
        )
        item["bases"] += int(row["sequence_length"])
        item["contigs"] += 1
        item["genomes"].add(row["genome_id"])
        item["sources"][row["source"]] += int(row["sequence_length"])
        item["labels"][row["main_label"]] += int(row["sequence_length"])
    return stats, bases, len(genomes_all), contigs


def pre_split_exclusions() -> dict[str, str]:
    """Resolve exclusions before balancing, so excluded data cannot affect split targets."""
    return json.loads((FINAL_MANIFEST.parent / "similarity_exclusions.json").read_text())


# 4. 큰 group부터 80/10/10 목표에 가장 부족한 split에 배정
def assign_groups(stats: dict[str, dict], ratios: dict[str, float], seed: int) -> dict[str, str]:
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for component, item in stats.items():
        source = item["sources"].most_common(1)[0][0]
        label = item["labels"].most_common(1)[0][0]
        strata[(source, label)].append(component)

    assignments: dict[str, str] = {}
    for stratum, components in sorted(strata.items()):
        total_bp = sum(stats[group]["bases"] for group in components)
        total_genomes = sum(len(stats[group]["genomes"]) for group in components)
        used = {split: {"bases": 0, "genomes": 0} for split in ratios}
        ordered = sorted(
            components,
            key=lambda group: (-stats[group]["bases"], deterministic_tie(group, "order", seed)),
        )
        for group in ordered:
            choices = []
            for split, ratio in ratios.items():
                target_bp = max(total_bp * ratio, 1)
                target_genomes = max(total_genomes * ratio, 1)
                # 이 group을 넣은 뒤 목표 대비 얼마나 차는지를 비교한다.
                score = (
                    (used[split]["bases"] + stats[group]["bases"]) / target_bp
                    + (used[split]["genomes"] + len(stats[group]["genomes"])) / target_genomes
                )
                choices.append((score, deterministic_tie(group, split, seed), split))
            split = min(choices)[2]
            assignments[group] = split
            used[split]["bases"] += stats[group]["bases"]
            used[split]["genomes"] += len(stats[group]["genomes"])

        print(
            f"[STRATUM] source={stratum[0]:20s} label={stratum[1]} | groups={len(components):,} | "
            + " | ".join(
                f"{split}={used[split]['genomes']:,} genomes/{used[split]['bases'] / 1e6:.1f} Mbp"
                for split in ratios
            )
        )
    return assignments


# 5. Frozen manifest와 group assignment 저장
def write_outputs(
    group_to_component: dict[str, str],
    conflict_hashes: set[str],
    stats: dict[str, dict],
    assignments: dict[str, str],
    pre_exclusions: dict[str, str],
) -> Counter:
    GROUP_ASSIGNMENTS.parent.mkdir(parents=True, exist_ok=True)
    with GROUP_ASSIGNMENTS.open("w", encoding="utf-8", newline="") as handle:
        fields = ["split_group", "split", "bases", "genomes", "contigs", "sources", "labels"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for group in sorted(assignments):
            item = stats[group]
            writer.writerow(
                {
                    "split_group": group,
                    "split": assignments[group],
                    "bases": item["bases"],
                    "genomes": len(item["genomes"]),
                    "contigs": item["contigs"],
                    "sources": ",".join(sorted(item["sources"])),
                    "labels": ",".join(sorted(item["labels"])),
                }
            )

    counts = Counter()
    temp = FINAL_MANIFEST.with_suffix(FINAL_MANIFEST.suffix + ".tmp")
    with gzip.open(RAW_MANIFEST, "rt", encoding="utf-8", newline="") as source_handle, gzip.open(
        temp, "wt", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.DictReader(source_handle, delimiter="\t")
        writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            if row["record_id"] in pre_exclusions:
                row["split"] = "quarantine"
                row["exclusion_reason"] = pre_exclusions[row["record_id"]]
            if row["canonical_sha256"] in conflict_hashes:
                row["split"] = "quarantine"
                row["exclusion_reason"] = "canonical_sequence_label_conflict"
            if row["split"] != "quarantine":
                component = group_to_component[row["base_group_id"]]
                row["similarity_component"] = component
                row["split_group"] = component
                row["split"] = assignments[component]
            counts[row["split"]] += 1
            writer.writerow(row)
    temp.replace(FINAL_MANIFEST)
    return counts


def source_class_split_summary() -> dict:
    """Show the actual retained records and genomes for each source/class/split."""
    tallies = defaultdict(lambda: {"records": 0, "genomes": set()})
    for row in read_tsv(FINAL_MANIFEST):
        key = (row["source"], row["main_label"], row["split"])
        tallies[key]["records"] += 1
        tallies[key]["genomes"].add(row["genome_id"])
    summary = {}
    for (source, label, split), item in sorted(tallies.items()):
        summary.setdefault(source, {}).setdefault(label, {})[split] = {
            "records": item["records"], "genomes": len(item["genomes"])
        }
    return summary


def main() -> int:
    args = parse_args()
    if not RAW_MANIFEST.exists():
        raise SystemExit(f"Missing {RAW_MANIFEST}; run 01_build_manifest.py first.")
    if (FINAL_MANIFEST.exists() or GROUP_ASSIGNMENTS.exists()) and not args.force:
        raise SystemExit("Frozen split output already exists. Use --force only for an intentional new split version.")

    config = load_yaml("split.yaml")
    audit_path = RAW_MANIFEST.parent / "similarity_audit.json"
    if not audit_path.exists():
        raise SystemExit("Mandatory similarity audit missing: run 02_similarity.py first.")
    audit = json.loads(audit_path.read_text())
    mirus_ani_edges = RAW_MANIFEST.parent / "mirus_genome_ani_edges.tsv"
    if (not audit.get("complete") or audit["raw_manifest_sha256"] != file_sha256(RAW_MANIFEST)
        or audit["edge_sha256"] != file_sha256(args.similarity_edges)
        or audit["exclusions_sha256"] != file_sha256(RAW_MANIFEST.parent / "similarity_exclusions.json")
        or not mirus_ani_edges.exists()
        or audit.get("mirus_ani_edges_sha256") != file_sha256(mirus_ani_edges)
        or audit["settings"] != load_yaml("preprocessing.yaml")["similarity"]):
        raise SystemExit("Similarity audit is stale; rerun it for the current inputs/settings.")
    ratios = {name: float(value) for name, value in config["ratios"].items()}
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise SystemExit(f"Split ratios must sum to 1.0, got {ratios}")

    log_path = PROJECT_ROOT / "logs" / "preprocess" / f"03_make_splits_{timestamp()}.log"
    tee = Tee(log_path)
    tee.start()
    started = print_run_header("preprocess/make_splits", [RAW_MANIFEST, args.similarity_edges], FINAL_MANIFEST)
    try:
        union, conflict_hashes, edge_count = build_components(args.similarity_edges)
        print(f"[GROUP] base_groups={len(union.parent):,} | label_conflict_hashes={len(conflict_hashes):,}")
        if edge_count:
            print(f"[GROUP] external near-duplicate edges applied={edge_count:,}")
        else:
            print("[WARN] no external near-duplicate edges; base groups and exact canonical hashes are enforced")

        names = component_names(union)
        pre_exclusions = pre_split_exclusions()
        stats, bases, genomes, contigs = collect_stats(names, conflict_hashes, set(pre_exclusions))
        assignments = assign_groups(stats, ratios, int(config["seed"]))
        counts = write_outputs(names, conflict_hashes, stats, assignments, pre_exclusions)
        write_json(PROJECT_ROOT / "data/splits/frozen.json", {
            "manifest_sha256": file_sha256(FINAL_MANIFEST),
            "raw_manifest_sha256": file_sha256(RAW_MANIFEST),
            "group_assignment_sha256": file_sha256(GROUP_ASSIGNMENTS),
            "split_config": config,
        })
        write_json(
            PROJECT_ROOT / "data" / "splits" / "split_summary.json",
            {"records": dict(counts), "groups": len(assignments),
             "near_duplicate_edges": edge_count,
             "by_source_and_main_label": source_class_split_summary()},
        )
        finish_stats(
            "preprocess/make_splits",
            started,
            PROJECT_ROOT / "data" / "splits" / "make_splits_runtime.json",
            bases=bases,
            genomes=genomes,
            contigs=contigs,
            input_bytes=RAW_MANIFEST.stat().st_size,
            extra={"records_by_split": dict(counts), "near_duplicate_edges": edge_count},
        )
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
