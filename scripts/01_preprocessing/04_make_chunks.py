#!/usr/bin/env python3
"""Create deterministic 5 kb main/sub datasets after the split is frozen."""

from __future__ import annotations

# 1. 경로 및 설정
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
RUN_STARTS = {}


# 이 스크립트에서 사용하는 FASTA·chunk·로그 보조 함수
def load_yaml(name):
    import yaml
    section = name.removesuffix(".yaml"); path = CONFIG_DIR / ("pipeline.yaml" if section in {"training", "preprocessing", "split", "inference"} else name)
    with path.open(encoding="utf-8") as handle: value = yaml.safe_load(handle)
    return value[section] if section in {"training", "preprocessing", "split", "inference"} else value
def project_path(value):
    path = Path(value); return path if path.is_absolute() else PROJECT_ROOT / path
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
    write_json(output_json, report); print(f"[DONE] stage={stage} | runtime={report['runtime_hms']} | genomes={genomes:,} | contigs={contigs:,} | chunks={chunks:,} | bases={bases:,} | throughput={report['mbp_per_second']:.3f} Mbp/s")
    return report
def existing_size(paths): return sum(path.stat().st_size if path.is_file() else sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) for path in paths if path.exists())
def open_text(path): return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(encoding="utf-8", newline="")
def read_tsv(path):
    with open_text(path) as handle: yield from csv.DictReader(handle, delimiter="\t")
def fasta_records(path):
    with open_text(path) as handle:
        header, parts = None, []
        for raw_line in handle:
            line = raw_line.strip()
            if not line: continue
            if line.startswith(">"):
                if header is not None: yield header, "".join(parts)
                header, parts = line[1:], []
            elif header is None: raise ValueError(f"Sequence found before FASTA header in {path}")
            else: parts.append(line)
        if header is not None: yield header, "".join(parts)
DNA_TRANS = str.maketrans({"U": "T"}); RC_TRANS = str.maketrans("ACGTN", "TGCAN")
def normalize_dna(sequence):
    bad = set(sequence.upper()) - set("ACGTURYSWKMBDHVN")
    if bad: raise ValueError(f"Invalid non-IUPAC DNA characters: {sorted(bad)}")
    return re.sub(r"[^ACGTN]", "N", sequence.upper().translate(DNA_TRANS))
def sequence_hashes(sequence):
    forward = hashlib.sha256(sequence.encode()).hexdigest(); reverse = sequence.translate(RC_TRANS)[::-1]
    return forward, hashlib.sha256(min(sequence, reverse).encode()).hexdigest()
def stable_id(*parts, length=20): return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:length]
def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()
def chunk_ranges(length, size, minimum_tail):
    if length <= 0: return
    if length <= size: yield 0, length; return
    full, remainder = divmod(length, size)
    if 0 < remainder < minimum_tail:
        for index in range(full - 1): yield index * size, (index + 1) * size
        start = (full - 1) * size; middle = start + (size + remainder) // 2
        yield start, middle; yield middle, length
    else:
        for start in range(0, length, size): yield start, min(start + size, length)


MANIFEST = PROJECT_ROOT / "data" / "manifests" / "genomes.tsv.gz"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "preprocessed"
SPLITS = ("train", "validation", "inference")
FIELDS = [
    "sequence", "labels", "sub_labels", "record_id", "source", "genome_id", "contig_id",
    "split_group", "chunk_id", "chunk_start_0based", "chunk_end_exclusive",
    "valid_length", "genome_total_bp", "sample_weight",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


# 2. Manifest에서 genome split과 원본 FASTA 목록 읽기
def load_manifest_index() -> tuple[dict[str, dict], dict[tuple[str, str], list[Path]], set[str]]:
    genomes: dict[str, dict] = {}
    source_paths: dict[tuple[str, str], list[Path]] = defaultdict(list)
    conflict_hashes: set[str] = set()
    seen_source_paths: set[tuple[str, str, str]] = set()

    for row in read_tsv(MANIFEST):
        source = row["source"]
        path = row["source_path"]
        selector = ""
        if source == "EVE_NCLDV":
            selector = "EVE-NCLDV__"
        elif source == "EVE_MIRUS":
            selector = "EVE-MIRUS__"
        key = (source, selector, path)
        if key not in seen_source_paths:
            source_paths[(source, selector)].append(project_path(path))
            seen_source_paths.add(key)

        if row["exclusion_reason"] == "canonical_sequence_label_conflict":
            conflict_hashes.add(row["canonical_sha256"])
        info = genomes.setdefault(
            row["genome_id"],
            {
                "split": "quarantine",
                "split_group": row["split_group"],
                "main_label": row["main_label"],
                "sub_label": row["sub_label"],
                "total_bp": 0,
            },
        )
        if row["split"] in SPLITS:
            info["split"] = row["split"]
            info["split_group"] = row["split_group"]
            info["total_bp"] += int(row["sequence_length"])
    return genomes, source_paths, conflict_hashes


def parse_genome_contig(source: str, header: str, filename: str = "") -> tuple[str, str]:
    token = header.split()[0]
    if source in {"MITO", "PLASTID"}:
        if filename != f"{source}__{token}.fna":
            raise ValueError(f"{source} FASTA accession/header mismatch: {filename} vs {token}")
        genome_id = f"{source}__{token}"
        return genome_id, genome_id
    left, *rest = token.split("|")
    contig = rest[0] if rest else left
    prefixes = {
        "ARC": "ARC__", "BAC": "BAC__", "EUK": "EUK__", "NCLDV": "NCLDV__",
        "MIRUS": "MIRUS__", "PHAGE": "PHAGE__", "EVE_NCLDV": "EVE-NCLDV__",
        "EVE_MIRUS": "EVE-MIRUS__",
    }
    raw = left.removeprefix(prefixes[source])
    if source.startswith("EVE_"):
        import re
        raw = re.sub(r"[_.]", "-", raw.removesuffix("_eves"))
    return f"{source}__{raw}", f"{source}__{contig}"


# 3. 5 kb와 마지막 variable-length tail 만들기
def open_writers(stack: ExitStack) -> tuple[dict[tuple[str, str], csv.DictWriter], list[Path]]:
    writers = {}
    outputs = []
    for task in ("main", "sub"):
        directory = OUTPUT_ROOT / task
        directory.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            path = directory / f"{split}.csv.gz"
            outputs.append(path)
            temp = path.with_suffix(path.suffix + ".tmp")
            handle = stack.enter_context(gzip.open(temp, "wt", encoding="utf-8", newline=""))
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writers[(task, split)] = writer
    return writers, outputs


def main() -> int:
    args = parse_args()
    if not MANIFEST.exists():
        raise SystemExit(f"Missing {MANIFEST}; run 03_make_splits.py first.")
    existing_outputs = [OUTPUT_ROOT / task / f"{split}.csv.gz" for task in ("main", "sub") for split in SPLITS]
    if any(path.exists() for path in existing_outputs) and not args.force:
        raise SystemExit("Chunk outputs already exist. Use --force to intentionally rebuild derived chunks.")

    prep = load_yaml("preprocessing.yaml")
    chunk_size = int(prep["chunking"]["size_nt"])
    minimum_tail = int(prep["chunking"]["minimum_tail_nt"])
    n_limit = float(prep["normalization"]["quarantine_if_n_fraction_above"])
    genomes, source_paths, conflict_hashes = load_manifest_index()
    allowed = {r["record_id"]: r["sequence_sha256"] for r in read_tsv(MANIFEST) if r["split"] in SPLITS}
    frozen = json.loads((PROJECT_ROOT / "data/splits/frozen.json").read_text())
    if frozen["manifest_sha256"] != file_sha256(MANIFEST):
        raise SystemExit("Frozen manifest checksum mismatch")

    input_paths = sorted({path for paths in source_paths.values() for path in paths})
    log_path = PROJECT_ROOT / "logs" / "preprocess" / f"04_make_chunks_{timestamp()}.log"
    tee = Tee(log_path)
    tee.start()
    started = print_run_header("preprocess/make_chunks", input_paths, OUTPUT_ROOT)

    counts = Counter()
    genomes_seen: set[str] = set()
    contigs_seen = 0
    try:
        with ExitStack() as stack:
            writers, outputs = open_writers(stack)
            for (source, selector), paths in sorted(source_paths.items()):
                for path in paths:
                    for header, raw_sequence in fasta_records(path):
                        if selector and not header.startswith(selector):
                            continue
                        record_id = "REC_" + stable_id(source, header)
                        if record_id not in allowed:
                            continue
                        genome_id, contig_id = parse_genome_contig(source, header, path.name)
                        info = genomes.get(genome_id)
                        if not info or info["split"] not in SPLITS:
                            continue

                        sequence = normalize_dna(raw_sequence)
                        if sequence_hashes(sequence)[0] != allowed[record_id]:
                            raise ValueError(f"Original sequence changed: {record_id}")
                        if not sequence or sequence.count("N") / len(sequence) > n_limit:
                            continue
                        _, canonical = sequence_hashes(sequence)
                        if canonical in conflict_hashes:
                            continue

                        genomes_seen.add(genome_id)
                        contigs_seen += 1
                        counts["bases"] += len(sequence)

                        kept_bp = 0
                        for chunk_number, (start, end) in enumerate(
                            chunk_ranges(len(sequence), chunk_size, minimum_tail), start=1
                        ):
                            subsequence = sequence[start:end]
                            valid_length = len(subsequence)
                            kept_bp += valid_length
                            # 모든 chunk를 사용하되 genome 전체 기여도와 짧은 tail 길이를 반영한다.
                            sample_weight = valid_length / max(info["total_bp"], 1)
                            common = {
                                "sequence": subsequence,
                                "record_id": record_id,
                                "source": source,
                                "genome_id": genome_id,
                                "contig_id": contig_id,
                                "split_group": info["split_group"],
                                "chunk_id": "CHK_" + stable_id(record_id, start, end),
                                "chunk_start_0based": start,
                                "chunk_end_exclusive": end,
                                "valid_length": valid_length,
                                "genome_total_bp": info["total_bp"],
                                "sample_weight": f"{sample_weight:.12g}",
                            }
                            main_row = {
                                **common,
                                "labels": info["main_label"],
                                "sub_labels": info["sub_label"],
                            }
                            writers[("main", info["split"])].writerow(main_row)
                            counts[f"main_{info['split']}_chunks"] += 1

                            if info["sub_label"] != "":
                                sub_row = {
                                    **common,
                                    "labels": info["sub_label"],
                                    "sub_labels": info["sub_label"],
                                }
                                writers[("sub", info["split"])].writerow(sub_row)
                                counts[f"sub_{info['split']}_chunks"] += 1
                        counts["omitted_short_tail_bp"] += len(sequence) - kept_bp

                    print(
                        f"[SOURCE] {source:20s} | genomes_seen={len(genomes_seen):,} | "
                        f"contigs={contigs_seen:,} | main_chunks={sum(counts[f'main_{s}_chunks'] for s in SPLITS):,}"
                    )

        # gzip handles are closed here; publish complete files atomically.
        for path in outputs:
            path.with_suffix(path.suffix + ".tmp").replace(path)

        summary = {key: int(value) for key, value in counts.items()}
        write_json(OUTPUT_ROOT / "chunk_summary.json", summary)
        write_json(OUTPUT_ROOT / "dataset_checksums.json", {
            "manifest_sha256": file_sha256(MANIFEST),
            "preprocessing": prep,
            "files": {str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in outputs},
        })
        finish_stats(
            "preprocess/make_chunks",
            started,
            OUTPUT_ROOT / "make_chunks_runtime.json",
            bases=counts["bases"],
            genomes=len(genomes_seen),
            contigs=contigs_seen,
            chunks=sum(counts[f"main_{split}_chunks"] for split in SPLITS),
            input_bytes=existing_size(input_paths),
            extra={"chunk_size_nt": chunk_size, "counts": summary},
        )
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
