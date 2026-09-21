#!/usr/bin/env python3
"""Read all eligible FASTA records and build the source-of-truth raw manifest."""

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
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
RUN_STARTS = {}


# 이 스크립트에서 사용하는 설정·FASTA·로그 보조 함수
def load_yaml(name: str) -> dict:
    import yaml
    section = name.removesuffix(".yaml")
    path = CONFIG_DIR / ("pipeline.yaml" if section in {"training", "preprocessing", "split", "inference"} else name)
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return value[section] if section in {"training", "preprocessing", "split", "inference"} else value


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def now_local() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8", buffering=1)
        self.stdout, self.stderr = sys.stdout, sys.stderr

    def start(self):
        sys.stdout = sys.stderr = self

    def write(self, text):
        self.stdout.write(text); self.stdout.flush(); self.handle.write(text); self.handle.flush()
        return len(text)

    def flush(self):
        self.stdout.flush(); self.handle.flush()

    def close(self):
        sys.stdout, sys.stderr = self.stdout, self.stderr
        self.handle.close()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_info() -> list[dict]:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
                                 "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [dict(zip(("physical_index", "name", "memory_total_mib", "memory_used_mib"),
                     (item.strip() for item in line.split(",")))) for line in result.stdout.splitlines()]


def print_run_header(stage: str, inputs: Iterable[Path], output: Path) -> float:
    print("=" * 100)
    print(f"[START] stage={stage} | local={now_local()} | utc={now_utc()}")
    print(f"[PATH] project={PROJECT_ROOT}\n[PATH] input={'; '.join(map(str, inputs))}\n[PATH] output={output}")
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    for row in gpu_info():
        print("[GPU] physical={physical_index} | name={name} | total={memory_total_mib} MiB | used_at_start={memory_used_mib} MiB".format(**row))
    print("=" * 100)
    started = time.perf_counter()
    RUN_STARTS[started] = {"started_local": now_local(), "started_utc": now_utc()}
    return started


def human_bytes(value: int) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(number) < 1024 or unit == "TiB": return f"{number:.2f} {unit}"
        number /= 1024


def finish_stats(stage: str, started: float, output_json: Path, *, bases=0, genomes=0,
                 contigs=0, chunks=0, input_bytes=0, extra=None) -> dict:
    seconds = max(time.perf_counter() - started, 1e-9); mbp = bases / 1_000_000
    report = {**RUN_STARTS.get(started, {}), "stage": stage, "completed_local": now_local(),
              "completed_utc": now_utc(), "runtime_seconds": seconds,
              "runtime_hms": str(dt.timedelta(seconds=round(seconds))), "input_bytes": int(input_bytes),
              "bases": int(bases), "genomes": int(genomes), "contigs": int(contigs),
              "sequences": int(contigs), "chunks": int(chunks), "mbp_per_second": mbp / seconds,
              "seconds_per_mbp": seconds / mbp if mbp else None,
              "peak_cpu_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "peak_gpu_memory_bytes": 0, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if extra: report.update(extra)
    write_json(output_json, report)
    print(f"[DONE] stage={stage} | runtime={report['runtime_hms']} | genomes={genomes:,} | contigs={contigs:,} | chunks={chunks:,} | bases={bases:,} | throughput={report['mbp_per_second']:.3f} Mbp/s | gpu_peak={human_bytes(report['peak_gpu_memory_bytes'])}")
    print(f"[DONE] stats={output_json} | local={report['completed_local']}")
    return report


def existing_size(paths: Iterable[Path]) -> int:
    return sum(path.stat().st_size if path.is_file() else sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) for path in paths if path.exists())


def open_text(path: Path, mode="rt") -> TextIO:
    return gzip.open(path, mode, encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(mode, encoding="utf-8", newline="")


def fasta_records(path: Path) -> Iterator[tuple[str, str]]:
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


DNA_TRANS = str.maketrans({"U": "T"})
RC_TRANS = str.maketrans("ACGTN", "TGCAN")


def normalize_dna(sequence: str) -> str:
    bad = set(sequence.upper()) - set("ACGTURYSWKMBDHVN")
    if bad: raise ValueError(f"Invalid non-IUPAC DNA characters: {sorted(bad)}")
    return re.sub(r"[^ACGTN]", "N", sequence.upper().translate(DNA_TRANS))


def sequence_hashes(sequence: str) -> tuple[str, str]:
    forward = hashlib.sha256(sequence.encode()).hexdigest()
    reverse = sequence.translate(RC_TRANS)[::-1]
    return forward, hashlib.sha256(min(sequence, reverse).encode()).hexdigest()


def stable_id(*parts: object, length=20) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:length]


def read_tsv(path: Path) -> Iterator[dict[str, str]]:
    with open_text(path) as handle: yield from csv.DictReader(handle, delimiter="\t")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def fasta_files(path: Path, glob_pattern=None) -> list[Path]:
    if path.is_file(): return [path]
    if glob_pattern: return sorted(item for item in path.glob(glob_pattern) if item.is_file())
    patterns = ("*.fna", "*.fa", "*.fasta", "*.fna.gz", "*.fa.gz", "*.fasta.gz")
    return sorted({item for pattern in patterns for item in path.rglob(pattern) if item.is_file()})


OUTPUT = PROJECT_ROOT / "data" / "manifests" / "records.raw.tsv.gz"

FIELDS = [
    "record_id", "source", "prefix", "source_path", "source_record_id",
    "genome_id", "contig_id", "record_type", "host_assembly_id", "votu",
    "main_label", "sub_label", "taxonomy_phylum", "taxonomy_class",
    "taxonomy_order", "taxonomy_family", "taxonomy_genus", "taxonomy_species",
    "taxonomy_source", "taxonomy_confidence", "sequence_length", "n_fraction",
    "sequence_sha256", "canonical_sha256", "base_group_id", "similarity_component",
    "split_group", "split", "exclusion_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--sources",
        nargs="*",
        help="Optional subset for a small pilot, e.g. --sources ARC NCLDV.",
    )
    return parser.parse_args()


# 2. Taxonomy와 제외 목록 읽기
def load_gv_labels(path: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    ranks = ["phylum", "class", "order", "family", "genus", "species"]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or "\t" not in line:
                continue
            key, lineage = line.rstrip("\n").split("\t", 1)
            parts = lineage.split("|")
            values = parts[1:7]
            labels[key] = {rank: values[index] if index < len(values) else "" for index, rank in enumerate(ranks)}
    return labels


def load_provisional(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    result = {}
    if not path.exists():
        return result
    for row in read_tsv(path):
        result[(row.get("group", ""), row.get("genome", ""))] = {
            rank: row.get(rank, "") for rank in ("phylum", "class", "order", "family", "genus", "species")
        }
    return result


def load_metadata(path: Path, key: str) -> dict[str, dict[str, str]]:
    metadata = {}
    for row in read_tsv(path):
        item_id = (row.get(key) or "").strip()
        if not item_id:
            continue
        if item_id in metadata:
            raise ValueError(f"Duplicate {key} in metadata: {path}: {item_id}")
        metadata[item_id] = row
    return metadata


def load_euk_exclusions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        row["genome_id"]
        for row in read_tsv(path)
        if row.get("decision", "").lower() == "exclude"
    }


def usable_genus(value: str) -> bool:
    token = (value or "").strip().lower()
    return token not in {"", "na", "nan", "none", "unknown", "unclassified", "(미분류)", "mirus_unclassified", "mirus", "ncldv", "g_"} and "unclassified" not in token


# 3. FASTA header를 genome/contig ID로 변환
def parse_header(source: str, header: str, filename: str = "") -> tuple[str, str, str]:
    token = header.split()[0]
    if source in {"MITO", "PLASTID"}:
        expected_filename = f"{source}__{token}.fna"
        if filename != expected_filename:
            raise ValueError(
                f"{source} FASTA accession/header mismatch: file={filename}, header={token}"
            )
        genome_id = f"{source}__{token}"
        return genome_id, genome_id, ""
    left, *rest = token.split("|")
    contig = rest[0] if rest else left

    expected = {
        "ARC": "ARC__", "BAC": "BAC__", "EUK": "EUK__", "NCLDV": "NCLDV__",
        "MIRUS": "MIRUS__", "PHAGE": "PHAGE__",
        "EVE_NCLDV": "EVE-NCLDV__", "EVE_MIRUS": "EVE-MIRUS__",
    }[source]
    if not left.startswith(expected):
        raise ValueError(f"Unexpected {source} header: {header[:160]}")
    raw_genome = left[len(expected):]

    if source.startswith("EVE_"):
        import re
        host = re.sub(r"[_.]", "-", raw_genome.removesuffix("_eves"))
        genome_id = f"{source}__{host}"
        return genome_id, f"{source}__{contig}", host
    genome_id = f"{source}__{raw_genome}"
    return genome_id, f"{source}__{contig}", ""


def taxonomy_for(
    source: str,
    raw_genome: str,
    metadata: dict[str, str],
    gv_labels: dict[str, dict[str, str]],
    provisional: dict[tuple[str, str], dict[str, str]],
) -> tuple[dict[str, str], str, str]:
    if source in {"ARC", "BAC", "EUK", "MITO", "PLASTID", "PHAGE"}:
        values = {
            "phylum": metadata.get("phylum_name", ""),
            "class": metadata.get("class_name", ""),
            "order": metadata.get("order_name", ""),
            "family": metadata.get("family_name", ""),
            "genus": metadata.get("genus_name", "") or metadata.get("selection_ictv_genus", ""),
            "species": metadata.get("species_name", ""),
        }
        return values, "snapshot_metadata", "curated"
    if source in {"EVE_NCLDV", "EVE_MIRUS"}:
        return {rank: "" for rank in ("phylum", "class", "order", "family", "genus", "species")}, "missing", "missing"

    prefix = "NCLDV" if source == "NCLDV" else "MIRUS"
    exact = gv_labels.get(f"{prefix}__{raw_genome}")
    if exact:
        return exact, "gv_labels", "exact_id"
    fallback = provisional.get((prefix, raw_genome))
    if fallback:
        return fallback, "provisional", "majority_derived"
    return {rank: "" for rank in ("phylum", "class", "order", "family", "genus", "species")}, "missing", "missing"


# 4. Source별 record를 같은 형식으로 stream
def source_inputs(source: str, spec: dict) -> tuple[list[Path], dict[str, dict[str, str]]]:
    path = project_path(spec["path"])
    files = fasta_files(path, spec.get("file_glob"))
    if not files:
        raise FileNotFoundError(f"No FASTA files for {source}: {path}")
    metadata = {}
    if spec.get("metadata_path") and source in {"ARC", "BAC", "EUK", "MITO", "PLASTID", "PHAGE"}:
        metadata = load_metadata(project_path(spec["metadata_path"]), "filename")
    if source in {"MITO", "PLASTID"}:
        listed = {name for name in metadata if name.startswith(f"{source}__")}
        found = {file.name for file in files}
        if found != listed:
            raise ValueError(
                f"{source} FASTA/metadata inventory mismatch: "
                f"FASTA_only={len(found - listed)}, metadata_only={len(listed - found)}"
            )
    return files, metadata


def main() -> int:
    args = parse_args()
    if OUTPUT.exists() and not args.force:
        raise SystemExit(f"{OUTPUT} already exists. Use --force to intentionally rebuild derived data.")

    data_config = load_yaml("data_sources.yaml")
    prep_config = load_yaml("preprocessing.yaml")
    active_sources = list(data_config["sources"])
    if args.sources:
        unknown = set(args.sources) - set(active_sources)
        if unknown:
            raise SystemExit(f"Unknown sources: {sorted(unknown)}")
        active_sources = args.sources

    gv_labels_path = project_path(data_config["metadata"]["gv_labels"])
    provisional_path = project_path(data_config["metadata"]["provisional_genome_taxonomy"])
    euk_exclusion_path = PROJECT_ROOT / "data" / "quarantine" / "eve_host_cellular_overlap.tsv"
    gv_labels = load_gv_labels(gv_labels_path)
    # Majority-derived labels may conservatively join NCLDV split groups, but optional
    # taxonomy training later accepts only authoritative gv_labels/snapshot_metadata.
    provisional = load_provisional(provisional_path)
    euk_exclusions = load_euk_exclusions(euk_exclusion_path)
    n_limit = float(prep_config["normalization"]["quarantine_if_n_fraction_above"])

    metadata_inputs = [gv_labels_path, provisional_path, euk_exclusion_path]
    source_files: list[Path] = [path for path in metadata_inputs if path.exists()]
    log_inputs: list[Path] = metadata_inputs.copy()
    for source in active_sources:
        spec = data_config["sources"][source]
        for raw_path in spec.get("supporting_files", []):
            support_path = project_path(raw_path)
            if not support_path.is_file():
                raise FileNotFoundError(f"Missing supporting file for {source}: {support_path}")
            source_files.append(support_path)
            log_inputs.append(support_path)
        log_inputs.append(project_path(spec["path"]))
        if spec.get("metadata_path"):
            log_inputs.append(project_path(spec["metadata_path"]))
    log_path = PROJECT_ROOT / "logs" / "preprocess" / f"01_build_manifest_{timestamp()}.log"
    tee = Tee(log_path)
    tee.start()
    started = print_run_header("preprocess/build_manifest", dict.fromkeys(log_inputs), OUTPUT)

    counts = Counter()
    record_ids = set()
    source_summary: dict[str, Counter] = {}
    temp = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    try:
        with gzip.open(temp, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
            writer.writeheader()

            for source in active_sources:
                spec = data_config["sources"][source]
                files, metadata_by_key = source_inputs(source, spec)
                source_files.extend(files)
                if spec.get("metadata_path"):
                    source_files.append(project_path(spec["metadata_path"]))
                summary = Counter()
                source_genomes = set()
                source_summary[source] = summary
                selector = spec.get("selector_prefix")

                for fasta_path in files:
                    file_metadata = metadata_by_key.get(fasta_path.name, {})
                    for header, raw_sequence in fasta_records(fasta_path):
                        if selector and not header.startswith(selector):
                            continue
                        sequence = normalize_dna(raw_sequence)
                        if not sequence:
                            summary["empty"] += 1
                            continue

                        genome_id, contig_id, host = parse_header(source, header, fasta_path.name)
                        raw_genome = genome_id.split("__", 1)[1]
                        metadata = file_metadata
                        if source == "PHAGE" and file_metadata:
                            raw_genome = file_metadata.get("uvig") or raw_genome

                        taxa, taxonomy_source, taxonomy_confidence = taxonomy_for(
                            source, raw_genome, metadata, gv_labels, provisional
                        )
                        forward_hash, canonical_hash = sequence_hashes(sequence)
                        n_fraction = sequence.count("N") / len(sequence)
                        exclusion = ""
                        if source == "EUK" and raw_genome in euk_exclusions:
                            exclusion = "eve_host_cellular_overlap"
                        elif n_fraction > n_limit:
                            exclusion = f"n_fraction_above_{n_limit}"
                        elif source == "PHAGE" and taxa.get("phylum") in {"Nucleocytoviricota", "Mirusviricota"}:
                            exclusion = "positive_virus_in_negative_source"
                        votu = metadata.get("votu", "")
                        if source == "NCLDV" and usable_genus(taxa.get("genus", "")):
                            base_group = f"NCLDV_GENUS::{taxa['genus']}"
                        elif source.startswith("EVE_"):
                            base_group = f"EVE_HOST::{host}"
                        elif source == "PHAGE" and votu:
                            base_group = f"PHAGE_VOTU::{votu}"
                        else:
                            base_group = f"GENOME::{genome_id}"

                        record_id = "REC_" + stable_id(source, header)
                        if record_id in record_ids:
                            raise ValueError(f"Duplicate record ID within source: {source}: {header}")
                        record_ids.add(record_id)
                        try:
                            stored_path = str(fasta_path.relative_to(PROJECT_ROOT))
                        except ValueError:
                            stored_path = str(fasta_path)
                        row = {
                            "record_id": record_id,
                            "source": source,
                            "prefix": source,
                            "source_path": stored_path,
                            "source_record_id": header,
                            "genome_id": genome_id,
                            "contig_id": contig_id,
                            "record_type": spec["record_type"],
                            "host_assembly_id": host,
                            "votu": votu,
                            "main_label": spec["main_label"],
                            "sub_label": "" if spec.get("sub_label") is None else spec["sub_label"],
                            **{f"taxonomy_{rank}": taxa.get(rank, "") for rank in ("phylum", "class", "order", "family", "genus", "species")},
                            "taxonomy_source": taxonomy_source,
                            "taxonomy_confidence": taxonomy_confidence,
                            "sequence_length": len(sequence),
                            "n_fraction": f"{n_fraction:.8f}",
                            "sequence_sha256": forward_hash,
                            "canonical_sha256": canonical_hash,
                            "base_group_id": base_group,
                            "similarity_component": "",
                            "split_group": "",
                            "split": "quarantine" if exclusion else "",
                            "exclusion_reason": exclusion,
                        }
                        writer.writerow(row)
                        source_genomes.add(genome_id)
                        summary["records"] += 1
                        summary["bases"] += len(sequence)
                        summary["quarantine"] += int(bool(exclusion))
                        counts["records"] += 1
                        counts["bases"] += len(sequence)

                summary["genomes"] = len(source_genomes)
                print(
                    f"[SOURCE] {source:20s} | records={summary['records']:,} | "
                    f"genomes={summary['genomes']:,} | bases={summary['bases']:,} "
                    f"| quarantine={summary['quarantine']:,}"
                )

        temp.replace(OUTPUT)
        unique_genomes = set()
        for row in read_tsv(OUTPUT):
            unique_genomes.add(row["genome_id"])
        summary_json = {
            source: {key: int(value) for key, value in values.items()}
            for source, values in source_summary.items()
        }
        write_json(PROJECT_ROOT / "data" / "manifests" / "source_summary.json", summary_json)
        write_json(PROJECT_ROOT / "data" / "manifests" / "inventory.json", {
            "sources": active_sources,
            "complete_source_inventory": set(active_sources) == set(data_config["sources"]),
            "raw_manifest_sha256": file_sha256(OUTPUT),
            "inputs": {str(p): {"bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in set(source_files)},
        })
        finish_stats(
            "preprocess/build_manifest",
            started,
            PROJECT_ROOT / "data" / "manifests" / "build_manifest_runtime.json",
            bases=counts["bases"],
            genomes=len(unique_genomes),
            contigs=counts["records"],
            input_bytes=existing_size(set(source_files)),
            extra={"manifest": str(OUTPUT), "sources": active_sources},
        )
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
