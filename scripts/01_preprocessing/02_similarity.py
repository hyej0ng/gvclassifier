#!/usr/bin/env python3
"""Pre-split Mirus genome ANI/AF and all-source fragment similarity audit."""

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
import shutil
import subprocess
import tempfile
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
RUN_STARTS = {}


# 이 스크립트에서 사용하는 설정·FASTA·hash·로그 보조 함수
def load_yaml(name):
    import yaml
    section = name.removesuffix(".yaml")
    path = CONFIG_DIR / ("pipeline.yaml" if section in {"training", "preprocessing", "split", "inference"} else name)
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
    def write(self, text):
        self.stdout.write(text); self.stdout.flush(); self.handle.write(text); self.handle.flush(); return len(text)
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
    write_json(output_json, report)
    print(f"[DONE] stage={stage} | runtime={report['runtime_hms']} | chunks={chunks:,} | bases={bases:,} | throughput={report['mbp_per_second']:.3f} Mbp/s")
    return report


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
def stable_id(*parts, length=20): return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:length]
def reverse_complement(sequence): return sequence.translate(RC_TRANS)[::-1]
def sequence_hashes(sequence):
    forward = hashlib.sha256(sequence.encode()).hexdigest(); reverse = reverse_complement(sequence)
    return forward, hashlib.sha256(min(sequence, reverse).encode()).hexdigest()
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

ROOT = PROJECT_ROOT / "data/manifests"
RAW = ROOT / "records.raw.tsv.gz"
EDGES = ROOT / "similarity_edges.tsv"
MIRUS_ANI_EDGES = ROOT / "mirus_genome_ani_edges.tsv"
WORK = PROJECT_ROOT / "data/preprocessed/similarity"


def stale_work_paths():
    """Return only known, derived leftovers from interrupted similarity runs."""
    if not WORK.exists():
        return []
    paths = []
    for path in WORK.iterdir():
        if (
            path.name == "audit_fragments.fna"
            or path.name.startswith(".mmseqs_")
            or path.name.startswith(".linclust_")
            or (path.name.startswith("hits_") and path.suffix == ".tsv")
        ):
            paths.append(path)
    edge_tmp = EDGES.with_suffix(".tmp")
    if edge_tmp.exists():
        paths.append(edge_tmp)
    return sorted(paths, key=lambda path: str(path))


def remove_stale_work(paths):
    """Delete only explicitly recognized derived files after --clean-stale."""
    allowed_roots = {WORK.resolve(), ROOT.resolve()}
    removed_bytes = 0
    for path in paths:
        if path.parent.resolve() not in allowed_roots:
            raise ValueError(f"Refusing to remove unexpected path: {path}")
        if path.is_symlink() or path.is_file():
            removed_bytes += path.stat().st_size
            path.unlink()
        elif path.is_dir():
            removed_bytes += sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            shutil.rmtree(path)
    return removed_bytes


def run_logged_command(command, log_path):
    """Run one external command and surface the log tail on failure."""
    print("[COMMAND] " + " ".join(map(str, command)), flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n[COMMAND] " + " ".join(map(str, command)) + "\n")
        handle.flush()
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        print("[ERROR] External command failed; last log lines:\n" + "\n".join(tail), flush=True)
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}. See {log_path}"
        )


def original_fragment_id(search_id, fragment_ids):
    """Validate and return one biological fragment ID emitted by Linclust."""
    if search_id not in fragment_ids:
        raise ValueError(f"Unexpected Linclust record ID: {search_id}")
    return search_id


def run_mirus_genome_ani(records, cfg, connect):
    """MIRUS의 모든 contig를 genome별로 묶고 skani ANI+AF edge를 만든다."""
    fields = [
        "left_genome", "right_genome", "ani", "af_shorter",
        "af_left", "af_right", "left_bp", "right_bp",
    ]
    ani_tmp = MIRUS_ANI_EDGES.with_suffix(".tmp")
    mirus_rows = [row for row in records.values() if row["source"] == "MIRUS"]
    with ani_tmp.open("w", encoding="utf-8", newline="") as detail_handle:
        detail_writer = csv.DictWriter(detail_handle, fieldnames=fields, delimiter="\t")
        detail_writer.writeheader()
        if not mirus_rows:
            ani_tmp.replace(MIRUS_ANI_EDGES)
            return {"genomes": 0, "comparisons_reported": 0, "accepted_pairs": 0, "bases": 0}

        executable = shutil.which(cfg["executable"])
        if not executable:
            raise SystemExit(
                "skani is required for the Mirus genome ANI/AF split: "
                "conda install -n GOholdout -c conda-forge -c bioconda 'skani>=0.3,<0.4'"
            )
        ani_cutoff = 100.0 * float(cfg["ani"])
        af_cutoff = 100.0 * float(cfg["aligned_fraction_shorter"])
        preset = str(cfg.get("preset", "medium"))
        if preset not in {"fast", "medium", "slow", "small-genomes"}:
            raise ValueError(f"Unsupported skani preset: {preset}")

        # 동일 source FASTA를 한 번만 읽으면서 genome별 임시 multi-FASTA를 만든다.
        rows_by_path = defaultdict(dict)
        for row in mirus_rows:
            rows_by_path[row["source_path"]][row["source_record_id"]] = row
        genome_ids = sorted({row["genome_id"] for row in mirus_rows})
        first_record = {}
        genome_bp = defaultdict(int)

        with tempfile.TemporaryDirectory(prefix=".mirus_skani_", dir=WORK) as temporary:
            temp_root = Path(temporary)
            genome_paths = {
                genome_id: temp_root / f"mirus_{index:06d}.fna"
                for index, genome_id in enumerate(genome_ids)
            }
            for source_path, header_rows in sorted(rows_by_path.items()):
                seen_headers = set()
                for header, raw in fasta_records(project_path(source_path)):
                    row = header_rows.get(header)
                    if row is None:
                        continue
                    sequence = normalize_dna(raw)
                    if sequence_hashes(sequence)[0] != row["sequence_sha256"]:
                        raise ValueError(f"Original Mirus sequence changed: {row['record_id']}")
                    genome_id = row["genome_id"]
                    # 한 genome의 모든 contig가 같은 파일에 들어가므로 skani가 genome 전체를 비교한다.
                    with genome_paths[genome_id].open("a", encoding="utf-8") as handle:
                        handle.write(f">{row['record_id']}\n{sequence}\n")
                    first_record.setdefault(genome_id, row["record_id"])
                    genome_bp[genome_id] += len(sequence)
                    seen_headers.add(header)
                missing = set(header_rows) - seen_headers
                if missing:
                    raise ValueError(f"Missing {len(missing)} Mirus records in {source_path}")

            genome_list = temp_root / "mirus_genomes.txt"
            with genome_list.open("w", encoding="utf-8") as handle:
                for genome_id in genome_ids:
                    handle.write(str(genome_paths[genome_id]) + "\n")
            path_to_genome = {str(path): genome_id for genome_id, path in genome_paths.items()}
            skani_output = temp_root / "skani_edges.tsv"
            command = [
                executable, "triangle", "-l", str(genome_list), "-o", str(skani_output),
                "--sparse", "--min-af", str(af_cutoff), f"--{preset}",
                "-t", str(cfg["threads"]),
            ]
            print("[COMMAND] " + " ".join(command), flush=True)
            with (WORK / "skani_mirus.log").open("w", encoding="utf-8") as skani_log:
                subprocess.run(command, stdout=skani_log, stderr=subprocess.STDOUT, check=True)

            comparisons = accepted = 0
            with skani_output.open(encoding="utf-8", newline="") as handle:
                for hit in csv.DictReader(handle, delimiter="\t"):
                    comparisons += 1
                    left = path_to_genome.get(hit["Ref_file"])
                    right = path_to_genome.get(hit["Query_file"])
                    if left is None or right is None:
                        raise ValueError("skani returned an unknown temporary Mirus genome path")
                    if left == right:
                        continue
                    ani = float(hit["ANI"])
                    af_left = float(hit["Align_fraction_ref"])
                    af_right = float(hit["Align_fraction_query"])
                    # 표준 vOTU 방식대로 두 genome 중 길이가 짧은 쪽의 aligned fraction을 사용한다.
                    af_shorter = af_left if genome_bp[left] <= genome_bp[right] else af_right
                    if ani < ani_cutoff or af_shorter < af_cutoff:
                        continue
                    detail_writer.writerow({
                        "left_genome": left,
                        "right_genome": right,
                        "ani": f"{ani:.6f}",
                        "af_shorter": f"{af_shorter:.6f}",
                        "af_left": f"{af_left:.6f}",
                        "af_right": f"{af_right:.6f}",
                        "left_bp": genome_bp[left],
                        "right_bp": genome_bp[right],
                    })
                    connect(first_record[left], first_record[right], "mirus_genome_ani_af")
                    accepted += 1

        ani_tmp.replace(MIRUS_ANI_EDGES)
        print(
            f"[MIRUS ANI] genomes={len(genome_ids):,} | comparisons_reported={comparisons:,} "
            f"| accepted_pairs={accepted:,} | ANI>={ani_cutoff:g}% | AF_shorter>={af_cutoff:g}%",
            flush=True,
        )
        return {
            "genomes": len(genome_ids), "comparisons_reported": comparisons,
            "accepted_pairs": accepted, "bases": sum(genome_bp.values()),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--clean-stale",
        action="store_true",
        help="Remove only recognized intermediate files left by an interrupted similarity run.",
    )
    args = parser.parse_args()
    settings = load_yaml("preprocessing.yaml")
    cfg = settings["similarity"]
    if EDGES.exists() and not args.force:
        raise SystemExit("Similarity results exist. Rebuilding requires --force.")
    executable = shutil.which(cfg["executable"])
    if not executable:
        raise SystemExit("MMseqs2 is required: conda install -c conda-forge -c bioconda mmseqs2")
    WORK.mkdir(parents=True, exist_ok=True)
    if cfg.get("method") != "linclust_connected_component":
        raise ValueError("similarity.method must be linclust_connected_component")
    if int(cfg.get("cluster_mode", -1)) != 1:
        raise ValueError("cluster_mode=1 is required so transitive near matches stay in one split")
    if cfg.get("strand_handling") != "mmseqs_nucleotide_native":
        raise ValueError("strand_handling must be mmseqs_nucleotide_native")
    memory_limit = str(cfg.get("memory_limit", ""))
    if not re.fullmatch(r"[1-9][0-9]*[KMGT]", memory_limit, flags=re.IGNORECASE):
        raise ValueError("similarity.memory_limit must look like 48G or 800M")

    stale_paths = stale_work_paths()
    if stale_paths and not args.clean_stale:
        preview = "\n".join(f"  - {path}" for path in stale_paths[:20])
        raise SystemExit(
            "Interrupted-run files exist. Inspect them, then rerun with --clean-stale "
            "to remove only these derived files:\n" + preview
        )
    if stale_paths:
        removed_bytes = remove_stale_work(stale_paths)
        print(
            f"[CLEAN] removed {len(stale_paths)} stale derived path(s), "
            f"{removed_bytes / (1024 ** 3):.2f} GiB",
            flush=True,
        )

    tee = Tee(PROJECT_ROOT / "logs/preprocess" / f"02_similarity_{timestamp()}.log")
    tee.start()
    started = print_run_header("preprocess/similarity", [RAW], WORK)
    try:
        records = {r["record_id"]: r for r in read_tsv(RAW) if r["split"] != "quarantine"}
        paths = defaultdict(set)
        for row in records.values():
            paths[row["source_path"]].add(row["source"])
        seen_hash = {}
        excluded = {}
        fragment_ids = set()
        written_edges = set()
        fragment_count = 0
        search_sequence_count = 0
        bases = 0
        fasta = WORK / "audit_fragments.fna"
        edge_tmp = EDGES.with_suffix(".tmp")
        with fasta.open("w") as fa, edge_tmp.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=["left_group", "right_group", "reason"], delimiter="\t")
            writer.writeheader()

            def connect(a, b, reason):
                ra, rb = records[a], records[b]
                la = (ra["main_label"], ra["sub_label"])
                lb = (rb["main_label"], rb["sub_label"])
                if la != lb:
                    excluded[a] = excluded[b] = "cross_label_" + reason
                    return
                pair = tuple(sorted((ra["base_group_id"], rb["base_group_id"])))
                if pair[0] != pair[1] and pair not in written_edges:
                    written_edges.add(pair)
                    writer.writerow({"left_group": pair[0], "right_group": pair[1], "reason": reason})

            # 2. 일반 Mirus는 5 kb 조각이 아니라 모든 contig를 합친 genome 전체 ANI+AF도 검사한다.
            mirus_stats = {"enabled": bool(cfg["mirus_genome_ani"]["enabled"])}
            if mirus_stats["enabled"]:
                mirus_stats.update(run_mirus_genome_ani(records, cfg["mirus_genome_ani"], connect))
            else:
                with MIRUS_ANI_EDGES.open("w", encoding="utf-8") as handle:
                    handle.write("left_genome\tright_genome\tani\taf_shorter\taf_left\taf_right\tleft_bp\tright_bp\n")

            # 3. 모든 source의 분할용 audit window를 만든다. 아직 학습 chunk는 만들지 않는다.
            total_paths = len(paths)
            for path_number, (source_path, sources) in enumerate(sorted(paths.items()), start=1):
                for header, raw in fasta_records(project_path(source_path)):
                    for source in sorted(sources):
                        rid = "REC_" + stable_id(source, header)
                        if rid not in records:
                            continue
                        sequence = normalize_dna(raw)
                        if sequence_hashes(sequence)[0] != records[rid]["sequence_sha256"]:
                            raise ValueError(f"Original sequence changed: {rid}")
                        for start, end in chunk_ranges(len(sequence), settings["chunking"]["size_nt"], settings["chunking"]["minimum_tail_nt"]):
                            fragment = sequence[start:end]
                            digest = sequence_hashes(fragment)[1]
                            if digest in seen_hash:
                                connect(rid, seen_hash[digest], "exact_fragment")
                                continue
                            seen_hash[digest] = rid
                            # 짧은 원본 contig도 보존하지만 근접유사도 검색은 500nt 이상만 수행한다.
                            if len(fragment) < settings["chunking"]["minimum_tail_nt"]:
                                continue
                            fid = rid + ":" + str(start)
                            fragment_ids.add(fid)
                            # MMseqs nucleotide Linclust는 reverse-complement strand도 처리한다.
                            # 한 fragment를 한 번만 기록해 검색 입력을 불필요하게 두 배로 만들지 않는다.
                            fa.write(f">{fid}\n{fragment}\n")
                            fragment_count += 1
                            search_sequence_count += 1
                            bases += len(fragment)
                if path_number % 100 == 0 or path_number == total_paths:
                    print(
                        f"[AUDIT] files={path_number:,}/{total_paths:,} | "
                        f"unique_fragments={fragment_count:,} | bases={bases:,}", flush=True
                    )

            # 4. 대용량 all-vs-all search 대신 Linclust connected components를 만든다.
            # identity/양방향 coverage 기준은 그대로 유지하고 split당 메모리를 제한한다.
            linclust_log = WORK / "linclust.log"
            linclust_log.write_text(
                f"[START] {now_local()} | fragments={fragment_count} | "
                f"search_sequences={search_sequence_count} | memory_limit={memory_limit}\n",
                encoding="utf-8",
            )
            temporary = tempfile.TemporaryDirectory(prefix=".linclust_", dir=WORK)
            cluster_stats = {
                "clusters": 0,
                "multi_record_clusters": 0,
                "mixed_label_clusters": 0,
                "record_links": 0,
            }
            try:
                temp_root = Path(temporary.name)
                sequence_db = temp_root / "fragment_db"
                cluster_db = temp_root / "fragment_cluster"
                cluster_tmp = temp_root / "work"
                cluster_tsv = temp_root / "clusters.tsv"

                run_logged_command(
                    [
                        executable, "createdb", str(fasta), str(sequence_db),
                        "--dbtype", "2", "--shuffle", "0", "--createdb-mode", "0",
                    ],
                    linclust_log,
                )
                run_logged_command(
                    [
                        executable, "linclust", str(sequence_db), str(cluster_db), str(cluster_tmp),
                        "--min-seq-id", str(cfg["identity"]),
                        "-c", str(cfg["coverage"]), "--cov-mode", "0",
                        "--cluster-mode", str(cfg["cluster_mode"]),
                        "--similarity-type", "2",
                        "--kmer-per-seq", str(cfg["kmer_per_sequence"]),
                        "--kmer-per-seq-scale", str(cfg["kmer_per_sequence_scale"]),
                        "--split-memory-limit", memory_limit,
                        "--threads", str(cfg["threads"]),
                        "--remove-tmp-files", "1",
                    ],
                    linclust_log,
                )
                run_logged_command(
                    [
                        executable, "createtsv", str(sequence_db), str(sequence_db),
                        str(cluster_db), str(cluster_tsv), "--threads", str(cfg["threads"]),
                    ],
                    linclust_log,
                )

                def process_cluster(search_members):
                    if not search_members:
                        return
                    cluster_stats["clusters"] += 1
                    record_ids = sorted({
                        original_fragment_id(member, fragment_ids).rsplit(":", 1)[0]
                        for member in search_members
                    })
                    if len(record_ids) < 2:
                        return
                    cluster_stats["multi_record_clusters"] += 1
                    labels = {
                        (records[record_id]["main_label"], records[record_id]["sub_label"])
                        for record_id in record_ids
                    }
                    if len(labels) > 1:
                        cluster_stats["mixed_label_clusters"] += 1
                        for record_id in record_ids:
                            excluded[record_id] = "cross_label_near_fragment"
                        return
                    anchor = record_ids[0]
                    for record_id in record_ids[1:]:
                        connect(anchor, record_id, "near_fragment_linclust")
                        cluster_stats["record_links"] += 1

                current_representative = None
                current_members = []
                with cluster_tsv.open(encoding="utf-8") as cluster_handle:
                    for line_number, raw_line in enumerate(cluster_handle, start=1):
                        fields = raw_line.rstrip("\n").split("\t")
                        if len(fields) != 2:
                            raise ValueError(f"Malformed Linclust TSV line {line_number}")
                        representative, member = fields
                        if current_representative is None:
                            current_representative = representative
                        elif representative != current_representative:
                            process_cluster(current_members)
                            current_representative = representative
                            current_members = []
                        current_members.append(member)
                process_cluster(current_members)
            finally:
                # 정상적인 Python 예외까지는 즉시 정리한다. SIGKILL 잔여물은 --clean-stale로만 삭제한다.
                temporary.cleanup()

            # 큰 audit FASTA는 최종 산출물이 아니므로 성공한 경우에만 지운다.
            fasta.unlink()
            print(
                f"[LINCLUST] fragments={fragment_count:,} | search_sequences={search_sequence_count:,} "
                f"| clusters={cluster_stats['clusters']:,} | "
                f"multi_record={cluster_stats['multi_record_clusters']:,} | "
                f"mixed_label={cluster_stats['mixed_label_clusters']:,}",
                flush=True,
            )
        edge_tmp.replace(EDGES)
        write_json(ROOT / "similarity_exclusions.json", excluded)
        write_json(ROOT / "similarity_audit.json", {
            "raw_manifest_sha256": file_sha256(RAW),
            "edge_sha256": file_sha256(EDGES),
            "exclusions_sha256": file_sha256(ROOT / "similarity_exclusions.json"),
            "mirus_ani_edges_sha256": file_sha256(MIRUS_ANI_EDGES),
            "settings": cfg, "complete": True,
            "units": ["Mirus_whole_genome_ANI_AF", "all_source_5kb_fragment_similarity"],
            "method": "skani_Mirus_genome_ANI_AF_plus_MMseqs_Linclust_bidirectional_fragments_plus_exact_hash",
            "mirus_genome_ani": mirus_stats,
            "linclust": cluster_stats,
            "fragment_count": fragment_count,
            "search_sequence_count": search_sequence_count,
        })
        finish_stats("preprocess/similarity", started, WORK / "runtime.json", bases=bases,
                     chunks=fragment_count,
                     extra={"excluded_records": len(excluded), "mirus_genome_ani": mirus_stats,
                            "linclust": cluster_stats, "search_sequence_count": search_sequence_count,
                            "memory_limit": memory_limit})
    finally:
        tee.close()


if __name__ == "__main__":
    main()
