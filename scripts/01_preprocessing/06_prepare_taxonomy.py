#!/usr/bin/env python3
"""Optional rank-specific datasets, using only authoritative labels and the primary frozen split."""

# 1. 경로 및 설정
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


# 이 스크립트에서 사용하는 설정·checksum 보조 함수
def load_yaml(name):
    import yaml
    section = name.removesuffix(".yaml"); path = CONFIG_DIR / ("pipeline.yaml" if section in {"training", "preprocessing", "split", "inference"} else name)
    with path.open(encoding="utf-8") as handle: value = yaml.safe_load(handle)
    return value[section] if section in {"training", "preprocessing", "split", "inference"} else value
def open_text(path): return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(encoding="utf-8", newline="")
def read_tsv(path):
    with open_text(path) as handle: yield from csv.DictReader(handle, delimiter="\t")
def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()
def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); temp.replace(path)
def verify_data(task, splits):
    root = PROJECT_ROOT / "data/preprocessed"
    qc = json.loads((root / "qc_report.json").read_text())
    fingerprints = json.loads((root / "dataset_checksums.json").read_text())
    if not qc.get("passed") or task not in {row["task"] for row in qc["tasks"] if row["passed"]}:
        raise ValueError("Preprocessing QC has not passed for this task")
    if qc.get("dataset_checksums_sha256") != file_sha256(root / "dataset_checksums.json"):
        raise ValueError("QC does not match current datasets")
    if fingerprints["preprocessing"] != load_yaml("preprocessing.yaml"):
        raise ValueError("Preprocessing config changed; regenerate datasets/QC")
    manifest = PROJECT_ROOT / "data/manifests/genomes.tsv.gz"
    if fingerprints["manifest_sha256"] != file_sha256(manifest): raise ValueError("Manifest changed after chunking")
    frozen = json.loads((PROJECT_ROOT / "data/splits/frozen.json").read_text())
    if frozen["split_config"] != load_yaml("split.yaml"): raise ValueError("Split config changed after freezing")
    inventory = json.loads((PROJECT_ROOT / "data/manifests/inventory.json").read_text())
    if not inventory["complete_source_inventory"]: raise ValueError("Pilot source subset cannot be used as a complete experiment")
    checked = {}
    for split in splits:
        key = f"data/preprocessed/{task}/{split}.csv.gz"; actual = file_sha256(PROJECT_ROOT / key)
        if actual != fingerprints["files"][key]: raise ValueError(f"Dataset checksum mismatch: {key}")
        checked[split] = actual
    return {"manifest_sha256": fingerprints["manifest_sha256"], "files": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["NCLDV", "MIRUS"], required=True)
    parser.add_argument("--rank", choices=["class", "order", "family", "genus"], required=True)
    args = parser.parse_args()
    parent = verify_data("sub", ["train", "validation", "inference"])
    output = PROJECT_ROOT / "data/preprocessed/taxonomy" / args.source / args.rank
    if output.exists():
        raise SystemExit("Taxonomy dataset already exists; preserve the frozen version.")
    labels_by_genome = defaultdict(set)
    train_genomes = defaultdict(set)
    for row in read_tsv(PROJECT_ROOT / "data/manifests/genomes.tsv.gz"):
        if row["source"] != args.source or row["split"] == "quarantine":
            continue
        value = row["taxonomy_" + args.rank].strip()
        if row["taxonomy_source"] not in {"gv_labels", "snapshot_metadata"}:
            continue
        if value.lower() in {"", "na", "unknown", "unclassified", "(미분류)",
                             "mirus", "ncldv", "c_", "o_", "f_", "g_"} or "unclassified" in value.lower():
            continue
        labels_by_genome[row["genome_id"]].add(value)
        if row["split"] == "train":
            train_genomes[value].add(row["genome_id"])
    reliable = {g: next(iter(values)) for g, values in labels_by_genome.items() if len(values) == 1}
    minimum = load_yaml("pipeline.yaml")["taxonomy"]["minimum_train_genomes_per_taxon"]
    vocabulary = sorted(name for name, genomes in train_genomes.items() if len(genomes) >= minimum)
    if len(vocabulary) < 2:
        raise SystemExit("Fewer than two usable training taxa at this rank; use a higher rank or curate labels.")
    label_to_id = {name:i for i,name in enumerate(vocabulary)}
    output.mkdir(parents=True)
    counts = {}
    for split in ["train", "validation", "inference"]:
        known = unseen = 0
        source_path = PROJECT_ROOT / "data/preprocessed/sub" / f"{split}.csv.gz"
        with gzip.open(source_path, "rt") as inp, gzip.open(output / f"{split}.csv.gz", "wt") as out, gzip.open(output / f"unknown_{split}.csv.gz", "wt") as unknown_out:
            reader = csv.DictReader(inp)
            writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
            unknown_writer = csv.DictWriter(unknown_out, fieldnames=reader.fieldnames)
            writer.writeheader()
            unknown_writer.writeheader()
            for row in reader:
                if row["source"] != args.source or row["genome_id"] not in reliable:
                    continue
                row["labels"] = label_to_id.get(reliable[row["genome_id"]], -1)
                if row["labels"] < 0:
                    unseen += 1
                    unknown_writer.writerow(row)
                    if split != "inference":
                        continue
                else:
                    known += 1
                writer.writerow(row)
        counts[split] = {"known_chunks": known, "unseen_taxon_chunks": unseen}
    if not counts["validation"]["known_chunks"]:
        raise SystemExit("No seen-taxon validation examples. This strict split cannot train this closed-set rank; use a higher rank.")
    write_json(output / "taxonomy.json", {"source": args.source, "rank": args.rank,
        "labels": vocabulary, "parent_manifest": parent["manifest_sha256"], "counts": counts,
        "files": {s: file_sha256(output / f"{s}.csv.gz") for s in ["train","validation","inference"]}})
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
