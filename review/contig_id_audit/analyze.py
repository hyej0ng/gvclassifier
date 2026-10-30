"""Read-only audit of contig aggregation key collisions across all sources."""
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data/manifests/genomes.tsv.gz"
KEEP_STEP = 1 << 32
MASK = KEEP_STEP - 1


def rows():
    with gzip.open(MANIFEST, "rt") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def key(row):
    return row["genome_id"] + "\x1f" + row["contig_id"]


def summary(groups):
    sizes = [len(rr) for rr in groups.values()]
    return {"records": sum(sizes), "contig_keys": len(sizes),
            "collision_groups": sum(n > 1 for n in sizes),
            "records_in_collision_groups": sum(n for n in sizes if n > 1),
            "merged_excess_records": sum(n - 1 for n in sizes)}


def main():
    counts = {}
    totals = defaultdict(Counter)
    for row in rows():
        kept = row["split"] != "quarantine"
        assert row["genome_id"].startswith(row["prefix"] + "__")
        k = key(row)
        counts[k] = counts.get(k, 0) + 1 + (KEEP_STEP if kept else 0)
        totals[row["prefix"]]["input_records"] += 1
        totals[row["prefix"]]["kept_records"] += kept
    print("First pass complete: all input records counted", flush=True)
    duplicate_keys = {k for k, n in counts.items() if (n & MASK) > 1}
    for k, n in counts.items():
        source = k.split("__", 1)[0]
        raw_n, kept_n = n & MASK, n >> 32
        totals[source]["input_unique_keys"] += 1
        totals[source]["kept_unique_keys"] += kept_n > 0
        for tag, size in (("input", raw_n), ("kept", kept_n)):
            totals[source][tag + "_collision_groups"] += size > 1
            totals[source][tag + "_collision_records"] += size if size > 1 else 0
            totals[source][tag + "_merged_excess"] += max(0, size - 1)
    del counts
    duplicates = defaultdict(list)
    for row in rows():
        if key(row) in duplicate_keys:
            duplicates[key(row)].append(row)
    print("Second pass complete: collision provenance collected", flush=True)

    details = []
    by_split = defaultdict(lambda: defaultdict(list))
    for k, rr in duplicates.items():
        source = rr[0]["prefix"]
        # Unique full tokens + common second pipe token identify the EVE failure mode.
        tokens = {r["source_record_id"].split()[0] for r in rr}
        suffixes = {"|".join(t.split("|")[2:]) for t in tokens}
        same_pipe_truncation = len(tokens) > 1 and len(suffixes) > 1
        for row in rr:
            details.append({"source": source, "split": row["split"],
                            "genome_id": row["genome_id"], "contig_id": row["contig_id"],
                            "record_id": row["record_id"], "sequence_length": row["sequence_length"],
                            "sequence_sha256": row["sequence_sha256"],
                            "source_record_id": row["source_record_id"],
                            "pipe_truncation_group": same_pipe_truncation})
            by_split[source, row["split"]][k].append(row)
        totals[source]["pipe_truncation_collision_groups"] += same_pipe_truncation
        totals[source]["distinct_sequence_collision_groups"] += len({r["sequence_sha256"] for r in rr}) > 1
        if len(tokens) == 1:
            totals[source]["same_token_collision_groups"] += 1

    fields = list(details[0])
    with (OUT / "collision_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(details)
    result = {"manifest": str(MANIFEST.relative_to(ROOT)),
              "sources": dict(sorted(totals.items())),
              "collision_records_by_split": {
                  source + "/" + split: summary(groups)
                  for (source, split), groups in sorted(by_split.items())},
              "examples": {
                  source: [{f: r[f] for f in ("genome_id", "contig_id", "source_record_id", "sequence_length")}
                           for rr in duplicates.values() if rr[0]["prefix"] == source for r in rr][:4]
                  for source in totals if totals[source]["input_collision_groups"]},
              "note": "All original records, including quarantine, are retained in this final manifest; "
                      "input counts are checked against source_summary.json."}
    original = json.loads((ROOT / "data/manifests/source_summary.json").read_text())
    for source, stats in totals.items():
        assert stats["input_records"] == original[source]["records"]
        assert stats["kept_records"] - stats["kept_unique_keys"] == stats["kept_merged_excess"]
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
