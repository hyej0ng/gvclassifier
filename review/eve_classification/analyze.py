"""Audit EVE holdout predictions and recover locus-level aggregation by record ID.

Existing inference and preprocessing outputs are read without modification.
"""
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = ("EVE_NCLDV", "EVE_MIRUS")
RUNS = {"main": "main1_10191118", "sub": "sub1_10191116"}


def ranges(length, size=5000, tail=500):
    if length <= size:
        yield 0, length
        return
    full, remainder = divmod(length, size)
    if 0 < remainder < tail:
        for index in range(full - 1):
            yield index * size, (index + 1) * size
        start = (full - 1) * size
        middle = start + (size + remainder) // 2
        yield start, middle
        yield middle, length
    else:
        for start in range(0, length, size):
            yield start, min(start + size, length)


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    records = defaultdict(list)
    with gzip.open(ROOT / "data/manifests/genomes.tsv.gz", "rt") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["prefix"] in SOURCES:
                records[row["prefix"], row["split"]].append(row)

    inventory = []
    chunk_map = {}
    inference_records = {}
    for source in SOURCES:
        for split in ("train", "validation", "inference", "total"):
            rows = (records[source, split] if split != "total" else
                    sum((records[source, s] for s in ("train", "validation", "inference")), []))
            inventory.append({
                "source": source, "split": split,
                "host_assemblies": len({r["genome_id"] for r in rows}),
                "eve_records_loci": len(rows),
                "existing_contig_keys": len({(r["genome_id"], r["contig_id"]) for r in rows}),
                "chunks": sum(sum(1 for _ in ranges(int(r["sequence_length"]))) for r in rows),
                "bases": sum(int(r["sequence_length"]) for r in rows),
            })
        for row in records[source, "inference"]:
            rid = row["record_id"]
            inference_records[rid] = row
            for start, end in ranges(int(row["sequence_length"])):
                digest = hashlib.sha256("\x1f".join(map(str, (rid, start, end))).encode()).hexdigest()[:20]
                cid = "CHK_" + digest
                assert cid not in chunk_map
                chunk_map[cid] = (rid, end - start)

    saved_metrics = []
    locus_metrics = []
    locus_predictions = []
    for task, run in RUNS.items():
        classes = 2 if task == "main" else 3
        weights = Counter()
        scores = defaultdict(lambda: [0.0] * classes)
        chunk_counts = Counter()
        for level in ("chunk", "contig", "genome"):
            frame = defaultdict(list)
            path = ROOT / "results" / run / task / level
            with gzip.open(path / "predictions.csv.gz", "rt") as handle:
                for row in csv.DictReader(handle):
                    if row["source"] in SOURCES:
                        frame[row["source"]].append(row)
            metrics = json.loads((path / "metrics.json").read_text())
            for source in SOURCES:
                rows = frame[source]
                pred = Counter(int(r["predicted_label"]) for r in rows)
                correct = sum(r["true_label"] == r["predicted_label"] for r in rows)
                assert len(rows) == metrics["by_source"][source]["n"]
                assert math.isclose(correct / len(rows), metrics["by_source"][source]["accuracy"])
                expected = next(x for x in inventory if x["source"] == source and x["split"] == "inference")
                assert sum(int(r["valid_length"]) for r in rows) == expected["bases"]
                saved_metrics.append({"task": task, "run": run, "source": source,
                                      "level": level, "n": len(rows), "correct": correct,
                                      "accuracy_recall": correct / len(rows),
                                      **{f"predicted_{i}": pred[i] for i in range(classes)}})
                if level == "chunk":
                    for row in rows:
                        rid, length = chunk_map[row["chunk_id"]]
                        assert length == int(row["valid_length"])
                        assert row["genome_id"] == inference_records[rid]["genome_id"]
                        assert source == inference_records[rid]["prefix"]
                        weights[rid] += length
                        chunk_counts[row["chunk_id"]] += 1
                        for i in range(classes):
                            scores[rid][i] += length * math.log(max(1e-12, min(1.0, float(row[f"p{i}"]))))
        assert set(chunk_counts) == set(chunk_map) and all(n == 1 for n in chunk_counts.values())
        assert set(weights) == set(inference_records)
        for rid, row in inference_records.items():
            assert weights[rid] == int(row["sequence_length"])
            predicted = max(range(classes), key=lambda i: scores[rid][i])
            target = 1 if task == "main" else int(row["sub_label"])
            locus_predictions.append({"task": task, "source": row["prefix"],
                                      "record_id": rid, "genome_id": row["genome_id"],
                                      "contig_id": row["contig_id"],
                                      "source_record_id": row["source_record_id"],
                                      "bases": weights[rid], "true_label": target,
                                      "predicted_label": predicted})
        for source in SOURCES:
            rows = [r for r in locus_predictions if r["task"] == task and r["source"] == source]
            pred = Counter(r["predicted_label"] for r in rows)
            correct = sum(r["true_label"] == r["predicted_label"] for r in rows)
            locus_metrics.append({"task": task, "run": run, "source": source,
                                  "level": "contig_record_id", "n": len(rows), "correct": correct,
                                  "accuracy_recall": correct / len(rows),
                                  **{f"predicted_{i}": pred[i] for i in range(classes)}})

    # Main/sub CSV columns differ, so JSON preserves the full prediction counts.
    result = {"runs": RUNS, "inventory": inventory, "saved_metrics": saved_metrics,
              "locus_metrics": locus_metrics,
              "verification": {"all_saved_metrics_match_prediction_csv": True,
                               "all_eve_holdout_chunks_mapped_once_per_task": True,
                               "all_locus_lengths_match_manifest": True},
              "notes": ["Primary contig results use original record_id (one FASTA EVE locus).",
                        "Existing contig results group by genome_id and contig_id; some IDs collide.",
                        "Genome means EVE records grouped within a host assembly, not its whole genome.",
                        "Source-subset accuracy equals target-class recall; source-specific precision is not estimated.",
                        "Main and sub were evaluated independently; these are not cascade metrics."]}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    write_csv("data_inventory.csv", inventory)
    write_csv("locus_predictions.csv", locus_predictions)
    for task in RUNS:
        rows = [r for r in saved_metrics + locus_metrics if r["task"] == task]
        write_csv(f"{task}_metrics.csv", rows)
    lines = [
        "# EVE main1/sub1 holdout 분석", "",
        "main: main1_10191118; sub: sub1_10191116. 두 모델은 독립적으로 평가했다.", "",
        "정답률은 해당 source의 target-class recall과 같다. source만 선택하면 참 라벨이 한 종류이므로, "
        "그 subset의 precision/F1을 EVE를 다른 source와 구별하는 성능으로 해석하지 않는다.", "",
        "contig 표는 원본 FASTA 레코드(record_id), 즉 EVE locus별로 재집계한 값이다. "
        "chunk_id를 frozen manifest의 record_id·좌표에서 복원해 모든 예측에 한 번씩 대응시켰다. "
        "집계는 기존 코드와 같은 valid_length 가중 평균 log-probability를 사용했다.", "",
        "기존 평가 코드는 genome_id+contig_id로 묶는다. EVE 헤더를 |로 나눈 첫 contig 토큰만 "
        "사용하면서 서로 다른 좌위에 같은 contig_id가 생긴다. holdout에서 EVE_NCLDV 151개가 "
        "59개로, EVE_MIRUS 28개가 16개로 합쳐졌다. chunk 및 genome 결과는 기존 값을 사용했다.", "",
        "EVE의 genome은 동일 host assembly 내 EVE들의 묶음이며 숙주 genome 전체의 예측이 아니다.", "",
    ]
    for task in RUNS:
        lines += [f"## {task}", "",
                  "| Source | 기준 | 평가 수 | 정답 수 | 정답률/Recall | 오분류 |",
                  "|---|---|---:|---:|---:|---|"]
        for source in SOURCES:
            for level in ("chunk", "contig_record_id", "genome"):
                r = next(r for r in saved_metrics + locus_metrics
                         if r["task"] == task and r["source"] == source and r["level"] == level)
                level_name = "contig(EVE locus)" if level == "contig_record_id" else level
                if task == "main":
                    errors = f"Cellular {r['predicted_0']:,}"
                elif source == "EVE_NCLDV":
                    errors = f"Mirus {r['predicted_1']:,}; PHAGE/Other Virus {r['predicted_2']:,}"
                else:
                    errors = f"NCLDV {r['predicted_0']:,}; PHAGE/Other Virus {r['predicted_2']:,}"
                lines.append(f"| {source} | {level_name} | {r['n']:,} | {r['correct']:,} | "
                             f"{r['accuracy_recall']:.2%} | {errors} |")
        lines.append("")
    lines += ["## 사용 EVE 데이터", "",
              "아래는 전처리 후 자료이며 main/sub가 같은 EVE 레코드를 사용한다. 합산하지 않는다.", "",
              "| Source | Split | Host assemblies | EVE loci | Chunks | Bases |",
              "|---|---|---:|---:|---:|---:|"]
    for r in inventory:
        lines.append(f"| {r['source']} | {r['split']} | {r['host_assemblies']:,} | "
                     f"{r['eve_records_loci']:,} | {r['chunks']:,} | {r['bases']:,} |")
    lines += ["", "## 기존 저장 contig 결과와 비교", "",
              "| Task | Source | 기존 contig 정답/수(정답률) | record_id별 재집계 정답/수(정답률) |",
              "|---|---|---:|---:|"]
    for new in locus_metrics:
        old = next(r for r in saved_metrics if r["task"] == new["task"]
                   and r["source"] == new["source"] and r["level"] == "contig")
        lines.append(f"| {new['task']} | {new['source']} | {old['correct']}/{old['n']} "
                     f"({old['accuracy_recall']:.2%}) | {new['correct']}/{new['n']} "
                     f"({new['accuracy_recall']:.2%}) |")
    lines += ["", "## 검증", "",
              "기존 12개 source/level 결과는 prediction CSV와 metrics JSON이 모두 일치했다. "
              "각 모델의 EVE holdout chunk 2,285개를 179개 원본 locus에 빠짐없이 한 번씩 대응시켰고, "
              "모든 locus의 chunk 길이 합이 manifest 길이와 일치했다.", "",
              "기존 데이터·평가 파일과 파이프라인 코드는 변경하지 않았다. 이 폴더에 분석 산출물만 생성했다.", ""]
    (OUT / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
