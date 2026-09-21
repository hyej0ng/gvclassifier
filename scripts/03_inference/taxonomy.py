#!/usr/bin/env python3
"""Optional single-rank taxonomy evaluation, independent of the primary classification score."""

# 1. 경로 및 설정
import argparse
import datetime as dt
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Iterable
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR / "04_evaluation"))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
RUN_STARTS = {}


# 이 스크립트에서 사용하는 설정·모델·검증·로그 보조 함수
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
    try:
        import torch
        gpu_peak = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
    except ImportError: gpu_peak = 0
    report = {**RUN_STARTS.get(started, {}), "stage": stage, "completed_local": now_local(), "completed_utc": now_utc(),
              "runtime_seconds": seconds, "runtime_hms": str(dt.timedelta(seconds=round(seconds))), "input_bytes": int(input_bytes),
              "bases": int(bases), "genomes": int(genomes), "contigs": int(contigs), "sequences": int(contigs), "chunks": int(chunks),
              "mbp_per_second": mbp / seconds, "seconds_per_mbp": seconds / mbp if mbp else None,
              "peak_cpu_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "peak_gpu_memory_bytes": gpu_peak, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if extra: report.update(extra)
    write_json(output_json, report); print(f"[DONE] stage={stage} | runtime={report['runtime_hms']} | chunks={chunks:,} | bases={bases:,}"); return report
def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()
def verify_data(task, splits):
    root = PROJECT_ROOT / "data/preprocessed"
    if task.startswith("taxonomy/"):
        verify_data("sub", ["train", "validation"])
        meta = json.loads((root / task / "taxonomy.json").read_text()); current = json.loads((root / "dataset_checksums.json").read_text())
        if meta["parent_manifest"] != current["manifest_sha256"]: raise ValueError("Taxonomy derives from another manifest")
        hashes = {}
        for split in splits:
            actual = file_sha256(root / task / f"{split}.csv.gz")
            if actual != meta["files"][split]: raise ValueError("Taxonomy dataset changed")
            hashes[split] = actual
        return {"manifest_sha256": meta["parent_manifest"], "files": hashes}
    qc = json.loads((root / "qc_report.json").read_text()); fingerprints = json.loads((root / "dataset_checksums.json").read_text())
    if not qc.get("passed") or task not in {row["task"] for row in qc["tasks"] if row["passed"]}: raise ValueError("Preprocessing QC has not passed for this task")
    if qc.get("dataset_checksums_sha256") != file_sha256(root / "dataset_checksums.json"): raise ValueError("QC does not match current datasets")
    if fingerprints["preprocessing"] != load_yaml("preprocessing.yaml"): raise ValueError("Preprocessing config changed; regenerate datasets/QC")
    if fingerprints["manifest_sha256"] != file_sha256(PROJECT_ROOT / "data/manifests/genomes.tsv.gz"): raise ValueError("Manifest changed after chunking")
    if json.loads((PROJECT_ROOT / "data/splits/frozen.json").read_text())["split_config"] != load_yaml("split.yaml"): raise ValueError("Split config changed after freezing")
    if not json.loads((PROJECT_ROOT / "data/manifests/inventory.json").read_text())["complete_source_inventory"]: raise ValueError("Pilot source subset cannot be used as a complete experiment")
    checked = {}
    for split in splits:
        key = f"data/preprocessed/{task}/{split}.csv.gz"; actual = file_sha256(PROJECT_ROOT / key)
        if actual != fingerprints["files"][key]: raise ValueError(f"Dataset checksum mismatch: {key}")
        checked[split] = actual
    return {"manifest_sha256": fingerprints["manifest_sha256"], "files": checked}
def predict_batches(model, tokenizer, dataset, batch_size, max_tokens, device):
    import torch
    model.eval()
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        encoded = tokenizer(batch["sequence"], padding=True, truncation=False, return_tensors="pt", return_token_type_ids=False)
        if encoded["input_ids"].shape[1] > max_tokens: raise ValueError("Token limit exceeded; no sequence truncation is allowed")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode(): probabilities = torch.softmax(model(**encoded).logits.float(), dim=-1).cpu().numpy()
        yield start, batch, probabilities
def load_finetuned_model(run, metadata, n_labels, device):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    cfg = metadata["training"]["model"]; repo_id, revision = cfg["name_or_path"], cfg.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision, trust_remote_code=True, padding_side="right", model_max_length=int(cfg["max_input_tokens"]), token=False)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, revision=revision, num_labels=n_labels, trust_remote_code=True, torch_dtype=torch.float32, ignore_mismatched_sizes=True, token=False)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id; model.to(device)
    return tokenizer, model, repo_id, revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--gpu", default=None)
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import numpy as np
    import torch
    from datasets import load_dataset
    from evaluation import normalize_scores, aggregate, metric_report, plot_cm
    run = project_path(args.run)
    meta = json.loads((run / "run_metadata.json").read_text())
    task = meta["task"]
    if not task.startswith("taxonomy/"):
        raise ValueError("Expected a taxonomy run")
    fingerprint = verify_data(task, ["inference"])
    if fingerprint["manifest_sha256"] != meta["data"]["manifest_sha256"]:
        raise ValueError("Different master split")
    names = meta["taxon_names"]
    threshold = meta["taxonomy"]["unknown_probability_threshold"]
    temperature = json.loads((run / "calibration.json").read_text())["temperature"]
    path = PROJECT_ROOT / "data/preprocessed" / task / "inference.csv.gz"
    output = PROJECT_ROOT / "results/taxonomy" / (run.name + "_" + timestamp())
    output.mkdir(parents=True)
    tee = Tee(PROJECT_ROOT / "logs/inference" / (output.name + ".log"))
    tee.start()
    started = print_run_header(task, [path, run], output)
    try:
        data = load_dataset("csv", data_files=str(path), split="train")
        frame = data.select_columns(["labels", "sub_labels", "source", "genome_id", "contig_id",
                                     "split_group", "valid_length", "chunk_id"]).to_pandas()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tokenizer, model, repo_id, revision = load_finetuned_model(
            run, meta, len(names), device
        )
        print(
            f"[MODEL] huggingface_repo={repo_id} | revision={revision} "
            f"| weights={run / 'best.pt'} | device={device}"
        )
        p = np.empty((len(data), len(names)), dtype=np.float32)
        for offset, batch, probs in predict_batches(model, tokenizer, data, load_yaml("inference.yaml")["batch_size"],
                                                    meta["training"]["model"]["max_input_tokens"], device):
            p[offset:offset+len(probs)] = normalize_scores(np.log(np.clip(probs, 1e-12, 1)) / temperature)
        cols = [f"m{i}" for i in range(len(names))]
        frame[cols] = p
        for level in ["chunk", "contig", "genome"]:
            table = aggregate(frame, cols, level)
            probabilities = table[cols].to_numpy()
            truth = table["labels"].astype(int).to_numpy()
            predicted = probabilities.argmax(1)
            accepted = probabilities.max(1) >= threshold
            known = truth >= 0
            report = {"threshold_frozen_before_inference": threshold, "known_n": int(known.sum()),
                      "unseen_n": int((~known).sum()), "prediction_coverage": float(accepted.mean()),
                      "unknown_rejection_rate": float((~accepted[~known]).mean()) if (~known).any() else None,
                      "known_unconditional_accuracy": float(((predicted == truth) & accepted)[known].mean()) if known.any() else None}
            if known.any():
                report["known_closed_set"] = metric_report(truth[known], probabilities[known], names)
                if len(names) <= 40:
                    plot_cm(report["known_closed_set"], names, output / f"{level}_known_confusion_matrix.png")
            table["predicted_taxon"] = [names[y] if accept else "unknown" for y, accept in zip(predicted, accepted)]
            table.to_csv(output / f"{level}_predictions.csv.gz", index=False)
            write_json(output / f"{level}_metrics.json", report)
        finish_stats(task, started, output / "runtime.json", bases=int(frame["valid_length"].sum()),
                     genomes=frame["genome_id"].nunique(),
                     contigs=frame[["genome_id", "contig_id"]].drop_duplicates().shape[0], chunks=len(frame),
                     input_bytes=path.stat().st_size)
    finally:
        tee.close()


if __name__ == "__main__":
    main()
