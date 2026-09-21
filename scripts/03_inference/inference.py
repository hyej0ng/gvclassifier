#!/usr/bin/env python3
"""Evaluate one independently trained model on its labeled inference split."""

# 1. 경로 및 설정
import argparse
import datetime as dt
import hashlib
import json
import os
import resource
import subprocess
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
    parser.add_argument("--task", choices=("main", "sub"), required=True)
    parser.add_argument("--run", required=True, help="Relative run directory, e.g. runs/main/main_v1")
    parser.add_argument("--name", default=None)
    parser.add_argument("--gpu", default=None)
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import numpy as np
    import torch
    from datasets import load_dataset
    from evaluation import normalize_scores, report_task

    cfg = load_yaml("inference.yaml")
    task = args.task
    run = project_path(args.run)
    fingerprints = verify_data(task, ["inference"])
    name = args.name or timestamp()
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("name must be a simple directory name")
    output = PROJECT_ROOT / "results" / name
    task_output = output / task
    if task_output.exists():
        raise SystemExit(
            f"Result directory already exists: {task_output}. "
            "Preserve the original holdout result or choose a new --name."
        )
    task_output.mkdir(parents=True)
    dataset_path = PROJECT_ROOT / "data/preprocessed" / task / "inference.csv.gz"
    tee = Tee(PROJECT_ROOT / "logs/inference" / f"{name}_{task}.log")
    tee.start()
    started = print_run_header(f"inference/{task}", [dataset_path, run], task_output)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        # 2. 선택한 task의 inference 데이터와 학습 run이 같은 split에서 왔는지 확인
        ds = load_dataset("csv", data_files=str(dataset_path), split="train")
        meta_columns = ["labels", "sub_labels", "source", "genome_id", "contig_id", "split_group", "valid_length", "chunk_id"]
        frame = ds.select_columns(meta_columns).to_pandas()
        processed_bases = 0
        metadata = json.loads((run / "run_metadata.json").read_text())
        if metadata["task"] != task or metadata["data"]["manifest_sha256"] != fingerprints["manifest_sha256"]:
            raise ValueError(f"The run is not a {task} model or uses a different split manifest")
        model_hash = file_sha256(run / "best.pt")
        n_labels = int(metadata["training"][task]["num_labels"])
        tokenizer, model, repo_id, revision = load_finetuned_model(
            run, metadata, n_labels, device
        )
        max_tokens = metadata["training"]["model"]["max_input_tokens"]
        temperature = json.loads((run / "calibration.json").read_text())["temperature"]
        probabilities = np.empty((len(ds), n_labels), dtype=np.float32)
        print(
            f"[MODEL] task={task} | huggingface_repo={repo_id} | revision={revision} "
            f"| weights={run / 'best.pt'} | device={device} | chunks={len(ds):,}"
        )
        next_log = 0.1

        # 3. 선택한 모델 하나만 batch 단위로 예측한다.
        for start, batch, p in predict_batches(model, tokenizer, ds, cfg["batch_size"], max_tokens, device):
            p = normalize_scores(np.log(np.clip(p, 1e-12, 1.0)) / temperature)
            probabilities[start:start+len(p)] = p
            processed_bases += sum(int(n) for n in batch["valid_length"])
            fraction = (start + len(p)) / len(ds)
            if fraction >= next_log or start + len(p) == len(ds):
                print(f"[LOG] task={task} | progress={fraction:.1%} | chunks={start+len(p):,}/{len(ds):,} | runtime={time.perf_counter()-started:.1f}s")
                next_log += .1
        probability_columns = [f"p{c}" for c in range(n_labels)]
        for column, values in zip(probability_columns, probabilities.T):
            frame[column] = values

        # 4. Chunk/contig/genome confusion matrix와 metric 저장
        frame.to_parquet(task_output / "chunk_probabilities.parquet", index=False)
        report_task(frame, output, task, probability_columns, cfg, load_yaml("split.yaml")["seed"])
        write_json(task_output / "inference_metadata.json", {
            "task": task, "model_sha256": model_hash, "data": fingerprints, "settings": cfg
        })
        finish_stats(f"inference/{task}", started, task_output / "runtime.json", bases=processed_bases,
                     genomes=frame["genome_id"].nunique(),
                     contigs=frame[["genome_id","contig_id"]].drop_duplicates().shape[0], chunks=len(frame),
                     input_bytes=dataset_path.stat().st_size,
                     extra={"unique_input_bases": int(frame["valid_length"].sum()),
                            "model_passes": 1, "device": device})
    finally:
        tee.close()


if __name__ == "__main__":
    main()
