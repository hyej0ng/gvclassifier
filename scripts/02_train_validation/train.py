#!/usr/bin/env python3
"""Fine-tune one GenomeOcean main or sub classifier with validation."""

from __future__ import annotations

# 1. 경로 및 설정
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
RUN_STARTS = {}
RC_TRANS = str.maketrans("ACGTN", "TGCAN")


# 이 스크립트에서 사용하는 설정·검증·로그 보조 함수
def load_yaml(name):
    import yaml
    section = name.removesuffix(".yaml"); path = CONFIG_DIR / ("pipeline.yaml" if section in {"training", "preprocessing", "split", "inference"} else name)
    with path.open(encoding="utf-8") as handle: value = yaml.safe_load(handle)
    return value[section] if section in {"training", "preprocessing", "split", "inference"} else value
def timestamp(): return dt.datetime.now().strftime("%Y%m%d-%H%M%S")
def now_local(): return dt.datetime.now().astimezone().isoformat(timespec="seconds")
def now_utc(): return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
def reverse_complement(sequence): return sequence.translate(RC_TRANS)[::-1]
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
def gpu_info():
    try: result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError): return []
    return [line for line in result.stdout.splitlines()]
def print_run_header(stage, inputs: Iterable[Path], output):
    print("=" * 100); print(f"[START] stage={stage} | local={now_local()} | utc={now_utc()}")
    print(f"[PATH] project={PROJECT_ROOT}\n[PATH] input={'; '.join(map(str, inputs))}\n[PATH] output={output}")
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    for line in gpu_info(): print(f"[GPU] {line}")
    print("=" * 100); started = time.perf_counter(); RUN_STARTS[started] = {"started_local": now_local(), "started_utc": now_utc()}; return started
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
    write_json(output_json, report); print(f"[DONE] stage={stage} | runtime={report['runtime_hms']} | chunks={chunks:,} | bases={bases:,} | throughput={report['mbp_per_second']:.3f} Mbp/s"); return report
def existing_size(paths): return sum(path.stat().st_size if path.is_file() else sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) for path in paths if path.exists())
def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()
def verify_data(task, splits):
    root = PROJECT_ROOT / "data/preprocessed"
    if task.startswith("taxonomy/"):
        verify_data("sub", ["train", "validation"]); meta = json.loads((root / task / "taxonomy.json").read_text()); current = json.loads((root / "dataset_checksums.json").read_text())
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("main", "sub", "taxonomy"), required=True)
    parser.add_argument("--source", choices=("NCLDV", "MIRUS"))
    parser.add_argument("--rank", choices=("class", "order", "family", "genus"))
    parser.add_argument("--run-name", default=None, help="Reusable name needed when --resume is used.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu", default=None, help="Set CUDA_VISIBLE_DEVICES before torch is imported.")
    parser.add_argument("--cpu", action="store_true", help="Small code-verification runs only; full data needs a GPU.")
    return parser.parse_args()


# 2. CSV 통계와 class weight 계산
def dataset_stats(dataset) -> dict:
    genomes = defaultdict(set)
    contigs = set()
    bases = 0
    for label, genome, contig, length in zip(
        dataset["labels"], dataset["genome_id"], dataset["contig_id"], dataset["valid_length"]
    ):
        genomes[int(label)].add(genome)
        contigs.add(contig)
        bases += int(length)
    return {
        "chunks": len(dataset),
        "bases": bases,
        "genomes": sum(len(values) for values in genomes.values()),
        "contigs": len(contigs),
        "genomes_per_class": {str(key): len(value) for key, value in genomes.items()},
    }


def sqrt_inverse_class_weights(genomes_per_class: dict[str, int], n_labels: int, power=0.5):
    import torch

    counts = [max(int(genomes_per_class.get(str(label), 0)), 1) for label in range(n_labels)]
    largest = max(counts)
    weights = [(largest / count) ** power for count in counts]
    mean = sum(weights) / len(weights)
    return torch.tensor([weight / mean for weight in weights], dtype=torch.float32)


# 3. Train에만 reverse-complement augmentation 적용
class SequenceCollator:
    def __init__(self, tokenizer, max_length: int, reverse_probability: float = 0.0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reverse_probability = reverse_probability

    def __call__(self, features: list[dict]):
        import random
        import torch

        sequences = []
        for feature in features:
            sequence = feature["sequence"]
            if self.reverse_probability and random.random() < self.reverse_probability:
                sequence = reverse_complement(sequence)
            sequences.append(sequence)
        batch = self.tokenizer(
            sequences,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        if batch["input_ids"].shape[1] > self.max_length:
            raise ValueError(
                f"Input has {batch['input_ids'].shape[1]} tokens, limit={self.max_length}. "
                "Increase training.model.max_input_tokens within the model context; no DNA was truncated."
            )
        batch["labels"] = torch.tensor([int(item["labels"]) for item in features], dtype=torch.long)
        batch["sample_weight"] = torch.tensor(
            [float(item["sample_weight"]) for item in features], dtype=torch.float32
        )
        batch["valid_length"] = torch.tensor([int(item["valid_length"]) for item in features])
        return batch


# 4. Genome weight와 class weight가 적용된 cross-entropy
def make_weighted_trainer_class(class_weights, label_smoothing: float):
    import torch
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            lengths = inputs.pop("valid_length")
            counter = "training_forward" if model.training else "validation_forward"
            if not hasattr(self, "forward_counts"):
                self.forward_counts = Counter()
            self.forward_counts[counter + "_bp"] += int(lengths.sum())
            self.forward_counts[counter + "_chunks"] += len(lengths)
            sample_weight = inputs.pop("sample_weight").to(model.device)
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            losses = torch.nn.functional.cross_entropy(
                logits,
                labels,
                weight=None,
                reduction="none",
                label_smoothing=label_smoothing,
            )
            # Weight normalization was calculated over the entire split, never per minibatch.
            loss = (losses * sample_weight).mean()
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


# 5. Validation chunk 확률을 genome 단위로 집계
def metric_function(validation_dataset, n_labels: int):
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

    genome_ids = validation_dataset["genome_id"]
    lengths = [int(value) for value in validation_dataset["valid_length"]]

    def compute(eval_prediction):
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        labels = np.asarray(eval_prediction.label_ids, dtype=int)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        chunk_pred = probabilities.argmax(axis=1)
        chunk_p, chunk_r, chunk_f1, _ = precision_recall_fscore_support(
            labels, chunk_pred, labels=list(range(n_labels)), average="macro", zero_division=0
        )

        log_sum = defaultdict(lambda: np.zeros(n_labels, dtype=np.float64))
        bp_sum = Counter()
        true_by_genome = {}
        for genome, length, probability, true_label in zip(genome_ids, lengths, probabilities, labels):
            log_sum[genome] += length * np.log(np.clip(probability, 1e-12, 1.0))
            bp_sum[genome] += length
            true_by_genome[genome] = int(true_label)
        genome_true, genome_pred = [], []
        for genome in sorted(log_sum):
            genome_true.append(true_by_genome[genome])
            genome_pred.append(int(np.argmax(log_sum[genome] / bp_sum[genome])))
        genome_p, genome_r, genome_f1, _ = precision_recall_fscore_support(
            genome_true, genome_pred, labels=list(range(n_labels)), average="macro", zero_division=0
        )
        return {
            "chunk_precision_macro": chunk_p,
            "chunk_recall_macro": chunk_r,
            "chunk_macro_f1": chunk_f1,
            "genome_precision_macro": genome_p,
            "genome_recall_macro": genome_r,
            "genome_macro_f1": genome_f1,
            "genome_balanced_accuracy": balanced_accuracy_score(genome_true, genome_pred),
        }

    return compute


# 6. 0.5 epoch마다 한 줄 log, history.csv, loss graph 저장
def make_progress_callback(run_dir: Path, total_epochs: int):
    from transformers import TrainerCallback

    class ProgressCallback(TrainerCallback):
        def __init__(self):
            self.last_train_loss = float("nan")
            self.last_lr = float("nan")
            self.evaluations = 0
            self.history_path = run_dir / "history.csv"
            if not self.history_path.exists():
                with self.history_path.open("w", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerow(
                        ["epoch", "train_loss", "val_loss", "genome_precision_macro", "genome_recall_macro", "genome_macro_f1", "learning_rate", "gpu_peak_gib"]
                    )

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" in logs:
                self.last_train_loss = float(logs["loss"])
            if "learning_rate" in logs:
                self.last_lr = float(logs["learning_rate"])

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            import torch

            metrics = metrics or {}
            self.evaluations += 1
            epoch = float(state.epoch or 0)
            val_loss = float(metrics.get("eval_loss", float("nan")))
            precision = float(metrics.get("eval_genome_precision_macro", float("nan")))
            recall = float(metrics.get("eval_genome_recall_macro", float("nan")))
            f1 = float(metrics.get("eval_genome_macro_f1", float("nan")))
            gpu_peak = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0
            gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "CPU")
            print(
                f"[LOG] epoch {epoch:05.1f}/{total_epochs:03d} | gpu={gpu} | mem_peak={gpu_peak:.2f} GiB "
                f"| train_loss={self.last_train_loss:.4f} | val_loss={val_loss:.4f} "
                f"| P={precision:.4f} | R={recall:.4f} | macro_F1={f1:.4f} | lr={self.last_lr:.3e}"
            )
            with self.history_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    [epoch, self.last_train_loss, val_loss, precision, recall, f1, self.last_lr, gpu_peak]
                )
            plot_history(self.history_path, run_dir / "loss_curve.png")

        def on_epoch_end(self, args, state, control, **kwargs):
            # 마지막 partial interval도 반드시 평가/저장해 last가 실제 마지막 상태가 되게 한다.
            if state.global_step % args.eval_steps:
                control.should_log = True
                control.should_evaluate = True
                control.should_save = True

    return ProgressCallback()


def plot_history(history_path: Path, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    epochs = [float(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(epochs, train_loss, marker="o", label="train loss")
    axis.plot(epochs, val_loss, marker="o", label="validation loss")
    axis.set(xlabel="epoch", ylabel="weighted cross-entropy loss", title="GenomeOcean loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


# 7. Trainer checkpoint에서 portable best.pt/last.pt 저장
def export_state_dict(checkpoint: Path, pt_path: Path) -> None:
    import torch
    safe_path = checkpoint / "model.safetensors"
    bin_path = checkpoint / "pytorch_model.bin"
    if safe_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safe_path), device="cpu")
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No model weights in checkpoint: {checkpoint}")
    torch.save(state_dict, pt_path)


def main() -> int:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # GPU 설정 뒤에 무거운 package를 import한다.
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    config = load_yaml("training.yaml")
    common = config["common"]
    if args.task == "taxonomy":
        if not args.source or not args.rank:
            raise ValueError("Taxonomy requires --source and --rank")
        args.task = f"taxonomy/{args.source}/{args.rank}"
        taxonomy_meta = json.loads((PROJECT_ROOT / "data/preprocessed" / args.task / "taxonomy.json").read_text())
        task_config = {"num_labels": len(taxonomy_meta["labels"]),
                       "max_epochs": load_yaml("pipeline.yaml")["taxonomy"]["max_epochs"]}
    else:
        task_config = config[args.task]
    seed = int(common["seed"])
    set_seed(seed)

    run_name = args.run_name or f"{args.task.replace('/', '_')}_{timestamp()}"
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise SystemExit("run-name must be a simple directory name")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SystemExit("This beginner baseline supports one GPU per run. Run main/sub separately.")
    run_dir = PROJECT_ROOT / "runs" / args.task / run_name
    checkpoint_dir = run_dir / "checkpoints"
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"Run directory exists: {run_dir}. Use a new --run-name or --resume.")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_path = PROJECT_ROOT / "data" / "preprocessed" / args.task / "train.csv.gz"
    validation_path = PROJECT_ROOT / "data" / "preprocessed" / args.task / "validation.csv.gz"
    if not train_path.exists() or not validation_path.exists():
        raise SystemExit("Preprocessed train/validation files are missing. Run preprocessing first.")
    data_fingerprint = verify_data(args.task, ["train", "validation"])

    log_path = PROJECT_ROOT / "logs" / "train" / f"{run_name}.log"
    tee = Tee(log_path)
    tee.start()
    started = print_run_header(
        f"train/{args.task}", [train_path, validation_path, Path(config["model"]["name_or_path"])], run_dir
    )

    try:
        if not torch.cuda.is_available() and not args.cpu:
            raise SystemExit("Training requires a visible CUDA GPU. Set --gpu or CUDA_VISIBLE_DEVICES.")
        if torch.cuda.is_available() and not args.cpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # 8. Dataset과 tokenizer/model 불러오기
        datasets = load_dataset(
            "csv", data_files={"train": str(train_path), "validation": str(validation_path)}
        )
        train_stats = dataset_stats(datasets["train"])
        validation_stats = dataset_stats(datasets["validation"])
        n_labels = int(task_config["num_labels"])
        class_weights = sqrt_inverse_class_weights(train_stats["genomes_per_class"], n_labels, common["class_weight_power"])
        for split in ["train", "validation"]:
            split_labels = set(int(v) for v in datasets[split]["labels"])
            if split_labels != set(range(n_labels)) and not (args.task.startswith("taxonomy/") and split == "validation"):
                raise ValueError(f"Missing class in {split}: {split_labels}")
            ds = datasets[split]
            base_weights = ds["sample_weight"] if common["genome_balancing"] else [1.0] * len(ds)
            weight_sum = sum(float(w) * float(class_weights[int(y)]) for w, y in zip(base_weights, ds["labels"]))
            norm = len(ds) / weight_sum
            datasets[split] = ds.map(lambda row: {"sample_weight": (float(row["sample_weight"]) if common["genome_balancing"] else 1.0) * float(class_weights[int(row["labels"])]) * norm})
        print(f"[DATA] train={json.dumps(train_stats, ensure_ascii=False)}")
        print(f"[DATA] validation={json.dumps(validation_stats, ensure_ascii=False)}")
        print(f"[LOSS] normalized_class_weights={class_weights.tolist()}")

        model_name = config["model"]["name_or_path"]
        print(f"[MODEL] huggingface_repo={model_name} | revision={config['model']['revision']}")
        max_length = int(config["model"]["max_input_tokens"])
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            model_max_length=max_length,
            use_fast=True,
            padding_side="right",
            revision=config["model"]["revision"],
            token=False,
        )
        # 5 kb가 잘리지 않는지 실제 tokenizer로 확인한다.
        audit_sequences = list(datasets["train"].select(range(min(32, len(datasets["train"]))))["sequence"])
        audit = tokenizer(audit_sequences, truncation=False, return_length=True, return_token_type_ids=False)
        audit_max = max(audit.get("length", [0]))
        if audit_max > max_length:
            raise SystemExit(f"Tokenizer audit failed: observed {audit_max} tokens > max_input_tokens={max_length}")
        print(f"[TOKENIZER] audited={len(audit_sequences)} chunks | maximum_tokens={audit_max}/{max_length}")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=n_labels,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            revision=config["model"]["revision"],
            ignore_mismatched_sizes=True,
            token=False,
        )
        context = getattr(model.config, "max_position_embeddings", max_length)
        if max_length > context:
            raise ValueError(f"max_input_tokens={max_length} exceeds model context={context}")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        if args.task.startswith("taxonomy/"):
            model.config.id2label = dict(enumerate(taxonomy_meta["labels"]))
            model.config.label2id = {name: i for i, name in enumerate(taxonomy_meta["labels"])}

        # 9. 0.5 epoch 간격으로 log/evaluation/checkpoint
        epochs = int(task_config["max_epochs"])
        batch_size = int(common["per_device_train_batch_size"])
        accumulation = int(common["gradient_accumulation_steps"])
        world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
        update_steps_per_epoch = max(1, math.ceil(len(datasets["train"]) / (batch_size * accumulation * world_size)))
        interval_fraction = float(common["log_and_validate_every_fraction_of_epoch"])
        interval_steps = max(1, round(update_steps_per_epoch * interval_fraction))
        patience_checks = max(1, round(float(common["early_stopping_patience_epochs"]) / interval_fraction))
        use_bf16 = not args.cpu and bool(torch.cuda.is_bf16_supported()) and str(common["precision"]).lower() == "bf16"
        if common["precision"] == "bf16" and not args.cpu and not use_bf16:
            raise ValueError("Selected GPU does not support BF16; set precision to fp32 or fp16 explicitly")

        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=int(common["per_device_validation_batch_size"]),
            gradient_accumulation_steps=accumulation,
            learning_rate=float(common["learning_rate"]),
            weight_decay=float(common["weight_decay"]),
            warmup_ratio=float(common["warmup_ratio"]),
            lr_scheduler_type=common["lr_scheduler"],
            max_grad_norm=float(common["max_grad_norm"]),
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            eval_steps=interval_steps,
            save_steps=interval_steps,
            logging_steps=interval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="genome_macro_f1",
            greater_is_better=True,
            save_total_limit=3,
            bf16=use_bf16,
            fp16=not args.cpu and common["precision"] == "fp16",
            use_cpu=args.cpu,
            optim="adamw_torch",
            dataloader_num_workers=int(common["dataloader_workers"]),
            remove_unused_columns=False,
            report_to=[],
            disable_tqdm=True,
            seed=seed,
            data_seed=seed,
        )

        WeightedTrainer = make_weighted_trainer_class(
            class_weights, float(common["label_smoothing"])
        )
        progress = make_progress_callback(run_dir, epochs)
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            data_collator=SequenceCollator(
                tokenizer,
                max_length,
                float(load_yaml("preprocessing.yaml")["train_augmentation"]["reverse_complement_probability"]),
            ),
            compute_metrics=metric_function(datasets["validation"], n_labels),
            callbacks=[
                progress,
                EarlyStoppingCallback(
                    early_stopping_patience=patience_checks,
                    early_stopping_threshold=float(common["early_stopping_min_delta"]),
                ),
            ],
            processing_class=tokenizer,
        )
        trainer.model_accepts_loss_kwargs = False

        # Validation에서는 augmentation을 끈다.
        original_evaluate = trainer.evaluate
        def evaluate_without_augmentation(*evaluate_args, **evaluate_kwargs):
            probability = trainer.data_collator.reverse_probability
            trainer.data_collator.reverse_probability = 0.0
            try:
                return original_evaluate(*evaluate_args, **evaluate_kwargs)
            finally:
                trainer.data_collator.reverse_probability = probability
        trainer.evaluate = evaluate_without_augmentation

        resume_checkpoint = get_last_checkpoint(str(checkpoint_dir)) if args.resume else None
        if args.resume and resume_checkpoint is None:
            raise ValueError("No checkpoint available to resume")
        metadata_path = run_dir / "run_metadata.json"
        metadata = {"task": args.task, "training": config, "data": data_fingerprint, "train_stats": train_stats,
                    "validation_stats": validation_stats, "class_weights": class_weights.tolist(),
                    "model_source": model_name, "actual_model_commit": getattr(model.config, "_commit_hash", None)}
        import importlib.metadata
        metadata["environment"] = {name: importlib.metadata.version(name) for name in
             ["torch", "transformers", "datasets", "huggingface-hub", "tokenizers", "accelerate"]}
        if args.task.startswith("taxonomy/"):
            metadata["taxonomy"] = load_yaml("pipeline.yaml")["taxonomy"]
            metadata["taxon_names"] = taxonomy_meta["labels"]
        if args.resume:
            previous = json.loads(metadata_path.read_text())
            if previous["training"] != config or previous["data"] != data_fingerprint:
                raise ValueError("Resume config/data changed. Start a new run.")
        else:
            write_json(metadata_path, metadata)
        from transformers.trainer_callback import PrinterCallback
        trainer.remove_callback(PrinterCallback)
        result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        trainer.save_state()
        # best 모델의 validation 예측만으로 temperature를 선택한다.
        trainer.data_collator.reverse_probability = 0.0
        prediction = trainer.predict(datasets["validation"])
        logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        scores_by_genome = defaultdict(lambda: np.zeros(n_labels, dtype=np.float64))
        lengths_by_genome = Counter()
        truths_by_genome = {}
        for logits_row, genome, length, label in zip(logits, datasets["validation"]["genome_id"],
                datasets["validation"]["valid_length"], datasets["validation"]["labels"]):
            scores_by_genome[genome] += int(length) * logits_row
            lengths_by_genome[genome] += int(length)
            truths_by_genome[genome] = int(label)
        score_matrix = np.stack([scores_by_genome[g] / lengths_by_genome[g] for g in scores_by_genome])
        truth_array = np.array([truths_by_genome[g] for g in scores_by_genome])
        choices = []
        for temperature in common["calibration_temperatures"]:
            scaled = score_matrix / float(temperature)
            scaled -= scaled.max(1, keepdims=True)
            log_prob = scaled - np.log(np.exp(scaled).sum(1, keepdims=True))
            nll = -float(log_prob[np.arange(len(truth_array)), truth_array].mean())
            choices.append((nll, float(temperature)))
        nll, temperature = min(choices)
        write_json(run_dir / "calibration.json", {"temperature": temperature,
            "fitted_on": "validation_genome_nll", "validation_nll": nll,
            "candidate_temperatures": common["calibration_temperatures"]})
        print(f"[CALIBRATION] temperature={temperature} | validation_genome_NLL={nll:.4f}")

        # 10. best와 last를 서로 다른 checkpoint에서 export
        completed_epochs = float(trainer.state.epoch or epochs)
        forward_counts = dict(trainer.forward_counts)
        best_metric = trainer.state.best_metric
        best_checkpoint = Path(trainer.state.best_model_checkpoint)
        last_checkpoint_text = get_last_checkpoint(str(checkpoint_dir))
        if not last_checkpoint_text:
            raise RuntimeError("No last checkpoint was saved")
        last_checkpoint = Path(last_checkpoint_text)
        write_json(run_dir / "checkpoint_selection.json", {
            "best": str(best_checkpoint.relative_to(run_dir)), "last": str(last_checkpoint.relative_to(run_dir)),
            "best_genome_macro_f1": best_metric, "completed_epochs": completed_epochs,
            "selection": "highest validation genome macro-F1; exact ties keep earlier checkpoint",
            "resume_checkpoint": str(last_checkpoint.relative_to(run_dir)),
        })
        print(f"[CHECKPOINT] best={best_checkpoint}")
        print(f"[CHECKPOINT] last={last_checkpoint}")
        del trainer, model
        torch.cuda.empty_cache()
        export_state_dict(best_checkpoint, run_dir / "best.pt")
        export_state_dict(last_checkpoint, run_dir / "last.pt")

        for config_path in (PROJECT_ROOT / "configs").glob("*.yaml"):
            snapshot_dir = run_dir / "config_snapshot"
            snapshot_dir.mkdir(exist_ok=True)
            shutil.copy2(config_path, snapshot_dir / config_path.name)

        processed_bp = forward_counts.get("training_forward_bp", 0) + forward_counts.get("validation_forward_bp", 0)
        finish_stats(
            f"train/{args.task}",
            started,
            run_dir / "runtime.json",
            bases=processed_bp,
            genomes=train_stats["genomes"],
            contigs=train_stats["contigs"],
            chunks=train_stats["chunks"],
            input_bytes=existing_size([train_path, validation_path]),
            extra={
                "unique_train_bases": train_stats["bases"],
                "unique_validation_bases": validation_stats["bases"],
                "validation_evaluations": progress.evaluations,
                "completed_epochs": completed_epochs,
                "actual_forward_counts_this_invocation": forward_counts,
                "throughput_includes_validation_and_checkpoint_export": True,
                "best_checkpoint": str(best_checkpoint),
                "last_checkpoint": str(last_checkpoint),
                "best_metric": best_metric,
                "model_source": model_name,
                "data_source": [str(train_path), str(validation_path)],
            },
        )
        return 0
    finally:
        tee.close()
if __name__ == "__main__":
    raise SystemExit(main())
