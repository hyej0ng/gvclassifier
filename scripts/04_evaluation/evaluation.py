"""Probability aggregation, confusion matrices and metrics for one model at three levels."""

# 1. 공통 label 및 확률 집계
import json
import sys
import numpy as np
from pathlib import Path


def write_json(path, value):
    """평가 결과를 임시 파일 작성 후 안전하게 교체한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)

LABELS = {"main": ["Cellular", "NCLDV/Mirus", "Other viruses"],
          "sub": ["NCLDV", "Mirus"]}


def normalize_scores(scores):
    values = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def aggregate(frame, columns, level):
    """Each base appears once: genome aggregation equals two-stage length weighting."""
    if level == "chunk":
        return frame.copy()
    keys = ["genome_id"] if level == "genome" else ["genome_id", "contig_id"]
    work = frame.copy()
    for column in columns:
        work[column] = np.log(np.clip(work[column], 1e-12, 1.0)) * work["valid_length"]
    reducer = {column: "sum" for column in columns}
    reducer.update({"valid_length": "sum", "labels": "first", "sub_labels": "first",
                    "source": "first", "split_group": "first"})
    result = work.groupby(keys, sort=False, dropna=False).agg(reducer).reset_index()
    result[columns] = normalize_scores(
        result[columns].to_numpy() / result["valid_length"].to_numpy()[:, None]
    )
    return result


# 2. 각 class precision/recall/F1, PR curve와 calibration
def metric_report(truth, probabilities, names, predictions=None, bins=10):
    from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                                 average_precision_score, precision_recall_curve, auc)
    truth = np.asarray(truth, dtype=int)
    predictions = probabilities.argmax(1) if predictions is None else np.asarray(predictions)
    k = len(names)
    cm = confusion_matrix(truth, predictions, labels=range(k))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=range(k), zero_division=0)
    classes = {}
    for c, name in enumerate(names):
        binary = truth == c
        pr_auc = ap = None
        if binary.any() and (~binary).any():
            p, r, _ = precision_recall_curve(binary, probabilities[:, c])
            pr_auc = float(auc(r, p))
            ap = float(average_precision_score(binary, probabilities[:, c]))
        classes[name] = {"precision": float(precision[c]), "recall": float(recall[c]),
                         "f1": float(f1[c]), "support": int(support[c]),
                         "pr_auc_trapezoidal": pr_auc, "average_precision": ap,
                         "positive_prevalence": float(binary.mean()) if len(binary) else None}
    # 가장 높은 확률로 선택한 class의 confidence가 실제 정답률과 맞는지 확인한다.
    confidence = probabilities[np.arange(len(truth)), predictions]
    correct = predictions == truth
    curve = []
    ece = 0.0
    for b in range(bins):
        mask = (confidence >= b / bins) & (confidence < (b + 1) / bins if b < bins-1 else confidence <= 1)
        if mask.any():
            conf, acc = float(confidence[mask].mean()), float(correct[mask].mean())
            ece += mask.mean() * abs(conf - acc)
            curve.append({"bin": b, "count": int(mask.sum()), "mean_confidence": conf, "accuracy": acc})
    return {"n": len(truth), "confusion_matrix": cm.tolist(), "classes": classes,
            "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall[support > 0].mean()) if (support > 0).any() else None,
            "accuracy": float(correct.mean()) if len(correct) else None,
            "ece": float(ece), "reliability": curve}


def plot_cm(report, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(report["confusion_matrix"])
    proportion = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(proportion, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i,j]:,}\n{100*proportion[i,j]:.1f}%", ha="center", va="center",
                    color="white" if proportion[i,j] > .5 else "black")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set(xlabel="Predicted", ylabel="True")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# 3. split_group 전체를 재표집하는 bootstrap CI
def group_bootstrap(truth, predicted, groups, n_classes, repeats, seed):
    from sklearn.metrics import confusion_matrix
    index = {}
    for i, group in enumerate(groups):
        index.setdefault(group, []).append(i)
    matrices = np.array([confusion_matrix(truth[idx], predicted[idx], labels=range(n_classes)) for idx in index.values()])
    if len(matrices) < 2 or repeats <= 0:
        return {"macro_f1_95ci": None, "groups": len(matrices)}
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        cm = matrices[rng.integers(len(matrices), size=len(matrices))].sum(axis=0)
        denom = cm.sum(0) + cm.sum(1)
        scores.append(np.divide(2 * np.diag(cm), denom, out=np.zeros(n_classes), where=denom > 0).mean())
    return {"macro_f1_95ci": np.quantile(scores, [.025, .975]).tolist(), "groups": len(matrices)}


# 4. 선택한 main 또는 sub 모델 하나의 평가 결과 저장
def report_task(frame, output, task, probability_columns, settings, seed=20260903):
    if task not in LABELS:
        raise ValueError(f"Unsupported classification task: {task}")
    output = Path(output)
    for level in ("chunk", "contig", "genome"):
        table = aggregate(frame, probability_columns, level)
        probabilities = table[probability_columns].to_numpy()
        truth = table["labels"].astype(int).to_numpy()
        predicted = probabilities.argmax(1)
        destination = output / task / level
        destination.mkdir(parents=True, exist_ok=True)
        report = metric_report(truth, probabilities, LABELS[task], predicted, settings["calibration_bins"])
        if level == "genome":
            report.update(group_bootstrap(truth, predicted, table["split_group"].to_numpy(), len(LABELS[task]),
                                           settings["bootstrap_replicates"], seed))
        report["by_source"] = {
            source: metric_report(truth[mask], probabilities[mask], LABELS[task], predicted[mask], settings["calibration_bins"])
            for source in table["source"].unique()
            for mask in [table["source"].to_numpy() == source]
        }
        lengths = table["valid_length"].to_numpy()
        report["by_length"] = {
            title: metric_report(truth[mask], probabilities[mask], LABELS[task], predicted[mask], settings["calibration_bins"])
            for title, mask in [("lt_500", lengths < 500), ("500_to_4999", (lengths >= 500) & (lengths < 5000)),
                                ("gte_5000", lengths >= 5000)] if mask.any()
        }
        write_json(destination / "metrics.json", report)
        plot_cm(report, LABELS[task], destination / "confusion_matrix.png")
        table["true_label"] = truth
        table["predicted_label"] = predicted
        for c in range(probabilities.shape[1]):
            table[f"probability_{c}"] = probabilities[:, c]
        table.to_csv(destination / "predictions.csv.gz", index=False)
        print(f"[RESULT] {task:5s} | level={level:6s} | N={len(truth):,} | macro_F1={report['macro_f1']:.4f}")
