#!/usr/bin/env python3
"""
Evaluate fine-tuned Qwen2-1.5B model on held-out test set.

Produces:
  - eval/qwen_metrics.json
  - eval/comparison.json       (side-by-side with baseline)
  - eval/qwen_confusion.png
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("evaluate_model")

MODEL_DIR = "models/qwen-injection-detector/best"
TEST_PATH = "data/processed/test.parquet"
BASELINE_PATH = "eval/baseline_metrics.json"
OUTPUT_DIR = Path("eval")


def main():
    cfg = load_config("configs/training_config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    base_name = cfg["model"]["base_model_name"]
    max_length = cfg["model"]["max_length"]
    model_dir = Path(MODEL_DIR)

    temperature_path = model_dir / "temperature.pt"
    if temperature_path.exists():
        temperature = torch.load(str(temperature_path)).item()
        logger.info("Loaded temperature: %.4f", temperature)
    else:
        temperature = 1.0
        logger.warning("No temperature file found, using T=1.0 (uncalibrated)")

    logger.info("Loading tokenizer: %s", base_name)
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        num_labels=2,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    logger.info("Loading LoRA adapters from: %s", model_dir)
    model = PeftModel.from_pretrained(model, str(model_dir))
    model = model.to(device)
    model.eval()
    logger.info("Model loaded. Parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    logger.info("Running warm-up inference...")
    dummy = tokenizer("Warm-up test.", return_tensors="pt")
    dummy = {k: v.to(device) for k, v in dummy.items()}
    with torch.no_grad():
        for _ in range(3):
            _ = model(**dummy)
    torch.cuda.synchronize()
    logger.info("Warm-up complete.")

    logger.info("Loading test data: %s", TEST_PATH)
    df = pd.read_parquet(TEST_PATH)
    logger.info("Test samples: %d", len(df))

    sources = df["source"].values
    test_ds = Dataset.from_pandas(df[["text", "label"]])

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=max_length, padding=False
        )

    test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(test_ds, batch_size=64, collate_fn=collator, shuffle=False)

    all_logits = []

    logger.info("Running inference on test set (%d samples)...", len(df))
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            outputs = model(**batch)
            all_logits.append(outputs.logits.cpu())

    logits = torch.cat(all_logits, dim=0).float()
    probs = torch.softmax(logits / temperature, dim=-1).numpy()
    preds = np.argmax(probs, axis=-1)
    gt = df["label"].values
    sources = df["source"].values

    acc = accuracy_score(gt, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt, preds, average=None, labels=[0, 1]
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(gt, preds, average="macro")
    try:
        auc = roc_auc_score(gt, probs[:, 1])
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(gt, preds)

    print(f"\n{'='*60}")
    print("  QWEN2-1.5B FINE-TUNED RESULTS")
    print(f"{'='*60}")
    print(f"  Temperature:    {temperature:.4f}")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Macro F1:       {macro_f1:.4f}")
    print(f"  ROC-AUC:        {auc:.4f}")
    print(f"\n  Per-class metrics:")
    print(f"    Benign (0):   precision={precision[0]:.4f}  recall={recall[0]:.4f}  f1={f1[0]:.4f}")
    print(f"    Injection (1): precision={precision[1]:.4f}  recall={recall[1]:.4f}  f1={f1[1]:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"                Predicted")
    print(f"               Benign  Inj")
    print(f"  Actual Benign  {cm[0,0]:>5d}  {cm[0,1]:>3d}")
    print(f"         Inj     {cm[1,0]:>5d}  {cm[1,1]:>3d}")

    results = {
        "model": "qwen2-1.5b-qlora",
        "temperature": round(float(temperature), 4),
        "test_size": len(df),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "roc_auc": round(float(auc), 4),
        "benign": {
            "precision": round(float(precision[0]), 4),
            "recall": round(float(recall[0]), 4),
            "f1": round(float(f1[0]), 4),
        },
        "injection": {
            "precision": round(float(precision[1]), 4),
            "recall": round(float(recall[1]), 4),
            "f1": round(float(f1[1]), 4),
        },
        "confusion_matrix": cm.tolist(),
        "per_source": {},
    }

    print(f"\n  Per-source breakdown:")
    for src in sorted(df["source"].unique()):
        mask = sources == src
        src_gt = gt[mask]
        src_pred = preds[mask]
        src_acc = accuracy_score(src_gt, src_pred)
        src_p, src_r, src_f, _ = precision_recall_fscore_support(
            src_gt, src_pred, average="binary", pos_label=1, zero_division=0
        )
        results["per_source"][src] = {
            "n": int(mask.sum()),
            "benign": int((src_gt == 0).sum()),
            "injection": int((src_gt == 1).sum()),
            "accuracy": round(float(src_acc), 4),
            "injection_precision": round(float(src_p), 4),
            "injection_recall": round(float(src_r), 4),
            "injection_f1": round(float(src_f), 4),
        }
        print(f"    {src:<20s} n={results['per_source'][src]['n']:>5d}  "
              f"acc={src_acc:.4f}  recall(inj)={src_r:.4f}")

    print()
    print(classification_report(gt, preds, target_names=["benign", "injection"]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    qwen_path = OUTPUT_DIR / "qwen_metrics.json"
    with open(qwen_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Metrics saved: %s", qwen_path)

    baseline_path = Path(BASELINE_PATH)
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)

        comparison = {
            "baseline": {
                "model": baseline["model"],
                "accuracy": baseline["accuracy"],
                "macro_f1": baseline["macro_f1"],
                "benign_recall": baseline["benign"]["recall"],
                "injection_recall": baseline["injection"]["recall"],
                "injection_f1": baseline["injection"]["f1"],
            },
            "qwen": {
                "model": "qwen2-1.5b-qlora",
                "accuracy": results["accuracy"],
                "macro_f1": results["macro_f1"],
                "benign_recall": results["benign"]["recall"],
                "injection_recall": results["injection"]["recall"],
                "injection_f1": results["injection"]["f1"],
            },
            "deltas": {},
        }

        for metric in ["accuracy", "macro_f1", "benign_recall", "injection_recall", "injection_f1"]:
            delta = comparison["qwen"][metric] - comparison["baseline"][metric]
            comparison["deltas"][metric] = round(float(delta), 4)

        comparison_path = OUTPUT_DIR / "comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info("Comparison saved: %s", comparison_path)

        print(f"\n{'='*60}")
        print("  COMPARISON: QWEN2-1.5B vs BASELINE")
        print(f"{'='*60}")
        print(f"  {'Metric':<25s} {'Baseline':<12s} {'Qwen2':<12s} {'Δ':<10s}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
        for metric in ["accuracy", "macro_f1", "benign_recall", "injection_recall", "injection_f1"]:
            b = comparison["baseline"][metric]
            q = comparison["qwen"][metric]
            d = comparison["deltas"][metric]
            arrow = "▲" if d > 0 else "▼"
            print(f"  {metric:<25s} {b:<12.4f} {q:<12.4f} {arrow} {abs(d):.4f}")
    else:
        logger.warning("Baseline metrics not found at %s, skipping comparison", BASELINE_PATH)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Benign", "Injection"],
                    yticklabels=["Benign", "Injection"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — Qwen2-1.5B QLoRA\n(acc={acc:.2%})")
        fig.tight_layout()
        fig_path = OUTPUT_DIR / "qwen_confusion.png"
        fig.savefig(fig_path, dpi=150)
        logger.info("Confusion matrix saved: %s", fig_path)
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping plot")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
