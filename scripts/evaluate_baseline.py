#!/usr/bin/env python3
"""
Evaluate deepset/deberta-v3-base-injection on our processed test set.

Produces:
  - eval/baseline_metrics.json  (aggregate + per-source metrics)
  - eval/baseline_confusion.png (confusion matrix plot)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "deepset/deberta-v3-base-injection"
TEST_PATH = "data/processed/test.parquet"
OUTPUT_DIR = Path("eval")


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


@torch.no_grad()
def predict(model, dataloader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    for batch in dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        all_logits.append(outputs.logits.cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    return preds, probs


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    print(f"Model parameters: {model.num_parameters():,}")

    label2id = model.config.label2id
    id2label = model.config.id2label
    print(f"Label mapping: {label2id}")

    print(f"\nLoading test data: {TEST_PATH}")
    df = pd.read_parquet(TEST_PATH)
    print(f"  Rows: {len(df)}")
    print(f"  Distribution:\n{df['label'].value_counts().to_string()}")

    dataset = TextDataset(df["text"].tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nRunning inference ({len(dataset)} samples)...")
    preds, probs = predict(model, dataloader)
    gt = df["label"].values

    acc = accuracy_score(gt, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt, preds, average=None, labels=[0, 1]
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        gt, preds, average="macro"
    )

    try:
        auc = roc_auc_score(gt, probs[:, 1])
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(gt, preds)

    print(f"\n{'='*60}")
    print("  BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  Macro F1:      {f1_macro:.4f}")
    print(f"  ROC-AUC:       {auc:.4f}")
    print(f"\n  Per-class metrics:")
    print(f"    Benign (0):   precision={precision[0]:.4f}  recall={recall[0]:.4f}  f1={f1[0]:.4f}")
    print(f"    Injection (1): precision={precision[1]:.4f}  recall={recall[1]:.4f}  f1={f1[1]:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"                Predicted")
    print(f"               Benign  Inj")
    print(f"  Actual Benign  {cm[0,0]:>5d}  {cm[0,1]:>3d}")
    print(f"         Inj     {cm[1,0]:>5d}  {cm[1,1]:>3d}")

    results = {
        "model": MODEL_NAME,
        "test_size": len(df),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_macro), 4),
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
        mask = df["source"] == src
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

    metrics_path = OUTPUT_DIR / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

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
        ax.set_title(f"Confusion Matrix — {MODEL_NAME.split('/')[-1]}\n(acc={acc:.2%})")
        fig.tight_layout()
        fig_path = OUTPUT_DIR / "baseline_confusion.png"
        fig.savefig(fig_path, dpi=150)
        print(f"Confusion matrix saved: {fig_path}")
    except ImportError:
        print("matplotlib/seaborn not installed, skipping plot")

    print("\nDone.")


if __name__ == "__main__":
    main()
