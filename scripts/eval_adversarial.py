#!/usr/bin/env python3
"""
Evaluate adversarial set against the live middleware server.

Usage:
    Terminal 1: python -m middleware.app
    Terminal 2: python scripts/eval_adversarial.py

Requires middleware on localhost:8080 with the same model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ADV_PATH = "data/processed/adversarial_eval.parquet"
OUTPUT_DIR = Path("eval")
MIDDLEWARE_URL = "http://localhost:8080"


def main():
    base_name = "Qwen/Qwen2-1.5B"

    df = pd.read_parquet(ADV_PATH)
    print(f"Loaded {len(df)} adversarial examples")

    labels = []
    confidences = []
    errors = []

    with httpx.Client(base_url=MIDDLEWARE_URL, timeout=30.0) as client:
        health = client.get("/health")
        print(f"Middleware health: {health.json()}")

        for i, row in df.iterrows():
            payload = {
                "model": base_name,
                "messages": [{"role": "user", "content": row["text"]}],
                "max_tokens": 1,
            }
            resp = client.post(
                "/chat/completions",
                json=payload,
            )
            if resp.status_code == 403:
                body = resp.json()
                labels.append(1)
                confidences.append(body.get("confidence", 1.0))
            elif resp.status_code == 502:
                labels.append(0)
                confidences.append(0.0)
                errors.append(i)
            elif resp.status_code == 200:
                labels.append(0)
                confidences.append(0.0)
            else:
                print(f"  Sample {i}: unexpected status {resp.status_code}")
                labels.append(-1)
                confidences.append(-1.0)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(df)}")

    gt = df["label"].values
    preds = np.array(labels)
    confs = np.array(confidences)

    valid = preds != -1
    gt = gt[valid]
    preds = preds[valid]
    confs = confs[valid]
    df_valid = df[valid]

    if errors:
        print(f"\nWARNING: {len(errors)} samples errored (LLM unreachable)")
        for i in errors[:5]:
            print(f"  [{i}] {df.iloc[i]['attack_type']}: {df.iloc[i]['text'][:60]}...")

    print(f"\n{'='*60}")
    print("  ADVERSARIAL EVAL RESULTS")
    print(f"{'='*60}")
    print(f"  Samples:       {len(gt)}")
    print(f"  Accuracy:      {accuracy_score(gt, preds):.4f}")

    prec, rec, f1, _ = precision_recall_fscore_support(
        gt, preds, average=None, labels=[0, 1]
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(gt, preds, average="macro")
    print(f"  Macro F1:      {macro_f1:.4f}")
    print(f"\n  Per-class:")
    print(f"    Benign (0):   precision={prec[0]:.4f}  recall={rec[0]:.4f}  f1={f1[0]:.4f}")
    print(f"    Injection (1): precision={prec[1]:.4f}  recall={rec[1]:.4f}  f1={f1[1]:.4f}")

    cm = confusion_matrix(gt, preds)
    print(f"\n  Confusion matrix:")
    print(f"                Predicted")
    print(f"               Benign  Inj")
    print(f"  Actual Benign  {cm[0,0]:>5d}  {cm[0,1]:>3d}")
    print(f"         Inj     {cm[1,0]:>5d}  {cm[1,1]:>3d}")

    print(f"\n  Per-attack-type accuracy:")
    for atype in sorted(df_valid["attack_type"].unique()):
        mask = df_valid["attack_type"] == atype
        atype_gt = gt[mask]
        atype_pred = preds[mask]
        atype_acc = accuracy_score(atype_gt, atype_pred)
        atype_n = int(mask.sum())
        atype_inj = int(atype_gt.sum())
        atype_correct = int((atype_gt == atype_pred).sum())
        print(f"    {atype:<22s} n={atype_n:>2d}  ({atype_inj:>2d} inj)  "
              f"acc={atype_acc:.2%}  ({atype_correct}/{atype_n})")

    print()
    print(classification_report(gt, preds, target_names=["benign", "injection"]))

    results = {
        "model": f"qwen2-1.5b-qlora_adversarial",
        "test_size": int(len(gt)),
        "accuracy": round(float(accuracy_score(gt, preds)), 4),
        "macro_f1": round(float(macro_f1), 4),
        "benign": {
            "precision": round(float(prec[0]), 4),
            "recall": round(float(rec[0]), 4),
            "f1": round(float(f1[0]), 4),
        },
        "injection": {
            "precision": round(float(prec[1]), 4),
            "recall": round(float(rec[1]), 4),
            "f1": round(float(f1[1]), 4),
        },
        "confusion_matrix": cm.tolist(),
        "per_category": {},
    }

    for atype in sorted(df_valid["attack_type"].unique()):
        mask = df_valid["attack_type"] == atype
        atype_gt = gt[mask]
        atype_pred = preds[mask]
        results["per_category"][atype] = {
            "n": int(mask.sum()),
            "accuracy": round(float(accuracy_score(atype_gt, atype_pred)), 4),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "adversarial_metrics.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved: {path}")


if __name__ == "__main__":
    main()
