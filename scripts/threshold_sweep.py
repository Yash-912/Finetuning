#!/usr/bin/env python3
"""
Threshold sweep for the calibrated detector: precision / recall / false-positive rate at each cutoff.

The middleware blocks when label==1 AND confidence >= THRESHOLD (default 0.85). That cutoff was never
justified. This script measures it on:
  - test.parquet          frozen in-distribution test set
  - benign_test.parquet   held-out hand-authored benign set (if present)
  - adversarial_eval.parquet  hand-written attack set (if present)

Outputs eval/threshold_sweep.json with one row per threshold and the recall at a 1% false-positive rate.

Usage:
    python scripts/threshold_sweep.py --model-dir models/qwen-injection-detector/best
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

DATA = Path("data/processed")
THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]


def load_model(model_dir: Path, base_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2, quantization_config=bnb)
    model.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(model, str(model_dir)).to(device).eval()
    return tok, model, device


@torch.no_grad()
def p_injection(tok, model, device, texts: list[str], temperature: float, max_length: int) -> np.ndarray:
    order = np.argsort([len(t) for t in texts])
    out = np.zeros(len(texts))
    for i in range(0, len(texts), 16):
        idx = order[i : i + 16]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits.float().cpu()
        out[idx] = torch.softmax(logits / temperature, dim=-1)[:, 1].numpy()
    return out


def sweep(y: np.ndarray, p: np.ndarray) -> list[dict]:
    """Rows: threshold, precision, recall (injection), false-positive rate (benign flagged)."""
    rows = []
    for t in THRESHOLDS:
        pred = p >= t
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        rows.append({
            "threshold": t,
            "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4),
            "fpr": round(fp / max(fp + tn, 1), 4),
        })
    return rows


def recall_at_fpr(y: np.ndarray, p: np.ndarray, target_fpr: float = 0.01) -> dict:
    """Highest recall achievable while the benign false-positive rate stays at or below target_fpr."""
    benign = np.sort(p[y == 0])[::-1]
    if len(benign) == 0:
        return {}
    k = int(np.floor(target_fpr * len(benign)))
    cutoff = benign[k] if k < len(benign) else 0.0
    recall = float((p[y == 1] > cutoff).mean()) if (y == 1).any() else float("nan")
    return {"target_fpr": target_fpr, "threshold": round(float(cutoff), 4), "recall": round(recall, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/qwen-injection-detector/best")
    ap.add_argument("--out", default="eval/threshold_sweep.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("configs/training_config.yaml").read_text())
    base, max_len = cfg["model"]["base_model_name"], cfg["model"]["max_length"]
    model_dir = Path(args.model_dir)
    temp_path = model_dir / "temperature.pt"
    temperature = torch.load(str(temp_path)).item() if temp_path.exists() else 1.0

    tok, model, device = load_model(model_dir, base)

    datasets = {}
    for name, fname in [("test", "test.parquet"), ("benign_test", "benign_test.parquet"),
                        ("adversarial", "adversarial_eval.parquet")]:
        path = DATA / fname
        if path.exists():
            datasets[name] = pd.read_parquet(path)

    results = {"model_dir": str(model_dir), "temperature": round(float(temperature), 4), "sets": {}}
    for name, df in datasets.items():
        y = df["label"].to_numpy()
        p = p_injection(tok, model, device, df["text"].tolist(), temperature, max_len)
        results["sets"][name] = {
            "n": int(len(df)),
            "injections": int((y == 1).sum()),
            "benign": int((y == 0).sum()),
            "by_threshold": sweep(y, p),
            "recall_at_1pct_fpr": recall_at_fpr(y, p),
        }
        print(name, results["sets"][name]["recall_at_1pct_fpr"])

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
