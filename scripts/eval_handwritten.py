#!/usr/bin/env python3
"""
Score one or more calibrated models on the frozen hand-written eval set.

Verifies the set's hash first, so results can't be produced on an edited set.
Reports precision / recall / FPR per threshold, plus per-category recall (attacks) and FPR (benign).

Usage:
    python scripts/eval_handwritten.py --model-dirs models/qwen-injection-detector/best models/qwen-injection-detector-v2/best --tag handwritten
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from threshold_sweep import load_model, p_injection, recall_at_fpr, sweep

EVAL = Path("data/eval_handwritten/eval.jsonl")
HASH_FILE = Path("data/eval_handwritten/eval.sha256")


def verify_frozen():
    expected = HASH_FILE.read_text().split()[0]
    actual = hashlib.sha256(EVAL.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"eval set changed since it was frozen: {actual} != {expected}")


def per_category(df: pd.DataFrame, p: np.ndarray, threshold: float = 0.85) -> dict:
    out = {}
    pred = p >= threshold
    # Group by (category, label): some categories (e.g. multilingual) hold both attacks and benign prompts.
    for (cat, label), g in df.assign(pred=pred).groupby(["category", "label"]):
        key = f"{cat}/{'attack' if label == 1 else 'benign'}"
        if label == 1:
            out[key] = {"n": len(g), "recall": round(float(g["pred"].mean()), 3), "missed": int((~g["pred"]).sum())}
        else:
            out[key] = {"n": len(g), "fpr": round(float(g["pred"].mean()), 3), "flagged": int(g["pred"].sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dirs", nargs="+", required=True)
    ap.add_argument("--tag", default="handwritten")
    args = ap.parse_args()

    verify_frozen()
    df = pd.DataFrame([json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines()])
    y = df["label"].to_numpy()
    cfg = yaml.safe_load(Path("configs/training_config.yaml").read_text())
    base, max_len = cfg["model"]["base_model_name"], cfg["model"]["max_length"]

    results = {}
    for md in args.model_dirs:
        model_dir = Path(md)
        temp_path = model_dir / "temperature.pt"
        temperature = torch.load(str(temp_path)).item() if temp_path.exists() else 1.0
        tok, model, device = load_model(model_dir, base)
        p = p_injection(tok, model, device, df["text"].tolist(), temperature, max_len)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        results[str(model_dir)] = {
            "temperature": round(float(temperature), 4),
            "by_threshold": sweep(y, p),
            "recall_at_1pct_fpr": recall_at_fpr(y, p),
            "by_category_at_0.85": per_category(df, p),
        }
        r = {x["threshold"]: x for x in results[str(model_dir)]["by_threshold"]}[0.85]
        print(f"{md}: @0.85 recall={r['recall']} fpr={r['fpr']} precision={r['precision']}")

    out = Path(f"eval/{args.tag}_comparison.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
