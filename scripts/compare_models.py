#!/usr/bin/env python3
"""
Score several detector versions on every eval set with one consistent protocol, then rank them.

Sets scored here (each model is loaded once):
  test            frozen in-distribution test split
  benign_test     held-out hand-authored benign prompts (benign only)
  adversarial     47 hand-written direct attacks (43 attack / 4 benign)
  handwritten     frozen hand-written set (data/eval_handwritten)
  indirect        frozen indirect-injection set (data/eval_indirect)
  indirect_long   frozen long-document indirect set (data/eval_indirect_long): reported, not ranked
BIPIA is too slow to rerun here, so its 0.85-threshold numbers are read from eval/bipia_*_results.json. A BIPIA
result is used only if it was produced by the same adapter weights (sha256), the same temperature and the same
--long-input mode as this run; otherwise BIPIA is left out and a warning is printed.

Ranking rule (fixed before the numbers were produced, so it cannot be tuned to a preferred winner):
  score = mean Youden's J (recall - FPR at --threshold) over the ranked sets that contain both classes
          (test, adversarial, handwritten, indirect, bipia)
  tie-break = lower pooled FPR over every benign prompt scored.
Every recall and FPR carries a 95% Wilson interval because most of these sets are small.

Usage:
    python scripts/compare_models.py MODEL_DIR [MODEL_DIR ...] [--threshold 0.85]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
from threshold_sweep import adapter_sha256, load_model, p_injection, recall_at_fpr

DATA = Path("data/processed")
FROZEN = {
    "handwritten": Path("data/eval_handwritten"),
    "indirect": Path("data/eval_indirect"),
    "indirect_long": Path("data/eval_indirect_long"),
}
RANKED = ["test", "adversarial", "handwritten", "indirect", "bipia"]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def load_frozen(set_dir: Path) -> pd.DataFrame:
    eval_path = set_dir / "eval.jsonl"
    expected = (set_dir / "eval.sha256").read_text().split()[0]
    if hashlib.sha256(eval_path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"{eval_path} changed since it was frozen")
    return pd.DataFrame([json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines()])


def load_sets() -> dict[str, pd.DataFrame]:
    sets = {
        "test": pd.read_parquet(DATA / "test.parquet"),
        "benign_test": pd.read_parquet(DATA / "benign_test.parquet"),
        "adversarial": pd.read_parquet(DATA / "adversarial_eval.parquet"),
    }
    sets.update({name: load_frozen(path) for name, path in FROZEN.items()})
    return sets


def summarize(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    n_attack, n_benign = int((y == 1).sum()), int((y == 0).sum())
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    out = {"n_attack": n_attack, "n_benign": n_benign, "tp": tp, "fp": fp}
    if n_attack:
        out["recall"], out["recall_ci95"] = round(tp / n_attack, 4), wilson(tp, n_attack)
    if n_benign:
        out["fpr"], out["fpr_ci95"] = round(fp / n_benign, 4), wilson(fp, n_benign)
    if n_attack and n_benign:
        out["auroc"] = round(float(roc_auc_score(y, p)), 4)
        out["recall_at_1pct_fpr"] = recall_at_fpr(y, p)
    return out


def load_bipia(
    adapter_hash: str, temperature: float, threshold: float, long_input: str, results_dir: Path = Path("eval")
) -> dict | None:
    """Read the BIPIA numbers eval_bipia.py produced for exactly this model and scoring mode, if present.

    Models are matched by adapter hash, not folder path: a path can later hold a different model.
    """
    for path in sorted(results_dir.glob("bipia_*_results.json")):
        res = json.loads(path.read_text())
        same_model = res.get("adapter_sha256") == adapter_hash and abs(res["temperature"] - temperature) < 1e-3
        if not same_model or res["threshold"] != threshold or res.get("long_input", "truncate") != long_input:
            continue
        o = res["overall"]
        tp, fp = round(o["recall"] * o["n_attack"]), round(o["fpr"] * o["n_benign"])
        return {
            "n_attack": o["n_attack"], "n_benign": o["n_benign"], "tp": tp, "fp": fp,
            "recall": o["recall"], "recall_ci95": wilson(tp, o["n_attack"]),
            "fpr": o["fpr"], "fpr_ci95": wilson(fp, o["n_benign"]),
            "source": str(path),
        }
    return None


def score(results: dict[str, dict]) -> dict:
    """Ranking score and pooled FPR for one model's per-set results."""
    j = {s: results[s]["recall"] - results[s]["fpr"] for s in RANKED if s in results and "fpr" in results[s] and "recall" in results[s]}
    fp = sum(r["fp"] for r in results.values() if r["n_benign"])
    benign = sum(r["n_benign"] for r in results.values())
    return {
        "youden_j_by_set": {s: round(v, 4) for s, v in j.items()},
        "sets_ranked": sorted(j),
        "score": round(float(np.mean(list(j.values()))), 4),
        "pooled_fpr": round(fp / benign, 4),
        "pooled_benign": benign,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dirs", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--long-input", choices=["truncate", "window"], default="truncate",
                    help="truncate: first 512 tokens only (original); window: read whole texts in overlapping windows")
    ap.add_argument("--out", type=Path, default=None, help="default eval/model_comparison[_window].json")
    args = ap.parse_args()

    windowed = args.long_input == "window"
    if args.out is None:
        args.out = Path("eval/model_comparison_window.json" if windowed else "eval/model_comparison.json")
    sets = load_sets()
    cfg = yaml.safe_load(Path("configs/training_config.yaml").read_text())
    base, max_len = cfg["model"]["base_model_name"], cfg["model"]["max_length"]

    models = {}
    for md in map(Path, args.model_dirs):
        temp_path = md / "temperature.pt"
        temperature = torch.load(str(temp_path)).item() if temp_path.exists() else 1.0
        tok, model, device = load_model(md, base)
        results = {}
        for name, df in sets.items():
            p = p_injection(tok, model, device, df["text"].tolist(), temperature, max_len, windowed=windowed)
            results[name] = summarize(df["label"].to_numpy(), p, args.threshold)
        del model
        torch.cuda.empty_cache()
        bipia = load_bipia(adapter_sha256(md), temperature, args.threshold, args.long_input)
        if bipia:
            results["bipia"] = bipia
        else:
            print(f"warning: no matching BIPIA result for {md} ({args.long_input}); BIPIA left out", file=sys.stderr)
        models[str(md)] = {
            "adapter_sha256": adapter_sha256(md),
            "temperature": round(float(temperature), 4),
            "sets": results,
            "ranking": score(results),
        }
        print(md, models[str(md)]["ranking"]["score"], models[str(md)]["ranking"]["pooled_fpr"], flush=True)

    order = sorted(models, key=lambda m: (-models[m]["ranking"]["score"], models[m]["ranking"]["pooled_fpr"]))
    report = {"threshold": args.threshold, "long_input": args.long_input, "ranked_sets": RANKED, "order": order, "models": models}
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}; best: {order[0]}")


if __name__ == "__main__":
    main()
