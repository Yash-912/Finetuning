"""Score models on a frozen indirect-injection eval set (data/eval_indirect or data/eval_indirect_long).

Usage:
    python scripts/eval_indirect.py MODEL_DIR [MODEL_DIR ...] [--set-dir data/eval_indirect_long] [--threshold 0.85]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, "scripts")
from threshold_sweep import load_model, p_injection, recall_at_fpr, sweep

DEFAULT_SET_DIR = Path("data/eval_indirect")
POSITIONS = ["start", "middle", "end", "embedded_in_table_or_code"]


def output_path(set_dir: Path, long_input: str = "truncate") -> Path:
    """eval/indirect_comparison.json for the short set, eval/indirect_long_comparison.json for the long one.
    Windowed runs get a _window suffix so they never overwrite the truncated results."""
    suffix = set_dir.name.removeprefix("eval_indirect")
    tag = "_window" if long_input == "window" else ""
    return Path("eval") / f"indirect{suffix}_comparison{tag}.json"


def verify_frozen(set_dir: Path) -> Path:
    eval_path = set_dir / "eval.jsonl"
    expected = (set_dir / "eval.sha256").read_text().split()[0]
    if hashlib.sha256(eval_path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"{eval_path} changed since it was frozen")
    return eval_path


def score_model(
    df: pd.DataFrame, model_dir: Path, base: str, max_len: int, threshold: float, windowed: bool = False
) -> dict:
    y = df["label"].to_numpy()
    temp_path = model_dir / "temperature.pt"
    temperature = torch.load(str(temp_path)).item() if temp_path.exists() else 1.0
    tok, model, device = load_model(model_dir, base)
    p = p_injection(tok, model, device, df["text"].tolist(), temperature, max_len, windowed=windowed)
    del model
    torch.cuda.empty_cache()
    pred = p >= threshold
    cats = {}
    for (cat, label), g in df.assign(pred=pred).groupby(["category", "label"]):
        if label == 1:
            cats[f"{cat}/attack"] = round(float(g["pred"].mean()), 3)
    return {
        "temperature": round(float(temperature), 4),
        f"at_{threshold}": {
            "recall": round(float(pred[y == 1].mean()), 4),
            "fpr": round(float(pred[y == 0].mean()), 4),
        },
        "by_threshold": sweep(y, p),
        "recall_at_1pct_fpr": recall_at_fpr(y, p),
        f"attack_recall_by_position_at_{threshold}": {
            pos: round(float(pred[(df["position"] == pos).to_numpy()].mean()), 4) for pos in POSITIONS
        },
        f"recall_by_category_at_{threshold}": cats,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dirs", nargs="+", help="model directories to score")
    ap.add_argument("--set-dir", type=Path, default=DEFAULT_SET_DIR, help="frozen eval set directory")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--long-input", choices=["truncate", "window"], default="truncate",
                    help="truncate: read only the first 512 tokens; window: read the whole text")
    args = ap.parse_args()

    eval_path = verify_frozen(args.set_dir)
    df = pd.DataFrame([json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines()])
    cfg = yaml.safe_load(Path("configs/training_config.yaml").read_text())
    base, max_len = cfg["model"]["base_model_name"], cfg["model"]["max_length"]

    out = {}
    for model_dir in map(Path, args.model_dirs):
        out[str(model_dir)] = score_model(df, model_dir, base, max_len, args.threshold, args.long_input == "window")
        print(model_dir, out[str(model_dir)][f"at_{args.threshold}"], flush=True)
    dest = output_path(args.set_dir, args.long_input)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
