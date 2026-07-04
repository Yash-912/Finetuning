#!/usr/bin/env python3
"""
Analyze processed dataset: print per-source stats, class balance, text length distribution.

Usage:
    python scripts/analyze_dataset.py --data-dir data/processed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def analyze(df: pd.DataFrame, name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Total rows:      {len(df)}")
    print(f"  Benign (0):      {(df['label'] == 0).sum()}")
    print(f"  Injection (1):   {(df['label'] == 1).sum()}")
    ratio = (df['label'] == 0).sum() / max((df['label'] == 1).sum(), 1)
    print(f"  Ratio (benign:inj): {ratio:.2f}:1")

    if "source" in df.columns:
        print(f"\n  Per-source breakdown:")
        for src, count in df["source"].value_counts().items():
            inj = ((df["source"] == src) & (df["label"] == 1)).sum()
            ben = ((df["source"] == src) & (df["label"] == 0)).sum()
            print(f"    {src:<25s} {count:>6d} total  ({ben} benign, {inj} injection)")

    lengths = df["text"].astype(str).str.len()
    print(f"\n  Text length (chars):")
    print(f"    min:    {lengths.min()}")
    print(f"    max:    {lengths.max()}")
    print(f"    mean:   {lengths.mean():.1f}")
    print(f"    median: {lengths.median():.0f}")
    print(f"    p95:    {lengths.quantile(0.95):.0f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze processed dataset")
    parser.add_argument("--data-dir", default="data/processed", help="Path to processed data dir")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        sys.exit(1)

    for split_name in ["train", "val", "test"]:
        path = data_dir / f"{split_name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            analyze(df, split_name)
        else:
            print(f"\n  {path} not found, skipping")

    adv_path = data_dir / "adversarial_eval.parquet"
    if adv_path.exists():
        df = pd.read_parquet(adv_path)
        analyze(df, "adversarial_eval")

    card_path = data_dir / "dataset_card.json"
    if card_path.exists():
        import json
        with open(card_path) as f:
            card = json.load(f)
        print(f"\n{'='*60}")
        print("  Dataset Card Summary")
        print(f"{'='*60}")
        print(f"  Final counts: {json.dumps(card.get('final_counts', {}), indent=4)}")
        print(f"  Class balance: {json.dumps(card.get('class_balance', {}), indent=4)}")


if __name__ == "__main__":
    main()
