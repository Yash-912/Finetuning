#!/usr/bin/env python3
"""
Merge hand-authored benign examples into the corpus WITHOUT touching the frozen test set.

Why not re-run prepare_dataset.py? It re-splits everything, so the test set would change and the
98.7% headline would stop being comparable. Instead:

  - authored benign rows are deduped and split 80/10/10 into train_add / val_add / benign_test
  - train_add and val_add are appended to data/processed/{train,val}.parquet (originals backed up)
  - data/processed/test.parquet is never modified
  - data/processed/benign_test.parquet is a NEW held-out set used to measure the fix

Input: data/authored/*.jsonl, one {"text": "..."} object per line, all benign (label=0).
The source name is the file stem (code, multilingual, hard_negatives, everyday).

Usage:
    python scripts/build_benign_diverse.py            # dry run: writes benign_test.parquet only
    python scripts/build_benign_diverse.py --apply    # also appends to train/val (after backup)
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("build_benign_diverse")

DATA_DIR = Path("data/processed")
AUTHORED_DIR = Path("data/authored")
SEED = 42


def load_authored() -> pd.DataFrame:
    frames = []
    for path in sorted(AUTHORED_DIR.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(rows)
        df = df[["text"]].assign(label=0, source=path.stem)
        logger.info("%-16s %d rows", path.stem, len(df))
        frames.append(df)
    if not frames:
        raise SystemExit(f"No .jsonl files in {AUTHORED_DIR}")
    return pd.concat(frames, ignore_index=True)


def clean(new: pd.DataFrame) -> pd.DataFrame:
    new["text"] = new["text"].astype(str).str.strip()
    new = new[new["text"].str.len() > 0].drop_duplicates(subset="text")

    # A benign-labelled row must not duplicate an existing row with any label: that would be a label conflict.
    existing = pd.concat([pd.read_parquet(DATA_DIR / f"{s}.parquet") for s in ("train", "val", "test")])
    seen = set(existing["text"].str.lower().str.strip())
    before = len(new)
    new = new[~new["text"].str.lower().str.strip().isin(seen)]
    logger.info("Dropped %d rows already present in the corpus", before - len(new))
    return new.reset_index(drop=True)


def split(new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, rest = train_test_split(new, test_size=0.2, random_state=SEED, stratify=new["source"])
    val, test = train_test_split(rest, test_size=0.5, random_state=SEED, stratify=rest["source"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Append train/val rows to the corpus (backs up originals)")
    args = ap.parse_args()

    new = clean(load_authored())
    logger.info("Clean benign rows: %d  %s", len(new), new["source"].value_counts().to_dict())
    train_add, val_add, benign_test = split(new)
    logger.info("train_add=%d val_add=%d benign_test=%d", len(train_add), len(val_add), len(benign_test))

    benign_test.to_parquet(DATA_DIR / "benign_test.parquet", index=False)
    logger.info("Wrote held-out benign test: %s", DATA_DIR / "benign_test.parquet")

    if not args.apply:
        logger.info("Dry run: not touching train/val. Re-run with --apply to append.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for split_name, add in [("train", train_add), ("val", val_add)]:
        path = DATA_DIR / f"{split_name}.parquet"
        shutil.copy2(path, DATA_DIR / f"{split_name}_pre_benign_{stamp}.parquet")
        merged = pd.concat([pd.read_parquet(path), add], ignore_index=True).sample(frac=1, random_state=SEED)
        merged.to_parquet(path, index=False)
        logger.info("%s: +%d rows -> %d total", split_name, len(add), len(merged))


if __name__ == "__main__":
    main()
