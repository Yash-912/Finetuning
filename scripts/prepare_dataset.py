#!/usr/bin/env python3
"""
Main data pipeline: loads all sources, deduplicates, balances, splits, exports.

Usage:
    python scripts/prepare_dataset.py --config configs/dataset_config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.data.loader import LOADERS
from src.data.deduplicator import deduplicate
from src.data.balancer import check_balance, stratified_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("prepare_dataset")


def build_dataset_card(
    before_stats: dict,
    after_stats: dict,
    config: dict,
) -> dict:
    return {
        "config": config,
        "per_source_before_dedup": before_stats,
        "final_counts": after_stats,
        "class_balance": {
            "benign": int(after_stats.get("benign", 0)),
            "injection": int(after_stats.get("injection", 0)),
            "ratio": round(
                after_stats.get("benign", 0) / max(after_stats.get("injection", 1), 1), 2
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare prompt injection dataset")
    parser.add_argument(
        "--config",
        default="configs/dataset_config.yaml",
        help="Path to dataset config YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    per_source = {}

    for name, source_cfg in cfg["sources"].items():
        if not source_cfg.get("enabled", True):
            logger.info("Skipping disabled source: %s", name)
            continue

        loader_fn = LOADERS.get(name)
        if loader_fn is None:
            logger.warning("No loader registered for '%s', skipping", name)
            continue

        logger.info("Loading %s...", name)
        df = loader_fn(source_cfg)
        per_source[name] = {
            "count": len(df),
            "benign": int((df["label"] == 0).sum()),
            "injection": int((df["label"] == 1).sum()),
        }
        logger.info("  -> %s: %d rows (%d benign, %d injection)", name, len(df), per_source[name]["benign"], per_source[name]["injection"])

        if "df_combined" not in locals():
            df_combined = df
        else:
            df_combined = pd.concat([df_combined, df], ignore_index=True)

    logger.info("Combined: %d rows before dedup", len(df_combined))
    logger.info("Label distribution before dedup:\n%s", df_combined["label"].value_counts().to_string())

    df_combined = deduplicate(df_combined, cfg["processing"])

    logger.info("Combined: %d rows after dedup", len(df_combined))
    logger.info("Label distribution after dedup:\n%s", df_combined["label"].value_counts().to_string())

    df_balanced = check_balance(df_combined, cfg["processing"].get("target_benign_to_injection_ratio", 2.0))

    train, val, test = stratified_split(df_balanced, cfg["split"])

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{split_name}.parquet"
        split_df.to_parquet(path, index=False)
        logger.info("Wrote %s: %d rows -> %s", split_name, len(split_df), path)

    adversarial_template = pd.DataFrame(columns=["text", "label", "attack_type", "notes"])
    adv_path = Path(cfg["output"]["adversarial_eval_path"])
    if not adv_path.exists():
        adversarial_template.to_parquet(adv_path, index=False)
        logger.info("Created empty adversarial eval template: %s", adv_path)
        logger.info("  -> Edit this file with 40-60 hand-authored examples")
    else:
        logger.info("Adversarial eval already exists: %s (%d rows)", adv_path, len(pd.read_parquet(adv_path)))

    card = build_dataset_card(
        before_stats=per_source,
        after_stats={
            "total": len(df_balanced),
            "benign": int((df_balanced["label"] == 0).sum()),
            "injection": int((df_balanced["label"] == 1).sum()),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        config=cfg,
    )

    card_path = Path(cfg["output"]["dataset_card_path"])
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2, default=str)
    logger.info("Wrote dataset card: %s", card_path)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
