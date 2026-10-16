#!/usr/bin/env python3
"""
Leave-one-source-out (LOSO) evaluation of the QLoRA injection detector.

For each source S: train on every OTHER source, then evaluate on S (never seen in training).
This tests whether the model learned "injection" or just "which dataset does this text come from".

Training is expensive (hours at full size on an RTX 4050), so by default each fold trains on a
label-stratified subsample for 1 epoch. Results are written per fold and the run is resumable.

Usage:
    python scripts/eval_loso.py                              # all sources, 4000 train samples, 1 epoch
    python scripts/eval_loso.py --sources gandalf --max-train-samples 64   # quick smoke test
    python scripts/eval_loso.py --dry-run                    # build splits/configs only

Outputs:
    data/loso/<source>/{train,val,held}.parquet, configs/loso/<source>.yaml (generated)
    models/loso/<source>/best                    (fold adapters; gitignored)
    eval/loso/<source>.json                      (per-fold metrics)
    eval/loso_results.json                       (aggregate)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.loso import make_loso_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("eval_loso")

DATA_DIR = Path("data/processed")
BASE_CONFIG = Path("configs/training_config.yaml")


def prepare_fold(pool: pd.DataFrame, src: str, args) -> Path:
    train, val, held = make_loso_split(pool, src, max_train_samples=args.max_train_samples, seed=args.seed)
    val = val.sample(n=min(len(val), args.max_val_samples), random_state=args.seed).reset_index(drop=True)
    fold_data = Path("data/loso") / src
    fold_data.mkdir(parents=True, exist_ok=True)
    train.to_parquet(fold_data / "train.parquet", index=False)
    val.to_parquet(fold_data / "val.parquet", index=False)
    held.to_parquet(fold_data / "held.parquet", index=False)

    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    steps_per_epoch = math.ceil(
        len(train) / (cfg["training"]["per_device_train_batch_size"] * cfg["training"]["gradient_accumulation_steps"])
    )
    total_steps = steps_per_epoch * args.epochs
    eval_every = max(1, min(100, total_steps // 3))
    cfg["training"].update(
        output_dir=f"models/loso/{src}",
        num_train_epochs=args.epochs,
        eval_steps=eval_every,
        save_steps=eval_every,
    )
    cfg["data"].update(
        train_path=str(fold_data / "train.parquet"),
        val_path=str(fold_data / "val.parquet"),
    )
    cfg_path = Path("configs/loso") / f"{src}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    logger.info("[%s] train=%d val=%d held=%d (steps/epoch=%d)", src, len(train), len(val), len(held), steps_per_epoch)
    return cfg_path


def predict(model_dir: Path, texts: list[str], base_name: str, max_length: int) -> np.ndarray:
    """Return P(injection) for each text, uncalibrated (T=1: no temperature is fit per fold)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

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

    # Sort by length so batches pad little; restore order afterwards.
    order = np.argsort([len(t) for t in texts])
    probs = np.zeros(len(texts))
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            idx = order[i : i + 16]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            p = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu().numpy()
            probs[idx] = p
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs


def score(labels: np.ndarray, p_inj: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    preds = (p_inj >= 0.5).astype(int)
    out = {
        "n": int(len(labels)),
        "classes_present": sorted(int(c) for c in np.unique(labels)),
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
    }
    if len(np.unique(labels)) == 2:
        out["macro_f1"] = round(float(f1_score(labels, preds, average="macro")), 4)
        out["roc_auc"] = round(float(roc_auc_score(labels, p_inj)), 4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="*", help="Subset of sources to hold out (default: all)")
    ap.add_argument("--max-train-samples", type=int, default=4000)
    ap.add_argument("--max-val-samples", type=int, default=500, help="Cap val set so periodic eval stays cheap")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="Prepare splits/configs, skip training")
    ap.add_argument("--force", action="store_true", help="Re-run folds that already have results")
    args = ap.parse_args()

    pool = pd.concat(
        [pd.read_parquet(DATA_DIR / f"{s}.parquet") for s in ("train", "val", "test")], ignore_index=True
    )
    sources = args.sources or sorted(pool["source"].unique())
    base_cfg = yaml.safe_load(BASE_CONFIG.read_text())

    out_dir = Path("eval/loso")
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in sources:
        result_path = out_dir / f"{src}.json"
        if result_path.exists() and not args.force:
            logger.info("[%s] already done, skipping (use --force to redo)", src)
            continue

        cfg_path = prepare_fold(pool, src, args)
        if args.dry_run:
            continue

        logger.info("[%s] training...", src)
        rc = subprocess.run([sys.executable, "scripts/train_qlora.py", "--config", str(cfg_path)]).returncode
        if rc != 0:
            logger.error("[%s] training failed (exit %d); continuing with next fold", src, rc)
            continue

        held = pd.read_parquet(Path("data/loso") / src / "held.parquet")
        p_inj = predict(Path(f"models/loso/{src}/best"), held["text"].tolist(),
                        base_cfg["model"]["base_model_name"], base_cfg["model"]["max_length"])
        res = score(held["label"].to_numpy(), p_inj)
        res.update(source=src, max_train_samples=args.max_train_samples, epochs=args.epochs)
        result_path.write_text(json.dumps(res, indent=2))
        logger.info("[%s] %s", src, res)

    if args.dry_run:
        return
    folds = {p.stem: json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))}
    Path("eval/loso_results.json").write_text(json.dumps(folds, indent=2))
    print(json.dumps(folds, indent=2))


if __name__ == "__main__":
    main()
