#!/usr/bin/env python3
"""
Learn optimal temperature scaling parameter on the validation set.

Produces:
  - models/qwen-injection-detector/best/temperature.pt
  - eval/calibration_metrics.json
  - eval/calibration_curve.png
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from peft import PeftModel
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("calibrate")

MODEL_DIR = "models/qwen-injection-detector/best"
VAL_PATH = "data/processed/val.parquet"
OUTPUT_DIR = Path("eval")


def compute_ece(logits: torch.Tensor, labels: torch.Tensor, T: float = 1.0, n_bins: int = 10) -> float:
    probs = torch.softmax(logits / T, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(labels)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = in_bin.sum().item()
        if bin_size == 0:
            continue
        bin_confidence = confidences[in_bin].mean().item()
        bin_accuracy = (predictions[in_bin] == labels[in_bin]).float().mean().item()
        ece += (bin_size / total) * abs(bin_confidence - bin_accuracy)
    return ece


def compute_nll(logits: torch.Tensor, labels: torch.Tensor, T: float) -> float:
    scaled = logits / T
    return nn.CrossEntropyLoss()(scaled, labels).item()


def main():
    logger.info("Loading config...")
    cfg = load_config("configs/training_config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    base_name = cfg["model"]["base_model_name"]
    max_length = cfg["model"]["max_length"]
    model_dir = Path(MODEL_DIR)

    logger.info("Loading tokenizer: %s", base_name)
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        num_labels=2,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    logger.info("Loading LoRA adapters from: %s", model_dir)
    model = PeftModel.from_pretrained(model, str(model_dir))
    model = model.to(device)
    model.eval()
    logger.info("Model loaded. Parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    logger.info("Loading validation data: %s", VAL_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    logger.info("Validation samples: %d", len(val_df))

    val_ds = Dataset.from_pandas(val_df[["text", "label"]])

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=max_length, padding=False
        )

    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(val_ds, batch_size=64, collate_fn=collator, shuffle=False)

    all_logits = []
    all_labels = []

    logger.info("Running inference on validation set...")
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(batch["labels"].cpu())

    logits = torch.cat(all_logits, dim=0).float()
    labels = torch.cat(all_labels, dim=0)
    logger.info("Collected logits: %s", str(logits.shape))

    logger.info("Grid search for optimal temperature...")
    T_candidates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    nll_values = {}
    for T in T_candidates:
        nll = compute_nll(logits, labels, T)
        nll_values[T] = nll
        logger.info("  T=%.2f → NLL=%.4f", T, nll)

    best_grid_T = min(T_candidates, key=lambda t: nll_values[t])
    logger.info("Best grid T: %.2f (NLL=%.4f)", best_grid_T, nll_values[best_grid_T])

    logger.info("Fine-tuning temperature...")
    bounds = (max(0.1, best_grid_T * 0.5), best_grid_T * 2.0)
    result = minimize_scalar(
        lambda t: compute_nll(logits, labels, t),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-3},
    )
    optimal_T = result.x
    logger.info("Optimal temperature: %.4f (NLL=%.4f)", optimal_T, result.fun)

    ece_before = compute_ece(logits, labels, T=1.0)
    ece_after = compute_ece(logits, labels, T=optimal_T)
    logger.info("ECE before (T=1.0): %.4f", ece_before)
    logger.info("ECE after  (T=%.4f): %.4f", optimal_T, ece_after)

    logger.info("Saving temperature: %s", model_dir / "temperature.pt")
    torch.save(torch.tensor(optimal_T), str(model_dir / "temperature.pt"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "optimal_temperature": round(float(optimal_T), 4),
        "ece_before": round(float(ece_before), 4),
        "ece_after": round(float(ece_after), 4),
        "improvement_pct": round(float((ece_before - ece_after) / max(ece_before, 1e-8) * 100), 2),
        "nll_before": round(float(compute_nll(logits, labels, 1.0)), 4),
        "nll_after": round(float(result.fun), 4),
        "nll_values": {str(k): round(float(v), 4) for k, v in nll_values.items()},
        "val_size": len(labels),
    }
    metrics_path = OUTPUT_DIR / "calibration_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved: %s", metrics_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        probs_before = torch.softmax(logits / 1.0, dim=-1)
        probs_after = torch.softmax(logits / optimal_T, dim=-1)
        conf_before, preds_before = probs_before.max(dim=-1)
        conf_after, preds_after = probs_after.max(dim=-1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        for ax, conf, preds, title, ece_val in [
            (ax1, conf_before, preds_before, f"Before Calibration (T=1.0)\nECE={ece_before:.2%}", ece_before),
            (ax2, conf_after, preds_after, f"After Calibration (T={optimal_T:.2f})\nECE={ece_after:.2%}", ece_after),
        ]:
            ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Perfect")
            n_bins = 10
            bin_edges = torch.linspace(0, 1, n_bins + 1)
            for i in range(n_bins):
                in_bin = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
                if in_bin.sum() == 0:
                    continue
                bin_conf = conf[in_bin].mean().item()
                bin_acc = (preds[in_bin] == labels[in_bin]).float().mean().item()
                error = abs(bin_conf - bin_acc)
                color = plt.cm.RdYlGn_r(min(error * 5, 1.0))
                ax.bar(bin_conf, bin_acc, width=0.08, color=color, edgecolor="black", alpha=0.8)
            ax.set_xlabel("Predicted Confidence")
            ax.set_ylabel("Actual Accuracy")
            ax.set_title(title)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(alpha=0.3)

        fig.tight_layout()
        fig_path = OUTPUT_DIR / "calibration_curve.png"
        fig.savefig(fig_path, dpi=150)
        logger.info("Calibration curve saved: %s", fig_path)
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")

    logger.info("Calibration complete. Optimal T = %.4f", optimal_T)


if __name__ == "__main__":
    main()
