#!/usr/bin/env python3
"""
QLoRA fine-tuning of Qwen2-1.5B for prompt injection classification.

Usage:
    python scripts/train_qlora.py --config configs/training_config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("train_qlora")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1]
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )

    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = float("nan")

    result = {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "roc_auc": round(float(auc), 4),
        "benign_f1": round(float(f1[0]), 4),
        "injection_f1": round(float(f1[1]), 4),
        "benign_recall": round(float(recall[0]), 4),
        "injection_recall": round(float(recall[1]), 4),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s (VRAM: %.1f GB)", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9)
        logger.info("CUDA capability: %d.%d", *torch.cuda.get_device_capability())

    model_cfg = cfg["model"]
    quant_cfg = cfg["quantization"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    logger.info("Loading tokenizer: %s", model_cfg["base_model_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["base_model_name"],
        quantization_config=bnb_config,
        num_labels=model_cfg["num_labels"],
        trust_remote_code=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(model)

    logger.info("Configuring LoRA (rank=%d)...", lora_cfg["r"])
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %s / %s (%.2f%%)",
                f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    logger.info("Loading data...")
    train_df = pd.read_parquet(data_cfg["train_path"])
    val_df = pd.read_parquet(data_cfg["val_path"])

    logger.info("Train: %d | Val: %d", len(train_df), len(val_df))

    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label"]])

    def tokenize_fn(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=model_cfg["max_length"],
            padding=False,
        )
        return result

    logger.info("Tokenizing...")
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        fp16=train_cfg["fp16"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        remove_unused_columns=train_cfg["remove_unused_columns"],
        report_to=train_cfg["report_to"],
        eval_strategy="steps",
        save_strategy="steps",
        logging_dir=str(Path(train_cfg["output_dir"]) / "logs"),
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving best model...")
    trainer.save_model(str(Path(train_cfg["output_dir"]) / "best"))
    tokenizer.save_pretrained(str(Path(train_cfg["output_dir"]) / "best"))

    logger.info("Training complete. Best model saved to: %s/best", train_cfg["output_dir"])

    logger.info("Final evaluation on val set:")
    val_metrics = trainer.evaluate()
    for k, v in val_metrics.items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
