from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_deepset(cfg: dict[str, Any]) -> pd.DataFrame:
    ds = load_dataset(cfg["hf_path"], split=cfg["split"])
    df = ds.to_pandas()[["text", "label"]]
    df["source"] = "deepset"
    return df


def load_s_labs(cfg: dict[str, Any]) -> pd.DataFrame:
    ds = load_dataset(cfg["hf_path"], split=cfg["split"])
    df = ds.to_pandas()[["text", "label"]]
    df["source"] = "s_labs"
    return df


def load_xtram1(cfg: dict[str, Any]) -> pd.DataFrame:
    ds = load_dataset(cfg["hf_path"], split=cfg["split"])
    df = ds.to_pandas()[["text", "label"]]
    df["source"] = "xtram1"

    injections = df[df["label"] == 1].copy()
    logger.info(
        "xTRam1: %d total, %d injection (keeping only injection)",
        len(df), len(injections),
    )
    return injections


def load_gandalf(cfg: dict[str, Any]) -> pd.DataFrame:
    ds = load_dataset(cfg["hf_path"], split=cfg["split"])
    df = ds.to_pandas()[["text"]]
    df["label"] = 1
    df["source"] = "gandalf"

    max_n = cfg.get("max_samples")
    if max_n and len(df) > max_n:
        df = df.sample(n=max_n, random_state=42)
        logger.info("Gandalf: subsampled to %d", len(df))
    return df


def load_hackaprompt(cfg: dict[str, Any]) -> pd.DataFrame:
    ds = load_dataset(cfg["hf_path"], split=cfg["split"])
    df = ds.to_pandas()

    if cfg.get("filter_successful", False):
        if "success" not in df.columns:
            logger.warning("hackaprompt: no 'success' column, skipping filter")
        else:
            before = len(df)
            df = df[df["success"] == True].copy()
            logger.info("hackaprompt: filtered to successful only: %d -> %d", before, len(df))

    if "prompt" in df.columns:
        df = df.rename(columns={"prompt": "text"})
    elif "text" not in df.columns:
        raise KeyError("hackaprompt: expected 'prompt' or 'text' column")

    df["label"] = 1
    df["source"] = "hackaprompt"

    max_n = cfg.get("max_samples")
    if max_n and len(df) > max_n:
        df = df.sample(n=max_n, random_state=42)
        logger.info("hackaprompt: subsampled to %d", len(df))
    return df[["text", "label", "source"]]


def load_no_robots(cfg: dict[str, Any]) -> pd.DataFrame:
    ds = load_dataset(cfg["hf_path"], split=cfg["split"])
    df = ds.to_pandas()

    texts = []
    for _, row in df.iterrows():
        msg = row["messages"]
        if isinstance(msg, list) and len(msg) > 0:
            first_user = next((m["content"] for m in msg if m["role"] == "user"), None)
            if first_user:
                texts.append(first_user)
            else:
                texts.append(msg[0].get("content", ""))
        else:
            texts.append(row.get("prompt", ""))

    result = pd.DataFrame({"text": texts, "label": 0, "source": "no_robots"})
    logger.info("no_robots: %d examples", len(result))
    return result


LOADERS = {
    "deepset_prompt_injections": load_deepset,
    "s_labs_prompt_injection": load_s_labs,
    "xtram1_safe_guard": load_xtram1,
    "gandalf_ignore_instructions": load_gandalf,
    "hackaprompt_submissions": load_hackaprompt,
    "no_robots": load_no_robots,
}
