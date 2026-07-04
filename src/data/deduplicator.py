from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

SOURCE_PRIORITY = {
    "s_labs": 0,
    "xtram1": 1,
    "no_robots": 2,
    "hackaprompt": 3,
    "gandalf": 4,
    "deepset": 5,
}


def exact_deduplicate(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    before = len(df)
    df["text_norm"] = df[text_col].str.strip().str.lower()
    df = df.sort_values("source", key=lambda s: s.map(SOURCE_PRIORITY))
    df = df.drop_duplicates(subset="text_norm", keep="first")
    df = df.drop(columns=["text_norm"])
    after = len(df)
    removed = before - after
    if removed:
        logger.info("Exact dedup: removed %d duplicates (%d -> %d)", removed, before, after)
    return df


def near_deduplicate(
    df: pd.DataFrame,
    threshold: float = 0.92,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 1024,
    text_col: str = "text",
) -> pd.DataFrame:
    before = len(df)
    if before < 2:
        return df

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Computing embeddings for %d texts...", before)

    texts = df[text_col].astype(str).tolist()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    sim_matrix = cosine_similarity(embeddings)

    keep = np.ones(before, dtype=bool)
    removed = []

    for i in range(before):
        if not keep[i]:
            continue
        for j in range(i + 1, before):
            if not keep[j]:
                continue
            if sim_matrix[i, j] >= threshold:
                prio_i = SOURCE_PRIORITY.get(df.iloc[i]["source"], 99)
                prio_j = SOURCE_PRIORITY.get(df.iloc[j]["source"], 99)
                if prio_i <= prio_j:
                    keep[j] = False
                    removed.append(j)
                else:
                    keep[i] = False
                    removed.append(i)
                    break

    after = keep.sum()
    logger.info(
        "Near-dedup (threshold=%.2f): removed %d duplicates (%d -> %d)",
        threshold, before - after, before, after,
    )
    return df.iloc[keep].reset_index(drop=True)


def deduplicate(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    logger.info("Starting deduplication...")
    df = exact_deduplicate(df)
    df = near_deduplicate(
        df,
        threshold=cfg.get("near_dup_threshold", 0.92),
        model_name=cfg.get("embedding_model", "all-MiniLM-L6-v2"),
        batch_size=cfg.get("batch_size", 1024),
    )
    return df.reset_index(drop=True)
