"""Leave-one-source-out (LOSO) split construction."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def make_loso_split(
    pool: pd.DataFrame,
    held_source: str,
    max_train_samples: int | None = None,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, held_out).

    The held-out source never appears in train or val. Train/val come from the remaining sources,
    stratified by label. If ``max_train_samples`` is set, train is subsampled (label-stratified) to
    that size so a full LOSO sweep is affordable on a single consumer GPU.
    """
    if held_source not in set(pool["source"]):
        raise ValueError(f"unknown source: {held_source!r}")

    held = pool[pool["source"] == held_source].reset_index(drop=True)
    rest = pool[pool["source"] != held_source]

    train, val = train_test_split(rest, test_size=val_frac, random_state=seed, stratify=rest["label"])

    if max_train_samples is not None and len(train) > max_train_samples:
        train, _ = train_test_split(
            train, train_size=max_train_samples, random_state=seed, stratify=train["label"]
        )

    return train.reset_index(drop=True), val.reset_index(drop=True), held
