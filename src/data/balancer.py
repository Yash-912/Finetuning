from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def check_balance(df: pd.DataFrame, target_ratio: float = 2.0) -> pd.DataFrame:
    n_benign = (df["label"] == 0).sum()
    n_inj = (df["label"] == 1).sum()
    current_ratio = n_benign / max(n_inj, 1)
    logger.info("Class balance: %d benign, %d injection (ratio=%.2f:1)", n_benign, n_inj, current_ratio)

    if current_ratio > target_ratio:
        excess = int(n_benign - target_ratio * n_inj)
        benign_ids = df[df["label"] == 0].index
        drop = benign_ids.to_series().sample(n=excess, random_state=42)
        df = df.drop(drop)
        logger.info("Downsampled benign by %d to hit target ratio %.1f:1", excess, target_ratio)
    elif current_ratio < target_ratio:
        logger.warning(
            "Benign ratio %.2f is below target %.1f — injection examples may be over-represented. "
            "Consider adding more benign data.",
            current_ratio, target_ratio,
        )
    return df


def stratified_split(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = cfg.get("random_seed", 42)
    train_r = cfg.get("train_ratio", 0.80)
    val_r = cfg.get("val_ratio", 0.10)

    strat_col = cfg.get("stratify_by", "source")
    if strat_col not in df.columns:
        logger.warning("Stratification column '%s' not found, using 'label'", strat_col)
        strat_col = "label"

    strat_labels = df[strat_col].astype(str)

    train_idx, temp_idx = train_test_split(
        range(len(df)),
        test_size=(1.0 - train_r),
        random_state=seed,
        stratify=strat_labels,
    )

    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    temp_strat = temp_df[strat_col].astype(str)

    val_frac = val_r / (val_r + (1.0 - train_r - val_r) + 1e-12)
    val_idx, test_idx = train_test_split(
        range(len(temp_df)),
        test_size=(1.0 - val_frac),
        random_state=seed + 1,
        stratify=temp_strat,
    )

    train = df.iloc[train_idx].reset_index(drop=True)
    val = temp_df.iloc[val_idx].reset_index(drop=True)
    test = temp_df.iloc[test_idx].reset_index(drop=True)

    logger.info("Split: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test
