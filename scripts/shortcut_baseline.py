#!/usr/bin/env python3
"""
Dataset-shortcut diagnostics (CPU only, ~seconds).

Answers: how much of the headline accuracy can be reached WITHOUT understanding injections?

Produces eval/shortcut_baseline.json with:
  - source_only:     predict the majority label of each example's *source* (no text used)
  - tfidf_in_dist:   TF-IDF + logistic regression on the standard train/val/test split
  - tfidf_loso:      leave-one-source-out - train on every other source, test on the held-out one

If source_only / tfidf_in_dist are close to the fine-tuned model's score, the benchmark is mostly
measuring source style. Compare tfidf_loso to the Qwen LOSO run (scripts/eval_loso.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

DATA_DIR = Path("data/processed")
OUTPUT = Path("eval/shortcut_baseline.json")


def _fit_tfidf(train: pd.DataFrame):
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    return model.fit(train["text"], train["label"])


def source_only(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Accuracy of predicting each row's source's majority training label. Uses no text."""
    majority = train.groupby("source")["label"].agg(lambda s: int(s.mean() >= 0.5))
    preds = test["source"].map(majority).fillna(0).astype(int)
    return {
        "accuracy": round(float(accuracy_score(test["label"], preds)), 4),
        "macro_f1": round(float(f1_score(test["label"], preds, average="macro")), 4),
        "majority_label_by_source": {k: int(v) for k, v in majority.items()},
    }


def tfidf_in_dist(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    preds = _fit_tfidf(train).predict(test["text"])
    return {
        "accuracy": round(float(accuracy_score(test["label"], preds)), 4),
        "macro_f1": round(float(f1_score(test["label"], preds, average="macro")), 4),
    }


def tfidf_loso(pool: pd.DataFrame) -> dict:
    """Hold out each source in turn. Single-class sources report accuracy == recall of that class."""
    results = {}
    for src in sorted(pool["source"].unique()):
        held = pool[pool["source"] == src]
        rest = pool[pool["source"] != src]
        preds = _fit_tfidf(rest).predict(held["text"])
        results[src] = {
            "n": int(len(held)),
            "classes_present": sorted(int(c) for c in held["label"].unique()),
            "accuracy": round(float(accuracy_score(held["label"], preds)), 4),
        }
    return results


def main():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    pool = pd.concat([train, val, test], ignore_index=True)

    results = {
        "source_only": source_only(pd.concat([train, val]), test),
        "tfidf_in_dist": tfidf_in_dist(pd.concat([train, val]), test),
        "tfidf_loso": tfidf_loso(pool),
    }

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
