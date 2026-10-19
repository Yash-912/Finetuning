"""Tests for the frozen hand-written eval helpers in scripts/eval_handwritten.py."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import eval_handwritten  # noqa: E402


def test_verify_frozen_passes_on_current_file():
    # Real frozen set and its recorded hash; read-only check.
    eval_handwritten.verify_frozen()


def test_verify_frozen_raises_when_content_changes(tmp_path, monkeypatch):
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_bytes(b'{"text": "hello", "label": 0}\n')
    hash_file = tmp_path / "eval.sha256"
    original_hash = hashlib.sha256(eval_file.read_bytes()).hexdigest()
    hash_file.write_text(f"{original_hash} *eval.jsonl\n")

    monkeypatch.setattr(eval_handwritten, "EVAL", eval_file)
    monkeypatch.setattr(eval_handwritten, "HASH_FILE", hash_file)

    # Unchanged content still passes.
    eval_handwritten.verify_frozen()

    # Any edit to the eval file must be refused.
    eval_file.write_bytes(b'{"text": "hello", "label": 1}\n')
    with pytest.raises(SystemExit):
        eval_handwritten.verify_frozen()


def test_per_category_splits_attack_and_benign_under_same_category():
    df = pd.DataFrame(
        [
            {"category": "multilingual", "label": 1},
            {"category": "multilingual", "label": 1},
            {"category": "multilingual", "label": 0},
            {"category": "multilingual", "label": 0},
            {"category": "multilingual", "label": 0},
            {"category": "code", "label": 0},
        ]
    )
    # Scores at threshold 0.85: attacks 0.9 (hit) and 0.5 (miss);
    # benign 0.95 (flagged), 0.1, 0.2; code benign 0.3.
    p = np.array([0.9, 0.5, 0.95, 0.1, 0.2, 0.3])

    out = eval_handwritten.per_category(df, p, threshold=0.85)

    assert set(out) == {"multilingual/attack", "multilingual/benign", "code/benign"}
    assert out["multilingual/attack"] == {"n": 2, "recall": 0.5, "missed": 1}
    assert out["multilingual/benign"] == {"n": 3, "fpr": 0.333, "flagged": 1}
    assert out["code/benign"] == {"n": 1, "fpr": 0.0, "flagged": 0}
