"""Pure-logic tests for the eval scripts (no GPU, no model loading): the numbers they report must be right."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import compare_models as cm  # noqa: E402
import eval_indirect as ei  # noqa: E402


def test_wilson_matches_known_values():
    lo, hi = cm.wilson(0, 58)
    assert lo == 0.0 and hi == pytest.approx(0.0621, abs=1e-4)
    lo, hi = cm.wilson(50, 100)
    assert (lo, hi) == pytest.approx((0.4038, 0.5962), abs=1e-4)
    lo, hi = cm.wilson(0, 4)
    assert hi == pytest.approx(0.4899, abs=1e-4)
    assert all(np.isnan(cm.wilson(0, 0)))


def test_summarize_counts_recall_and_fpr_at_threshold():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.99, 0.90, 0.85, 0.40, 0.86, 0.10, 0.10, 0.10, 0.10, 0.10])
    out = cm.summarize(y, p, threshold=0.85)
    assert (out["tp"], out["n_attack"], out["recall"]) == (3, 4, 0.75)  # 0.85 counts: the cutoff is inclusive
    assert (out["fp"], out["n_benign"], out["fpr"]) == (1, 6, round(1 / 6, 4))
    assert 0.5 < out["auroc"] <= 1.0


def test_summarize_single_class_has_no_cross_class_metrics():
    out = cm.summarize(np.zeros(5, dtype=int), np.full(5, 0.1), threshold=0.85)
    assert "recall" not in out and "auroc" not in out
    assert out["fpr"] == 0.0


def test_ranking_score_is_mean_youden_j_over_ranked_sets_with_both_classes():
    results = {
        "test": {"n_attack": 10, "n_benign": 10, "tp": 9, "fp": 1, "recall": 0.9, "fpr": 0.1},
        "adversarial": {"n_attack": 10, "n_benign": 4, "tp": 10, "fp": 0, "recall": 1.0, "fpr": 0.0},
        "indirect_long": {"n_attack": 5, "n_benign": 5, "tp": 0, "fp": 5, "recall": 0.0, "fpr": 1.0},  # not ranked
        "benign_test": {"n_attack": 0, "n_benign": 20, "tp": 0, "fp": 0, "fpr": 0.0},  # benign only
    }
    out = cm.score(results)
    assert out["sets_ranked"] == ["adversarial", "test"]
    assert out["score"] == pytest.approx((0.8 + 1.0) / 2)
    assert out["pooled_fpr"] == pytest.approx((1 + 0 + 5 + 0) / (10 + 4 + 5 + 20), abs=1e-4)


def _write_bipia(dir_: Path, name: str, **fields):
    base = {"adapter_sha256": "aaa", "temperature": 2.7795, "threshold": 0.85, "long_input": "truncate",
            "overall": {"n_attack": 100, "n_benign": 10, "recall": 0.5, "fpr": 0.1}}
    (dir_ / name).write_text(json.dumps({**base, **fields}))


def test_bipia_result_is_matched_by_model_hash_not_by_folder(tmp_path):
    _write_bipia(tmp_path, "bipia_v1_results.json", adapter_sha256="v1hash", temperature=3.5)
    _write_bipia(tmp_path, "bipia_v4_results.json", adapter_sha256="v4hash")
    got = cm.load_bipia("v4hash", 2.7795, 0.85, "truncate", results_dir=tmp_path)
    assert got is not None and got["source"].endswith("bipia_v4_results.json")
    # A model with unknown weights never inherits another model's BIPIA numbers
    assert cm.load_bipia("someone-else", 2.7795, 0.85, "truncate", results_dir=tmp_path) is None


def test_bipia_result_requires_same_temperature_threshold_and_long_input_mode(tmp_path):
    _write_bipia(tmp_path, "bipia_v4_results.json")
    assert cm.load_bipia("aaa", 2.7795, 0.85, "truncate", results_dir=tmp_path) is not None
    assert cm.load_bipia("aaa", 3.0, 0.85, "truncate", results_dir=tmp_path) is None  # recalibrated
    assert cm.load_bipia("aaa", 2.7795, 0.90, "truncate", results_dir=tmp_path) is None
    assert cm.load_bipia("aaa", 2.7795, 0.85, "window", results_dir=tmp_path) is None


def test_bipia_result_without_long_input_field_counts_as_truncated(tmp_path):
    _write_bipia(tmp_path, "bipia_old_results.json")
    data = json.loads((tmp_path / "bipia_old_results.json").read_text())
    del data["long_input"]
    (tmp_path / "bipia_old_results.json").write_text(json.dumps(data))
    assert cm.load_bipia("aaa", 2.7795, 0.85, "truncate", results_dir=tmp_path) is not None


def test_indirect_output_names_do_not_overwrite_each_other():
    assert ei.output_path(Path("data/eval_indirect")) == Path("eval/indirect_comparison.json")
    assert ei.output_path(Path("data/eval_indirect_long")) == Path("eval/indirect_long_comparison.json")
    assert ei.output_path(Path("data/eval_indirect_long"), "window") == Path("eval/indirect_long_comparison_window.json")


def test_frozen_set_check_rejects_a_changed_file(tmp_path):
    import hashlib

    (tmp_path / "eval.jsonl").write_bytes(b'{"text": "a", "label": 0}\n')
    (tmp_path / "eval.sha256").write_text(hashlib.sha256((tmp_path / "eval.jsonl").read_bytes()).hexdigest() + " *eval.jsonl\n")
    assert ei.verify_frozen(tmp_path) == tmp_path / "eval.jsonl"
    (tmp_path / "eval.jsonl").write_bytes(b'{"text": "b", "label": 0}\n')
    with pytest.raises(SystemExit):
        ei.verify_frozen(tmp_path)
