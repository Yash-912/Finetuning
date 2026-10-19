"""Regression: retrain must read the new model's folder from the training config, not assume production's."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from retrain import BEST_DIR, trained_best_dir  # noqa: E402


def _cfg(tmp_path, output_dir):
    p = tmp_path / "train.yaml"
    p.write_text(yaml.safe_dump({"training": {"output_dir": output_dir}}))
    return str(p)


def test_default_config_writes_to_production_folder(tmp_path):
    cfg = _cfg(tmp_path, "models/qwen-injection-detector")
    assert trained_best_dir(cfg).resolve() == BEST_DIR.resolve()


def test_custom_output_dir_is_respected(tmp_path):
    cfg = _cfg(tmp_path, "models/qwen-injection-detector-v2")
    trained = trained_best_dir(cfg)
    assert trained == Path("models/qwen-injection-detector-v2/best")
    assert trained.resolve() != BEST_DIR.resolve()


@pytest.mark.parametrize("out", ["models/x", "models/x/"])
def test_trailing_slash_does_not_change_result(tmp_path, out):
    assert trained_best_dir(_cfg(tmp_path, out)) == Path("models/x/best")
