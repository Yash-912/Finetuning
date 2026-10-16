import pandas as pd

from src.data.balancer import check_balance, stratified_split
from src.data.deduplicator import exact_deduplicate


def _frame(n_benign: int, n_inj: int) -> pd.DataFrame:
    rows = [{"text": f"b{i}", "label": 0, "source": "no_robots"} for i in range(n_benign)]
    rows += [{"text": f"i{i}", "label": 1, "source": "xtram1"} for i in range(n_inj)]
    return pd.DataFrame(rows)


def test_check_balance_downsamples_benign_to_ratio():
    out = check_balance(_frame(500, 100), target_ratio=2.0)
    assert (out["label"] == 1).sum() == 100
    assert (out["label"] == 0).sum() == 200


def test_check_balance_leaves_underrepresented_benign_alone():
    df = _frame(50, 100)
    assert len(check_balance(df, target_ratio=2.0)) == len(df)


def test_stratified_split_is_disjoint_and_complete():
    df = _frame(400, 200)
    cfg = {"train_ratio": 0.8, "val_ratio": 0.1, "stratify_by": "source", "random_seed": 42}
    train, val, test = stratified_split(df, cfg)
    assert len(train) + len(val) + len(test) == len(df)
    assert set(train.text).isdisjoint(val.text) and set(train.text).isdisjoint(test.text)
    assert set(val.text).isdisjoint(test.text)
    assert abs(len(train) / len(df) - 0.8) < 0.02


def test_exact_dedup_ignores_case_and_whitespace():
    df = pd.DataFrame(
        {
            "text": ["Hello world", "  hello WORLD ", "other"],
            "label": [0, 0, 0],
            "source": ["no_robots", "no_robots", "no_robots"],
        }
    )
    out = exact_deduplicate(df)
    assert len(out) == 2
