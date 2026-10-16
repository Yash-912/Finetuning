import pandas as pd
import pytest

from src.data.loso import make_loso_split


@pytest.fixture
def pool():
    rows = []
    for src, label, n in [("a", 0, 300), ("b", 1, 200), ("c", 0, 100), ("c", 1, 100)]:
        rows += [{"text": f"{src}{label}-{i}", "label": label, "source": src} for i in range(n)]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("held", ["a", "b", "c"])
def test_held_out_source_never_leaks(pool, held):
    train, val, heldout = make_loso_split(pool, held)
    assert set(heldout["source"]) == {held}
    assert held not in set(train["source"]) | set(val["source"])
    assert len(heldout) == (pool["source"] == held).sum()
    assert set(train["text"]).isdisjoint(val["text"])


def test_train_is_subsampled_and_label_stratified(pool):
    train, _, _ = make_loso_split(pool, "a", max_train_samples=100)
    assert len(train) == 100
    assert set(train["label"]) == {0, 1}


def test_split_is_deterministic(pool):
    t1, _, _ = make_loso_split(pool, "b", max_train_samples=80)
    t2, _, _ = make_loso_split(pool, "b", max_train_samples=80)
    assert t1["text"].tolist() == t2["text"].tolist()


def test_unknown_source_raises(pool):
    with pytest.raises(ValueError):
        make_loso_split(pool, "zzz")
