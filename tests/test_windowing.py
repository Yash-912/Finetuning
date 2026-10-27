"""Sliding-window scoring: coverage, the window cap, and that an injection past the first window is found."""
from __future__ import annotations

import numpy as np
import pytest

from middleware.windowing import score_windowed, window_spans

INJECTION_TOKEN = 999


def fake_score(batch):
    """Stand-in model: a window containing INJECTION_TOKEN scores 0.97, any other window 0.02."""
    return np.array([0.97 if INJECTION_TOKEN in ids else 0.02 for ids in batch])


def test_short_text_is_one_window():
    assert window_spans(10, window=512, stride=256) == ([(0, 10)], True)
    assert window_spans(512, window=512, stride=256) == ([(0, 512)], True)


def test_windows_cover_every_token_and_end_at_the_last_token():
    for n in (513, 700, 1024, 1500, 5000):
        spans, covered = window_spans(n, window=512, stride=256)
        assert covered
        assert spans[0][0] == 0 and spans[-1][1] == n
        assert all(e - s == 512 for s, e in spans)
        reached = set()
        for s, e in spans:
            reached.update(range(s, e))
        assert reached == set(range(n))


def test_cap_keeps_first_and_last_window_and_reports_partial_read():
    spans, covered = window_spans(100_000, window=512, stride=256, max_windows=10)
    assert not covered
    assert len(spans) == 10
    assert spans[0][0] == 0 and spans[-1][1] == 100_000


def test_zero_cap_means_unlimited():
    spans, covered = window_spans(20_000, window=512, stride=256, max_windows=0)
    assert covered and len(spans) > 70


def test_invalid_stride_is_rejected():
    with pytest.raises(ValueError):
        window_spans(1000, window=512, stride=513)


@pytest.mark.parametrize("position", [0, 400, 511, 600, 1500, 2999])
def test_injection_anywhere_is_found_but_first_window_alone_misses_late_ones(position):
    ids = [1] * 3000
    ids[position] = INJECTION_TOKEN
    windowed, _ = score_windowed([ids], fake_score, window=512, stride=256)
    truncated, _ = score_windowed([ids[:512]], fake_score, window=512, stride=256)
    assert windowed[0] == pytest.approx(0.97)
    assert truncated[0] == pytest.approx(0.97 if position < 512 else 0.02)


def test_scores_are_per_text_in_input_order():
    clean, dirty, short = [1] * 2000, [1] * 2000, [1] * 5
    dirty[1800] = INJECTION_TOKEN
    scores, partly_read = score_windowed([clean, dirty, short], fake_score, batch_size=4)
    assert list(np.round(scores, 2)) == [0.02, 0.97, 0.02]
    assert partly_read == []


def test_single_window_text_scores_like_a_direct_call():
    ids = [1] * 40
    ids[7] = INJECTION_TOKEN
    scores, _ = score_windowed([ids], fake_score)
    assert scores[0] == fake_score([ids])[0]


def test_capped_text_is_reported_as_partly_read():
    scores, partly_read = score_windowed([[1] * 50_000, [1] * 10], fake_score, max_windows=5)
    assert partly_read == [0]
    assert len(scores) == 2
