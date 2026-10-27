"""Sliding-window scoring so inputs longer than the model's 512-token window are fully read.

A text is split into overlapping windows of at most `window` tokens. Each window is scored on its own and the
text's injection probability is the highest window score, so an injection anywhere in the text can raise the
alarm. Texts that fit in one window are scored exactly as before.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

DEFAULT_WINDOW = 512
DEFAULT_STRIDE = 256
DEFAULT_MAX_WINDOWS = 128  # 128 windows at stride 256 reads about 32k tokens; 0 means no cap


def window_spans(
    n_tokens: int, window: int = DEFAULT_WINDOW, stride: int = DEFAULT_STRIDE, max_windows: int = 0
) -> tuple[list[tuple[int, int]], bool]:
    """Return ([start, end) token spans, fully_covered).

    The last window always ends at the last token. If more than `max_windows` windows are needed, evenly spaced
    windows are kept (first and last always included) and fully_covered is False.
    """
    if window <= 0 or stride <= 0 or stride > window:
        raise ValueError("need 0 < stride <= window")
    if n_tokens <= window:
        return [(0, n_tokens)], True
    starts = list(range(0, n_tokens - window + 1, stride))
    if starts[-1] != n_tokens - window:
        starts.append(n_tokens - window)
    covered = True
    if max_windows and len(starts) > max_windows:
        keep = sorted({int(round(i)) for i in np.linspace(0, len(starts) - 1, max_windows)})
        starts = [starts[i] for i in keep]
        covered = False
    return [(s, s + window) for s in starts], covered


def score_windowed(
    token_ids: list[list[int]],
    score_fn: Callable[[list[list[int]]], np.ndarray],
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    max_windows: int = 0,
    batch_size: int = 16,
) -> tuple[np.ndarray, list[int]]:
    """Score each token-id list with its highest window score.

    score_fn maps a batch of token-id lists (each at most `window` long) to injection probabilities.
    Returns (scores, indices of texts that exceeded the window cap and were only partly read).
    """
    pieces: list[list[int]] = []
    owner: list[int] = []
    partly_read: list[int] = []
    for i, ids in enumerate(token_ids):
        spans, covered = window_spans(len(ids), window, stride, max_windows)
        if not covered:
            partly_read.append(i)
        for s, e in spans:
            pieces.append(ids[s:e])
            owner.append(i)

    order = np.argsort([len(p) for p in pieces], kind="stable")
    piece_scores = np.zeros(len(pieces))
    for b in range(0, len(pieces), batch_size):
        idx = order[b : b + batch_size]
        piece_scores[idx] = score_fn([pieces[j] for j in idx])

    scores = np.zeros(len(token_ids))
    np.maximum.at(scores, np.asarray(owner, dtype=int), piece_scores)
    return scores, partly_read
