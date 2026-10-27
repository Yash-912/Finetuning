"""BIPIA train/dev split and window cropping: no leakage between splits, and labels match what the model can see."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_bipia_v5 as b  # noqa: E402


class WordTokenizer:
    """Tokenizer stand-in: one token per whitespace-separated word, with character offsets."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offsets, pos = [], 0
        for word in text.split(" "):
            offsets.append((pos, pos + len(word)))
            pos += len(word) + 1
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def test_category_split_is_disjoint_complete_and_deterministic():
    cats = [f"cat{i}" for i in range(15)]
    train, dev = b.split_categories(cats, 0.3, seed=1)
    assert set(train).isdisjoint(dev) and set(train) | set(dev) == set(cats)
    assert len(dev) == 4  # round(15 * 0.3)
    assert b.split_categories(cats, 0.3, seed=1) == (train, dev)
    assert b.split_categories(cats, 0.3, seed=2) != (train, dev)


def test_category_split_never_empties_either_side():
    train, dev = b.split_categories(["a", "b"], 0.99, seed=0)
    assert len(train) == 1 and len(dev) == 1
    train, dev = b.split_categories(["a", "b", "c"], 0.0, seed=0)
    assert len(dev) == 1


def test_context_split_is_disjoint_and_duplicates_cannot_straddle_it():
    contexts = [f"context number {i}" for i in range(50)] + ["context number 3"] * 5
    train, dev = b.split_contexts(contexts, 0.3, seed=1, max_dev=None)
    assert set(train).isdisjoint(dev)
    assert len(train) + len(dev) == 50  # duplicates collapsed to one entry each


def test_dev_context_cap():
    contexts = [f"c{i}" for i in range(1000)]
    _, dev = b.split_contexts(contexts, 0.3, seed=1, max_dev=60)
    assert len(dev) == 60


def test_short_text_is_not_cropped():
    text = "one two three"
    assert b.crop_to_window(WordTokenizer(), text, "two", random.Random(0), window=10) == (text, False)


@pytest.mark.parametrize("where", ["start", "middle", "end"])
def test_cropped_poisoned_text_always_keeps_the_attack(where):
    attack = "ATTACK_PHRASE_HERE"
    words = [f"w{i}" for i in range(400)]
    position = {"start": 0, "middle": 200, "end": 400}[where]
    text = " ".join(words[:position] + [attack] + words[position:])
    cropped, was_cropped = b.crop_to_window(WordTokenizer(), text, attack, random.Random(0), window=50)
    assert was_cropped
    assert attack in cropped
    assert len(cropped.split(" ")) == 50


def test_cropped_clean_text_is_a_window_of_the_original():
    text = " ".join(f"w{i}" for i in range(300))
    cropped, was_cropped = b.crop_to_window(WordTokenizer(), text, None, random.Random(3), window=40)
    assert was_cropped and cropped in text and len(cropped.split(" ")) == 40


def test_build_rows_labels_and_counts():
    ctxs = [" ".join(f"w{i}_{j}" for i in range(120)) for j in range(6)]
    attacks = {"A": ["do bad thing a"], "B": ["do bad thing b"], "C": ["do bad thing c"]}
    rows, prompts, cropped = b.build_rows(
        WordTokenizer(), ctxs, attacks, poisons_per_context=2, source="bipia_train_email",
        rng=random.Random(0), seed=7,
    )
    assert len(rows) == 6 * 3  # one clean + two poisoned per context
    assert sum(r["label"] for r in rows) == 12
    assert prompts <= {"do bad thing a", "do bad thing b", "do bad thing c"}
    assert all(r["source"] == "bipia_train_email" for r in rows)
    assert cropped == 0  # 120 words fit in the default 512-token window
