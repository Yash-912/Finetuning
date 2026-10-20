#!/usr/bin/env python3
"""
Build the v3 training set: v2 training data plus BIPIA *train* split injections.

BIPIA's test split is kept out of training completely:
  - test contexts are never used for training (train-side contexts that also appear in the test split
    are dropped, so no test context is seen during training);
  - test attack prompts are never used for training. The test split's attack categories are already
    disjoint from the train split's, so the test split is also a category-level holdout.

Validation stays on data/processed/val.parquet (no BIPIA). Model selection never touches BIPIA test.

Outputs:
    data/processed/v3/train.parquet    columns: text, label, source
    data/processed/v3/build_card.json  counts and leakage checks

Usage:
    python scripts/build_bipia_v3.py --bipia-dir path/to/BIPIA
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_bipia import POSITIONS, TASKS, context_text, insert, read_jsonl  # noqa: E402

V2_TRAIN = Path("data/processed/train.parquet")
DEFAULT_MAX_CONTEXTS_PER_TASK = 150  # v3: BIPIA table has 900 train contexts; cap it so it does not dominate the mix


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bipia-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/v3"))
    ap.add_argument("--max-contexts-per-task", type=int, default=DEFAULT_MAX_CONTEXTS_PER_TASK,
                    help="0 keeps every train-side context")
    ap.add_argument("--poisons-per-context", type=int, default=0,
                    help="attack categories sampled per context (one prompt each); 0 means every category (v3)")
    args = ap.parse_args()
    max_contexts = args.max_contexts_per_task or None
    out_dir = args.out_dir

    benchmark = args.bipia_dir / "benchmark"
    rng = random.Random(args.seed)

    # Test-side material that must never reach training. TASKS already points at the test files.
    test_context_hashes = set()
    for ctx_file, _ in TASKS.values():
        for s in read_jsonl(benchmark / ctx_file):
            test_context_hashes.add(sha1(context_text(s)))
    test_prompts = set()
    for atk_file in ["text_attack_test.json", "code_attack_test.json"]:
        for prompts in json.loads((benchmark / atk_file).read_text(encoding="utf-8")).values():
            test_prompts.update(prompts)

    rows = []
    removed_overlap = Counter()
    train_attack_prompts = set()
    for task, (ctx_file, atk_file) in TASKS.items():
        train_ctx_file = ctx_file.replace("test.jsonl", "train.jsonl")
        train_atk_file = atk_file.replace("_test", "_train")
        contexts = read_jsonl(benchmark / train_ctx_file)
        attacks = json.loads((benchmark / train_atk_file).read_text(encoding="utf-8"))

        kept = []
        for s in contexts:
            ctx = context_text(s)
            if sha1(ctx) in test_context_hashes:
                removed_overlap[task] += 1
                continue
            kept.append(ctx)
        if max_contexts and len(kept) > max_contexts:
            kept = rng.sample(kept, max_contexts)

        for ctx in kept:
            rows.append({"text": ctx, "label": 0, "source": f"bipia_train_{task}"})
            categories = list(attacks.values())
            if args.poisons_per_context:
                categories = rng.sample(categories, k=min(args.poisons_per_context, len(categories)))
            for prompts in categories:
                prompt = rng.choice(prompts)
                train_attack_prompts.add(prompt)
                position = rng.choice(POSITIONS)
                rows.append({
                    "text": insert(position, ctx, prompt, args.seed),
                    "label": 1,
                    "source": f"bipia_train_{task}",
                })

    # Leakage checks: fail loudly rather than train on test material
    leaked_prompts = train_attack_prompts & test_prompts
    assert not leaked_prompts, f"{len(leaked_prompts)} test attack prompts appear in training"
    leaked_contexts = sum(sha1(r["text"]) in test_context_hashes for r in rows if r["label"] == 0)
    assert leaked_contexts == 0, f"{leaked_contexts} test contexts appear in training"

    bipia_df = pd.DataFrame(rows)
    v2_df = pd.read_parquet(V2_TRAIN)[["text", "label", "source"]]
    train_df = pd.concat([v2_df, bipia_df], ignore_index=True).sample(frac=1.0, random_state=args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", index=False)

    card = {
        "seed": args.seed,
        "v2_train_rows": len(v2_df),
        "bipia_rows": len(bipia_df),
        "bipia_injections": int(bipia_df["label"].sum()),
        "bipia_clean": int((bipia_df["label"] == 0).sum()),
        "total_rows": len(train_df),
        "total_label_counts": {str(k): int(v) for k, v in train_df["label"].value_counts().items()},
        "max_contexts_per_task": max_contexts,
        "poisons_per_context": args.poisons_per_context or "all categories",
        "train_contexts_dropped_as_test_overlap": dict(removed_overlap),
        "leakage_checks": {
            "test_attack_prompts_in_train": len(leaked_prompts),
            "test_contexts_in_train": leaked_contexts,
        },
        "validation": "data/processed/val.parquet (unchanged, no BIPIA)",
        "bipia_eval": "BIPIA test split, all attack categories (disjoint from train categories)",
    }
    (out_dir / "build_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
