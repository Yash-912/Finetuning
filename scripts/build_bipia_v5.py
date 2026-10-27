#!/usr/bin/env python3
"""
Build the v5 training and validation sets with a proper train / dev / test split of BIPIA.

Split (all deterministic from --seed):
  test  BIPIA's own test split, untouched. Its contexts, attack prompts and attack categories never reach train or dev.
  dev   held out from BIPIA's train split, disjoint from train by attack CATEGORY, attack prompt and context.
        Used for validation, checkpoint choice, temperature fitting and cutoff choice, so the test split is never
        used for any decision.
  train the rest of BIPIA's train split.

Long contexts are cropped to a model-sized window instead of being cut at the start. A poisoned row keeps the
window around the inserted attack, a clean row keeps a random window, so the label always matches what the model
can see. (With plain truncation, an attack inserted at the end of a long context was cut off and the row was still
labelled "attack", which contradicted the clean copy of the same context.)

Outputs in --out-dir (default data/processed/v5):
    train.parquet       v2 training data + BIPIA train rows            columns: text, label, source
    val.parquet         data/processed/val.parquet + BIPIA dev rows    columns: text, label, source
    split_manifest.json categories, prompts and context hashes per split
    build_card.json     counts, crop statistics and leakage checks

Usage:
    python scripts/build_bipia_v5.py --bipia-dir path/to/BIPIA
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_bipia import POSITIONS, context_text, insert, read_jsonl  # noqa: E402

V2_TRAIN = Path("data/processed/train.parquet")
VAL = Path("data/processed/val.parquet")
WINDOW = 512
# task -> (train context file, train attack file, test context file, test attack file), relative to benchmark/
TASKS = {
    "email": ("email/train.jsonl", "text_attack_train.json", "email/test.jsonl", "text_attack_test.json"),
    "table": ("table/train.jsonl", "text_attack_train.json", "table/test.jsonl", "text_attack_test.json"),
    "code": ("code/train.jsonl", "code_attack_train.json", "code/test.jsonl", "code_attack_test.json"),
}


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def split_categories(categories: list[str], dev_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Hold out whole attack categories for dev. Always leaves at least one category on each side."""
    names = sorted(categories)
    random.Random(seed).shuffle(names)
    n_dev = min(max(1, round(len(names) * dev_fraction)), len(names) - 1)
    return sorted(names[n_dev:]), sorted(names[:n_dev])


def split_contexts(contexts: list[str], dev_fraction: float, seed: int, max_dev: int | None) -> tuple[list[str], list[str]]:
    """Hold out whole contexts for dev, identified by content hash so duplicates cannot straddle the split."""
    unique = sorted({sha1(c): c for c in contexts}.items())
    random.Random(seed).shuffle(unique)
    n_dev = min(max(1, round(len(unique) * dev_fraction)), len(unique) - 1)
    if max_dev:
        n_dev = min(n_dev, max_dev)
    return [c for _, c in unique[n_dev:]], [c for _, c in unique[:n_dev]]


def crop_to_window(tok, text: str, focus: str | None, rng: random.Random, window: int = WINDOW) -> tuple[str, bool]:
    """Return (text, was_cropped). Keeps the window of `window` tokens around `focus` (the inserted attack)
    or, if focus is None, a random window. Text that already fits is returned unchanged."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    if n <= window:
        return text, False
    if focus is not None and (loc := text.find(focus)) >= 0:
        starts = [o[0] for o in offsets]
        centre = min(max(bisect.bisect_right(starts, loc + len(focus) // 2) - 1, 0), n - 1)
        first = min(max(centre - window // 2, 0), n - window)
    else:
        first = rng.randint(0, n - window)
    return text[offsets[first][0] : offsets[first + window - 1][1]], True


def build_rows(
    tok, contexts: list[str], attacks_by_category: dict[str, list[str]], poisons_per_context: int,
    source: str, rng: random.Random, seed: int,
) -> tuple[list[dict], set[str], int]:
    """One clean row per context plus `poisons_per_context` poisoned rows, each cropped to a window."""
    rows, prompts_used, cropped = [], set(), 0
    categories = sorted(attacks_by_category)
    for ctx in contexts:
        text, was_cropped = crop_to_window(tok, ctx, None, rng)
        rows.append({"text": text, "label": 0, "source": source})
        cropped += was_cropped
        for category in rng.sample(categories, k=min(poisons_per_context, len(categories))):
            prompt = rng.choice(attacks_by_category[category])
            prompts_used.add(prompt)
            poisoned = insert(rng.choice(POSITIONS), ctx, prompt, seed)
            text, was_cropped = crop_to_window(tok, poisoned, prompt, rng)
            rows.append({"text": text, "label": 1, "source": source})
            cropped += was_cropped
    return rows, prompts_used, cropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bipia-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/v5"))
    ap.add_argument("--dev-fraction", type=float, default=0.3)
    ap.add_argument("--max-dev-contexts-per-task", type=int, default=60,
                    help="cap dev contexts per task so the 900-context table task does not dominate dev (0 = no cap)")
    ap.add_argument("--poisons-per-context", type=int, default=2, help="same as v4")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
    benchmark = args.bipia_dir / "benchmark"
    rng = random.Random(args.seed)

    test_context_hashes: set[str] = set()
    test_prompts: set[str] = set()
    test_categories: set[str] = set()
    for _, _, test_ctx_file, test_atk_file in TASKS.values():
        test_context_hashes.update(sha1(context_text(s)) for s in read_jsonl(benchmark / test_ctx_file))
        for category, prompts in json.loads((benchmark / test_atk_file).read_text(encoding="utf-8")).items():
            test_categories.add(category)
            test_prompts.update(prompts)

    # Categories are split once per attack file so email and table share the same text-attack split
    attack_files = {atk: json.loads((benchmark / atk).read_text(encoding="utf-8")) for _, atk, _, _ in TASKS.values()}
    # A category name that also appears in BIPIA's test split (e.g. "Language Translation") is a near-duplicate
    # attack type of the test set, so it is removed from train and dev.
    shared_with_test = sorted({c for attacks in attack_files.values() for c in attacks} & test_categories)
    attack_files = {
        atk: {c: prompts for c, prompts in attacks.items() if c not in test_categories}
        for atk, attacks in attack_files.items()
    }
    category_split = {}
    for name, attacks in attack_files.items():
        train_cats, dev_cats = split_categories(list(attacks), args.dev_fraction, args.seed)
        category_split[name] = {"train": train_cats, "dev": dev_cats}

    train_rows, dev_rows = [], []
    used = {"train": set(), "dev": set()}
    context_hashes = {"train": set(), "dev": set()}
    cropped = {"train": 0, "dev": 0}
    dropped_test_overlap = {}
    for task, (ctx_file, atk_file, _, _) in TASKS.items():
        contexts = [context_text(s) for s in read_jsonl(benchmark / ctx_file)]
        kept = [c for c in contexts if sha1(c) not in test_context_hashes]
        dropped_test_overlap[task] = len(contexts) - len(kept)
        train_ctx, dev_ctx = split_contexts(kept, args.dev_fraction, args.seed, args.max_dev_contexts_per_task or None)
        for split, ctxs, out in [("train", train_ctx, train_rows), ("dev", dev_ctx, dev_rows)]:
            attacks = {c: attack_files[atk_file][c] for c in category_split[atk_file][split]}
            rows, prompts, n_cropped = build_rows(
                tok, ctxs, attacks, args.poisons_per_context, f"bipia_{split}_{task}", rng, args.seed
            )
            out.extend(rows)
            used[split] |= prompts
            cropped[split] += n_cropped
            context_hashes[split] |= {sha1(c) for c in ctxs}

    train_cats = {c for v in category_split.values() for c in v["train"]}
    dev_cats = {c for v in category_split.values() for c in v["dev"]}
    leakage = {
        "train_dev_context_overlap": len(context_hashes["train"] & context_hashes["dev"]),
        "train_test_context_overlap": len(context_hashes["train"] & test_context_hashes),
        "dev_test_context_overlap": len(context_hashes["dev"] & test_context_hashes),
        "train_dev_prompt_overlap": len(used["train"] & used["dev"]),
        "train_test_prompt_overlap": len(used["train"] & test_prompts),
        "dev_test_prompt_overlap": len(used["dev"] & test_prompts),
        "train_dev_category_overlap": len(train_cats & dev_cats),
        "train_test_category_overlap": len((train_cats | dev_cats) & test_categories),
    }
    assert not any(leakage.values()), f"split leaks: {leakage}"

    bipia_train, bipia_dev = pd.DataFrame(train_rows), pd.DataFrame(dev_rows)
    v2_train = pd.read_parquet(V2_TRAIN)[["text", "label", "source"]]
    val = pd.read_parquet(VAL)[["text", "label", "source"]]
    train = pd.concat([v2_train, bipia_train], ignore_index=True).sample(frac=1.0, random_state=args.seed)
    val_all = pd.concat([val, bipia_dev], ignore_index=True).sample(frac=1.0, random_state=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.out_dir / "train.parquet", index=False)
    val_all.to_parquet(args.out_dir / "val.parquet", index=False)
    manifest = {
        "seed": args.seed,
        "attack_categories": category_split,
        "test_attack_categories": sorted(test_categories),
        "train_attack_prompts": sorted(used["train"]),
        "dev_attack_prompts": sorted(used["dev"]),
        "context_sha1": {s: sorted(h) for s, h in context_hashes.items()},
    }
    (args.out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    card = {
        "seed": args.seed,
        "window_tokens": WINDOW,
        "poisons_per_context": args.poisons_per_context,
        "v2_train_rows": len(v2_train),
        "bipia_train_rows": len(bipia_train),
        "bipia_train_injections": int(bipia_train["label"].sum()),
        "bipia_train_clean": int((bipia_train["label"] == 0).sum()),
        "bipia_dev_rows": len(bipia_dev),
        "bipia_dev_injections": int(bipia_dev["label"].sum()),
        "bipia_dev_clean": int((bipia_dev["label"] == 0).sum()),
        "rows_cropped_to_window": cropped,
        "train_total_rows": len(train),
        "val_total_rows": len(val_all),
        "contexts_dropped_as_test_overlap": dropped_test_overlap,
        "train_categories_dropped_for_sharing_a_name_with_test": shared_with_test,
        "train_attack_categories": len(train_cats),
        "dev_attack_categories": len(dev_cats),
        "leakage_checks": leakage,
        "test": "BIPIA test split, unchanged; to be reported once (see eval_bipia.py --final)",
    }
    (args.out_dir / "build_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
