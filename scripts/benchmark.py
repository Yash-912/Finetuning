#!/usr/bin/env python3
"""
Single-prompt latency and VRAM benchmark for one detector model directory.

Loads the model, runs 5 warm-up inferences, then times 200 single-prompt predictions over the
texts in data/eval_handwritten/eval.jsonl (cycled if fewer than 200). Reports p50/p95/p99 latency
in ms and peak torch.cuda.max_memory_allocated in MB when running on CUDA.

Usage:
    python scripts/benchmark.py models/qwen-injection-detector/best
    python scripts/benchmark.py models/qwen-injection-detector-v2/best --out eval/benchmark.json

With --out, the result is merged into that JSON file as one entry per model_dir (an existing
entry for the same model_dir is replaced).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from threshold_sweep import load_model, p_injection  # noqa: E402

EVAL = Path("data/eval_handwritten/eval.jsonl")
WARMUP = 5
N_TIMED = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="Adapter directory, e.g. models/qwen-injection-detector/best")
    ap.add_argument("--out", default=None, help="Merge the result into this JSON file (one entry per model_dir)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cfg = yaml.safe_load(Path("configs/training_config.yaml").read_text())
    base, max_len = cfg["model"]["base_model_name"], cfg["model"]["max_length"]

    texts = [json.loads(line)["text"] for line in EVAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    timed_texts = [texts[i % len(texts)] for i in range(N_TIMED)]

    on_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if on_cuda else "cpu"

    temp_path = model_dir / "temperature.pt"
    temperature = torch.load(str(temp_path)).item() if temp_path.exists() else 1.0

    tok, model, device = load_model(model_dir, base)
    weights_mb = None
    if on_cuda:
        torch.cuda.synchronize()
        weights_mb = torch.cuda.memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()

    for i in range(WARMUP):
        p_injection(tok, model, device, [texts[i % len(texts)]], temperature, max_len)

    latencies_ms = []
    for text in timed_texts:
        start = time.perf_counter()
        p_injection(tok, model, device, [text], temperature, max_len)
        if on_cuda:
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    lat = np.asarray(latencies_ms)
    entry = {
        "model_dir": str(model_dir),
        "device": device_name,
        "n_warmup": WARMUP,
        "n_timed": N_TIMED,
        "latency_ms": {
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
            "p99": round(float(np.percentile(lat, 99)), 2),
            "mean": round(float(lat.mean()), 2),
        },
        "peak_allocated_mb": None,
        "weights_allocated_mb": None,
    }
    if on_cuda:
        entry["weights_allocated_mb"] = round(weights_mb, 1)
        # Peak during the timed loop only (reset after load, before warm-up).
        entry["peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)

    print(json.dumps(entry, indent=2))

    if args.out:
        out_path = Path(args.out)
        existing = json.loads(out_path.read_text()) if out_path.exists() else []
        existing = [e for e in existing if e.get("model_dir") != entry["model_dir"]]
        existing.append(entry)
        out_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
