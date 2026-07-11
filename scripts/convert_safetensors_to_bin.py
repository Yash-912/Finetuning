#!/usr/bin/env python3
"""
Convert the cached safetensors file to PyTorch .bin format.
Run ONCE to avoid the Windows page file issue with memory-mapped loading.

Usage: venv\Scripts\python scripts\convert_safetensors_to_bin.py
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import torch

SNAP = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2-1.5B" / "snapshots" / "8a16abf2848eda07cc5253dec660bf1ce007ad7a"
SAFETENSORS_PATH = SNAP / "model.safetensors"
BIN_PATH = SNAP / "pytorch_model.bin"


def main():
    print(f"Reading safetensors header from: {SAFETENSORS_PATH}")
    file_size = os.path.getsize(SAFETENSORS_PATH)
    print(f"File size: {file_size / 1e9:.2f} GB")

    with open(SAFETENSORS_PATH, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_len).decode("utf-8")
        header = json.loads(header_json)

    tensor_keys = [k for k in header if k != "__metadata__"]
    n_tensors = len(tensor_keys)
    print(f"Found {n_tensors} tensors in safetensors file")

    dtype_map = {
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }

    state_dict = {}
    with open(SAFETENSORS_PATH, "rb") as f:
        for i, key in enumerate(tensor_keys):
            info = header[key]
            dtype_str = info["dtype"]
            dtype = dtype_map.get(dtype_str, torch.float32)
            shape = tuple(info["shape"])
            offset_start, offset_end = info["data_offsets"]
            n_bytes = offset_end - offset_start

            f.seek(8 + header_len + offset_start)
            tensor_bytes = f.read(n_bytes)

            tensor = torch.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
            state_dict[key] = tensor

            if (i + 1) % 50 == 0:
                print(f"  Loaded {i + 1}/{n_tensors} tensors ({tensor.shape})")

    print(f"Saving to: {BIN_PATH}")
    torch.save(state_dict, str(BIN_PATH))
    print(f"Done. Saved {len(state_dict)} tensors to {BIN_PATH}")


if __name__ == "__main__":
    main()
