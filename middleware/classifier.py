from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from middleware.windowing import DEFAULT_MAX_WINDOWS, DEFAULT_STRIDE, score_windowed

logger = logging.getLogger("classifier")


class InjectionClassifier:
    """One instance per adapter directory; production and shadow must be distinct objects."""

    def __init__(
        self,
        model_path: str = "models/qwen-injection-detector/best",
        max_length: int = 512,
        long_input: str = "window",
        window_stride: int = DEFAULT_STRIDE,
        max_windows: int = DEFAULT_MAX_WINDOWS,
    ):
        if long_input not in ("window", "truncate"):
            raise ValueError("long_input must be 'window' or 'truncate'")
        self._ready = False
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.long_input = long_input
        self.window_stride = window_stride
        self.max_windows = max_windows
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        self._warmup()
        self._ready = True

    def _load_model(self):
        logger.info("Loading tokenizer...")
        base_name = "Qwen/Qwen2-1.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(base_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading 4-bit quantized model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            base_name,
            num_labels=2,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        model.config.pad_token_id = self.tokenizer.pad_token_id

        logger.info("Loading LoRA adapters from: %s", self.model_path)
        model = PeftModel.from_pretrained(model, str(self.model_path))
        self.model = model.to(self.device)
        self.model.eval()

        temperature_path = self.model_path / "temperature.pt"
        if temperature_path.exists():
            self.temperature = torch.load(str(temperature_path)).item()
            logger.info("Loaded temperature: %.4f", self.temperature)
        else:
            self.temperature = 1.0
            logger.warning("No temperature file, using T=1.0 (uncalibrated)")

        total = sum(p.numel() for p in self.model.parameters())
        logger.info("Model loaded: %s parameters on %s", f"{total:,}", self.device)

    def _warmup(self):
        logger.info("Running warm-up...")
        dummy = "Warm-up inference."
        for _ in range(3):
            self.predict(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        logger.info("Warm-up complete.")

    @torch.no_grad()
    def _score_batch(self, batch_ids: list[list[int]]) -> np.ndarray:
        batch = self.tokenizer.pad({"input_ids": batch_ids}, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits = self.model(**batch).logits.float()
        return torch.softmax(logits / self.temperature, dim=-1)[:, 1].cpu().numpy()

    def predict(self, text: str) -> tuple[int, float]:
        """Return (label, confidence in that label).

        Text longer than max_length tokens is read in overlapping windows and scored by its highest window,
        so an injection anywhere in the text is seen. long_input="truncate" restores the old first-window-only read.
        """
        ids = self.tokenizer(text)["input_ids"] or [self.tokenizer.pad_token_id]
        if self.long_input == "truncate":
            ids = ids[: self.max_length]
        scores, partly_read = score_windowed(
            [ids], self._score_batch, window=self.max_length, stride=self.window_stride, max_windows=self.max_windows
        )
        if partly_read:
            logger.warning("Input of %d tokens exceeds the %d-window cap; only part of it was read", len(ids), self.max_windows)
        p_injection = float(scores[0])
        label = int(p_injection >= 0.5)
        return label, p_injection if label else 1.0 - p_injection

    @property
    def is_ready(self) -> bool:
        return self._ready
