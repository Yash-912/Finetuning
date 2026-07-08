from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger("classifier")


class InjectionClassifier:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: str = "models/qwen-injection-detector/best", max_length: int = 512):
        if self._initialized:
            return
        self._initialized = True
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        self._warmup()

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
        torch.cuda.synchronize()
        logger.info("Warm-up complete.")

    @torch.no_grad()
    def predict(self, text: str) -> tuple[int, float]:
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        logits = logits.float()
        probs = torch.softmax(logits / self.temperature, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, predicted_class].item()
        return predicted_class, confidence

    @property
    def is_ready(self) -> bool:
        return self._initialized
