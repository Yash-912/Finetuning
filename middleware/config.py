from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_endpoint: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    threshold: float = 0.85
    mode: str = "soft_flag"
    model_path: str = "models/qwen-injection-detector/best"
    max_length: int = 512
    long_input: str = "window"  # "window": read inputs longer than max_length in overlapping windows; "truncate": first window only
    window_stride: int = 256
    max_windows: int = 128  # 0 means no cap
    host: str = "0.0.0.0"
    port: int = 8080

    shadow_enabled: bool = False
    shadow_model_path: str = ""
    review_queue_path: str = "data/review_queue.db"
    review_near_threshold: float = 0.1

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
