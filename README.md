# Prompt Injection Detector

A lightweight, real-time prompt injection detection guardrail that sits as middleware in front of any OpenAI-compatible LLM endpoint. Fine-tuned from Qwen2-1.5B using QLoRA for sequence classification.

## Architecture

```
Client Request
      │
      ▼
┌─────────────────────┐
│  FastAPI Middleware   │
│  ┌─────────────────┐ │
│  │ Injection        │ │──► logs/metrics ──► Prometheus ──► Grafana
│  │ Classifier       │ │
│  │ (Qwen2-1.5B      │ │
│  │  QLoRA, seq-cls) │ │
│  └─────────────────┘ │
│         │             │
│   score ≥ threshold?  │
│    ┌────┴────┐        │
│   Yes        No       │
│    │          │        │
│  Block/    Forward to  │
│  Flag      LLM endpoint│
└─────────────────────┘
                │
                ▼
        OpenAI-compatible
           LLM Endpoint
```

## Project Structure

```
├── configs/              # Training & dataset configs, Prometheus config
├── data/                 # Raw and processed datasets
├── eval/                 # Evaluation results, calibration curves, metrics
├── learning/             # Documentation notes on calibration, deployment, etc.
├── middleware/            # FastAPI middleware + classifier serving
│   ├── app.py            # FastAPI application with /chat/completions proxy
│   ├── classifier.py     # QLoRA-tuned Qwen2 classification model
│   ├── config.py         # Pydantic-based settings (.env)
│   └── example_client.py # Usage example
├── models/               # Fine-tuned checkpoint (LoRA adapters)
├── scripts/              # Data pipeline, training, evaluation scripts
│   ├── prepare_dataset.py
│   ├── train_qlora.py
│   ├── evaluate_baseline.py / evaluate_model.py
│   ├── calibrate.py
│   ├── build_adversarial_eval.py
│   └── analyze_dataset.py
└── src/
    ├── data/             # Dataset loaders, deduplication, balancing
    └── utils/            # Shared utilities (config loading)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -r middleware/requirements.txt
```

### 2. Prepare the dataset

```bash
python scripts/prepare_dataset.py --config configs/dataset_config.yaml
```

### 3. Train the model

```bash
python scripts/train_qlora.py --config configs/training_config.yaml
```

### 4. Run the middleware

```bash
# Configure via .env or environment variables
export LLM_ENDPOINT="https://api.openai.com/v1"
export LLM_API_KEY="sk-..."
export THRESHOLD=0.85
export MODE="soft_flag"

python -m middleware.app
```

Or use Docker Compose for the full monitoring stack:

```bash
docker compose up -d
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# Middleware: http://localhost:8080
```

## Usage

Point your LLM client at `http://localhost:8080/v1` instead of the default OpenAI endpoint:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

In `soft_flag` mode (default), detected injections are forwarded with an `X-Injection-Detected` header. In `hard_block` mode, a 403 is returned.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_ENDPOINT` | `https://api.openai.com/v1` | Upstream LLM endpoint |
| `LLM_API_KEY` | `""` | API key for upstream LLM |
| `THRESHOLD` | `0.85` | Confidence threshold for flagging |
| `MODE` | `soft_flag` | `soft_flag` or `hard_block` |
| `MODEL_PATH` | `models/qwen-injection-detector/best` | Path to LoRA checkpoint |
| `PORT` | `8080` | Middleware listen port |

## Training Data

The model is trained on a combination of public prompt injection datasets:

- `deepset/prompt-injections`
- `S-Labs/prompt-injection-dataset`
- `xTRam1/safe-guard-prompt-injection`
- `Lakera/gandalf_ignore_instructions`
- `HuggingFaceH4/no_robots` (benign examples)

A hand-authored adversarial eval set targeting obfuscation, encoding, translation, and multi-turn attacks is used for evaluation only.

## Monitoring

- **Prometheus** metrics: request count, blocked/flagged count, confidence histogram, latency histogram
- **Grafana** dashboards: injection attempt rates, confidence drift, latency percentiles
- **Metrics endpoint**: `GET /metrics`

## Evaluation Results

The eval directory contains metrics, confusion matrices, and calibration curves from training runs. Baseline comparisons against zero-shot classifiers are included.

## License

MIT
