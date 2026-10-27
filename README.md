# Prompt Injection Detector

An end-to-end ML system that detects prompt injection attacks in LLM traffic. It fine-tunes **Qwen2-1.5B with QLoRA** as a binary sequence classifier (benign vs. injection), calibrates its confidence with temperature scaling, and serves it as a **FastAPI reverse proxy** that screens every message before forwarding traffic to any OpenAI-compatible LLM endpoint. The system ships with Prometheus/Grafana monitoring, shadow-mode evaluation, a human review queue, and an automated retraining loop.

## Status

| | |
|---|---|
| **Production model** | **v4** (`models/qwen-injection-detector/best`). Four versions (v1 to v4) are compared in [Model versions](#model-versions-and-the-production-choice) |
| **Direct injection** | Solid: 94.8% of attacks caught at 0.3% false alarms on the in-distribution test split; 90.7% on 47 hand-written attacks and 92.6% on a 112-prompt set, with no false alarms on the hand-written benign prompts |
| **Indirect injection** (instructions hidden in documents) | **Weak.** 86.6% on BIPIA, but only 27.5% and 15.0% on two hand-written indirect sets. Documented limitation |
| **Long inputs** | The proxy reads text longer than 512 tokens in overlapping windows. Windowed results for v3 and v4 are **not measured yet** |
| **CI** | Green on `main`: lint, 71 unit tests, Docker build |
| **Prepared, not done** | v5 training (cleaner BIPIA split), CPU Docker image, real-LLM proxy demo, Hugging Face publishing, PINT benchmark |

## How It Works

```
                 ┌──────────────────────────────────────────────────────┐
                 │                     TRAINING PIPELINE                │
                 │                                                      │
  HuggingFace ──►│  loader.py ──► deduplicator.py ──► balancer.py       │
  datasets (5)   │                                      │               │
                 │                          stratified_split (80/10/10) │
                 │                                            │         │
                 │              train_qlora.py ◄──────────────┘         │
                 │  Qwen2-1.5B + 4-bit NF4 + LoRA (SEQ_CLS)             │
                 │                        │                             │
                 │                   calibrate.py                       │
                 │            temperature scaling on val NLL            │
                 └────────────────────────┼─────────────────────────────┘
                                          ▼
                    models/qwen-injection-detector/best
                    (LoRA adapter + tokenizer + temperature.pt)
                                          │
                 ┌────────────────────────▼─────────────────────────────┐
                 │                  SERVING PIPELINE                    │
                 │                                                      │
  Client ───────►│  FastAPI :8080  /chat/completions                    │
  (OpenAI SDK)   │        │                                             │
                 │        ├── InjectionClassifier.predict()             │
                 │        │   (4-bit Qwen2 + LoRA, 512-token windows)   │
                 │        │                                             │
                 │        ├── label=1 & conf ≥ threshold?               │
                 │        │     hard_block → 403 JSON error             │
                 │        │     soft_flag  → X-Injection-Detected hdr   │
                 │        │                                             │
                 │        ├── shadow classifier (optional, parallel)    │
                 │        └── ReviewQueue (SQLite WAL) logs everything  │
                 │                                                      │
                 │  /health   → readiness + mode + threshold            │
                 │  /metrics  → Prometheus exposition                   │
                 └────────────────────────┬─────────────────────────────┘
                                          ▼
                            Upstream LLM API (OpenAI-compatible)
```

## Repository Layout

| Path | Purpose |
|---|---|
| `configs/dataset_config.yaml` | Data source registry, dedup/balance/split parameters |
| `configs/training_config.yaml` | Base model, quantization, LoRA, and Trainer hyperparameters (v1) |
| `configs/training_config_v2.yaml` to `_v5.yaml` | Same recipe with each version's data paths and output folder |
| `configs/prometheus.yml` | Prometheus scrape config (5s interval against middleware `:8080/metrics`) |
| `configs/grafana/` | Provisioned Grafana datasource and "Injection Detector" dashboard |
| `docker-compose.yml` | Prometheus (`:9090`) + Grafana (`:3000`) containers |
| `src/data/loader.py` | Per-source HuggingFace dataset loaders (normalize to `text`, `label`, `source`) |
| `src/data/deduplicator.py` | Exact dedup + embedding-based near-dedup (cosine similarity) |
| `src/data/balancer.py` | Benign:injection ratio enforcement and stratified train/val/test split |
| `src/utils/config.py` | YAML config loader |
| `scripts/prepare_dataset.py` | Full data pipeline: load → dedup → balance → split → parquet + dataset card |
| `scripts/train_qlora.py` | QLoRA fine-tuning of Qwen2-1.5B for sequence classification |
| `scripts/calibrate.py` | Temperature scaling via grid search + bounded NLL minimization (`--val-path` to calibrate on another validation set) |
| `scripts/evaluate_baseline.py` | Zero-shot evaluation of `deepset/deberta-v3-base-injection` on the test set |
| `scripts/evaluate_model.py` | Test-set evaluation of the fine-tuned model + baseline comparison + confusion matrix plot |
| `scripts/shortcut_baseline.py` | Source-only and TF-IDF baselines + leave-one-source-out for TF-IDF (CPU, seconds) |
| `scripts/eval_loso.py` | Leave-one-source-out training/eval for the Qwen QLoRA model (resumable, GPU) |
| `scripts/build_adversarial_eval.py` | 47 hand-authored adversarial examples across 12 attack categories |
| `scripts/eval_adversarial.py` | Replays the adversarial set through the live middleware HTTP API |
| `middleware/app.py` | FastAPI reverse proxy with hard-block/soft-flag modes and Prometheus metrics |
| `middleware/classifier.py` | `InjectionClassifier` (4-bit base + LoRA adapter + temperature; reads long input in windows) |
| `middleware/windowing.py` | Sliding-window splitting and max-pool scoring for inputs longer than 512 tokens |
| `middleware/config.py` | Pydantic settings loaded from environment / `.env` |
| `middleware/example_client.py` | One-line OpenAI SDK integration demo |
| `scripts/review_queue.py` | SQLite-backed human labeling queue + interactive CLI |
| `scripts/compare_models.py` | Scores several model versions on every eval set with one protocol, 95% intervals, and a fixed ranking rule |
| `scripts/eval_bipia.py` | Builds poisoned contexts from BIPIA's test split and scores a model on them (`--long-input window` to read whole texts) |
| `scripts/eval_indirect.py` | Scores models on the frozen indirect-injection sets |
| `scripts/eval_handwritten.py` | Scores models on the frozen 112-prompt hand-written set |
| `scripts/threshold_sweep.py` | Shared scoring code (`p_injection`, `adapter_sha256`) and precision/recall/FPR at each cutoff |
| `scripts/build_bipia_v3.py`, `build_bipia_v5.py` | Training-set builders: v3/v4 (BIPIA train split) and v5 (train/dev/test split, long contexts cropped around the attack) |
| `scripts/benchmark.py` | Single-prompt latency and GPU-memory benchmark |
| `scripts/smoke_middleware.py` | End-to-end check of the real proxy and model against a fake upstream (`--mode hard_block`/`soft_flag`) |
| `scripts/retrain.py` | Continuous retraining: merge labels → retrain → recalibrate → promote to shadow |
| `tests/` | pytest suite, 71 tests (fake classifier + mock upstream; CPU-only) |
| `docs/` | PRD and learning notes |
| `Dockerfile` | GPU (CUDA) image for the detector proxy. No CPU image yet |
| `.gitattributes` | Keeps LF line endings on the frozen eval files so their checksums hold on every OS |
| `eval/` | Generated metrics JSON, model comparison, calibration curves, confusion matrices, smoke-test results |
| `data/eval_handwritten/`, `data/eval_indirect/`, `data/eval_indirect_long/` | Frozen eval sets, each with an `eval.sha256` checked before scoring |
| `data/processed/` | Train/val/test/adversarial parquet files + dataset card |

## Dataset Pipeline

**Sources** (registered in `configs/dataset_config.yaml`, all pulled from HuggingFace):

| Source | Role |
|---|---|
| `deepset/prompt-injections` | Mixed benign + injection |
| `S-Labs/prompt-injection-dataset` | Mixed benign + injection |
| `xTRam1/safe-guard-prompt-injection` | Injection-only (loader keeps `label == 1` rows) |
| `Lakera/gandalf_ignore_instructions` | Injection-only, subsampled to 350 |
| `HuggingFaceH4/no_robots` | Benign-only (first user turn of each conversation) |
| `hackaprompt/hackaprompt-dataset` | Injection-only, disabled in config; filters to successful submissions when enabled |

**Processing**:
1. Each source is normalized to `(text, label, source)` — `label=0` benign, `label=1` injection.
2. **Exact dedup**: lowercase/stripped text, duplicates resolved by source priority order.
3. **Near-dedup**: `all-MiniLM-L6-v2` sentence embeddings, cosine similarity ≥ **0.92** flagged; lower-priority source row dropped.
4. **Balancing**: benign downsampled to a **2:1 benign:injection** target ratio.
5. **Split**: stratified by source into **80% / 10% / 10%** train/val/test (seed 42).
6. Outputs `train.parquet`, `val.parquet`, `test.parquet`, plus a `dataset_card.json` capturing per-source counts before dedup and final class balance. An empty adversarial-eval template is created if none exists.

Final processed corpus (v1): **21,963 examples** (14,642 benign / 7,321 injection) → 17,570 train / 2,196 val / 2,197 test.

Later versions add data on top of this: v2 adds hand-authored benign prompts (code, multilingual, hard negatives, everyday text); v3 and v4 add rows built from the BIPIA indirect-injection benchmark. The v4 training set has **22,137 rows** (14,315 benign / 7,822 injection). The validation and test splits stay free of BIPIA.

## Training

`scripts/train_qlora.py` trains **Qwen/Qwen2-1.5B** as a 2-label sequence classifier:

- **Quantization**: 4-bit NF4 with double quantization, fp16 compute (`bitsandbytes`)
- **LoRA**: rank 8, alpha 16, dropout 0.1 targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`; classifier head trained fully (`SEQ_CLS` task type)
- **Optimization**: 3 epochs, effective batch size 32 (8 × grad-accum 4), lr 2e-4, cosine schedule, 3% warmup, weight decay 0.01, fp16, gradient checkpointing
- **Selection**: best checkpoint by validation **macro F1**, saved to the config's `output_dir`
- **Versions**: `configs/training_config_v4.yaml` (production) uses the same hyperparameters with the v4 data, 2,076 steps, roughly 6 to 8 hours on the RTX 4050 Laptop GPU. The 512-token input limit applies to training: longer texts were cut at the start

## Calibration

Neural nets are overconfident, and this model was no exception. `scripts/calibrate.py`:

1. Collects validation logits once.
2. Grid-searches temperature over `{0.5 … 5.0}` by negative log-likelihood, then refines with bounded scalar minimization (`scipy.optimize.minimize_scalar`).
3. Saves the scalar as `models/qwen-injection-detector/best/temperature.pt`; inference divides logits by T before softmax.

Result (v1): optimal **T = 3.51**, ECE reduced **65.9%** (0.0058 → 0.0020), NLL 0.064 → 0.027. Plots before/after reliability diagrams to `eval/calibration_curve.png`.

Production v4: **T = 2.7795**, ECE 0.0075 → 0.0032, NLL 0.0601 → 0.0321 (`eval/calibration_metrics_v4.json`). It was fitted on the validation set, which has no indirect-injection samples, so calibration on documents is unverified. `calibrate.py --val-path data/processed/v5/val.parquet` fits it on a validation set that includes indirect samples (used by the prepared v5 recipe).

## Results

These first tables are the original v1 results. Production is now v4; see "Model versions and the production choice" below for all four versions.

Test set (n = 2,197), from `eval/comparison.json`:

| Metric | Baseline (`deepset/deberta-v3-base-injection`) | Fine-tuned Qwen2-1.5B QLoRA | Δ |
|---|---|---|---|
| Accuracy | 0.6636 | **0.9873** | ▲ 0.3237 |
| Macro F1 | 0.6623 | **0.9857** | ▲ 0.3234 |
| ROC-AUC | 0.7643 | **0.9979** | — |
| Benign recall | 0.5473 | **0.9918** | ▲ 0.4445 |
| Injection recall | 0.8942 | **0.9783** | ▲ 0.0841 |
| Injection F1 | 0.6407 | **0.9810** | ▲ 0.3403 |

The baseline collapses on out-of-distribution benign data (26.96% accuracy on the `no_robots` source), which is exactly the failure mode fine-tuning fixes.

### Generalization: how much of 98.7% is real?

The corpus is source-skewed: `no_robots` is all benign, `xtram1` and `gandalf` are all injection, and only `s_labs` and `deepset` have both classes. So I measured how far dataset style alone goes (`scripts/shortcut_baseline.py`) and how the model does on a source it never trained on (`scripts/eval_loso.py`):

| Check | Result |
|---|---|
| Predict from source label only, no text | 77.3% accuracy |
| TF-IDF + logistic regression, standard split | 96.9% (Qwen: 98.7%) |

| Held-out source | Qwen QLoRA | TF-IDF |
|---|---|---|
| gandalf (injection only) | 98.5% | 94.1% |
| s_labs | 88.1% (AUC 0.952) | 78.5% |
| deepset | 81.5% (AUC 0.917) | 74.8% |
| xtram1 (injection only) | 80.3% | 73.0% |
| no_robots (benign only) | 56.9% | 60.5% |

Takeaways: the model generalizes better than a bag-of-words baseline (so it learned more than source style), but 98.7% is an in-distribution number. Folds trained on 4,000 samples for 1 epoch, uncalibrated, so these are lower bounds. The weakest case is benign text unlike the training mix: without `no_robots`, 43% of ordinary instructions are flagged. Next step: more diverse benign data.

### Model versions and the production choice

Production is **v4** (`models/qwen-injection-detector/best`, copied from `models/qwen-injection-detector-v4/best` on 2026-10-26; the previous v1 adapter is in git history). Versions:

- **v1**: original QLoRA fine-tune on the five-source corpus.
- **v2**: v1 plus hand-authored benign prompts (code, multilingual, hard negatives, everyday text).
- **v3**: v2 plus the BIPIA train split (email, table and code indirect injections), 3,335 injected vs 239 clean.
- **v4**: v3 rebalanced to 1,978 injected vs 989 clean BIPIA rows (`configs/training_config_v4.yaml`, `data/processed/v4/build_card.json`).

All four were scored the same way by `scripts/compare_models.py` at threshold 0.85, each with its own learned temperature (v1 3.510, v2 3.087, v3 2.635, v4 2.780). Raw output: `eval/model_comparison.json`. Counts are in brackets because most sets are small.

| Set | Metric | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| Test split (in-distribution) | Attack recall | 94.8% (699/737) | 95.1% (701/737) | 96.2% (709/737) | 94.8% (699/737) |
| | Benign FPR | 0.4% (6/1460) | 0.3% (5/1460) | 0.3% (5/1460) | 0.3% (5/1460) |
| Hand-written direct attacks, 47 items | Attack recall | 93.0% (40/43) | 86.1% (37/43) | 83.7% (36/43) | 90.7% (39/43) |
| | Benign FPR | 25.0% (1/4) | 0.0% (0/4) | 0.0% (0/4) | 0.0% (0/4) |
| Hand-written set, 112 prompts | Attack recall | 96.3% (52/54) | 96.3% (52/54) | 92.6% (50/54) | 92.6% (50/54) |
| | Benign FPR | 6.9% (4/58) | 0.0% (0/58) | 0.0% (0/58) | 0.0% (0/58) |
| BIPIA test split (indirect: email, table, code) | Attack recall | 14.7% (1214/8250) | 9.8% (811/8250) | 93.4% (7703/8250) | 86.6% (7144/8250) |
| | Benign FPR | 1.5% (3/200) | 1.5% (3/200) | 13.0% (26/200) | 0.5% (1/200) |
| Indirect set, 240 items | Attack recall | 25.0% (30/120) | 16.7% (20/120) | 29.2% (35/120) | 27.5% (33/120) |
| | Benign FPR | 6.7% (8/120) | 5.0% (6/120) | 4.2% (5/120) | 3.3% (4/120) |
| Long-document indirect set, 120 items (not ranked) | Attack recall | 3.3% (2/60) | 5.0% (3/60) | 21.7% (13/60) | 15.0% (9/60) |
| | Benign FPR | 1.7% (1/60) | 3.3% (2/60) | 6.7% (4/60) | 1.7% (1/60) |
| Held-out benign set, 200 prompts | Benign FPR | 4.5% (9/200) | 0.0% (0/200) | 0.0% (0/200) | 0.0% (0/200) |
| **Ranking score (mean Youden J)** | | 0.567 | 0.594 | 0.755 | **0.776** |
| Pooled FPR, all benign prompts | | 1.52% | 0.76% | 1.90% | 0.52% |

**Selection rule**, fixed in `scripts/compare_models.py` before the new numbers were produced: rank by the mean of (recall − FPR) over the test split, the 47-item set, the 112-prompt set, the 240-item indirect set and BIPIA; break ties by lower pooled FPR. v4 ranks first and has the lowest pooled FPR. Per-set 95% Wilson intervals are in the JSON.

How to read it:

- v3 catches the most BIPIA attacks but flags 13% of clean BIPIA inputs. v4 gives up about 7 points of BIPIA recall to cut that to 0.5%.
- v4 recovers most of what v2 lost on the 47-item set (90.7% vs 86.1%), with no false alarm on its 4 benign lookalikes; v1 flagged 1 of 4.
- On the in-distribution test split v4 matches v1 (94.8% recall, 0.3% FPR). On the 112-prompt set v4 misses 2 more attacks than v1 (50/54 vs 52/54) but has no false alarms, where v1 has 4 of 58.
- The sets from the same author process as the training data (benign held-out set, 112-prompt set, 47-item set) are optimistic, and each has 4 to 58 benign items, so the 0% FPR figures have wide intervals.

**Known limitation: indirect injection.** Detection of injections hidden inside retrieved or long documents is weak in every version: the best catch rate is 29% on the 240-item indirect set and 22% on the 120-item long-document set (v4: 27.5% and 15.0%). The long set has 60 attacks, so the v3-versus-v4 gap there is within noise. v4 is a reasonable first-pass filter for direct injection, not a defence against injection through retrieved content.

Things to keep in mind when reading the BIPIA numbers:

- BIPIA has only **125 distinct attack strings per split** (75 text, 50 code). The 8,250 test "attacks" are those strings inserted into many contexts and positions, so the sample size overstates the independent evidence and the confidence intervals on BIPIA are far too tight.
- BIPIA's test split was used to choose v4 over v3, so v4's BIPIA score is not a clean test.
- The attack category "Language Translation" appears in both BIPIA's train and test lists (different prompts). The earlier claim of fully disjoint categories was wrong for that one category. The v5 builder drops it from train and dev.
- With the 512-token limit, an attack inserted at the end of a long context was cut off but the row was still labelled "attack". v3's 13% false-alarm rate on BIPIA came almost entirely from clean documents over 512 tokens (58% flagged vs 0.6% for shorter ones).

### Long inputs

The model was trained on at most 512 tokens. The proxy now reads longer text in overlapping 512-token windows (stride 256), scores each window, and uses the highest window score (`middleware/windowing.py`, up to 128 windows, about 32,000 tokens; `LONG_INPUT=truncate` restores the old behaviour). Text that fits in one window scores exactly as before, and in a measured run windowed and truncating modes took the same time for short prompts. Cost grows with length (one forward pass per window).

What is measured so far: windowed BIPIA recall for v1 (17.5% vs 14.7% truncated) and v2 (11.5% vs 9.8%), false alarms unchanged at 1.5%. **Not measured yet:** windowed results for v3 and v4, and whether taking the highest of many windows raises false alarms on long benign documents. All v4 numbers in this README were measured with the truncating read, which is identical for inputs that fit in 512 tokens.

### Training protocol for v5 (prepared, not trained)

`scripts/build_bipia_v5.py` splits BIPIA into train / dev / test with no shared attack categories, attack prompts or contexts (checked by assertions): the official test split stays untouched, and a dev slice held out from the train split is added to the validation set. Long contexts are cropped to a 512-token window around the inserted attack, so the label matches what the model sees. `configs/training_config_v5.yaml` points at the result. The plan is to train once, fit the temperature and choose the cutoff on dev, and report the test split once. Because the test split already influenced the v3 to v4 choice, v5's score on it will not be a pristine test.

PINT was not run (the dataset requires an access request).

### Adversarial Evaluation

47 hand-crafted examples spanning 12 categories (direct injection, DAN/roleplay, hypothetical framing, Base64/ROT13/leetspeak, multi-language FR/ZH/HI/KO, system-prompt override, payload splitting, few-shot poisoning, benign lookalikes, zero-width/Unicode tricks, structured-data payloads, token smuggling). Run through the live middleware (`eval_adversarial.py`):

Original v1 results through the live middleware:

| Metric | v1 |
|---|---|
| Accuracy | 0.8936 |
| Injection recall | 0.9535 (at the model's own argmax; 93.0% at the 0.85 cutoff) |
| Benign recall (lookalikes) | 0.25 |
| Perfect categories | direct injection, DAN, Base64, multi-lang, special chars, structured data, token smuggling, few-shot |

At the 0.85 cutoff, v4 catches 39 of 43 attacks (90.7%) and flags 0 of the 4 benign lookalikes, against v1's 40 of 43 and 1 of 4 (see the versions table). With only 4 benign items, the false-alarm figure is very uncertain. v1's weakness was over-blocking benign prompts that merely *mention* injection keywords, which motivated the soft-flag default mode and the review queue below.

## Serving Middleware

`middleware/app.py` exposes three endpoints:

- `POST /chat/completions` — OpenAI-compatible proxy. Concatenates all message contents, classifies them, then either blocks or forwards.
- `GET /health` — model readiness, active mode, threshold.
- `GET /metrics` — Prometheus metrics.

**Enforcement modes** (`MODE` setting):
- `hard_block` — returns `403` with `{"error": "Prompt injection detected", "confidence": ..., "injection_detected": true}`
- `soft_flag` (default) — forwards the request but attaches `X-Injection-Detected: <confidence>` response header

A request is treated as injection only when `label == 1` **and** calibrated confidence ≥ `THRESHOLD` (default 0.85).

**Long input**: messages are concatenated and, if longer than `MAX_LENGTH` tokens, read in overlapping windows (stride `WINDOW_STRIDE`, at most `MAX_WINDOWS` windows). The request is scored by its highest window. See "Long inputs" above.

**Shadow mode** (`SHADOW_ENABLED=true`): loads a second classifier (typically the newly retrained candidate promoted by `retrain.py --shadow`). Every request runs through both models; agreement/disagreement counts and absolute confidence deltas are exported as metrics, so a new model can be validated on live traffic without risk.

**Prometheus metrics** exposed at `/metrics`:

| Metric | Type | Description |
|---|---|---|
| `injection_requests_total` | counter | Total requests screened |
| `injection_blocked_total` | counter | Hard-blocked requests |
| `injection_flagged_total` | counter | Soft-flagged requests |
| `injection_confidence` | histogram | Confidence score distribution |
| `injection_latency_seconds` | histogram | Classifier latency |
| `injection_shadow_agreements_total` | counter | Shadow/prod label agreement |
| `injection_shadow_disagreements_total` | counter | Shadow/prod label disagreement |
| `injection_shadow_confidence_delta` | histogram | \|prod − shadow\| confidence gap |

### Configuration (environment variables or `.env`)

| Variable | Default | Description |
|---|---|---|
| `LLM_ENDPOINT` | `https://api.openai.com/v1` | Upstream OpenAI-compatible API |
| `LLM_API_KEY` | *(empty)* | Bearer key forwarded upstream |
| `THRESHOLD` | `0.85` | Confidence cutoff for block/flag |
| `MODE` | `soft_flag` | `soft_flag` or `hard_block` |
| `MODEL_PATH` | `models/qwen-injection-detector/best` | LoRA adapter directory |
| `MAX_LENGTH` | `512` | Window size in tokens (the length the model was trained on) |
| `LONG_INPUT` | `window` | `window` reads long input in overlapping windows; `truncate` reads only the first `MAX_LENGTH` tokens |
| `WINDOW_STRIDE` | `256` | Tokens between window starts |
| `MAX_WINDOWS` | `128` | Cap on windows per request (0 = no cap); a warning is logged when it is hit |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | Server bind address |
| `SHADOW_ENABLED` | `false` | Enable parallel shadow classifier |
| `SHADOW_MODEL_PATH` | *(empty)* | Adapter dir for shadow model |
| `REVIEW_QUEUE_PATH` | `data/review_queue.db` | SQLite queue location |

## Human Review & Retraining Loop

1. Every proxied request (with production and shadow predictions) lands in a SQLite WAL-backed **review queue** (`scripts/review_queue.py`).
2. Label pending items interactively:
   ```bash
   python scripts/review_queue.py          # i=injection, b=benign, s=skip, q=quit
   python scripts/review_queue.py --stats  # queue statistics
   python scripts/review_queue.py --export data/reviewed_labels.jsonl
   ```
3. Close the loop:
   ```bash
   python scripts/retrain.py --shadow
   ```
   This merges human-labeled examples back into the corpus, dedups/rebalances/re-splits (originals backed up with timestamps), reruns training + calibration, optionally copies the result to the **shadow** slot, and writes `models/qwen-injection-detector/retrain_summary.json`. If training fails, the previous splits are restored automatically.

## Monitoring Stack

```bash
docker compose up -d
```

- **Prometheus** on `http://localhost:9090` — scrapes the middleware every 5s via `host.docker.internal:8080`, 7-day retention
- **Grafana** on `http://localhost:3000` (anonymous admin auth for local dev) — datasource and "Injection Detector" dashboard auto-provisioned from `configs/grafana/`

## Getting Started

Requires Python 3.10+ and a CUDA GPU for training and inference. The 4-bit loading path is built for CUDA; a CPU deployment is not supported yet.

```bash
# 1. Training dependencies
pip install -r requirements.txt

# 2. Build the dataset
python scripts/prepare_dataset.py --config configs/dataset_config.yaml
python scripts/analyze_dataset.py    # sanity-check per-source stats

# 3. Baseline + training + calibration
python scripts/evaluate_baseline.py
python scripts/train_qlora.py --config configs/training_config.yaml
python scripts/calibrate.py

# 4. Evaluate on test set
python scripts/evaluate_model.py

# 5. Serve the detector
pip install -r middleware/requirements.txt
python -m middleware.app

# 6. Adversarial eval against the live server
python scripts/build_adversarial_eval.py
python scripts/eval_adversarial.py

# 7. Compare model versions on every eval set (needs the version folders under models/)
python scripts/compare_models.py models/qwen-injection-detector-v1/best models/qwen-injection-detector-v4/best

# 8. End-to-end check of the real proxy (hard_block or soft_flag), then latency
python scripts/smoke_middleware.py --mode soft_flag
python scripts/benchmark.py models/qwen-injection-detector/best

# 9. Optional monitoring stack
docker compose up -d
```

Weights and training data are not in the repository, except the production adapter in `models/qwen-injection-detector/best/`. `eval_bipia.py` and `build_bipia_v5.py` also need a clone of [microsoft/BIPIA](https://github.com/microsoft/BIPIA) (`--bipia-dir`) and `nltk`.

### Frozen eval sets

`data/eval_handwritten`, `data/eval_indirect` and `data/eval_indirect_long` each carry an `eval.sha256`. The eval scripts refuse to run if the file's hash differs. `.gitattributes` forces LF line endings on these files, because a CRLF checkout would otherwise change the hash. If you ever regenerate one, record the new hash deliberately and say why.

### Client Integration

Point any OpenAI SDK client at the middleware — no other code changes needed:

```python
from openai import OpenAI
from middleware import wrap_openai_client

client = wrap_openai_client(
    OpenAI(api_key="..."),
    endpoint="http://localhost:8080",
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
if "X-Injection-Detected" in getattr(response, "headers", {}):
    ...  # flagged in soft_flag mode
```

Hard-blocked requests raise a `403` API error carrying `"Prompt injection detected"`.

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest              # 71 tests; no GPU or network needed
ruff check .
```

The suite (`tests/`) uses a fake classifier and a mock upstream transport, so it runs on CPU in seconds and in CI (`.github/workflows/ci.yml`). It covers:
hard-block / soft-flag behaviour and the inclusive threshold, upstream failure (502), not-ready (503), shadow-vs-production disagreement metrics, review-queue persistence and id uniqueness across restarts, retrain-adjacent data utilities (balancing, stratified split, dedup), multimodal text extraction, sliding-window scoring, the BIPIA train/dev split and crop logic, and the eval-script arithmetic (confidence intervals, recall/FPR, ranking, hash-based result matching, frozen-set checks). The smoke test and the real model are not part of CI.

### Docker

```bash
docker compose up -d                       # Prometheus + Grafana
docker compose --profile serve up -d       # + the GPU detector proxy (needs NVIDIA Container Toolkit)
```

The detector image is GPU-only (CUDA base image, 4-bit `bitsandbytes`) and copies only the production adapter. It downloads the `Qwen/Qwen2-1.5B` base model (about 3 GB) from Hugging Face on first start. CI builds the image but never runs it. A CPU image would need the adapter merged into a full-precision base model and its scores re-checked, because v4 was trained and calibrated on 4-bit weights.

## Notes

- Training checkpoints (`checkpoint-*`, optimizer states) are gitignored. The small final LoRA adapter in `models/qwen-injection-detector/best/` is kept in the repo so the service runs after a clone. `scripts/retrain.py` never overwrites it: new models land in `models/qwen-injection-detector/candidates/<timestamp>/` (and `shadow/` with `--shadow`) and are promoted manually; the test set is frozen across retrains.
- `convert_safetensors_to_bin.py` is a one-off Windows workaround converting the cached HF safetensors snapshot to `.bin` to avoid page-file issues with memory-mapped loading.
- The middleware loads the 4-bit quantized base model at startup and warms up with 3 dummy inferences before reporting ready; requests received earlier get a `503`.
- The adapter in `models/qwen-injection-detector/best/` is now v4. v1 is in git history. Results are matched to a model by the SHA-256 of its adapter weights (`adapter_sha256`), not by folder path.
- Every proxied prompt is logged to the review queue (SQLite). Decide on retention and privacy before using real user data.
- Proxy latency depends on the GPU state. On the development laptop the same test ran at about 150 ms per request when cool and about 600 ms when the GPU was power-capped and thermally throttled. With a shadow model every request runs two models.
