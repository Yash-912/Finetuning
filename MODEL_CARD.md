# Model Card: Prompt Injection Detector (v4)

The model in production is **v4**. This card describes v4. Where a number comes from an earlier version, it says so.

## Model Details

- **Base model**: Qwen2-1.5B
- **Fine-tuning method**: QLoRA (4-bit NF4 quantization, LoRA rank 8, alpha 16, dropout 0.1, on the q/k/v/o projections)
- **Task**: Sequence classification, 2 labels (benign, injection)
- **Trainable parameters**: 2,182,144 of 890,801,664 counted parameters (0.24%). The count is taken on the 4-bit model, so it is below the 1.5B of the full-precision base
- **Temperature scaling**: T = 2.7795, fitted on the 2,396-row validation set (see Calibration)
- **Decision rule**: flag an input as an injection when P(injection) is at least 0.85 (the proxy default)
- **Input handling**: the model was trained on inputs of at most 512 tokens. In the proxy, longer inputs are read in overlapping 512-token windows (stride 256) and scored by the highest window. See "Long inputs"
- **Training**: 3 epochs, 2,076 optimizer steps, effective batch size 32 (8 x 4 accumulation), learning rate 2e-4 with a cosine schedule, fp16, gradient checkpointing. The best checkpoint by macro-F1 on the validation set was kept
- **Hardware**: NVIDIA RTX 4050 Laptop GPU (6 GB). Training took roughly 6 to 8 hours, estimated from step times.
- **Version history**: v1 (original), v2 (adds hand-authored benign prompts), v3 (adds BIPIA indirect-injection training data), v4 (v3 rebalanced). v5 (a cleaner BIPIA split) is prepared but not trained

## Intended Use

A guardrail that flags prompt injection in requests to an LLM application. It runs as a FastAPI proxy that inspects messages before forwarding them to the upstream LLM.

### Use cases

- Flagging or blocking direct injection attempts before they reach the LLM
- Sending suspicious requests to human review (shadow mode and the review queue)
- Tracking injection attempt rates over time with Prometheus and Grafana

### Out of scope

- Reliable detection of indirect injection (instructions hidden in documents, emails, web pages or tool output). It is measured and weak, see "Known limitations"
- Content moderation, or general text classification
- Images, audio and other non-text input
- Use without calibration on your own traffic and a shadow-mode trial

## Training Data

The v4 training set has 22,137 rows: 14,315 benign and 7,822 injection.

| Part | Rows | Notes |
|---|---|---|
| v2 training data | 19,170 | The original five sources plus hand-authored benign prompts |
| BIPIA train split | 2,967 | 1,978 injected and 989 clean. Every train context, two attack prompts per context |

The original five sources (v1 corpus, from `data/processed/dataset_card.json`):

| Source | Role |
|---|---|
| deepset/prompt-injections | Injection and benign |
| S-Labs/prompt-injection-dataset | Injection and benign |
| xTRam1/safe-guard-prompt-injection | Injection only |
| Lakera/gandalf_ignore_instructions | Injection only |
| HuggingFaceH4/no_robots | Benign only |

Hand-authored benign prompts cover code, multilingual text, hard negatives (benign text that mentions words like "ignore") and everyday requests. Preprocessing: exact deduplication, near-duplicate removal at 0.92 cosine similarity (all-MiniLM-L6-v2), and a stratified train/validation/test split by source.

Validation (`data/processed/val.parquet`, 2,396 rows) and test (`data/processed/test.parquet`, 2,197 rows) contain no BIPIA.

**BIPIA** (Benchmark for Indirect Prompt Injection Attacks) builds poisoned documents by inserting an attack prompt into an email, a table or a code file. Three of its five tasks are used. The other two ship without their context data. BIPIA has only 125 distinct attack strings per split (75 text and 50 code), so its large sample counts overstate how much independent evidence it gives.

## Evaluation Results

All numbers use the 0.85 cutoff and each model's own temperature. Counts are in brackets because most sets are small.

How the numbers were measured: each set was scored reading only the first 512 tokens of every input. This is identical to the proxy's behavior for inputs that fit in 512 tokens, which covers almost every set here. It differs only for long inputs (the long-document set and about 45% of BIPIA table contexts). The windowed results for v3 and v4 have not been measured yet, see "Long inputs".

### v4 results

| Set | Recall (attacks caught) | False-alarm rate |
|---|---|---|
| Test split, in-distribution (737 attacks, 1,460 benign) | 94.8% (699/737), 95% interval 93.0 to 96.2% | 0.3% (5/1460) |
| Held-out benign set (200 benign) | n/a | 0.0% (0/200) |
| 112-prompt hand-written set (54 attacks, 58 benign) | 92.6% (50/54) | 0.0% (0/58), upper bound 6.2% |
| 47-item hand-written set (43 attacks, 4 benign) | 90.7% (39/43) | 0.0% (0/4), upper bound 49% |
| BIPIA test split (8,250 attacks, 200 clean) | 86.6% (7,144/8,250) | 0.5% (1/200) |
| 240-item indirect set (120 attacks, 120 benign) | 27.5% (33/120) | 3.3% (4/120) |
| 120-item long-document set (60 attacks, 60 benign) | 15.0% (9/60) | 1.7% (1/60) |

On the test split, v4 reaches accuracy 98.0%, macro-F1 0.978 and ROC-AUC 0.9986. At a 1% false-alarm budget it catches 98.0% of attacks. These are in-distribution numbers. See "Generalization to unseen sources" before relying on them.

### All versions, side by side

@@TABLE@@
**Selection rule**, fixed in `scripts/compare_models.py` before the final numbers were produced: rank by the mean of (recall minus false-alarm rate) over the test split, the 47-item set, the 112-prompt set, the 240-item indirect set and BIPIA. Break ties by the lower false-alarm rate over all clean prompts. v4 ranks first.

Caveats about that choice:

- v3 and v4 trained on BIPIA's train split and v1 and v2 did not, so BIPIA favors v3 and v4. Without BIPIA the scores are v4 0.755, v3 0.743, v2 0.722, v1 0.676. v4 still leads, by a small margin that is within noise on these set sizes
- The v3-to-v4 change (rebalancing the BIPIA rows) was made after looking at v3's results on BIPIA's test split, so v4's BIPIA score is optimistic
- All four hand-written sets are small and were written by one author process, so a 0% false-alarm figure on 4 to 58 items says little

How the versions differ:

- v1 flagged benign lookalikes (4.5% on the held-out benign set)
- v2 fixed that but missed more framing-style attacks (roleplay, hypotheticals, encodings): 86.1% on the 47-item set
- v3 learned BIPIA (93.4% recall) but flagged 13.0% of clean BIPIA inputs. Almost all of that came from clean documents longer than 512 tokens (58% flagged against 0.6% for shorter ones)
- v4 keeps most of v3's BIPIA recall (86.6%) with 0.5% false alarms

### Calibration

Temperature scaling was fitted on the validation set, which contains no indirect-injection samples:

| Metric | Before (T = 1.0) | After (T = 2.7795) |
|---|---|---|
| ECE | 0.0075 | 0.0032 |
| NLL | 0.0601 | 0.0321 |

Because the validation set has no indirect samples, calibration on documents is unverified. The prepared v5 recipe fits the temperature on a validation set that includes indirect samples.

### Generalization to unseen sources

This diagnostic was run on the v1 recipe, not on v4. The training data is source-skewed: `no_robots` is all benign, `xtram1` and `gandalf` are all injection, and only `s_labs` and `deepset` contain both classes. Two checks (`scripts/shortcut_baseline.py`, `scripts/eval_loso.py`):

- Predicting from the source label alone, with no text, scores 77.3% accuracy on the test split
- A TF-IDF and logistic regression baseline scores 96.9% on the same split

Leave-one-source-out (train on the other four sources, test on the held-out one; 4,000 training samples and 1 epoch per fold, uncalibrated, so a lower bound):

| Held-out source | Classes | Qwen QLoRA | TF-IDF baseline |
|---|---|---|---|
| gandalf | injection | 98.5% | 94.1% |
| s_labs | both | 88.1% (macro-F1 0.879, AUC 0.952) | 78.5% |
| deepset | both | 81.5% (macro-F1 0.778, AUC 0.917) | 74.8% |
| xtram1 | injection | 80.3% | 73.0% |
| no_robots | benign | 56.9% | 60.5% |

The model beats the bag-of-words baseline on unseen sources, so it learns more than source style. But accuracy on unseen sources is far below the in-distribution figure. With `no_robots` removed, it labels 43% of ordinary benign instructions as injections. Expect a drop on traffic unlike the training mix, and validate on your own data.

### Baseline comparison (v1, zero-shot deepset/deberta-v3-base-injection)

| Metric | Zero-shot | Fine-tuned v1 |
|---|---|---|
| Accuracy | 66.36% | 98.73% |
| Macro F1 | 0.6623 | 0.9857 |
| Injection recall | 89.42% | 97.83% |
| Benign recall | 54.73% | 99.18% |

This comparison has not been repeated for v4.

### Long inputs

The model reads at most 512 tokens at a time. In the proxy, longer text is split into overlapping 512-token windows (stride 256). Each window is scored, and the text gets its highest window score. Up to 128 windows are read (about 32,000 tokens); beyond that a warning is logged and only part of the text is read. Set `LONG_INPUT=truncate` to restore the old first-512-tokens behavior.

What is known:

- For text that fits in one window, scoring is identical to truncation, and in a measured proxy run the two modes took the same time
- Windowed BIPIA results exist only for v1 and v2: recall 17.5% against 14.7% for v1 and 11.5% against 9.8% for v2, both with 1.5% false alarms
- Windowed results for v3 and v4, and whether taking the highest of many windows raises false alarms on long benign documents, have **not** been measured. The v4 numbers above are for the truncating read
- Cost grows with length: one forward pass per window

### Deployment checks (live proxy, v4 with v1 as the shadow model)

`scripts/smoke_middleware.py` replays the 112-prompt hand-written set through the real proxy against a local fake LLM:

| Mode | Result |
|---|---|
| hard_block | 50 of 54 attacks blocked, 0 of 58 benign flagged, all 58 benign forwarded with the upstream reply intact |
| soft_flag (the default mode) | no request blocked, all 112 forwarded, 50 flagged requests carry the `X-Injection-Detected` header, 0 benign flagged |
| both modes | all 112 requests logged to the review queue with production and shadow predictions; shadow agreed on 107 and disagreed on 5 |

Forwarding to a real LLM endpoint has not been tested.

## Known limitations

- **Indirect injection is weak.** v4 catches 86.6% of BIPIA attacks but only 27.5% of the 240-item hand-written indirect set and 15.0% of the 120-item long-document set. The BIPIA figure mostly reflects BIPIA's templated attacks and should not be read as general indirect-injection detection. The model scores the text it is given. It does not see retrieved context unless that is passed to it
- **BIPIA is not a clean test.** Its test split was used to choose between v3 and v4. One attack category name ("Language Translation") appears in both its train and test lists. The earlier claim that the categories were fully disjoint was wrong for that one category
- **Small, single-author test sets.** The hand-written sets hold 47 to 240 items from one author process, so their false-alarm rates have wide uncertainty
- **Source skew.** Most benign training data comes from one source (`no_robots`); see "Generalization to unseen sources"
- **Benign lookalikes.** Earlier versions over-blocked benign prompts that mention words like "ignore". v4 had 0 false alarms on the small lookalike sets, but they hold only 4 and 58 items
- **Framing attacks.** Roleplay, hypothetical and encoded attacks are where v2 and v3 were weakest. v4 recovers most of that (90.7% on the 47-item set) but misses some
- **Domain shift.** Performance on other LLMs, task domains and languages is unknown. Training data is mostly English

## Bias and Fairness

- Training data is mostly English. Coverage of non-English and code-switched input is small
- The hand-written attack sets are mostly English, so multilingual attacks likely have higher miss rates
- Nothing here has been checked for demographic bias; the task is about the form of the text, not about people

## Deployment Recommendations

1. Start in **shadow mode** (`SHADOW_ENABLED=true`) with human review for at least 500 requests
2. The default mode is `soft_flag`, which forwards flagged requests with a header. Use `hard_block` only after validating on your own traffic
3. Treat confidence between 0.5 and 0.8 as low-confidence and route it to review
4. Do not rely on this model alone against injection in retrieved documents
5. Prompts are logged to the review queue (SQLite). Decide on retention and privacy before using real user data
6. Retrain monthly, or after 5,000 or more human-labeled reviews are collected

## Performance and Resources

- **Inference** (`scripts/benchmark.py`, v4, single prompt, 4-bit, RTX 4050 Laptop, idle GPU): median 142 ms, p95 202 ms, p99 340 ms, peak GPU memory 1.2 GB
- Latency depends heavily on the GPU state. On this laptop the GPU was power-capped and thermally throttled in later runs, and the same proxy test took about 600 ms per request instead of about 150 ms. With a shadow model loaded, every request runs two models
- **Quantization**: 4-bit NF4 reduces the base model from about 3 GB to about 0.9 GB
- **Training emissions**: not measured
- The Dockerfile is GPU-only (CUDA). A CPU deployment needs a different model format; it is not provided

## Reproducing

```bash
pytest                                              # 71 unit tests, no GPU needed
python scripts/compare_models.py \
  models/qwen-injection-detector-v1/best models/qwen-injection-detector-v2/best \
  models/qwen-injection-detector-v3/best models/qwen-injection-detector-v4/best
python scripts/smoke_middleware.py --mode soft_flag   # needs the GPU and model weights
```

Weights and training data are not in the repository.

## Citation

```bibtex
@software{prompt_injection_detector_2025,
  author = {Yash Desai},
  title = {Prompt Injection Detector: QLoRA Fine-tuned Qwen2-1.5B},
  year = {2025},
  url = {https://github.com/yash/prompt-injection-detector}
}
```
