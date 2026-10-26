# Model Card: Prompt Injection Detector

## Model Details

- **Base model**: Qwen2-1.5B (1.5B parameters)
- **Fine-tuning method**: QLoRA (4-bit NF4 quantization, rank 8)
- **Task**: Sequence classification (2 labels: benign, injection)
- **Trainable parameters**: 2.18M / 890M (0.24%)
- **Production version**: v4 (see "Model versions and the production choice"); the training data and test-split tables below describe v1 as originally reported
- **Temperature scaling**: Learned T = 2.7795 (v4); v1 was 3.5104
- **Hardware**: NVIDIA RTX 4050 Laptop GPU (6.4 GB VRAM)
- **Training time**: ~4 hours (3 epochs)

## Intended Use

Real-time guardrail to detect prompt injection attacks in LLM application requests. Designed as a FastAPI middleware layer that inspects incoming messages before forwarding to the upstream LLM.

### Use Cases

- Blocking jailbreak attempts before they reach the LLM
- Flagging suspicious requests for human review (shadow mode)
- Monitoring injection attempt rates over time via Prometheus/Grafana

### Out-of-Scope

- Generative text classification or content filtering
- Reliable detection of indirect prompt injection (retrieval-augmented generation context poisoning): measured and weak, see "Model versions and the production choice"
- Multi-modal inputs (images, audio)
- Production use without calibration and shadow-mode validation

## Training Data

| Source | Role | Samples (after dedup) |
|--------|------|----------------------|
| deepset/prompt-injections | Injection + benign | ~3,800 |
| S-Labs/prompt-injection-dataset | Injection + benign | ~4,200 |
| xTRam1/safe-guard-prompt-injection | Injection only | ~2,500 |
| Lakera/gandalf_ignore_instructions | Injection only | 350 |
| HuggingFaceH4/no_robots | Benign only | ~13,000 |

**Preprocessing**:
- Exact deduplication (case-insensitive, whitespace-normalized)
- Near-dedup at 0.92 cosine similarity (all-MiniLM-L6-v2)
- Balanced to ~2:1 benign:injection ratio
- Stratified 80/10/10 split with source-based stratification

**Final counts**: train=17,570, val=2,196, test=2,197

## Evaluation Results

### Test Set (n=2,197)

| Metric | Value |
|--------|-------|
| Accuracy | 98.73% |
| Macro F1 | 0.9857 |
| Injection Recall | 97.83% |
| Benign Recall | 99.18% |
| ROC AUC | 0.9979 |

These are **in-distribution** numbers (same five sources in train and test). See "Generalization to unseen sources" below before relying on them.

### Calibration

| Metric | Before (T=1.0) | After (T=3.51) |
|--------|----------------|-----------------|
| ECE | 0.0058 | 0.0020 |
| NLL | 0.0643 | 0.0272 |

### Generalization to unseen sources

The training data is source-skewed: `no_robots` is 100% benign, `xtram1` and `gandalf` are 100% injection, and only `s_labs` and `deepset` contain both classes. Two diagnostics (`scripts/shortcut_baseline.py`, `scripts/eval_loso.py`):

- Predicting from the source label alone, with no text, scores **77.3%** accuracy on the test set.
- A TF-IDF + logistic regression baseline scores **96.9%** in-distribution, 1.8 points under the fine-tuned model.

Leave-one-source-out (train on the other four sources, test on the held-out one; 4,000 training samples and 1 epoch per fold, uncalibrated, so a lower bound):

| Held-out source | Classes | Qwen QLoRA | TF-IDF baseline |
|---|---|---|---|
| gandalf | injection | 98.5% | 94.1% |
| s_labs | both | 88.1% (macro-F1 0.879, AUC 0.952) | 78.5% |
| deepset | both | 81.5% (macro-F1 0.778, AUC 0.917) | 74.8% |
| xtram1 | injection | 80.3% | 73.0% |
| no_robots | benign | 56.9% | 60.5% |

The model beats the bag-of-words baseline on unseen sources, so it learns more than source style. But accuracy on unseen sources is far below 98.7%, and with `no_robots` removed it labels 43% of ordinary benign instructions as injections. Expect that kind of drop on traffic unlike the training mix, and validate on your own data.

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

**Known limitation: long-document indirect injection.** Detection of injections hidden inside retrieved or long documents is weak in every version: the best catch rate is 29% on the 240-item indirect set and 22% on the 120-item long-document set (v4: 27.5% and 15.0%). The long set has 60 attacks, so the v3-versus-v4 gap there is within noise. v4 is a reasonable first-pass filter for direct injection, not a defence against injection through retrieved content. No claim is made for documents longer than the 512-token model window.

PINT was not run (the dataset requires an access request).

### Baseline Comparison (deberta-v3-base-injection, zero-shot)

| Metric | Zero-Shot | Fine-tuned (ours) |
|--------|-----------|-------------------|
| Accuracy | 66.36% | 98.73% |
| Macro F1 | 0.6623 | 0.9857 |
| Injection Recall | 89.42% | 97.83% |
| Benign Recall | 54.73% | 99.18% |

### Limitations

- **Adversarial robustness**: On a 47-example hand-written set (12 attack categories) run through the live middleware, accuracy is 89.4% and injection recall 95.4%. Benign-lookalike recall is only 1 of 4, so benign prompts that merely mention words like "ignore" are over-blocked. The set is small and written by the model's author, so treat it as a smoke test, not a benchmark.
- **Source skew**: Most benign training data comes from one source (`no_robots`); see "Generalization to unseen sources".
- **no_robots collapse**: The zero-shot baseline achieves only 27% accuracy on the no_robots subset (all benign, classified as injection). The fine-tuned model resolves this.
- **Indirect injection**: Evaluated on BIPIA and two frozen indirect sets. v4 catches 86.6% of BIPIA test attacks but only 27.5% of the 240-item hand-written indirect set and 15.0% of the 120-item long-document set. Treat this as a known limitation. The model inspects the text it is given, not retrieved context unless that is passed in.
- **Domain shift**: Performance on out-of-distribution prompts (different LLM, different task domains) is unknown.

## Bias & Fairness

- Training data is English-only (primarily LLM chat formats)
- The no_robots subset provides diverse instruction-following bening examples, but coverage of non-English or code-switching inputs is minimal
- Adversarial examples are English-only — multilingual injection attacks likely have higher miss rates

## Deployment Recommendations

1. Start in **shadow mode** (`SHADOW_ENABLED=true`) with a human review process for at least 500 requests
2. Validate on your own traffic distribution before enabling `hard_block` mode
3. Use calibrated confidence scores: treat predictions with confidence 0.5–0.8 as low-confidence and route to review
4. Retrain monthly or after 5,000+ human-labeled reviews are collected

## Environmental Impact

- **Training emissions**: ~0.4 kg CO2e (estimated, 4 hours on RTX 4050 at ~65W)
- **Inference** (measured by `scripts/benchmark.py`, single prompt, 4-bit, RTX 4050 Laptop, idle GPU): median ~150–160 ms, peak allocated VRAM ~1.2 GB for both v1 and v2. v3 and v4 have the same architecture; they were not benchmarked separately. Tail latency is noisy: p95 ranged from about 175 ms to 720 ms across clean runs, so treat it as a range. The earlier "~35 ms" figure was not measured and has been removed.
- **Quantization**: 4-bit NF4 reduces model size from ~3 GB to ~0.9 GB for the base model

## Citation

```bibtex
@software{prompt_injection_detector_2025,
  author = {Yash Desai},
  title = {Prompt Injection Detector: QLoRA Fine-tuned Qwen2-1.5B},
  year = {2025},
  url = {https://github.com/yash/prompt-injection-detector}
}
```
