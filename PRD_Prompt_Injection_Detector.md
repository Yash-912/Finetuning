# Product Requirements Document
## LLM Guardrail: Real-Time Prompt Injection Detector

**Version:** 1.0
**Status:** Draft
**Owner:** [Your Name]
**Last Updated:** July 2026

---

## 1. Problem Statement

LLM-powered applications are increasingly exposed to **prompt injection attacks** — adversarial inputs designed to override system instructions, exfiltrate data, or hijack model behavior. This risk is amplified in RAG and agentic systems where untrusted content (documents, web pages, tool outputs) is injected into the context window indirectly.

Most existing mitigations either:
- Rely on the LLM itself to "notice" the attack (unreliable, and doubles cost/latency), or
- Use static regex/keyword filters (trivially bypassed via paraphrasing, encoding, or translation).

There is no lightweight, low-latency, continuously-updated guardrail that sits **in front of** any LLM endpoint and screens input before it reaches the model.

## 2. Goals & Non-Goals

### Goals
- Detect direct and (where feasible) indirect prompt injection attempts with high recall at an acceptable false-positive rate.
- Run as a **middleware layer** in front of any OpenAI-compatible endpoint, adding minimal latency.
- Be deployable on modest hardware (single consumer GPU or CPU) — this is a portfolio/self-hosted project, not a hyperscale service.
- Provide observability (attempt rates, confidence distributions, drift signals) so the system's health is visible, not a black box.
- Demonstrate a full MLOps lifecycle: data → fine-tuning → eval → deployment → monitoring → retraining trigger — end to end, reproducible.

### Non-Goals (v1)
- Guaranteed robustness against adaptive, white-box adversaries who have access to the model weights.
- Fully automated closed-loop retraining without human review (v1 will flag for review, not auto-deploy).
- Multi-modal injection detection (image/audio-based injections) — text only in v1.
- Enterprise-scale throughput (1000s of req/sec) — target is realistic single-node throughput.

## 3. Success Metrics

| Metric | Target (v1) | Notes |
|---|---|---|
| Recall (catch rate) on held-out test set | ≥ 90% | Prioritized over precision — missed injection is costlier than a false alarm |
| Precision on held-out test set | ≥ 85% | Avoid blocking legitimate users too often |
| Recall on hand-crafted adversarial set (obfuscated/paraphrased) | ≥ 60% | Honest stretch target — this is where most classifiers fail; report even if it's lower, that's a real finding |
| P50 inference latency | ≤ 50ms | Measured on target deployment hardware, not laptop dev box |
| P99 inference latency | ≤ 150ms | |
| System uptime (middleware) | 99% (best-effort, portfolio SLA) | Not a real production SLA — framed honestly |

**Important framing for the portfolio narrative:** the adversarial-set recall number is expected to be meaningfully lower than the in-distribution number. Reporting that gap honestly, and explaining *why* (dataset size, known-pattern overfitting), is a stronger signal of ML maturity than hiding it.

## 4. Users & Use Cases

**Primary user:** a developer integrating an LLM into their application who wants a drop-in safety layer.

**Use cases:**
1. Chatbot/agent developer wraps their OpenAI-compatible endpoint with the middleware; injected user messages are blocked or flagged before reaching the LLM.
2. RAG pipeline owner screens retrieved document chunks for embedded injection payloads before they're placed in context.
3. Security/ML engineer (reviewer persona) monitors the Grafana dashboard for injection attempt trends and reviews flagged misclassifications for retraining.

## 5. System Architecture

```
Client Request
      │
      ▼
┌─────────────────────┐
│  FastAPI Middleware   │
│  ┌─────────────────┐ │
│  │ Injection        │ │
│  │ Classifier       │ │──► logs/metrics ──► Prometheus ──► Grafana
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

### Components
1. **Classifier model** — Qwen2-1.5B fine-tuned via QLoRA (4-bit) with a sequence-classification head, binary output (injection / benign) with calibrated confidence score.
2. **FastAPI middleware** — wraps any OpenAI-compatible `/chat/completions` endpoint. Single decorator/proxy pattern for integration.
3. **Monitoring stack** — Prometheus metrics exporter + Grafana dashboards (injection rate over time, confidence score distribution, latency percentiles).
4. **Retraining trigger pipeline** — logs low-confidence / disputed predictions to a review queue; human-reviewed corrections feed a retraining dataset; retraining is manually triggered in v1 (see Non-Goals).
5. **Shadow mode harness** — new classifier versions run in parallel against production traffic (predictions logged, not enforced) before promotion.

## 6. Data

| Source | Role | Size |
|---|---|---|
| `deepset/prompt-injections` | Primary training data | 662 examples |
| `JasperLS/prompt-injections` | Supplemental training data | ~460 examples |
| `Lakera/gandalf_ignore_instructions` | Supplemental training data | ~1000+ examples |
| Hand-authored adversarial eval set | Held-out, never trained on | 40-60 examples (obfuscation, translation, multi-turn, encoding tricks) |
| Synthetic augmentation (LLM-generated paraphrases) | Training data expansion | TBD, target 2-3x base dataset |

**Known limitations (documented, not hidden):**
- Combined dataset is small (~2,000 examples) for a 1.5B parameter model — overfitting risk is real and will be measured, not assumed away.
- Gandalf dataset is drawn from a single game context and may not generalize to production system prompts.
- Class balance and near-duplicate leakage across sources will be explicitly checked before the train/val/test split.

## 7. Model Requirements

- **Base model:** Qwen2-1.5B
- **Fine-tuning method:** QLoRA, 4-bit NF4 quantization, LoRA rank 8-16 on attention projections
- **Task head:** Sequence classification (binary), not generative decoding — required to meet latency target
- **Training hardware:** Single RTX 4050 (6GB VRAM), laptop-class — this constraint is a first-class requirement, not an afterthought, and will be documented in the model card
- **Output:** Binary label + calibrated confidence score (not just argmax)

## 8. Deployment Requirements

- **Interface:** FastAPI middleware, single import/one-line wrap around existing OpenAI-compatible client calls
- **Configurability:** Caller can set confidence threshold; default threshold chosen via precision/recall tradeoff analysis on val set, not arbitrarily
- **Response on detection:** configurable — hard block (return 403-style error) or soft flag (pass through with a warning header/log entry), default to soft-flag in v1 to avoid over-blocking real users while the classifier is still young
- **Latency budget:** classifier inference must fit inside the 50ms P50 target — quantized inference (ONNX/INT8 export post-QLoRA-merge) evaluated if raw PyTorch inference doesn't hit target

## 9. Observability

- **Metrics exported:** request count, injection-flagged count, confidence score histogram, latency histogram, model version tag
- **Dashboards (Grafana):**
  - Injection attempt rate over time
  - Confidence score distribution (drift indicator — a shifting distribution suggests new attack patterns)
  - Latency percentiles
- **Alerting (basic v1):** spike in flagged rate, spike in latency, classifier service health check failures

## 10. Retraining & Model Lifecycle

1. Low-confidence predictions (score near threshold) and any predictions overturned by human review are logged to a review queue.
2. Reviewed/corrected examples are periodically batched into a retraining dataset.
3. Retraining is triggered manually in v1 (explicitly scoped — see Non-Goals) by running the fine-tuning pipeline against the updated dataset.
4. New model version is evaluated against the frozen test set + adversarial set before being eligible for shadow mode.
5. Shadow mode: new version runs alongside production version, predictions logged but not enforced, for a defined evaluation window.
6. Promotion to production is a manual decision gated on shadow-mode metrics meeting or exceeding the current production model.

## 11. Risks & Open Questions

| Risk | Mitigation / Status |
|---|---|
| Small dataset limits generalization | Documented as a known limitation; synthetic augmentation planned; adversarial eval set reports the real gap |
| 1.5B model may not hit 50ms on CPU | Will benchmark early; quantized export (ONNX/INT8) as fallback; latency target may be revised with evidence rather than assumed |
| False positives blocking legitimate users | Default to soft-flag, not hard-block, in v1; threshold tunable by integrator |
| No ground-truth signal for production misclassification | v1 uses human review queue, not automated detection, for this — explicitly scoped as a limitation, not solved |
| Adaptive/white-box attackers can bypass a known-architecture classifier | Out of scope for v1; documented as a threat model boundary |

## 12. Milestones

| Phase | Deliverable |
|---|---|
| 1 | Data pipeline: merge, dedupe, split, hand-author adversarial eval set |
| 2 | QLoRA fine-tuning script + baseline model + eval report (including honest adversarial-set numbers) |
| 3 | FastAPI middleware + latency benchmarking + threshold tuning |
| 4 | Monitoring stack (Prometheus/Grafana) wired to middleware |
| 5 | Shadow-mode harness + retraining pipeline (manual trigger) |
| 6 | Documentation: model card, architecture diagram, README with one-line integration example |
