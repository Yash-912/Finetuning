#!/usr/bin/env python3
"""
End-to-end smoke test of the real middleware process with the real production model.

Starts a local fake OpenAI-compatible upstream, launches `python -m middleware.app` against it with a
shadow model and a throwaway review queue (the real data/review_queue.db is never touched), replays the
frozen hand-written eval set through POST /chat/completions, and checks:
  - benign prompts are forwarded and the upstream's body comes back unchanged
  - flagged prompts are blocked (hard_block) or tagged with X-Injection-Detected (soft_flag)
  - every request is logged to the review queue with production and shadow predictions
  - the shadow agreement/disagreement counters on /metrics add up to the number of requests
  - client-observed latency percentiles

Needs a GPU-capable environment with the model weights; it is a manual check, not part of CI.

Usage:
    python scripts/smoke_middleware.py [--mode hard_block|soft_flag] [--shadow-model DIR] [--out eval/smoke_middleware.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
import numpy as np

EVAL = Path("data/eval_handwritten/eval.jsonl")
UPSTREAM_PORT, MIDDLEWARE_PORT = 9099, 8090
UPSTREAM_REPLY = {"id": "fake-1", "object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class FakeUpstream(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps(UPSTREAM_REPLY).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def wait_ready(client: httpx.Client, proc: subprocess.Popen, timeout: float = 240.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"middleware exited early with code {proc.returncode}")
        try:
            if client.get("/health").json().get("model_loaded"):
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise SystemExit("middleware did not become ready in time")


def metric(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith(name + " ") or line.startswith(name + "_total "):
            return float(line.split()[-1])
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["hard_block", "soft_flag"], default="hard_block")
    ap.add_argument("--model", default="models/qwen-injection-detector/best")
    ap.add_argument("--shadow-model", default="models/qwen-injection-detector-v1/best")
    ap.add_argument("--no-shadow", action="store_true", help="run without a shadow model (isolates production latency)")
    ap.add_argument("--out", type=Path, default=Path("eval/smoke_middleware.json"))
    args = ap.parse_args()

    rows = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines() if line.strip()]

    upstream = HTTPServer(("127.0.0.1", UPSTREAM_PORT), FakeUpstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    tmp = Path(tempfile.mkdtemp(prefix="smoke_"))
    queue_path = tmp / "review_queue.db"
    env = {
        **os.environ,
        "LLM_ENDPOINT": f"http://127.0.0.1:{UPSTREAM_PORT}",
        "LLM_API_KEY": "smoke-test-key",
        "MODE": args.mode,
        "MODEL_PATH": args.model,
        "SHADOW_ENABLED": "false" if args.no_shadow else "true",
        "SHADOW_MODEL_PATH": args.shadow_model,
        "REVIEW_QUEUE_PATH": str(queue_path),
        "HOST": "127.0.0.1",
        "PORT": str(MIDDLEWARE_PORT),
    }
    log_path = tmp / "middleware.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen([sys.executable, "-m", "middleware.app"], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{MIDDLEWARE_PORT}", timeout=60.0) as client:
            wait_ready(client, proc)
            results, latencies = [], []
            for r in rows:
                payload = {"model": "smoke", "messages": [{"role": "user", "content": r["text"]}]}
                t0 = time.perf_counter()
                resp = client.post("/chat/completions", json=payload)
                latencies.append((time.perf_counter() - t0) * 1000)
                results.append({
                    "label": r["label"],
                    "status": resp.status_code,
                    "flag_header": "x-injection-detected" in {k.lower() for k in resp.headers},
                    "forwarded_body_ok": resp.status_code == 200 and resp.json() == UPSTREAM_REPLY,
                })
            metrics_text = client.get("/metrics").text
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        upstream.shutdown()

    blocked = [x["status"] == 403 for x in results]
    flagged = [x["status"] == 403 or x["flag_header"] for x in results]
    y = np.array([x["label"] for x in results])
    detected = np.array(flagged)
    other = [x["status"] for x in results if x["status"] not in (200, 403)]
    con = sqlite3.connect(queue_path)
    queue_rows, with_shadow = con.execute("SELECT COUNT(*), SUM(shadow_pred IS NOT NULL) FROM reviews").fetchone()
    con.close()
    agree, disagree = metric(metrics_text, "injection_shadow_agreements"), metric(metrics_text, "injection_shadow_disagreements")
    predict_count = metric(metrics_text, "injection_latency_seconds_count")
    predict_ms = 1000 * metric(metrics_text, "injection_latency_seconds_sum") / max(predict_count, 1)

    report = {
        "mode": args.mode,
        "model": args.model,
        "shadow_model": None if args.no_shadow else args.shadow_model,
        "n_requests": len(results),
        "attack_detected": int(detected[y == 1].sum()),
        "attacks": int((y == 1).sum()),
        "benign_flagged": int(detected[y == 0].sum()),
        "benign": int((y == 0).sum()),
        "blocked_count": int(sum(blocked)),
        "unexpected_statuses": other,
        "benign_forwarded_with_upstream_body": int(sum(x["forwarded_body_ok"] for x, lab in zip(results, y) if lab == 0)),
        "review_queue_rows": queue_rows,
        "review_queue_rows_with_shadow": int(with_shadow or 0),
        "shadow_agreements": agree,
        "shadow_disagreements": disagree,
        "server_side_production_predict_mean_ms": round(predict_ms, 1),
        "latency_ms": {k: round(float(np.percentile(latencies, q)), 1) for k, q in [("p50", 50), ("p95", 95), ("p99", 99)]},
    }
    checks = {
        "no unexpected statuses": not other,
        "every request logged": queue_rows == len(results),
        "every row has a shadow prediction": args.no_shadow or report["review_queue_rows_with_shadow"] == len(results),
        "shadow counters add up": args.no_shadow or agree + disagree == len(results),
        "benign passes reach upstream unchanged": report["benign_forwarded_with_upstream_body"] == report["benign"] - report["benign_flagged"]
        if args.mode == "hard_block" else True,
    }
    if args.mode == "soft_flag":
        # soft_flag must never block: every request, flagged or not, is forwarded and its upstream reply returned intact
        checks["soft_flag never blocks"] = report["blocked_count"] == 0
        checks["every request forwarded with the upstream body"] = all(x["forwarded_body_ok"] for x in results)
        checks["flagged prompts carry the X-Injection-Detected header"] = int(sum(x["flag_header"] for x in results)) == int(detected.sum())
    report["checks"] = checks
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        print(f"FAILED checks; middleware log: {log_path}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
