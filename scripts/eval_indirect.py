import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
sys.path.insert(0, "scripts")
from threshold_sweep import load_model, p_injection, recall_at_fpr, sweep

EVAL = Path("data/eval_indirect/eval.jsonl")
expected = Path("data/eval_indirect/eval.sha256").read_text().split()[0]
assert hashlib.sha256(EVAL.read_bytes()).hexdigest() == expected, "indirect set changed"
df = pd.DataFrame([json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines()])
y = df["label"].to_numpy()
cfg = yaml.safe_load(Path("configs/training_config.yaml").read_text())
base, max_len = cfg["model"]["base_model_name"], cfg["model"]["max_length"]
out = {}
for md in sys.argv[1:]:
    md = Path(md)
    tp = md / "temperature.pt"
    T = torch.load(str(tp)).item() if tp.exists() else 1.0
    tok, model, device = load_model(md, base)
    p = p_injection(tok, model, device, df["text"].tolist(), T, max_len)
    del model; torch.cuda.empty_cache()
    pred = p >= 0.85
    cats = {}
    for (cat, lab), g in df.assign(pred=pred).groupby(["category", "label"]):
        if lab == 1:
            cats[f"{cat}/attack"] = round(float(g["pred"].mean()), 3)
    out[str(md)] = {
        "temperature": round(float(T), 4),
        "at_0.85": {"recall": round(float(pred[y == 1].mean()), 4), "fpr": round(float(pred[y == 0].mean()), 4)},
        "by_threshold": sweep(y, p),
        "recall_at_1pct_fpr": recall_at_fpr(y, p),
        "attack_recall_by_position_at_0.85": {
            pos: round(float(pred[(df["position"] == pos).to_numpy()].mean()), 4)
            for pos in ["start", "middle", "end", "embedded_in_table_or_code"]
        },
        "recall_by_category_at_0.85": cats,
    }
    print(md, out[str(md)]["at_0.85"], flush=True)
Path("eval/indirect_comparison.json").write_text(json.dumps(out, indent=2))
print("wrote eval/indirect_comparison.json")
