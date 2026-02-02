import json
import pandas as pd

PIPELINES = [
    ("B1-RecordLinkage", "results/rl_B1_metrics.json", "timing_seconds.training", "timing_seconds.inference"),
    ("B2-RecordLinkage", "results/rl_B2_metrics.json", "timing_seconds.training", "timing_seconds.inference"),
    ("B1-Dedupe",        "results/dedupe_B1.json",     "training_time",          "inference_time"),
    ("B2-Dedupe",        "results/dedupe_B2.json",     "training_time",          "inference_time"),
    ("B1-Ditto",         "results/ditto_B1.json",      "training_time",          "inference_time"),
    ("B2-Ditto",         "results/ditto_B2.json",      "training_time",          "inference_time"),
]

def get_nested(d, path):
    parts = path.split(".")
    cur = d
    for p in parts:
        cur = cur[p]
    return cur

rows = []
for name, path, train_key, inf_key in PIPELINES:
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)

    # RecordLinkage ha tempi sotto timing_seconds (chiavi nested)
    if "." in train_key:
        train_time = float(get_nested(m, train_key))
        inf_time = float(get_nested(m, inf_key))
    else:
        train_time = float(m[train_key])
        inf_time = float(m[inf_key])

    rows.append({
        "Pipeline": name,
        "Precision": float(m["precision"]),
        "Recall": float(m["recall"]),
        "F1": float(m["f1"]),
        "Train_time_s": train_time,
        "Inference_time_s": inf_time
    })

df = pd.DataFrame(rows)
df.to_csv("results/final_comparison.csv", index=False)
print(df)
print("\nCreato: results/final_comparison.csv")
