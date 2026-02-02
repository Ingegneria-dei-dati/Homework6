import argparse
import json
import time
import pandas as pd
import dedupe
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def load_records(records_path):
    df = pd.read_csv(records_path, low_memory=False)
    if "record_id" not in df.columns:
        raise ValueError("records deve avere 'record_id'")
    df = df.set_index("record_id")

    # Dedupe: meglio stringhe (anche per numeri, li può gestire come Price)
    records = {}
    for rid, row in df.iterrows():
        records[rid] = {
            "make": "" if pd.isna(row.get("make")) else str(row.get("make")),
            "model": "" if pd.isna(row.get("model")) else str(row.get("model")),
            "year": "" if pd.isna(row.get("year")) else str(int(row.get("year"))) if str(row.get("year")).isdigit() else str(row.get("year")),
            "price": "" if pd.isna(row.get("price")) else str(row.get("price")),
            "mileage": "" if pd.isna(row.get("mileage")) else str(row.get("mileage")),
            "state": "" if pd.isna(row.get("state")) else str(row.get("state")),
            "fuel_type": "" if pd.isna(row.get("fuel_type")) else str(row.get("fuel_type")),
            "transmission": "" if pd.isna(row.get("transmission")) else str(row.get("transmission")),
        }
    return records


def load_pairs(pairs_path):
    df = pd.read_csv(pairs_path)
    for c in ["id_l", "id_r", "label"]:
        if c not in df.columns:
            raise ValueError(f"{pairs_path} deve avere colonne id_l,id_r,label")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return df


def main():
    p = argparse.ArgumentParser(description="Dedupe: training + inference + metriche + tempi")
    p.add_argument("--records", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out_metrics", required=True)
    p.add_argument("--threshold", type=float, default=0.5, help="Soglia decisione match (default 0.5)")
    args = p.parse_args()

    print("Caricamento record...")
    records = load_records(args.records)

    print("Caricamento train/test pairs...")
    train_df = load_pairs(args.train)
    test_df = load_pairs(args.test)

    # Training data per dedupe
    match = list(zip(train_df.loc[train_df["label"] == 1, "id_l"], train_df.loc[train_df["label"] == 1, "id_r"]))
    distinct = list(zip(train_df.loc[train_df["label"] == 0, "id_l"], train_df.loc[train_df["label"] == 0, "id_r"]))

    print(f"Train match: {len(match)}  Train distinct: {len(distinct)}")

    fields = [
        {"field": "make", "type": "String"},
        {"field": "model", "type": "String"},
        {"field": "year", "type": "Exact"},
        {"field": "price", "type": "Price"},
        {"field": "mileage", "type": "Price"},
        {"field": "state", "type": "Exact"},
        {"field": "fuel_type", "type": "Exact"},
        {"field": "transmission", "type": "Exact"},
    ]

    deduper = dedupe.Dedupe(fields)
    deduper.prepare_training(records, {"match": match, "distinct": distinct})

    print("Training Dedupe...")
    t_train_start = time.perf_counter()
    deduper.train()
    t_train_end = time.perf_counter()

    print("Inferenza su test...")
    y_true = test_df["label"].to_numpy()
    y_pred = []

    t_inf_start = time.perf_counter()
    for _, r in test_df.iterrows():
        a = records.get(r["id_l"])
        b = records.get(r["id_r"])
        if a is None or b is None:
            y_pred.append(0)
            continue
        score = deduper.score(a, b)
        y_pred.append(1 if score >= args.threshold else 0)
    t_inf_end = time.perf_counter()

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "training_time": float(t_train_end - t_train_start),
        "inference_time": float(t_inf_end - t_inf_start),
    }

    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Metriche salvate in:", args.out_metrics)


if __name__ == "__main__":
    main()
