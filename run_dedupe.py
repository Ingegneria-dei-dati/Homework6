


'''import os
import json
import dedupe
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

# ================= CONFIG =================
DATASET_PATH = "dataset/dataset_for_training.csv"
TRAIN_PAIRS = "dataset/splits/train_pairs.csv"
VAL_PAIRS   = "dataset/splits/val_pairs.csv"
TEST_PAIRS  = "dataset/splits/test_pairs.csv"

MODEL_PATH  = "dedupe_model"
os.makedirs(MODEL_PATH, exist_ok=True)

CHUNK_SIZE = 100_000
SEED = 42
np.random.seed(SEED)

FIELDS = [
    "make","model","year","price","mileage",
    "fuel_type","transmission","body_type","drive","condition"
]

# ================= UTILS =================
def clean_val(val, dtype):
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ["nan", "none", "", "0", "0.0"]:
        return None
    try:
        return dtype(val)
    except:
        return None

def norm_str(val):
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    return s if s not in ["", "nan", "none"] else None

def load_records(ids_needed):
    """Carica solo i record necessari, tutti i campi del data model presenti."""
    cols = ["id","make","model","year","price","mileage",
            "fuel_type","transmission","body_type","drive","condition"]
    records = {}
    for chunk in pd.read_csv(DATASET_PATH, usecols=cols, chunksize=CHUNK_SIZE):
        chunk["id_str"] = chunk["id"].astype(str)
        sub = chunk[chunk["id_str"].isin(ids_needed)]
        if sub.empty:
            continue
        for row in sub.itertuples(index=False):
            rid = str(row.id)
            records[rid] = {f: None for f in FIELDS}  # inizializza tutte le chiavi
            records[rid]["make"] = norm_str(row.make)
            records[rid]["model"] = norm_str(row.model)
            records[rid]["year"] = clean_val(row.year,int)
            records[rid]["price"] = clean_val(row.price,float)
            records[rid]["mileage"] = clean_val(row.mileage,float)
            records[rid]["fuel_type"] = norm_str(getattr(row,"fuel_type",None))
            records[rid]["transmission"] = norm_str(getattr(row,"transmission",None))
            records[rid]["body_type"] = norm_str(getattr(row,"body_type",None))
            records[rid]["drive"] = norm_str(getattr(row,"drive",None))
            records[rid]["condition"] = norm_str(getattr(row,"condition",None))
    return records

def pairs_to_dedupe_format(df, records):
    match, distinct = [], []
    for r in df.itertuples(index=False):
        a, b, y = str(r.id1), str(r.id2), int(r.label)
        if a not in records or b not in records:
            continue
        if y == 1:
            match.append((a,b))
        else:
            distinct.append((a,b))
    return {"match": match, "distinct": distinct}

def score_pairs(linker, df, records):
    y_true = df["label"].astype(int).to_numpy()
    probs  = []
    for r in df.itertuples(index=False):
        a, b = str(r.id1), str(r.id2)
        if a in records and b in records:
            probs.append(linker.score(records[a],records[b]))
        else:
            probs.append(0.0)
    return y_true, np.array(probs)

def best_threshold(y_val, p_val):
    prec, rec, thr = precision_recall_curve(y_val,p_val)
    f1 = (2*prec*rec)/(prec+rec+1e-12)
    if len(thr)==0: return 0.5
    return thr[np.argmax(f1[:-1])]

# ================= MAIN =================
def main():
    train_df = pd.read_csv(TRAIN_PAIRS)
    val_df   = pd.read_csv(VAL_PAIRS)
    test_df  = pd.read_csv(TEST_PAIRS)

    # --- IDs necessari ---
    ids_needed = set(train_df.id1.astype(str)) | set(train_df.id2.astype(str)) | \
                 set(val_df.id1.astype(str))   | set(val_df.id2.astype(str)) | \
                 set(test_df.id1.astype(str))  | set(test_df.id2.astype(str))
    print("Carico record necessari:", len(ids_needed))

    records = load_records(ids_needed)
    print("Record in RAM:", len(records))

    # --- Definizione campi Dedupe ---
    dedupe_fields = [
        dedupe.variables.String("make"),
        dedupe.variables.String("model"),
        dedupe.variables.Exact("year", has_missing=True),
        dedupe.variables.Price("price", has_missing=True),
        dedupe.variables.Price("mileage", has_missing=True),
        dedupe.variables.Exact("fuel_type", has_missing=True),
        dedupe.variables.Exact("transmission", has_missing=True),
        dedupe.variables.Exact("body_type", has_missing=True),
        dedupe.variables.Exact("drive", has_missing=True),
        dedupe.variables.Exact("condition", has_missing=True)
    ]

    linker = dedupe.RecordLink(dedupe_fields)

    # --- Training ---
    training_pairs = pairs_to_dedupe_format(train_df, records)
    print("Preparazione training Dedupe...")
    linker.prepare_training(records, training_pairs)
    linker.train()
    print("Training completato.")

    # --- Threshold validation ---
    y_val, p_val = score_pairs(linker, val_df, records)
    thr = best_threshold(y_val, p_val)
    print(f"Soglia ottimale su validation: {thr:.4f}")

    # --- Test evaluation ---
    y_test, p_test = score_pairs(linker, test_df, records)
    y_pred = (p_test>=thr).astype(int)

    print("\n=== RISULTATI TEST ===")
    print("Precision:", round(precision_score(y_test,y_pred),4))
    print("Recall:   ", round(recall_score(y_test,y_pred),4))
    print("F1:       ", round(f1_score(y_test,y_pred),4))
    print("Test pairs:", len(y_test))

if __name__=="__main__":
    main()'''
