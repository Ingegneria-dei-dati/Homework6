import os
import json
from itertools import combinations
import pandas as pd

# ======================================================
# CONFIGURAZIONE GOOGLE DRIVE
# ======================================================
DRIVE_DATASET_ROOT = "/content/drive/MyDrive/Homework6_data/dataset"

DATASET_PATH = os.path.join(DRIVE_DATASET_ROOT, "dataset_for_training.csv")
SPLITS_DIR   = os.path.join(DRIVE_DATASET_ROOT, "splits")
BLOCKS_DIR   = os.path.join(DRIVE_DATASET_ROOT, "blocks")
GT_PATH      = os.path.join(DRIVE_DATASET_ROOT, "ground_truth_map.json")

# ======================================================
# OUTPUT FAIR-DA4ER (repo clonato su Colab)
# ======================================================
FAIR_ROOT = "/content/FAIR-DA4ER"
OUT_BASE  = os.path.join(FAIR_ROOT, "data", "cars_base")
OUT_B1    = os.path.join(FAIR_ROOT, "data", "cars_B1")
OUT_B2    = os.path.join(FAIR_ROOT, "data", "cars_B2")

FIELDS = [
    "make", "model_norm_full", "year", "price", "mileage",
    "fuel_type", "transmission", "body_type", "drive", "color"
]

# ======================================================
# UTILS
# ======================================================
def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if s in ("nan", "none", "null", "unknown"):
        return ""
    return s

def serialize_record(row):
    parts = []
    for f in FIELDS:
        v = clean(row.get(f, ""))
        parts.append(f"COL {f} VAL {v}")
    return " ".join(parts)

def load_id_to_vin(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_map = json.load(f)  # vin -> [ids]
    return {str(i): vin for vin, ids in gt_map.items() for i in ids}

def load_blocks(path):
    with open(path, "r", encoding="utf-8") as f:
        b = json.load(f)
    return {k: [str(x) for x in v] for k, v in b.items()}

def write_pairs_txt(df_pairs, recs, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for _, r in df_pairs.iterrows():
            id1, id2, lab = str(r["id1"]), str(r["id2"]), int(r["label"])
            if id1 not in recs or id2 not in recs:
                continue
            w.write(f"{recs[id1]}\t{recs[id2]}\t{lab}\n")
            n += 1
    print(f"[WRITE] {out_path}: {n} righe")

def write_blocked_pairs_txt(blocks, recs, id_to_vin, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for _, ids in blocks.items():
            ids = [i for i in ids if i in recs]
            if len(ids) < 2:
                continue
            for a, b in combinations(ids, 2):
                va, vb = id_to_vin.get(a), id_to_vin.get(b)
                lab = 1 if (va is not None and va == vb) else 0
                w.write(f"{recs[a]}\t{recs[b]}\t{lab}\n")
                n += 1
    print(f"[WRITE] {out_path}: {n} righe (NO sampling)")

# ======================================================
# MAIN
# ======================================================
def main():
    print(" Carico dataset da Google Drive...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df["id"] = df["id"].astype(str)

    print(" Serializzazione record...")
    recs = {row["id"]: serialize_record(row) for _, row in df.iterrows()}

    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train_pairs.csv"))
    val_df   = pd.read_csv(os.path.join(SPLITS_DIR, "val_pairs.csv"))
    test_df  = pd.read_csv(os.path.join(SPLITS_DIR, "test_pairs.csv"))

    # BASE
    write_pairs_txt(train_df, recs, os.path.join(OUT_BASE, "train.txt"))
    write_pairs_txt(val_df,   recs, os.path.join(OUT_BASE, "valid.txt"))
    write_pairs_txt(test_df,  recs, os.path.join(OUT_BASE, "test.txt"))

    id_to_vin = load_id_to_vin(GT_PATH)

    # B1
    blocks_val  = load_blocks(os.path.join(BLOCKS_DIR, "blocking_B1_val.json"))
    blocks_test = load_blocks(os.path.join(BLOCKS_DIR, "blocking_B1_test.json"))
    write_pairs_txt(train_df, recs, os.path.join(OUT_B1, "train.txt"))
    write_blocked_pairs_txt(blocks_val,  recs, id_to_vin, os.path.join(OUT_B1, "valid.txt"))
    write_blocked_pairs_txt(blocks_test, recs, id_to_vin, os.path.join(OUT_B1, "test.txt"))

    # B2
    blocks_val  = load_blocks(os.path.join(BLOCKS_DIR, "blocking_B2_val.json"))
    blocks_test = load_blocks(os.path.join(BLOCKS_DIR, "blocking_B2_test.json"))
    write_pairs_txt(train_df, recs, os.path.join(OUT_B2, "train.txt"))
    write_blocked_pairs_txt(blocks_val,  recs, id_to_vin, os.path.join(OUT_B2, "valid.txt"))
    write_blocked_pairs_txt(blocks_test, recs, id_to_vin, os.path.join(OUT_B2, "test.txt"))

    print("✅ Preparazione dati Ditto completata")

if __name__ == "__main__":
    main()
