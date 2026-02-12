import os
import json
import pandas as pd
from itertools import combinations
import time

# ===== CONFIG =====
PATH_SU_DRIVE = "/content/drive/MyDrive/dataset"

DATASET_PATH = os.path.join(PATH_SU_DRIVE, "dataset_for_training.csv")
SPLITS_DIR   = os.path.join(PATH_SU_DRIVE, "splits")
BLOCKS_DIR   = os.path.join(PATH_SU_DRIVE, "blocks")

FAIR_ROOT = "/content/FAIR-DA4ER"
OUT_B1 = os.path.join(FAIR_ROOT, "data", "cars_B1")
OUT_B2 = os.path.join(FAIR_ROOT, "data", "cars_B2")

FIELDS = [
    "make", "model", "year", "price", "mileage", "fuel_type", 
    "transmission", "body_type", "drive", "color", 
    "engine_cylinders", "latitude", "longitude", "description"
]

CHUNK_SIZE = 50_000  # per serializzazione

# ===== UTILITIES =====
def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_recs_map():
    """Serializza i record in stringhe Ditto a chunk per non esplodere RAM"""
    print("🚀 Serializzazione streaming...")
    recs = {}
    for chunk in pd.read_csv(DATASET_PATH, usecols=["id"] + FIELDS,
                             chunksize=CHUNK_SIZE, low_memory=False):
        chunk["id"] = chunk["id"].astype(str)
        cols = []
        for f in FIELDS:
            if f in chunk.columns:
                cols.append("COL " + f + " VAL " +
                            chunk[f].fillna("").astype(str).str.lower().str.strip())
        serialized = pd.concat(cols, axis=1).agg(" ".join, axis=1)
        recs.update(dict(zip(chunk["id"], serialized)))
        print(f"  ✔ chunk caricato ({len(recs)} record totali)")
    return recs

def write_pairs_from_blocks(blocks, recs, id_to_vin, out_path, log_every=100_000):
    """Scrive coppie dai blocchi e stampa avanzamento ogni N coppie"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    count = 0
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        for ids in blocks.values():
            valid_ids = [str(i) for i in ids if str(i) in recs]
            if len(valid_ids) < 2:
                continue

            for a, b in combinations(valid_ids, 2):
                va, vb = id_to_vin.get(a), id_to_vin.get(b)
                label = 1 if (va and vb and va == vb) else 0
                f.write(f"{recs[a]}\t{recs[b]}\t{label}\n")
                count += 1

                if count % log_every == 0:
                    elapsed = time.time() - start_time
                    print(f"  ⚡ {count} coppie generate ({elapsed:.1f}s)")

    print(f"  ✅ {os.path.basename(out_path)}: {count} coppie totali")

# ===== MAIN =====
def main():
    t0 = time.time()

    print("🚀 1. Build record map")
    recs = build_recs_map()

    print("🚀 2. Load ground truth")
    gt = load_json(os.path.join(PATH_SU_DRIVE, "ground_truth_map.json"))
    id_to_vin = {str(i): vin for vin, ids in gt.items() for i in ids}

    print("🚀 3. Load train pairs")
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train_pairs.csv"))

    for strat, out_dir in [("B1", OUT_B1), ("B2", OUT_B2)]:
        print(f"\n📦 Strategia {strat}")

        # TRAIN
        os.makedirs(out_dir, exist_ok=True)
        train_path = os.path.join(out_dir, "train.txt")
        with open(train_path, "w", encoding="utf-8") as f:
            for r in train_df.itertuples(index=False):
                id1, id2 = str(r.id1), str(r.id2)
                if id1 in recs and id2 in recs:
                    f.write(f"{recs[id1]}\t{recs[id2]}\t{int(r.label)}\n")
        print(f"  ✅ train.txt scritto")

        # VAL / TEST
        for split in ["val", "test"]:
            block_file = os.path.join(BLOCKS_DIR, f"blocking_{strat}_{split}.json")
            blocks = load_json(block_file)
            out_file = os.path.join(out_dir, "valid.txt" if split=="val" else "test.txt")
            write_pairs_from_blocks(blocks, recs, id_to_vin, out_file)

    print(f"\n✨ Completato in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
