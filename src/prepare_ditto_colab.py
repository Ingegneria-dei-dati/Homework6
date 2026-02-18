import os
import json
import pandas as pd
from itertools import combinations
import time
import gc

# ================= CONFIG =================
# Percorsi di Input (Drive)
PATH_SU_DRIVE = "/content/drive/MyDrive/dataset"
DATASET_PATH = os.path.join(PATH_SU_DRIVE, "dataset_for_training.csv")
SPLITS_DIR   = os.path.join(PATH_SU_DRIVE, "splits")
BLOCKS_DIR   = os.path.join(PATH_SU_DRIVE, "blocks")

# Percorsi di Output (Drive - Permanenti)
DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/FAIR_DATA_OUTPUT'
OUT_B1 = os.path.join(DRIVE_OUTPUT_DIR, "cars_B1")
OUT_B2 = os.path.join(DRIVE_OUTPUT_DIR, "cars_B2")

FIELDS = ["make", "model", "year", "price", "mileage", "fuel_type", 
          "transmission", "body_type", "drive", "color", 
          "engine_cylinders", "latitude", "longitude"]

CHUNK_SIZE = 50_000

def build_recs_map():
    print(" Serializzazione streaming dei record...")
    recs = {}
    # Usiamo 'id' o 'listing_id' a seconda del tuo CSV
    for chunk in pd.read_csv(DATASET_PATH, usecols=["id"] + FIELDS, chunksize=CHUNK_SIZE, low_memory=False):
        chunk["id"] = chunk["id"].astype(str)
        cols = ["COL " + f + " VAL " + chunk[f].fillna("").astype(str).str.lower().str.strip() for f in FIELDS]
        serialized = pd.concat(cols, axis=1).agg(" ".join, axis=1)
        recs.update(dict(zip(chunk["id"], serialized)))
    return recs

def main():
    t0 = time.time()
    recs = build_recs_map()
    
    gt = json.load(open(os.path.join(PATH_SU_DRIVE, "ground_truth_map.json")))
    id_to_vin = {str(i): vin for vin, ids in gt.items() for i in ids}
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train_pairs.csv"))

    for strat, out_dir in [("B1", OUT_B1), ("B2", OUT_B2)]:
        print(f"\n Strategia {strat} -> {out_dir}")
        # Scrittura TRAIN
        with open(os.path.join(out_dir, "train.txt"), "w") as f:
            for r in train_df.itertuples(index=False):
                id1, id2 = str(r.id1), str(r.id2)
                if id1 in recs and id2 in recs:
                    f.write(f"{recs[id1]}\t{recs[id2]}\t{int(r.label)}\n")
        
        # Scrittura VAL/TEST in streaming
        for split in ["val", "test"]:
            blocks = json.load(open(os.path.join(BLOCKS_DIR, f"blocking_{strat}_{split}.json")))
            out_file = os.path.join(out_dir, "valid.txt" if split=="val" else "test.txt")
            with open(out_file, "w") as f:
                for ids in blocks.values():
                    valid_ids = [str(i) for i in ids if str(i) in recs]
                    if len(valid_ids) < 2: continue
                    for a, b in combinations(valid_ids, 2):
                        va, vb = id_to_vin.get(a), id_to_vin.get(b)
                        f.write(f"{recs[a]}\t{recs[b]}\t{1 if (va and vb and va==vb) else 0}\n")
            del blocks
            gc.collect()

    print(f" Completato in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
