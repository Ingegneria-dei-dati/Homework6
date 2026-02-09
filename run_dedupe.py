import pandas as pd
import dedupe
import numpy as np
import json
import os
import random
from itertools import combinations, islice
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "dataset/dataset_for_training.csv"
GT_PATH = "dataset/ground_truth_map.json"
TRAIN_PAIRS_PATH = "dataset/splits/train_pairs.csv"
BLOCKS_DIR = "dataset/blocks"

CHUNKSIZE = 100_000
MAX_PAIRS_PER_BLOCK = 30_000
GLOBAL_PAIR_LIMIT = 300_000
BATCH_SIZE = 20_000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# UTILS
# ============================================================

def clean_val(val, dtype):
    if pd.isna(val) or str(val).lower() in ["nan", "none", "", "0", "0.0"]:
        return None
    try:
        return dtype(val)
    except:
        return None


def limited_pairs(ids, max_pairs):
    return islice(combinations(ids, 2), max_pairs)


def cheap_filter(r1, r2):
    """Filtro deterministico pre-dedupe (fondamentale)"""
    if r1["brand"] != r2["brand"]:
        return False

    if r1["year"] and r2["year"]:
        if abs(r1["year"] - r2["year"]) > 1:
            return False

    if r1["price"] and r2["price"]:
        if abs(r1["price"] - r2["price"]) / max(r1["price"], r2["price"]) > 0.5:
            return False

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # FASE 1: ID NECESSARI
    # --------------------------------------------------------
    print("Fase 1: Raccolta ID necessari...")

    ids_to_load = set()
    train_df = pd.read_csv(TRAIN_PAIRS_PATH)

    ids_to_load.update(train_df["id1"].astype(str))
    ids_to_load.update(train_df["id2"].astype(str))

    for fname in ["blocking_B1.json", "blocking_B2.json"]:
        with open(os.path.join(BLOCKS_DIR, fname)) as f:
            blocks = json.load(f)
            for ids in blocks.values():
                ids_to_load.update(map(str, ids))

    print(f"ID totali da caricare: {len(ids_to_load)}")

    # --------------------------------------------------------
    # FASE 2: CARICAMENTO DATI
    # --------------------------------------------------------
    print("\nFase 2: Caricamento dataset...")

    cols = ["id", "brand", "model_norm_full", "year", "price", "mileage"]
    data_d = {}

    for chunk in pd.read_csv(DATASET_PATH, usecols=cols, chunksize=CHUNKSIZE):
        chunk["id_str"] = chunk["id"].astype(str)
        filtered = chunk[chunk["id_str"].isin(ids_to_load)]

        for _, row in filtered.iterrows():
            data_d[row["id"]] = {
                "brand": str(row["brand"]).strip().lower() if pd.notnull(row["brand"]) else None,
                "model_norm_full": str(row["model_norm_full"]).strip().lower() if pd.notnull(row["model_norm_full"]) else None,
                "year": clean_val(row["year"], int),
                "price": clean_val(row["price"], float),
                "mileage": clean_val(row["mileage"], float),
            }

    print(f"Record in RAM: {len(data_d)}")

    # --------------------------------------------------------
    # FASE 3: TRAINING DEDUPE (BILANCIATO)
    # --------------------------------------------------------
    print("\nFase 3: Training dedupe...")

    fields = [
        dedupe.variables.String("brand"),
        dedupe.variables.String("model_norm_full"),
        dedupe.variables.Price("year", has_missing=True),
        dedupe.variables.Price("price", has_missing=True),
        dedupe.variables.Price("mileage", has_missing=True),
    ]

    linker = dedupe.RecordLink(fields)

    matches, distinct = [], []

    for _, row in train_df.iterrows():
        id1, id2 = row["id1"], row["id2"]
        if id1 in data_d and id2 in data_d:
            pair = (data_d[id1], data_d[id2])
            if row["label"] == 1:
                matches.append(pair)
            else:
                distinct.append(pair)

    # ⚠️ BILANCIAMENTO (CRITICO)
    n = min(len(matches), len(distinct))
    matches = random.sample(matches, n)
    distinct = random.sample(distinct, n)

    X = linker.data_model.distances(matches + distinct)
    y = np.array([1] * n + [0] * n)

    linker.classifier.fit(X, y)
    print(f"Training completato su {2*n} coppie bilanciate")

    # --------------------------------------------------------
    # FASE 4: GROUND TRUTH
    # --------------------------------------------------------
    with open(GT_PATH) as f:
        gt_map = json.load(f)

    id_to_vin = {str(i): vin for vin, ids in gt_map.items() for i in ids}

    # --------------------------------------------------------
    # FASE 5: VALUTAZIONE CORRETTA
    # --------------------------------------------------------
    results = []

    for block_file in ["blocking_B2.json", "blocking_B1.json"]:
        label = block_file.replace(".json", "").replace("blocking_", "")
        print(f"\n>>> Valutazione {label}")

        with open(os.path.join(BLOCKS_DIR, block_file)) as f:
            blocks = json.load(f)

        y_true, probs_all = [], []
        batch, batch_true = [], []
        total = 0

        for ids in blocks.values():
            valid = [i for i in ids if i in data_d]
            if len(valid) < 2:
                continue

            for id1, id2 in limited_pairs(valid, MAX_PAIRS_PER_BLOCK):

                r1, r2 = data_d[id1], data_d[id2]
                if not cheap_filter(r1, r2):
                    continue

                batch.append(((id1, r1), (id2, r2)))

                v1 = id_to_vin.get(str(id1))
                v2 = id_to_vin.get(str(id2))
                batch_true.append(1 if (v1 == v2 and v1 is not None) else 0)

                total += 1

                if len(batch) >= BATCH_SIZE:
                    scored = linker.score(batch)
                    probs_all.extend(scored["score"].astype(float))
                    y_true.extend(batch_true)
                    batch, batch_true = [], []

                if total >= GLOBAL_PAIR_LIMIT:
                    break

            if total >= GLOBAL_PAIR_LIMIT:
                break

        if batch:
            scored = linker.score(batch)
            probs_all.extend(scored["score"].astype(float))
            y_true.extend(batch_true)

        # ----------------------------------------------------
        # CALIBRAZIONE SOGLIA
        # ----------------------------------------------------
        prec, rec, thr = precision_recall_curve(y_true, probs_all)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        best = np.argmax(f1)
        best_thr = thr[best]

        y_pred = (np.array(probs_all) >= best_thr).astype(int)

        results.append({
            "Strategy": f"Dedupe-{label}",
            "Threshold": round(best_thr, 4),
            "Precision": round(precision_score(y_true, y_pred), 4),
            "Recall": round(recall_score(y_true, y_pred), 4),
            "F1": round(f1_score(y_true, y_pred), 4),
            "PairsEvaluated": total
        })

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("REPORT FINALE DEDUPE (CORRETTO)")
    print("=" * 80)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()




'''import pandas as pd
import dedupe
import numpy as np
import json
import os
import time
from sklearn.metrics import precision_score, recall_score, f1_score

DATASET_PATH = "dataset/dataset_for_training.csv"
GT_PATH = "dataset/ground_truth_map.json"
TRAIN_PAIRS_PATH = "dataset/splits/train_pairs.csv"
BLOCKS_DIR = "dataset/blocks"

def clean_val(val, dtype):
    if pd.isna(val) or str(val).lower() in ["nan", "none", "", "0", "0.0"]:
        return None
    try: return dtype(val)
    except: return None

def main():
    print("Fase 1: Caricamento ID necessari...")
    ids_to_load = set()
    train_df = pd.read_csv(TRAIN_PAIRS_PATH)
    ids_to_load.update(train_df['id1'].astype(str))
    ids_to_load.update(train_df['id2'].astype(str))
    
    for b_file in ["blocking_B1.json", "blocking_B2.json"]:
        path = os.path.join(BLOCKS_DIR, b_file)
        if os.path.exists(path):
            with open(path) as f:
                blocks = json.load(f)
                for block_ids in blocks.values():
                    ids_to_load.update(map(str, block_ids))

    print("Fase 2: Caricamento dati (Chunking)...")
    cols = ['id', 'brand', 'model_norm_full', 'year', 'price', 'mileage']
    data_d = {}
    for chunk in pd.read_csv(DATASET_PATH, low_memory=False, usecols=cols, chunksize=100000):
        chunk['id_str'] = chunk['id'].astype(str)
        filtered = chunk[chunk['id_str'].isin(ids_to_load)]
        for _, row in filtered.iterrows():
            data_d[row['id']] = {
                'brand': str(row['brand']).strip().lower() if pd.notnull(row['brand']) else None,
                'model_norm_full': str(row['model_norm_full']).strip().lower() if pd.notnull(row['model_norm_full']) else None,
                'year': clean_val(row['year'], int),
                'price': clean_val(row['price'], float),
                'mileage': clean_val(row['mileage'], float)
            }
    
    print(f"Record in RAM: {len(data_d)}")

    print("Fase 3: Training classificatore...")
    fields = [
        dedupe.variables.String("brand"),
        dedupe.variables.String("model_norm_full"),
        dedupe.variables.Price("year", has_missing=True),
        dedupe.variables.Price("price", has_missing=True),
        dedupe.variables.Price("mileage", has_missing=True)
    ]
    linker = dedupe.RecordLink(fields)

    matches, distinct = [], []
    for _, row in train_df.iterrows():
        id1, id2 = row['id1'], row['id2']
        if id1 in data_d and id2 in data_d:
            pair = (data_d[id1], data_d[id2])
            if row['label'] == 1: matches.append(pair)
            else: distinct.append(pair)

    X_train = linker.data_model.distances(matches + distinct)
    y_train = np.array([1] * len(matches) + [0] * len(distinct))
    linker.classifier.fit(X_train, y_train)
    print("Training completato.")

    with open(GT_PATH) as f:
        gt_map = json.load(f)
    id_to_vin = {str(i): vin for vin, ids in gt_map.items() for i in ids}

    results = []
    # Valutiamo prima B2 perché è più veloce, poi B1
    for block_name in ["blocking_B2.json", "blocking_B1.json"]:
        label = block_name.replace(".json", "").replace("blocking_", "")
        print(f"\n>>> Valutazione Strategia: {label}")
        
        threshold = 0.33 if "B2" in label else 0.1225
        y_true, y_pred = [], []
        
        with open(os.path.join(BLOCKS_DIR, block_name)) as f:
            blocks = json.load(f)

        batch_data, batch_true = [], []
        total_count = 0
        
        for ids in blocks.values():
            valid = [i for i in ids if i in data_d]
            if len(valid) > 1:
                for i in range(len(valid)):
                    for j in range(i + 1, len(valid)):
                        id1, id2 = valid[i], valid[j]
                        batch_data.append(((id1, data_d[id1]), (id2, data_d[id2])))
                        v1, v2 = id_to_vin.get(str(id1)), id_to_vin.get(str(id2))
                        batch_true.append(1 if (v1 == v2 and v1 is not None) else 0)
                        
                        total_count += 1
                        if len(batch_data) >= 5000:
                            scored = linker.score(batch_data)
                            probs = scored['score'].astype(float)
                            y_pred.extend((probs >= threshold).astype(int))
                            y_true.extend(batch_true)
                            batch_data, batch_true = [], []
                            if total_count % 10000 == 0:
                                print(f"   Analizzate {total_count} coppie...")

        if batch_data:
            scored = linker.score(batch_data)
            probs = scored['score'].astype(float)
            y_pred.extend((probs >= threshold).astype(int))
            y_true.extend(batch_true)

        results.append({
            "Strategy": f"Dedupe-{label}",
            "Precision": f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
            "Recall": f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
            "F1": f"{f1_score(y_true, y_pred, zero_division=0):.4f}"
        })

    print("\n" + "="*60)
    print("REPORT FINALE DEDUPE")
    print("="*60)
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()'''