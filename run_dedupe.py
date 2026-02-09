


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


import os
import json
import time
from itertools import combinations
from typing import Dict, Any, Iterable, Tuple, List

import numpy as np
import pandas as pd
import dedupe
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

# ============================================================
# CONFIG (coerente col tuo repo)
# ============================================================

DATASET_PATH = os.path.join("dataset", "dataset_for_training.csv")
GT_PATH = os.path.join("dataset", "ground_truth_map.json")
SPLITS_DIR = os.path.join("dataset", "splits")
BLOCKS_DIR = os.path.join("dataset", "blocks")

# colonne che usiamo per Dedupe (devono esistere in dataset_for_training.csv)
DEDUPE_COLS = [
    "id",
    "make",
    "model_norm_full",   # nel tuo repo esiste già questa normalizzazione
    "year",
    "price",
    "mileage",
    "fuel_type",
    "transmission",
    "body_type",
    "drive",
    "color",
]

# batch per scoring (non è campionamento: è solo per non esplodere in RAM)
SCORE_BATCH_SIZE = 50_000


# ============================================================
# UTILS
# ============================================================

def clean_str(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in ("", "nan", "none", "null", "unknown"):
        return None
    return s

def clean_int(x):
    if pd.isna(x):
        return None
    try:
        return int(float(x))
    except:
        return None

def clean_float(x):
    if pd.isna(x):
        return None
    try:
        return float(x)
    except:
        return None

def load_id_to_vin(gt_path: str) -> Dict[str, str]:
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_map = json.load(f)
    # gt_map: vin -> [ids]
    return {str(i): vin for vin, ids in gt_map.items() for i in ids}

def iter_block_pairs(block_ids: List[str]) -> Iterable[Tuple[str, str]]:
    # tutte le combinazioni (NO sampling)
    for a, b in combinations(block_ids, 2):
        yield a, b

def load_blocks(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        blocks = json.load(f)
    # normalizziamo a string
    return {k: [str(x) for x in v] for k, v in blocks.items()}

def load_records_subset(dataset_path: str, ids_needed: set) -> Dict[str, Dict[str, Any]]:
    """
    Carica SOLO i record necessari (per training + validation/test + blocchi).
    Questo non è campionamento: è un filtro deterministico sugli ID che effettivamente servono.
    """
    df = pd.read_csv(dataset_path, usecols=[c for c in DEDUPE_COLS if c != "id"] + ["id"], low_memory=False)
    df["id"] = df["id"].astype(str)
    df = df[df["id"].isin(ids_needed)]

    data = {}
    for _, r in df.iterrows():
        data[r["id"]] = {
            "make": clean_str(r.get("make")),
            "model_norm_full": clean_str(r.get("model_norm_full")),
            "year": clean_int(r.get("year")),
            "price": clean_float(r.get("price")),
            "mileage": clean_float(r.get("mileage")),
            "fuel_type": clean_str(r.get("fuel_type")),
            "transmission": clean_str(r.get("transmission")),
            "body_type": clean_str(r.get("body_type")),
            "drive": clean_str(r.get("drive")),
            "color": clean_str(r.get("color")),
        }
    return data

def pairs_from_csv(pairs_path: str) -> pd.DataFrame:
    df = pd.read_csv(pairs_path)
    df["id1"] = df["id1"].astype(str)
    df["id2"] = df["id2"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


# ============================================================
# TRAIN DEDUPE (NO sampling) + class weights (per sbilanciamento)
# ============================================================

def train_dedupe_classifier(linker: dedupe.RecordLink,
                            data_d: Dict[str, Dict[str, Any]],
                            train_pairs_df: pd.DataFrame) -> None:
    matches = []
    distinct = []

    for _, row in train_pairs_df.iterrows():
        id1, id2, lab = row["id1"], row["id2"], row["label"]
        if id1 not in data_d or id2 not in data_d:
            continue
        pair = (data_d[id1], data_d[id2])
        if lab == 1:
            matches.append(pair)
        else:
            distinct.append(pair)

    if len(matches) == 0 or len(distinct) == 0:
        raise RuntimeError(
            f"Training impossibile: match={len(matches)} distinct={len(distinct)}. "
            f"Controlla la ground-truth e gli split."
        )

    X = linker.data_model.distances(matches + distinct)
    y = np.array([1] * len(matches) + [0] * len(distinct))

    # NO downsampling: pesi bilanciati
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)

    t0 = time.time()
    linker.classifier.fit(X, y, sample_weight=sample_weight)
    t1 = time.time()

    print(f"[TRAIN] coppie usate: {len(y)} (match={len(matches)} distinct={len(distinct)})")
    print(f"[TRAIN] tempo training classifier: {t1 - t0:.2f}s")


# ============================================================
# SCORE + THRESHOLD TUNING SU VALIDATION
# ============================================================

def score_blocks(linker: dedupe.RecordLink,
                 data_d: Dict[str, Dict[str, Any]],
                 blocks: Dict[str, List[str]],
                 id_to_vin: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Ritorna:
      y_true, y_score, inference_time_seconds
    """
    batch = []
    y_true = []
    y_score = []

    t0 = time.time()

    for _, ids in blocks.items():
        ids_valid = [i for i in ids if i in data_d]
        if len(ids_valid) < 2:
            continue

        for id1, id2 in iter_block_pairs(ids_valid):
            r1, r2 = data_d[id1], data_d[id2]

            # coppia nel formato che dedupe.score si aspetta:
            batch.append(((id1, r1), (id2, r2)))

            v1 = id_to_vin.get(id1)
            v2 = id_to_vin.get(id2)
            y_true.append(1 if (v1 is not None and v1 == v2) else 0)

            if len(batch) >= SCORE_BATCH_SIZE:
                scored = linker.score(batch)
                y_score.extend(scored["score"].astype(float).tolist())
                batch = []

    if batch:
        scored = linker.score(batch)
        y_score.extend(scored["score"].astype(float).tolist())

    t1 = time.time()
    return np.array(y_true, dtype=int), np.array(y_score, dtype=float), (t1 - t0)

def best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    # precision_recall_curve produce thr con len = len(prec)-1
    best_i = int(np.nanargmax(f1[:-1]))
    return float(thr[best_i])


def eval_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_score >= thr).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


# ============================================================
# MAIN: B1/B2 con split train/val/test
# ============================================================

def run(strategy: str):
    assert strategy in ("B1", "B2")

    train_pairs_path = os.path.join(SPLITS_DIR, "train_pairs.csv")
    val_pairs_path = os.path.join(SPLITS_DIR, "val_pairs.csv")
    test_pairs_path = os.path.join(SPLITS_DIR, "test_pairs.csv")

    blocks_train_path = os.path.join(BLOCKS_DIR, f"blocking_{strategy}_train.json")
    blocks_val_path = os.path.join(BLOCKS_DIR, f"blocking_{strategy}_val.json")
    blocks_test_path = os.path.join(BLOCKS_DIR, f"blocking_{strategy}_test.json")

    for p in [DATASET_PATH, GT_PATH, train_pairs_path, val_pairs_path, test_pairs_path,
              blocks_train_path, blocks_val_path, blocks_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Manca il file richiesto: {p}")

    train_df = pairs_from_csv(train_pairs_path)
    val_df = pairs_from_csv(val_pairs_path)
    test_df = pairs_from_csv(test_pairs_path)

    # (1) IDs necessari = tutti quelli che compaiono in train/val/test + quelli nei blocchi
    ids_needed = set(train_df["id1"]).union(train_df["id2"]) \
        .union(val_df["id1"]).union(val_df["id2"]) \
        .union(test_df["id1"]).union(test_df["id2"])

    blocks_train = load_blocks(blocks_train_path)
    blocks_val = load_blocks(blocks_val_path)
    blocks_test = load_blocks(blocks_test_path)

    for bl in (blocks_train, blocks_val, blocks_test):
        for ids in bl.values():
            ids_needed.update(ids)

    print(f"[{strategy}] ID necessari totali: {len(ids_needed)}")

    # (2) Carico record richiesti
    data_d = load_records_subset(DATASET_PATH, ids_needed)
    print(f"[{strategy}] Record caricati: {len(data_d)}")

    # (3) Ground truth VIN (serve SOLO per y_true di validation/test)
    id_to_vin = load_id_to_vin(GT_PATH)

    # (4) Definizione campi Dedupe
    # NB: Exact per year; Price per price/mileage (numerici); String per categorici
    fields = [
        dedupe.variables.String("make", has_missing=True),
        dedupe.variables.String("model_norm_full", has_missing=True),

        dedupe.variables.Exact("year", has_missing=True),
        dedupe.variables.Price("price", has_missing=True),
        dedupe.variables.Price("mileage", has_missing=True),

        dedupe.variables.String("fuel_type", has_missing=True),
        dedupe.variables.String("transmission", has_missing=True),
        dedupe.variables.String("body_type", has_missing=True),
        dedupe.variables.String("drive", has_missing=True),
        dedupe.variables.String("color", has_missing=True),
    ]

    linker = dedupe.RecordLink(fields)

    # (5) TRAIN (no sampling)
    train_dedupe_classifier(linker, data_d, train_df)

    # (6) Tune threshold su VALIDATION usando blocchi (NO sampling)
    y_true_val, y_score_val, t_inf_val = score_blocks(linker, data_d, blocks_val, id_to_vin)
    thr = best_threshold_by_f1(y_true_val, y_score_val)
    val_metrics = eval_at_threshold(y_true_val, y_score_val, thr)

    print(f"[{strategy}] VALID - pairs={len(y_true_val)} thr={thr:.4f} "
          f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} F1={val_metrics['f1']:.4f} "
          f"(inference_time={t_inf_val:.2f}s)")

    # (7) TEST con la soglia trovata
    y_true_test, y_score_test, t_inf_test = score_blocks(linker, data_d, blocks_test, id_to_vin)
    test_metrics = eval_at_threshold(y_true_test, y_score_test, thr)

    print(f"[{strategy}] TEST  - pairs={len(y_true_test)} thr={thr:.4f} "
          f"P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} F1={test_metrics['f1']:.4f} "
          f"(inference_time={t_inf_test:.2f}s)")

    return {
        "strategy": strategy,
        "threshold": thr,
        "val": {**val_metrics, "pairs": int(len(y_true_val)), "inference_time_s": float(t_inf_val)},
        "test": {**test_metrics, "pairs": int(len(y_true_test)), "inference_time_s": float(t_inf_test)},
    }


def main():
    t0 = time.time()
    r_b1 = run("B1")
    r_b2 = run("B2")
    t1 = time.time()

    print("\n" + "=" * 80)
    print("REPORT FINALE (DEDUPE, NO SAMPLING)")
    print("=" * 80)
    df = pd.DataFrame([
        {
            "Pipeline": "B1-dedupe",
            "Thr": r_b1["threshold"],
            "Val_F1": r_b1["val"]["f1"],
            "Test_F1": r_b1["test"]["f1"],
            "Test_P": r_b1["test"]["precision"],
            "Test_R": r_b1["test"]["recall"],
            "Test_pairs": r_b1["test"]["pairs"],
            "Test_infer_s": r_b1["test"]["inference_time_s"],
        },
        {
            "Pipeline": "B2-dedupe",
            "Thr": r_b2["threshold"],
            "Val_F1": r_b2["val"]["f1"],
            "Test_F1": r_b2["test"]["f1"],
            "Test_P": r_b2["test"]["precision"],
            "Test_R": r_b2["test"]["recall"],
            "Test_pairs": r_b2["test"]["pairs"],
            "Test_infer_s": r_b2["test"]["inference_time_s"],
        }
    ])
    print(df.to_string(index=False))
    print(f"\nTempo totale: {t1 - t0:.2f}s")


if __name__ == "__main__":
    main()
