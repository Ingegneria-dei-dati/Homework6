"""
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
    "model",   # nel tuo repo esiste già questa normalizzazione
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
    '''
    Carica SOLO i record necessari (per training + validation/test + blocchi).
    Questo non è campionamento: è un filtro deterministico sugli ID che effettivamente servono.
    '''
    df = pd.read_csv(dataset_path, usecols=[c for c in DEDUPE_COLS if c != "id"] + ["id"], low_memory=False)
    df["id"] = df["id"].astype(str)
    df = df[df["id"].isin(ids_needed)]

    data = {}
    for _, r in df.iterrows():
        data[r["id"]] = {
            "make": clean_str(r.get("make")),
            "model": clean_str(r.get("model")),
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
    ''''''
    Ritorna:
      y_true, y_score, inference_time_seconds
    '''
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
        dedupe.variables.String("model", has_missing=True),

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
"""




'''import os
import json
import time
from itertools import combinations
from typing import Dict, Any, Iterable, Tuple, List

import numpy as np
import pandas as pd
import dedupe
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = os.path.join("dataset", "dataset_for_training.csv")
GT_PATH = os.path.join("dataset", "ground_truth_map.json")
SPLITS_DIR = os.path.join("dataset", "splits")
BLOCKS_DIR = os.path.join("dataset", "blocks")

DEDUPE_COLS = [
    "id",
    "make",
    "model",
    "year",
    "price",
    "mileage",
    "fuel_type",
    "transmission",
    "body_type",
    "drive",
    "color",
]

SCORE_BATCH_SIZE = 50_000

# sampling SOLO per prepare_training (workflow classico)
PREPARE_TRAINING_SAMPLE_SIZE = 20_000
BLOCKED_PROPORTION = 0.5  # tipico in dedupe; puoi aumentare a 0.5 se vuoi più focus su blocchi


# ============================================================
# CLEANING
# ============================================================

def clean_str(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in ("", "nan", "none", "null", "unknown", "<na>"):
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


# ============================================================
# IO UTILS
# ============================================================

def load_id_to_vin(gt_path: str) -> Dict[str, str]:
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_map = json.load(f)  # vin -> [ids]
    return {str(i): vin for vin, ids in gt_map.items() for i in ids}

def load_blocks(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        blocks = json.load(f)
    return {k: [str(x) for x in v] for k, v in blocks.items()}

def iter_block_pairs(block_ids: List[str]) -> Iterable[Tuple[str, str]]:
    for a, b in combinations(block_ids, 2):
        yield a, b

def pairs_from_csv(pairs_path: str) -> pd.DataFrame:
    df = pd.read_csv(pairs_path)
    df["id1"] = df["id1"].astype(str)
    df["id2"] = df["id2"].astype(str)
    df["label"] = df["label"].astype(int)
    return df

def load_records_subset(dataset_path: str, ids_needed: set) -> Dict[str, Dict[str, Any]]:
    """
    Carica SOLO i record necessari (deterministico, non sampling).
    """
    usecols = [c for c in DEDUPE_COLS if c != "id"] + ["id"]
    df = pd.read_csv(dataset_path, usecols=usecols, low_memory=False)
    df["id"] = df["id"].astype(str)
    df = df[df["id"].isin(ids_needed)]

    data = {}
    # itertuples è molto più veloce di iterrows
    for r in df.itertuples(index=False):
        # accesso per nome: r.id, r.make ...
        rid = str(getattr(r, "id"))
        data[rid] = {
            "make": clean_str(getattr(r, "make", None)),
            "model": clean_str(getattr(r, "model", None)),
            "year": clean_int(getattr(r, "year", None)),
            "price": clean_float(getattr(r, "price", None)),
            "mileage": clean_float(getattr(r, "mileage", None)),
            "fuel_type": clean_str(getattr(r, "fuel_type", None)),
            "transmission": clean_str(getattr(r, "transmission", None)),
            "body_type": clean_str(getattr(r, "body_type", None)),
            "drive": clean_str(getattr(r, "drive", None)),
            "color": clean_str(getattr(r, "color", None)),
        }
    return data


# ============================================================
# DEDUPE CLASSIC TRAINING
# ============================================================

def build_fields():
    # “classico” Dedupe: definisci le variabili + has_missing=True
    return [
        dedupe.variables.String("make", has_missing=True),
        dedupe.variables.String("model", has_missing=True),

        dedupe.variables.Exact("year", has_missing=True),
        dedupe.variables.Price("price", has_missing=True),
        dedupe.variables.Price("mileage", has_missing=True),

        dedupe.variables.String("fuel_type", has_missing=True),
        dedupe.variables.String("transmission", has_missing=True),
        dedupe.variables.String("body_type", has_missing=True),
        dedupe.variables.String("drive", has_missing=True),
        dedupe.variables.String("color", has_missing=True),
    ]

def build_labeled_pairs(train_pairs_df: pd.DataFrame,
                        data_d: Dict[str, Dict[str, Any]]) -> Tuple[List[Tuple[dict, dict]], List[Tuple[dict, dict]]]:
    matches = []
    distinct = []
    skipped = 0

    for row in train_pairs_df.itertuples(index=False):
        id1, id2, lab = row.id1, row.id2, row.label
        if id1 not in data_d or id2 not in data_d:
            skipped += 1
            continue
        pair = (data_d[id1], data_d[id2])
        if lab == 1:
            matches.append(pair)
        else:
            distinct.append(pair)

    if skipped:
        print(f"[TRAIN] ⚠️ coppie saltate (ID mancanti nei record): {skipped}")

    if len(matches) == 0 or len(distinct) == 0:
        raise RuntimeError(f"Training impossibile: match={len(matches)} distinct={len(distinct)}")

    return matches, distinct

def train_dedupe_classic(data_d: Dict[str, Dict[str, Any]],
                         train_pairs_df: pd.DataFrame) -> Tuple[dedupe.Dedupe, float]:
    """
    Workflow classico:
      - Dedupe(fields)
      - mark_pairs(match/distinct)
      - prepare_training(sample_size=..., blocked_proportion=...)
      - train()
    """
    fields = build_fields()
    deduper = dedupe.Dedupe(fields, num_cores=None)

    matches, distinct = build_labeled_pairs(train_pairs_df, data_d)
    print(f"[TRAIN] coppie etichettate: {len(matches)+len(distinct)} (match={len(matches)} distinct={len(distinct)})")

    # inserisco esempi supervisionati (qui NON campioniamo le coppie)
    deduper.mark_pairs({"match": matches, "distinct": distinct})

    # sampling solo per stimare blocchi / attive learning internamente
    ss = min(PREPARE_TRAINING_SAMPLE_SIZE, len(data_d))
    print(f"[TRAIN] prepare_training: sample_size={ss}, blocked_proportion={BLOCKED_PROPORTION}")

    t0 = time.time()
    deduper.prepare_training(data_d, sample_size=ss, blocked_proportion=BLOCKED_PROPORTION)
    deduper.train()
    t1 = time.time()

    return deduper, (t1 - t0)


# ============================================================
# SCORING (pairwise) usando classifier interno dopo train()
# ============================================================

def score_blocks_pairwise(deduper: dedupe.Dedupe,
                          data_d: Dict[str, Dict[str, Any]],
                          blocks: Dict[str, List[str]],
                          id_to_vin: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Dedupe “classico” non nasce per restituire score su coppie candidate,
    ma dopo train() abbiamo:
      - deduper.data_model.distances(pairs)
      - deduper.classifier.predict_proba(distances)[:,1]
    Quindi valutiamo B1/B2 come nel tuo script.
    """
    y_true = []
    y_score = []
    pair_batch = []

    t0 = time.time()

    def flush():
        nonlocal pair_batch, y_score
        if not pair_batch:
            return
        X = deduper.data_model.distances(pair_batch)
        proba = deduper.classifier.predict_proba(X)[:, 1]
        y_score.extend(proba.astype(float).tolist())
        pair_batch = []

    for ids in blocks.values():
        ids_valid = [i for i in ids if i in data_d]
        if len(ids_valid) < 2:
            continue

        for id1, id2 in iter_block_pairs(ids_valid):
            pair_batch.append((data_d[id1], data_d[id2]))

            v1 = id_to_vin.get(id1)
            v2 = id_to_vin.get(id2)
            y_true.append(1 if (v1 is not None and v1 == v2) else 0)

            if len(pair_batch) >= SCORE_BATCH_SIZE:
                flush()

    flush()
    t1 = time.time()

    return np.array(y_true, dtype=int), np.array(y_score, dtype=float), (t1 - t0)


# ============================================================
# THRESHOLD + METRICS
# ============================================================

def best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
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
# MAIN: B1/B2
# ============================================================

def run(strategy: str):
    assert strategy in ("B1", "B2")

    train_pairs_path = os.path.join(SPLITS_DIR, "train_pairs.csv")
    val_pairs_path   = os.path.join(SPLITS_DIR, "val_pairs.csv")
    test_pairs_path  = os.path.join(SPLITS_DIR, "test_pairs.csv")

    blocks_train_path = os.path.join(BLOCKS_DIR, f"blocking_{strategy}_train.json")
    blocks_val_path   = os.path.join(BLOCKS_DIR, f"blocking_{strategy}_val.json")
    blocks_test_path  = os.path.join(BLOCKS_DIR, f"blocking_{strategy}_test.json")

    for p in [DATASET_PATH, GT_PATH, train_pairs_path, val_pairs_path, test_pairs_path,
              blocks_train_path, blocks_val_path, blocks_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Manca il file richiesto: {p}")

    train_df = pairs_from_csv(train_pairs_path)
    val_df   = pairs_from_csv(val_pairs_path)
    test_df  = pairs_from_csv(test_pairs_path)

    # IDs necessari: split + blocchi
    ids_needed = set(train_df["id1"]).union(train_df["id2"]) \
        .union(val_df["id1"]).union(val_df["id2"]) \
        .union(test_df["id1"]).union(test_df["id2"])

    blocks_train = load_blocks(blocks_train_path)
    blocks_val   = load_blocks(blocks_val_path)
    blocks_test  = load_blocks(blocks_test_path)

    for bl in (blocks_train, blocks_val, blocks_test):
        for ids in bl.values():
            ids_needed.update(ids)

    print(f"[{strategy}] ID necessari totali: {len(ids_needed)}")

    data_d = load_records_subset(DATASET_PATH, ids_needed)
    print(f"[{strategy}] Record caricati: {len(data_d)}")

    id_to_vin = load_id_to_vin(GT_PATH)

    # TRAIN classico
    t0 = time.time()
    deduper, train_time = train_dedupe_classic(data_d, train_df)
    t1 = time.time()
    print(f"[{strategy}] tempo training (prepare_training+train): {train_time:.2f}s (totale wrapper: {t1-t0:.2f}s)")

    # VALID
    y_true_val, y_score_val, t_inf_val = score_blocks_pairwise(deduper, data_d, blocks_val, id_to_vin)
    thr = best_threshold_by_f1(y_true_val, y_score_val)
    val_metrics = eval_at_threshold(y_true_val, y_score_val, thr)

    print(f"[{strategy}] VALID - pairs={len(y_true_val)} thr={thr:.4f} "
          f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} F1={val_metrics['f1']:.4f} "
          f"(inference_time={t_inf_val:.2f}s)")

    # TEST
    y_true_test, y_score_test, t_inf_test = score_blocks_pairwise(deduper, data_d, blocks_test, id_to_vin)
    test_metrics = eval_at_threshold(y_true_test, y_score_test, thr)

    print(f"[{strategy}] TEST  - pairs={len(y_true_test)} thr={thr:.4f} "
          f"P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} F1={test_metrics['f1']:.4f} "
          f"(inference_time={t_inf_test:.2f}s)")

    return {
        "strategy": strategy,
        "threshold": thr,
        "train_time_s": float(train_time),
        "val": {**val_metrics, "pairs": int(len(y_true_val)), "inference_time_s": float(t_inf_val)},
        "test": {**test_metrics, "pairs": int(len(y_true_test)), "inference_time_s": float(t_inf_test)},
    }


def main():
    t0 = time.time()
    r_b1 = run("B1")
    r_b2 = run("B2")
    t1 = time.time()

    print("\n" + "=" * 80)
    print("REPORT FINALE (DEDUPE CLASSIC: prepare_training + train)")
    print("=" * 80)

    df = pd.DataFrame([
        {
            "Pipeline": "B1-dedupe-classic",
            "Thr": r_b1["threshold"],
            "Train_s": r_b1["train_time_s"],
            "Val_F1": r_b1["val"]["f1"],
            "Test_F1": r_b1["test"]["f1"],
            "Test_P": r_b1["test"]["precision"],
            "Test_R": r_b1["test"]["recall"],
            "Test_pairs": r_b1["test"]["pairs"],
            "Test_infer_s": r_b1["test"]["inference_time_s"],
        },
        {
            "Pipeline": "B2-dedupe-classic",
            "Thr": r_b2["threshold"],
            "Train_s": r_b2["train_time_s"],
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
    main()'''


import os
import json
import time
import random
import numpy as np
import pandas as pd
import dedupe
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, f1_score

# ============================================================
# CONFIGURAZIONE (Fedele alla Relazione)
# ============================================================
DATASET_PATH = os.path.join("dataset", "dataset_for_training.csv")
GT_PATH = os.path.join("dataset", "ground_truth_map.json")
BLOCKS_DIR = os.path.join("dataset", "blocks")
SPLITS_DIR = os.path.join("dataset", "splits")

# Categorie per variabili Categorical (Sezione 6.1.1)
FUEL_CATS = ['gasoline', 'diesel', 'electric', 'hybrid', 'other']
TRANS_CATS = ['automatic', 'manual', 'other']

# Parametri per garantire tempi "Trascurabili" (Table 10)
SCORE_BATCH_SIZE = 20000 
TRAIN_SAMPLE_SIZE = 2000 

# ============================================================
# UTILS DI PULIZIA
# ============================================================
def clean_val(x, t=str):
    if pd.isna(x): return None
    try: return t(x)
    except: return None

def load_records_subset(ids_needed):
    print(f"[IO] Caricamento di {len(ids_needed)} record da CSV...")
    # Carichiamo latitude e longitude per la variabile LatLong citata in relazione
    cols = ["id", "make", "model", "year", "price", "mileage", "fuel_type", 
            "transmission", "latitude", "longitude", "body_type", "color"]
    
    df = pd.read_csv(DATASET_PATH, usecols=cols, low_memory=False)
    df["id"] = df["id"].astype(str)
    df = df[df["id"].isin(ids_needed)]
    
    data = {}
    for r in df.itertuples(index=False):
        data[str(r.id)] = {
            "make": clean_val(r.make),
            "model": clean_val(r.model),
            "year": clean_val(r.year, float),
            "price": clean_val(r.price, float),
            "mileage": clean_val(r.mileage, float),
            "fuel_type": clean_val(r.fuel_type) if clean_val(r.fuel_type) in FUEL_CATS else "other",
            "transmission": clean_val(r.transmission) if clean_val(r.transmission) in TRANS_CATS else "other",
            "location": (clean_val(r.latitude, float), clean_val(r.longitude, float)),
            "body_type": clean_val(r.body_type),
            "color": clean_val(r.color)
        }
    return data

# ============================================================
# TRAINING (Sbloccato: index_predicates=False)
# ============================================================
def train_dedupe(data_d, train_df):
    # Definiamo i campi esattamente come descritto nella Sezione 6.1.1
    fields = [
        dedupe.variables.String("make", has_missing=True),
        dedupe.variables.String("model", has_missing=True),
        dedupe.variables.Price("year", has_missing=True),
        dedupe.variables.Price("price", has_missing=True),
        dedupe.variables.Price("mileage", has_missing=True),
        dedupe.variables.LatLong("location", has_missing=True),
        dedupe.variables.Categorical("fuel_type", categories=FUEL_CATS, has_missing=True),
        dedupe.variables.Categorical("transmission", categories=TRANS_CATS, has_missing=True)
    ]
    
    linker = dedupe.RecordLink(fields)
    
    matches, distinct = [], []
    for row in train_df.itertuples(index=False):
        id1, id2 = str(row.id1), str(row.id2)
        if id1 in data_d and id2 in data_d:
            pair = (data_d[id1], data_d[id2])
            if row.label == 1: matches.append(pair)
            else: distinct.append(pair)

    print(f"[TRAIN] Coppie per addestramento: {len(matches)} match, {len(distinct)} distinct")
    linker.mark_pairs({"match": matches, "distinct": distinct})
    
    # SBLOCCO: sample_size piccolo rende prepare_training istantaneo
    print("[TRAIN] Inizio prepare_training...")
    linker.prepare_training(data_d, data_d, sample_size=TRAIN_SAMPLE_SIZE)
    
    print("[TRAIN] Addestramento classifier...")
    t0 = time.time()
    # index_predicates=False evita che Dedupe cerchi blocchi; usiamo i vostri JSON
    linker.train(index_predicates=False, num_cores=None)
    t1 = time.time()
    
    return linker, (t1 - t0)

# ============================================================
# EVALUATION (Step 0.05 - Sezione 6.1.3)
# ============================================================
def evaluate_strategy(linker, data_d, blocks, id_to_vin, label):
    y_true, y_score, batch = [], [], []
    t0 = time.time()
    
    print(f"[SCORE] Inizio scoring {label}...")
    for ids in blocks.values():
        ids_v = [str(i) for i in ids if str(i) in data_d]
        if len(ids_v) < 2: continue
        for id1, id2 in combinations(ids_v, 2):
            batch.append((data_d[id1], data_d[id2]))
            v1, v2 = id_to_vin.get(id1), id_to_vin.get(id2)
            y_true.append(1 if (v1 and v1 == v2) else 0)
            
            if len(batch) >= SCORE_BATCH_SIZE:
                y_score.extend(linker.score(batch)['score'].tolist())
                batch = []
    
    if batch: y_score.extend(linker.score(batch)['score'].tolist())
    
    y_true, y_score = np.array(y_true), np.array(y_score)
    
    # Ricerca soglia ottima step 0.05 come da relazione
    best_f1, best_thr = -1, 0.3
    for thr in np.arange(0.1, 1.0, 0.05):
        f1 = f1_score(y_true, (y_score >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
            
    p = precision_score(y_true, (y_score >= best_thr).astype(int), zero_division=0)
    r = recall_score(y_true, (y_score >= best_thr).astype(int), zero_division=0)
    
    return {"f1": best_f1, "p": p, "r": r, "thr": best_thr, "time": time.time()-t0, "pairs": len(y_true)}

# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(strat):
    print(f"\n{'='*20} AVVIO PIPELINE {strat} {'='*20}")
    
    with open(GT_PATH) as f:
        gt = json.load(f)
    id_to_vin = {str(i): v for v, ids in gt.items() for i in ids}
    
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train_pairs.csv"))
    val_bl = json.load(open(os.path.join(BLOCKS_DIR, f"blocking_{strat}_val.json")))
    test_bl = json.load(open(os.path.join(BLOCKS_DIR, f"blocking_{strat}_test.json")))
    
    # Raccogliamo ID necessari
    ids_needed = set(train_df["id1"].astype(str)).union(train_df["id2"].astype(str))
    for b in [val_bl, test_bl]:
        for b_ids in b.values(): ids_needed.update([str(i) for i in b_ids])
    
    data_d = load_records_subset(ids_needed)
    
    # Training
    linker, t_train = train_dedupe(data_d, train_df)
    
    # Valutazione
    res = evaluate_strategy(linker, data_d, test_bl, id_to_vin, strat)
    
    return {
        "Pipeline": f"{strat}-dedupe",
        "F1": res["f1"], "P": res["p"], "R": res["r"],
        "Thr": res["thr"], "Train_s": t_train, "Inf_s": res["time"], "Pairs": res["pairs"]
    }

def main():
    results = [run_pipeline("B1"), run_pipeline("B2")]
    
    print("\n" + "="*80)
    print("REPORT FINALE DEDUPE (COERENTE CON RELAZIONE)")
    print("="*80)
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()