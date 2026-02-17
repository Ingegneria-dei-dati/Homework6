'''import gc
import os
import sys
import pandas as pd
import json
import random
import time
import multiprocessing
from sklearn.metrics import precision_recall_fscore_support

try:
    import dedupe
except ImportError:
    print("ERROR: install dedupe -> pip install dedupe[performance]")
    sys.exit(1)

# ============================================================
# CONFIGURAZIONE
# ============================================================

DATASET_PATH = "dataset/dataset_for_training.csv"
BLOCK_DIR = "dataset/blocks"
BLOCKINGS = ["B2", "B1"]
MODEL_DIR = "models"
METRICS_DIR = "dataset/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

MAX_DISTINCT = 200000
NEGATIVE_RATIO = 2.0

multiprocessing.set_start_method("spawn", force=True)

# ============================================================
# DATA CLEANING
# ============================================================

COLS_STRING = ['make', 'model', 'engine_cylinders', 'body_type', 'color', 'location']
COLS_NUMERIC = ['year', 'price', 'mileage', 'latitude', 'longitude']
COLS_CATEGORICAL = ['fuel_type', 'drive', 'transmission']
ALL_COLS = COLS_STRING + COLS_NUMERIC + COLS_CATEGORICAL


def to_clean_string(val):
    if pd.isnull(val):
        return None
    return str(val).strip()

def normalize_categorical(val, valid_values, default='other'):
    if pd.isnull(val) or str(val).strip() == '':
        return default
    v = str(val).strip().lower()
    return next((x for x in valid_values if x.lower() == v), default)

def load_and_clean_data():
    print(f"-> Caricamento dataset da {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    for col in COLS_STRING:
        if col in df.columns:
            df[col] = df[col].apply(to_clean_string)
    for col in COLS_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Normalizza valori categorical
    fuel_categories = ['gas','other','diesel','hybrid','electric','biodiesel',
                       'flex fuel vehicle','compressed natural gas','propane']
    if 'fuel_type' in df.columns:
        df['fuel_type'] = df['fuel_type'].apply(lambda x: normalize_categorical(x, fuel_categories, 'other'))
    drive_categories = ['rwd','4wd','fwd','awd','4x2']
    if 'drive' in df.columns:
        df['drive'] = df['drive'].apply(lambda x: normalize_categorical(x, drive_categories, '4x2'))
    transmission_categories = ['other','automatic','manual','cvt','dual clutch']
    if 'transmission' in df.columns:
        df['transmission'] = df['transmission'].apply(lambda x: normalize_categorical(x, transmission_categories, 'other'))

    # Location come stringa tupla
    df['location'] = df.apply(
        lambda r: str((r['latitude'], r['longitude'])) if pd.notna(r.get('latitude')) and pd.notna(r.get('longitude')) else None,
        axis=1
    )

    df['id'] = df['id'].astype(str)
    df = df.drop_duplicates(subset=['id']).set_index('id')
    print(f"   [OK] {len(df)} record pronti.")
    return df


# ============================================================
# RECORD DICTIONARY
# ============================================================

def create_dedupe_dict(df, ids):
    df_subset = df.loc[df.index.isin(ids)]
    records = {}

    for idx, row in df_subset.iterrows():
        rec = {}
        for col in ALL_COLS:
            if col in df_subset.columns:
                val = row[col]
                if pd.isna(val):
                    rec[col] = None
                elif col in COLS_NUMERIC:
                    rec[col] = float(val)
                else:
                    rec[col] = str(val)
            else:
                rec[col] = None
        records[str(idx)] = rec

    return records


# ============================================================
# STREAMING TRAINING BUILDER (CORRETTO)
# ============================================================

def build_training_from_blocking(block_json_path, pairs_df, df):

    with open(block_json_path, "r") as f:
        blocks = json.load(f)

    all_ids = df.index.tolist()
    records = create_dedupe_dict(df, all_ids)

    true_matches = {
        (str(r.id1), str(r.id2))
        for r in pairs_df.itertuples(index=False)
        if r.label == 1
    }

    matches = []
    distinct_sample = []
    total_seen_distinct = 0

    print("-> Streaming candidate generation...")

    for block_ids in blocks.values():

        ids = [str(i) for i in block_ids if str(i) in records]
        n = len(ids)

        for i in range(n):
            for j in range(i + 1, n):

                id1 = ids[i]
                id2 = ids[j]

                if (id1, id2) in true_matches or (id2, id1) in true_matches:
                    matches.append((records[id1], records[id2]))
                else:
                    total_seen_distinct += 1

                    if len(distinct_sample) < MAX_DISTINCT:
                        distinct_sample.append((id1, id2))
                    else:
                        r = random.randint(0, total_seen_distinct)
                        if r < MAX_DISTINCT:
                            distinct_sample[r] = (id1, id2)

    target = min(int(len(matches) * NEGATIVE_RATIO), len(distinct_sample))
    distinct_sample = distinct_sample[:target]
    distinct = [(records[i], records[j]) for i, j in distinct_sample]

    print(f"   Matches: {len(matches)}")
    print(f"   Distinct sampled: {len(distinct)}")

    return {"match": matches, "distinct": distinct}, records


# ============================================================
# DEDUPE FIELD CONFIG
# ============================================================

def get_dedupe_fields():
    return [
        dedupe.variables.String("make", has_missing=True),
        dedupe.variables.String("model", has_missing=True),
        dedupe.variables.String("engine_cylinders", has_missing=True),
        dedupe.variables.String("body_type", has_missing=True),
        dedupe.variables.String("color", has_missing=True),
        dedupe.variables.String("location", has_missing=True),
        dedupe.variables.Price("year", has_missing=True),
        dedupe.variables.Price("price", has_missing=True),
        dedupe.variables.Price("mileage", has_missing=True),
        dedupe.variables.Price("latitude", has_missing=True),
        dedupe.variables.Price("longitude", has_missing=True),
        dedupe.variables.Categorical(
            "fuel_type",
            categories=['gas','other','diesel','hybrid','electric',
                        'biodiesel','flex fuel vehicle','compressed natural gas','propane'],
            has_missing=True
        ),
        dedupe.variables.Categorical(
            "drive",
            categories=['rwd','4wd','fwd','awd','4x2'],
            has_missing=True
        ),
        dedupe.variables.Categorical(
            "transmission",
            categories=['other','automatic','manual','cvt','dual clutch'],
            has_missing=True
        )
    ]


# ============================================================
# EVALUATION
# ============================================================

def evaluate(deduper, pairs_dict, label):

    print(f"\n[{label}] Evaluation")

    y_true, y_pred = [], []
    start = time.time()

    for recA, recB in pairs_dict["match"]:
        score = deduper.score([(recA, recB)])[0]
        y_true.append(1)
        y_pred.append(1 if score >= 0.5 else 0)

    for recA, recB in pairs_dict["distinct"]:
        score = deduper.score([(recA, recB)])[0]
        y_true.append(0)
        y_pred.append(1 if score >= 0.5 else 0)

    infer_time = time.time() - start

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average="binary",
        zero_division=0
    )

    print(f"   Precision: {p:.3f}")
    print(f"   Recall: {r:.3f}")
    print(f"   F1: {f1:.3f}")
    print(f"   Inference time: {infer_time:.2f}s")

    return p, r, f1, infer_time


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_and_clean_data()
    n_cores = multiprocessing.cpu_count()
    print(f"\nDetected {n_cores} cores\n")

    for blocking in BLOCKINGS:

        print(f"\n================ BLOCKING: {blocking} ================")

        train_json = os.path.join(BLOCK_DIR, f"blocking_{blocking}_train.json")
        val_json   = os.path.join(BLOCK_DIR, f"blocking_{blocking}_val.json")
        test_json  = os.path.join(BLOCK_DIR, f"blocking_{blocking}_test.json")

        train_pairs_df = pd.read_csv("dataset/splits/train_pairs.csv")
        val_pairs_df   = pd.read_csv("dataset/splits/val_pairs.csv")
        test_pairs_df  = pd.read_csv("dataset/splits/test_pairs.csv")

        # -------- TRAIN --------
        train_pairs, train_dict = build_training_from_blocking(
            train_json, train_pairs_df, df
        )

        deduper = dedupe.Dedupe(
            get_dedupe_fields(),
            num_cores=n_cores,
            in_memory=True
        )

        print("-> Marking pairs")
        deduper.mark_pairs(train_pairs)

        print("-> Preparing training")
        
        # Punto A: Creiamo un universo di training ridotto (Sub-sampling)
        # Non serve analizzare 2M di record per capire i pesi, 50k sono sufficienti 
        all_ids = list(df.index)
        sample_ids = random.sample(all_ids, min(5000, len(all_ids)))
        
        # Creiamo un dizionario temporaneo leggero solo per questi 50k record
        train_data_sample = create_dedupe_dict(df, sample_ids)
        
        # Passiamo il campione ridotto: questo sblocca la CPU e la RAM 
        deduper.prepare_training(train_data_sample, sample_size=min(1000, len(train_data_sample)))

        # -------- TRAINING MODEL --------
        print("-> Training model (Logistic Regression)...")
        start_train = time.time()
        
        # index_predicates=False evita il loop infinito sui 2M di record 
        deduper.train(index_predicates=False) 
        
        train_time = time.time() - start_train
        print(f"   [OK] Training time: {train_time:.2f}s")

        # Pulizia memoria: eliminiamo il dizionario campione per liberare RAM per la valutazione
        del train_data_sample
        gc.collect() 

        # -------- SAVE MODEL --------
        model_path = os.path.join(MODEL_DIR, f"dedupe_model_{blocking}.pickle")
        with open(model_path, "wb") as f:
            deduper.write_settings(f)
        print(f"Model saved: {model_path}")

        # -------- EVALUATION (Punto 4.H) --------
        # Carichiamo le coppie dai blocchi JSON della Ground Truth [cite: 13, 20]
        val_pairs, _ = build_training_from_blocking(val_json, val_pairs_df, df)
        test_pairs, _ = build_training_from_blocking(test_json, test_pairs_df, df)

        # Valutazione delle performance (4.H): Precision, Recall, F1 [cite: 24]
        vp, vr, vf, vt = evaluate(deduper, val_pairs, f"VAL-{blocking}")
        tp, tr, tf, tt = evaluate(deduper, test_pairs, f"TEST-{blocking}")

        # Salvataggio metriche per la relazione finale [cite: 26]
        metrics = {
            "blocking": blocking,
            "train_time_sec": train_time,
            "VAL": {"precision": vp, "recall": vr, "f1": vf, "inference_time_sec": vt},
            "TEST": {"precision": tp, "recall": tr, "f1": tf, "inference_time_sec": tt}
        }

        metrics_path = os.path.join(METRICS_DIR, f"metrics_{blocking}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved: {metrics_path}")


# ============================================================

if __name__ == "__main__":
    main()'''











'''import gc
import os
import sys
import pandas as pd
import json
import time
import random
from sklearn.metrics import precision_recall_fscore_support

try:
    import dedupe
except ImportError:
    print("ERROR: install dedupe -> pip install dedupe[performance]")
    sys.exit(1)

# ============================================================
# CONFIGURAZIONE
# ============================================================
DATASET_PATH = "dataset/dataset_for_training.csv"
SPLITS_DIR = "dataset/splits"
RESULTS_DIR = "dataset/results"
MODEL_DIR = "models"
BLOCKINGS = ["B1", "B2"]

COLS_STRING = ["make", "model"]
COLS_NUMERIC = ["year", "price", "mileage"]
COLS_CATEGORICAL = ["fuel_type"]
ALL_COLS = COLS_STRING + COLS_NUMERIC + COLS_CATEGORICAL

FUEL_CATEGORIES = ["gas", "other", "diesel", "hybrid", "electric", "biodiesel", "flex fuel vehicle", "compressed natural gas", "propane"]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# UTILITIES
# ============================================================
def normalize_fuel(val):
    if pd.isnull(val): return "other"
    v = str(val).strip().lower()
    if v in ["gasoline", "benzina", "petrol"]: return "gas"
    return v if v in FUEL_CATEGORIES else "other"

def clean_record(row):
    rec = {}
    for col in ALL_COLS:
        val = row.get(col)
        if col == "fuel_type":
            rec[col] = normalize_fuel(val)
        elif pd.isna(val):
            rec[col] = None
        elif col in COLS_STRING:
            rec[col] = str(val).strip()
        elif col in COLS_NUMERIC:
            try: rec[col] = float(val)
            except: rec[col] = None
    return rec

# ============================================================
# MAIN
# ============================================================
def main():
    print("-> Caricamento dataset...")
    df_full = pd.read_csv(DATASET_PATH, low_memory=False)
    df_full['id'] = df_full['id'].astype(str)
    df_full = df_full.drop_duplicates(subset=['id']).set_index('id')
    
    for blocking in BLOCKINGS:
        print(f"\n===== PROCESSAMENTO: {blocking} =====")
        
        train_csv = os.path.join(SPLITS_DIR, "train_pairs.csv")
        val_csv = os.path.join(SPLITS_DIR, "val_pairs.csv")
        test_csv = os.path.join(SPLITS_DIR, "test_pairs.csv")
        
        # 1. Caricamento coppie e identificazione ID
        def get_needed_ids(paths):
            ids = set()
            for p in paths:
                if os.path.exists(p):
                    tmp = pd.read_csv(p, dtype={"id1": str, "id2": str})
                    ids.update(tmp.id1.tolist())
                    ids.update(tmp.id2.tolist())
            return ids

        needed_ids = get_needed_ids([train_csv, val_csv, test_csv])
        
        # 2. Creazione records (TUTTI gli ID necessari devono essere qui)
        records = {str(idx): clean_record(df_full.loc[idx]) 
                   for idx in needed_ids if idx in df_full.index}
        
        print(f"Dizionario creato con {len(records)} record.")

        # 3. Setup Dedupe
        fields = [
            dedupe.variables.String("make", has_missing=True),
            dedupe.variables.String("model", has_missing=True),
            dedupe.variables.Price("year", has_missing=True),
            dedupe.variables.Price("price", has_missing=True),
            dedupe.variables.Price("mileage", has_missing=True),
            dedupe.variables.Categorical("fuel_type", categories=FUEL_CATEGORIES, has_missing=True)
        ]
        deduper = dedupe.Dedupe(fields)
        
        # 4. PREPARE TRAINING
        # Usiamo un sample_size piccolo per velocizzare l'indicizzazione
        print("-> Preparazione training...")
        deduper.prepare_training(records, sample_size=5000)
        
        # 5. CARICAMENTO COPPIE PER MARK_PAIRS
        def load_pairs(path):
            if not os.path.exists(path): return {"match": [], "distinct": []}
            p_df = pd.read_csv(path, dtype={"id1": str, "id2": str})
            out = {"match": [], "distinct": []}
            for r in p_df.itertuples():
                id1, id2 = str(r.id1), str(r.id2)
                if id1 in records and id2 in records:
                    label = "match" if r.label == 1 else "distinct"
                    out[label].append((records[id1], records[id2]))
            return out

        train_pairs = load_pairs(train_csv)

        # 6. TRAINING con misura tempi
        print("-> Inizio Training...")
        start_train = time.time()
        
        # Mark pairs popola l'active learner correttamente
        deduper.mark_pairs(train_pairs)
        
        # Train esegue la Logistic Regression sulle distanze calcolate
        # index_predicates=False evita il NoIndexError durante il training
        deduper.train(index_predicates=False)
        
        train_time = time.time() - start_train
        print(f"   [OK] Tempo di training: {train_time:.4f}s")

        # 7. EVALUATION
        metrics = {
            "blocking": blocking,
            "train_time_sec": train_time,
            "results": {}
        }

        for split_label, path in [("VAL", val_csv), ("TEST", test_csv)]:
            p_data = load_pairs(path)
            all_pairs = p_data["match"] + p_data["distinct"]
            if not all_pairs: continue
            
            y_true = [1]*len(p_data["match"]) + [0]*len(p_data["distinct"])
            
            print(f"-> Inference su {split_label} ({len(all_pairs)} coppie)...")
            start_inf = time.time()
            # Calcolo degli score (inferenza)
            scores = deduper.score(all_pairs)
            inf_time = time.time() - start_inf
            
            y_pred = [1 if s >= 0.5 else 0 for s in scores]
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            
            metrics["results"][split_label] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1_score": round(f1, 4),
                "inference_time_sec": round(inf_time, 4)
            }
            print(f"   [{split_label}] F1: {f1:.3f} | Time: {inf_time:.2f}s")

        # 8. Salvataggio
        with open(os.path.join(RESULTS_DIR, f"metrics_{blocking}.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        
        with open(os.path.join(MODEL_DIR, f"dedupe_model_{blocking}.pickle"), "wb") as f:
            deduper.write_settings(f)
            
        del records, train_pairs, deduper
        gc.collect()

if __name__ == "__main__":
    main()'''







import json
import time
import random
from pathlib import Path

import pandas as pd
import dedupe

# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DATASET_PATH = "dataset/dataset_for_training.csv"
TRAIN_PAIRS_PATH = "dataset/splits/train_pairs.csv"
TEST_PAIRS_PATH  = "dataset/splits/test_pairs.csv"
BLOCKS_DIR = Path("dataset/blocks")
OUTPUT_DIR = Path("output/dedupe_results")

MAX_MATCHES = 20000
MAX_DISTINCT = 10000

# Thresholds da testare
THRESHOLDS = [0.4, 0.5, 0.6]

PIPELINES = [
    {
        "name": "P1_full",
        "fields": [
            dedupe.variables.String("make", has_missing=True),
            dedupe.variables.String("model", has_missing=True),
            dedupe.variables.Price("price", has_missing=True),
            dedupe.variables.Price("mileage", has_missing=True),
            dedupe.variables.Price("year", has_missing=True),
        ],
    },
    {
        "name": "P2_minimal",
        "fields": [
            dedupe.variables.String("make", has_missing=True),
            dedupe.variables.String("model", has_missing=True),
            dedupe.variables.Price("year", has_missing=True),
        ],
    },
]

# ============================================================
# HELPERS
# ============================================================

def clean_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, (float, int)):
        return v
    return str(v).strip().lower() or None

def build_entity_dict(df):
    records = {}
    for row in df.itertuples(index=False):
        rid = str(row.id)
        rec = {col: clean_value(getattr(row, col)) for col in df.columns if col != "id"}
        records[rid] = rec
    return records

def load_blocks(split, strategy):
    path = BLOCKS_DIR / f"blocking_{strategy}_{split}.json"
    print(f"-> Loading blocks: {path}")
    with open(path) as f:
        blocks = json.load(f)
    print(f"   Loaded {len(blocks):,} blocks")
    return blocks

def build_candidate_pairs(blocks):
    pairs = set()
    for block_ids in blocks.values():
        ids = list(map(str, block_ids))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.add((ids[i], ids[j]))
    return pairs

def build_training_pairs(train_df, candidate_pairs, records):
    print("-> Building training pairs (with sampling)...")
    true_matches = {
        (str(r.id1), str(r.id2)) for r in train_df.itertuples(index=False) if r.label == 1
    }

    matches, distinct = [], []
    for id1, id2 in candidate_pairs:
        if id1 not in records or id2 not in records:
            continue
        pair = (records[id1], records[id2])
        if (id1, id2) in true_matches or (id2, id1) in true_matches:
            matches.append(pair)
        else:
            distinct.append(pair)

    print(f"   Raw matches:  {len(matches):,}")
    print(f"   Raw distinct: {len(distinct):,}")

    if len(matches) > MAX_MATCHES:
        matches = random.sample(matches, MAX_MATCHES)
        print(f"   Sampled matches: {len(matches):,}")
    if len(distinct) > MAX_DISTINCT:
        distinct = random.sample(distinct, MAX_DISTINCT)
        print(f"   Sampled distinct: {len(distinct):,}")

    return {"match": matches, "distinct": distinct}

def evaluate(pred_pairs, test_df):
    truth = {(str(r.id1), str(r.id2)) for r in test_df.itertuples(index=False) if r.label == 1}
    tp = len(pred_pairs & truth)
    fp = len(pred_pairs - truth)
    fn = len(truth - pred_pairs)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0
    return precision, recall, f1, tp, fp, fn

# ============================================================
# CORE
# ============================================================

def run_pipeline(pipeline, blocking):
    print("\n" + "=" * 60)
    print(f"PIPELINE: {pipeline['name']} | BLOCKING: {blocking}")
    print("=" * 60)

    # --- Load datasets ---
    entities_df = pd.read_csv(DATASET_PATH, engine="python")
    train_df = pd.read_csv(TRAIN_PAIRS_PATH)
    test_df  = pd.read_csv(TEST_PAIRS_PATH)

    print(f"   Entities:  {len(entities_df):,}")
    print(f"   Train GT:  {len(train_df):,}")
    print(f"   Test GT:   {len(test_df):,}")

    # --- Build entity dict ---
    records = build_entity_dict(entities_df)
    print(f"   Records loaded: {len(records):,}")

    # --- Train phase ---
    print("\n[TRAIN PHASE]")
    train_blocks = load_blocks("train", blocking)
    train_candidates = build_candidate_pairs(train_blocks)
    print(f"   Candidate pairs: {len(train_candidates):,}")

    training_pairs = build_training_pairs(train_df, train_candidates, records)

    training_file = Path("temp_training.json")
    with open(training_file, "w") as f:
        json.dump(training_pairs, f)

    linker = dedupe.Dedupe(pipeline["fields"], num_cores=4)
    print("-> prepare_training()")
    with open(training_file) as f:
        linker.prepare_training(records, training_file=f)
    print("-> train()")
    t0 = time.time()
    linker.train()
    train_time = time.time() - t0
    print(f"   Training completed in {train_time:.2f}s")

    # --- Test phase ---
    print("\n[TEST PHASE]")
    test_blocks = load_blocks("test", blocking)
    test_candidates = build_candidate_pairs(test_blocks)
    test_ids = {i for pair in test_candidates for i in pair}
    test_records = {i: records[i] for i in test_ids if i in records}
    print(f"   Test records: {len(test_records):,}")

    results_list = []

    for threshold in THRESHOLDS:
        print(f"\n-> Clustering with threshold={threshold}")
        t0 = time.time()
        clusters = linker.partition(test_records, threshold=threshold)
        infer_time = time.time() - t0

        pred_pairs = set()
        for cluster, _ in clusters:
            ids = list(cluster)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pred_pairs.add((ids[i], ids[j]))

        precision, recall, f1, tp, fp, fn = evaluate(pred_pairs, test_df)
        print(f"Threshold={threshold} | Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

        results_list.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "inference_time": infer_time
        })

    # cleanup
    training_file.unlink(missing_ok=True)

    return {
        "pipeline": pipeline["name"],
        "blocking": blocking,
        "train_time": train_time,
        "results_per_threshold": results_list
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for blocking in ["B1", "B2"]:
        for pipeline in PIPELINES:
            summary = run_pipeline(pipeline, blocking)
            results.append(summary)

    out_path = OUTPUT_DIR / "manual_blocking_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ All experiments completed.")
