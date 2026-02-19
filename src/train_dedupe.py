import json
import pandas as pd
import math
import time
import itertools
import random
from pathlib import Path
import dedupe

# ================= CONFIG =================
DATASET_PATH = Path("dataset/dataset_for_training.csv")
BLOCKS_DIR = Path("dataset/blocks")
GROUND_TRUTH_PATH = Path("dataset/ground_truth_map.json")
OUTPUT_DIR = Path("dataset/results/dedupe")
MODELS_DIR = Path("dataset/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Limiti per test rapido
MAX_RECORDS = 25000
MAX_BLOCKS = 200
MAX_MATCHES = 5000
MAX_DISTINCT = 15000
THRESHOLDS = [0.4, 0.5, 0.6]

STRING_COLS = ["make", "model"]
NUMERIC_COLS = ["year", "price", "mileage"]

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

BLOCKING_TYPES = ["B1", "B2"]

# ================= HELPERS =================
def clean_val(v, numeric=False):
    if pd.isna(v):
        return None
    if numeric:
        try:
            return float(v)
        except:
            return None
    # string normalization: converti le stringhe vuote in None
    if isinstance(v, str):
        v = v.strip().lower()
        return v if v else None
    return str(v)

def load_dataset():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(DATASET_PATH, nrows=MAX_RECORDS, dtype=str)
    df["record_id"] = df["id"]
    df.set_index("record_id", inplace=True)
    
    # Normalizzazione separata per stringhe e numerici
    for col in STRING_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: clean_val(x, numeric=False))
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: clean_val(x, numeric=True))
    
    print(f"[INFO] Dataset loaded: {len(df)} records")
    return df

def load_ground_truth():
    print("[INFO] Loading ground truth...")
    with open(GROUND_TRUTH_PATH) as f:
        gt_map = json.load(f)
    pairs = []
    for vin, ids in gt_map.items():
        ids = list(map(str, ids))
        for a, b in itertools.combinations(ids, 2):
            pairs.append({"id_A": a, "id_B": b, "label": 1})
    print(f"[INFO] Positive pairs generated: {len(pairs)}")
    return pd.DataFrame(pairs)

def load_blocks(split="train", block_type="B1"):
    path = BLOCKS_DIR / f"blocking_{block_type}_{split}.json"
    print(f"[INFO] Loading blocks: {path}")
    with open(path) as f:
        blocks = json.load(f)
    limited_blocks = dict(list(blocks.items())[:MAX_BLOCKS])
    print(f"[INFO] Using {len(limited_blocks)} blocks for testing")
    return limited_blocks

def build_dedupe_records(df):
    records = {}
    for idx, row in df.iterrows():
        rec = {}
        # Strings: Keep None for missing
        for col in STRING_COLS:
            rec[col] = row.get(col, None)
            
        # Numerics: Explicitly catch NaN and convert to 0 or None
        for col in NUMERIC_COLS:
            val = row.get(col)
            try:
                f_val = float(val)
                # Check if the float itself is NaN
                if math.isnan(f_val):
                    rec[col] = 0.0 
                else:
                    rec[col] = f_val
            except (TypeError, ValueError):
                rec[col] = 0.0
                
        records[idx] = rec
    print(f"[INFO] {len(records)} records prepared for Dedupe")
    return records


def build_candidate_pairs(blocks):
    pairs = []
    for ids in blocks.values():
        if len(ids) < 2: continue
        # Se un blocco ha più di 300 record, saltalo o campionalo
        if len(ids) > 300: 
            print(f"[SKIP] Blocco troppo grande: {len(ids)} record")
            continue 
        for a, b in itertools.combinations(ids, 2):
            pairs.append((str(a), str(b)))
    return pairs


def calculate_metrics(y_true, y_pred):
    tp = sum((p==1 and t==1) for p,t in zip(y_pred, y_true))
    fp = sum((p==1 and t==0) for p,t in zip(y_pred, y_true))
    fn = sum((p==0 and t==1) for p,t in zip(y_pred, y_true))
    precision = tp/(tp+fp) if tp+fp else 0
    recall = tp/(tp+fn) if tp+fn else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
    return precision, recall, f1

# ================= MAIN =================
def run_experiment():
    # 1. Caricamento Dati Iniziale (una sola volta)
    df = load_dataset()
    gt_df = load_ground_truth()
    records = build_dedupe_records(df)
    
    # Creiamo un set per lookup veloce del Ground Truth
    gt_set = set()
    for r in gt_df.itertuples():
        gt_set.add(tuple(sorted((str(r.id_A), str(r.id_B)))))

    all_results = []

    # 2. Ciclo sulle Pipeline (P1, P2)
    for pipe in PIPELINES:
        # 3. Ciclo sui tipi di Blocking (B1, B2)
        for b_type in BLOCKING_TYPES:
            exp_id = f"{pipe['name']}_{b_type}"
            print(f"\n{'='*30}\nESPERIMENTO: {exp_id}\n{'='*30}")

            # --- Training ---
            print(f"[INFO] Preparazione training per {exp_id}...")
            train_blocks = load_blocks("train", b_type)
            train_pairs_ids = build_candidate_pairs(train_blocks)
            
            matches, distinct = [], []
            for a, b in train_pairs_ids:
                rec_a, rec_b = records.get(a), records.get(b)
                if rec_a and rec_b:
                    if tuple(sorted((a, b))) in gt_set:
                        matches.append((rec_a, rec_b))
                    else:
                        distinct.append((rec_a, rec_b))

            matches = random.sample(matches, min(len(matches), MAX_MATCHES))
            distinct = random.sample(distinct, min(len(distinct), MAX_DISTINCT))

            print(f"[INFO] Avvio Training...")
            start_train = time.time()
            deduper = dedupe.Dedupe(pipe["fields"], num_cores=2)
            deduper.mark_pairs({"match": matches, "distinct": distinct})
            deduper.prepare_training(records, sample_size=2000)
            deduper.train()
            train_time = time.time() - start_train
            print(f"[INFO] Training completato in {train_time:.2f}s")

            # --- Salvataggio Modello ---
            model_path = MODELS_DIR / f"{exp_id}.settings"
            with open(model_path, "wb") as f:
                deduper.write_settings(f)

            # --- Inference & Sensitivity Analysis ---
            print(f"[INFO] Caricamento test blocks per {b_type}...")
            test_blocks = load_blocks("test", b_type)
            test_pairs_ids = build_candidate_pairs(test_blocks)
            
            # Preparazione batch per velocizzare lo scoring
            to_score = []
            y_true = []
            for a, b in test_pairs_ids:
                rec_a, rec_b = records.get(a), records.get(b)
                if rec_a and rec_b:
                    to_score.append(((a, rec_a), (b, rec_b)))
                    y_true.append(1 if tuple(sorted((a, b))) in gt_set else 0)

            print(f"[INFO] Scoring di {len(to_score)} coppie...")
            start_inf = time.time()
            if to_score:
                scores = deduper.score(to_score)["score"]
            else:
                scores = []
            inf_time = time.time() - start_inf

            # --- Analisi di Sensibilità sulle Soglie ---
            best_f1_exp = 0
            best_thr_exp = 0
            
            for thr in THRESHOLDS:
                y_pred = [1 if s >= thr else 0 for s in scores]
                p, r, f1 = calculate_metrics(y_true, y_pred)
                
                if f1 > best_f1_exp:
                    best_f1_exp = f1
                    best_thr_exp = thr
                
                # Log del risultato per questa soglia
                res = {
                    "experiment": exp_id,
                    "pipeline": pipe["name"],
                    "blocking": b_type,
                    "threshold": thr,
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "train_time_sec": round(train_time, 2),
                    "inference_time_sec": round(inf_time, 2),
                    "test_pairs_count": len(to_score)
                }
                all_results.append(res)
                print(f"  > Thr {thr} | F1: {f1:.3f} | P: {p:.3f} | R: {r:.3f}")

            print(f"[OK] Esperimento {exp_id} concluso. Best F1: {best_f1_exp:.3f} (Soglia {best_thr_exp})")

    # 3. SALVATAGGIO FINALE
    print(f"\n{'='*50}\n[COMPLETATO] Salvataggio risultati finali...\n{'='*50}")
    results_df = pd.DataFrame(all_results)
    
    # Salviamo il CSV globale per il confronto
    output_path = OUTPUT_DIR / "final_benchmark_results.csv"
    results_df.to_csv(output_path, index=False)
    
    # Stampiamo un piccolo riassunto a video dei migliori per pipeline
    summary = results_df.sort_values('f1', ascending=False).drop_duplicates('experiment')
    print("\nRIASSUNTO MIGLIORI PERFORMANCE PER CONFIGURAZIONE:")
    print(summary[['experiment', 'threshold', 'f1', 'precision', 'recall']])
    print(f"\nReport completo salvato in: {output_path}")

if __name__ == "__main__":
    run_experiment()