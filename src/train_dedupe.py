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
OUTPUT_DIR = Path("dataset/results/dedupe_results")

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

    print("\n All experiments completed.")
