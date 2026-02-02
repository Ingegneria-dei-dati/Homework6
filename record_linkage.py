import argparse
import json
import time
import numpy as np
import pandas as pd

import recordlinkage
from recordlinkage.index import Block

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# ============================================================
# FUNZIONI UTILI
# ============================================================

def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza alcune colonne testuali/numeriche per rendere più stabile
    il confronto tra record (riduce rumore tipo maiuscole/spazi).
    """
    # Stringhe: make/model in lowercase, state in uppercase
    if 'make' in df.columns:
        df['make'] = df['make'].astype(str).str.lower().str.strip()
    if 'model' in df.columns:
        df['model'] = df['model'].astype(str).str.lower().str.strip()
    if 'state' in df.columns:
        df['state'] = df['state'].astype(str).str.upper().str.strip()

    # year: numerico intero (0 se mancante)
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)

    # price/mileage: numerici float (NaN se non convertibile)
    for col in ['price', 'mileage']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def build_index(blocking: str, df_left: pd.DataFrame, df_right: pd.DataFrame):
    """
    Costruisce le candidate pairs usando UNA sola strategia di blocking per volta:
    - B1: make + year + state
    - B2: make + model + year
    """
    indexer = recordlinkage.Index()

    if blocking == "B1":
        indexer.add(Block(['make', 'year', 'state']))
    elif blocking == "B2":
        indexer.add(Block(['make', 'model', 'year']))
    else:
        raise ValueError("blocking deve essere 'B1' oppure 'B2'")

    candidate_links = indexer.index(df_left, df_right)
    return candidate_links


def compute_features(pairs_index, df_left: pd.DataFrame, df_right: pd.DataFrame) -> pd.DataFrame:
    """
    Definisce le regole di confronto (4.E) e calcola la matrice feature.
    Le feature sono quelle “classiche” del tuo script.
    """
    compare = recordlinkage.Compare()

    # Regole di confronto (schema mediato)
    compare.string('model', 'model', method='jarowinkler', threshold=0.85, label='model_score')
    compare.numeric('price', 'price', method='gauss', offset=500, label='price_score')
    compare.numeric('mileage', 'mileage', method='gauss', offset=5000, label='mileage_score')
    compare.exact('transmission', 'transmission', label='trans_exact')
    compare.exact('fuel_type', 'fuel_type', label='fuel_exact')

    # Calcolo feature
    features = compare.compute(pairs_index, df_left, df_right)

    # Pulizia minima: NaN -> 0 (utile per Logistic Regression)
    features = features.fillna(0.0)

    return features


def load_pairs_csv(path: str) -> pd.DataFrame:
    """
    Carica il CSV delle coppie candidate e labels.
    Deve avere colonne: id_l, id_r, label
    """
    pairs = pd.read_csv(path)
    required = {'id_l', 'id_r', 'label'}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"{path} manca colonne richieste: {missing}")

    # Normalizza label in int 0/1
    pairs['label'] = pd.to_numeric(pairs['label'], errors='coerce').fillna(0).astype(int)
    pairs['label'] = pairs['label'].clip(0, 1)

    return pairs


def pairs_df_to_multiindex(pairs: pd.DataFrame) -> pd.MultiIndex:
    """
    Converte un DataFrame con colonne id_l/id_r in un MultiIndex
    compatibile con recordlinkage (index di coppie).
    """
    return pd.MultiIndex.from_frame(pairs[['id_l', 'id_r']])


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Record linkage completo (features + training + evaluation) per B1 o B2")
    parser.add_argument("--blocking", choices=["B1", "B2"], required=True,
                        help="Strategia di blocking da usare (B1 o B2).")
    parser.add_argument("--records", default="dataset/ground_truth_records_no_vin.csv",
                        help="CSV record (NO VIN) con record_id e source.")
    parser.add_argument("--pairs_train", required=True, help="CSV coppie train con colonne id_l,id_r,label")
    parser.add_argument("--pairs_test", required=True, help="CSV coppie test con colonne id_l,id_r,label")
    parser.add_argument("--features_out", default=None,
                        help="(Opzionale) Path dove salvare le feature del test (CSV).")
    parser.add_argument("--pred_out", default="results/preds.csv",
                        help="CSV output predizioni per le coppie del test.")
    parser.add_argument("--metrics_out", default="results/metrics.json",
                        help="JSON output con metriche e tempi.")
    args = parser.parse_args()

    # ------------------------------------------------------------
    # 1) CARICAMENTO RECORD
    # ------------------------------------------------------------
    print("Caricamento record...")
    df = pd.read_csv(args.records, low_memory=False)

    if 'record_id' not in df.columns or 'source' not in df.columns:
        raise ValueError("Il file records deve contenere almeno 'record_id' e 'source'.")

    # Normalizzazione campi (come fai tu)
    df = normalize_records(df)

    # Split per sorgente (cross-source)
    df_left = df[df['source'] == 'craigslist'].set_index('record_id')
    df_right = df[df['source'] == 'us_used_cars'].set_index('record_id')
    del df  # libera RAM

    print(f"Record left (craigslist): {len(df_left)}")
    print(f"Record right (us_used_cars): {len(df_right)}")

    # ------------------------------------------------------------
    # 2) CARICAMENTO COPPIE TRAIN/TEST (ground-truth a coppie)
    # ------------------------------------------------------------
    print("\nCaricamento coppie train/test...")
    train_pairs = load_pairs_csv(args.pairs_train)
    test_pairs = load_pairs_csv(args.pairs_test)

    # Importante: teniamo solo coppie che esistono davvero negli indici
    # (evita errori se qualche id non c’è)
    left_ids = set(df_left.index)
    right_ids = set(df_right.index)

    train_pairs = train_pairs[train_pairs['id_l'].isin(left_ids) & train_pairs['id_r'].isin(right_ids)]
    test_pairs = test_pairs[test_pairs['id_l'].isin(left_ids) & test_pairs['id_r'].isin(right_ids)]

    print(f"Coppie train: {len(train_pairs)} (match%={train_pairs['label'].mean()*100:.2f}%)")
    print(f"Coppie test:  {len(test_pairs)} (match%={test_pairs['label'].mean()*100:.2f}%)")

    if len(train_pairs) == 0 or len(test_pairs) == 0:
        raise ValueError("Train o test vuoti dopo il filtro sugli ID. Controlla i file pairs_*.")

    # ------------------------------------------------------------
    # 3) (4.D) BLOCKING: qui lo usiamo solo come 'controllo' opzionale
    #     e come info sul numero candidate pairs teoriche del blocco.
    #     La vera valutazione la facciamo sulle coppie dei file train/test.
    # ------------------------------------------------------------
    print(f"\n[Info] Generazione candidate pairs da blocking {args.blocking} (solo per statistiche)...")
    t0 = time.perf_counter()
    candidate_links = build_index(args.blocking, df_left, df_right)
    t1 = time.perf_counter()
    print(f"Candidate pairs (blocking {args.blocking}): {len(candidate_links)} in {t1 - t0:.2f}s")

    # ------------------------------------------------------------
    # 4) (4.E) FEATURE ENGINEERING su train e test (regole di confronto)
    # ------------------------------------------------------------
    # Convertiamo le coppie train/test in MultiIndex
    train_index = pairs_df_to_multiindex(train_pairs)
    test_index = pairs_df_to_multiindex(test_pairs)

    print("\nCalcolo feature TRAIN...")
    t_feat_train_start = time.perf_counter()
    X_train = compute_features(train_index, df_left, df_right)
    t_feat_train_end = time.perf_counter()

    print("Calcolo feature TEST...")
    t_feat_test_start = time.perf_counter()
    X_test = compute_features(test_index, df_left, df_right)
    t_feat_test_end = time.perf_counter()

    y_train = train_pairs['label'].to_numpy()
    y_test = test_pairs['label'].to_numpy()

    # ------------------------------------------------------------
    # 5) TRAINING: Logistic Regression sulle feature
    # ------------------------------------------------------------
    print("\nTraining modello (Logistic Regression)...")
    t_train_start = time.perf_counter()

    # class_weight='balanced' aiuta se hai tanti più 0 che 1
    clf = LogisticRegression(max_iter=200, class_weight='balanced', n_jobs=None)
    clf.fit(X_train.values, y_train)

    t_train_end = time.perf_counter()

    # ------------------------------------------------------------
    # 6) INFERENZA + VALUTAZIONE
    # ------------------------------------------------------------
    print("Predizione su TEST...")
    t_inf_start = time.perf_counter()
    proba = clf.predict_proba(X_test.values)[:, 1]
    pred = (proba >= 0.5).astype(int)  # soglia standard 0.5, puoi tararla su validation
    t_inf_end = time.perf_counter()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average='binary', zero_division=0
    )
    cm = confusion_matrix(y_test, pred)  # [[TN, FP],[FN, TP]]

    print("\n=== RISULTATI ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:")
    print(cm)

    # ------------------------------------------------------------
    # 7) SALVATAGGI
    # ------------------------------------------------------------
    # Predizioni coppie test
    out_pred = test_pairs[['id_l', 'id_r', 'label']].copy()
    out_pred['proba'] = proba
    out_pred['pred'] = pred

    # crea cartella results se non esiste
    import os
    os.makedirs(os.path.dirname(args.pred_out) or ".", exist_ok=True)
    out_pred.to_csv(args.pred_out, index=False)
    print(f"\nPredizioni salvate in: {args.pred_out}")

    # (Opzionale) salva features test
    if args.features_out:
        os.makedirs(os.path.dirname(args.features_out) or ".", exist_ok=True)
        X_test.to_csv(args.features_out)
        print(f"Feature test salvate in: {args.features_out}")

    # Metriche + tempi
    metrics = {
        "blocking": args.blocking,
        "n_records_left": int(len(df_left)),
        "n_records_right": int(len(df_right)),
        "n_candidate_pairs_blocking": int(len(candidate_links)),
        "n_train_pairs": int(len(train_pairs)),
        "n_test_pairs": int(len(test_pairs)),
        "match_rate_train": float(train_pairs["label"].mean()),
        "match_rate_test": float(test_pairs["label"].mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "timing_seconds": {
            "blocking_index_only_stats": float(t1 - t0),
            "feature_train": float(t_feat_train_end - t_feat_train_start),
            "feature_test": float(t_feat_test_end - t_feat_test_start),
            "training": float(t_train_end - t_train_start),
            "inference": float(t_inf_end - t_inf_start),
        }
    }

    os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metriche salvate in: {args.metrics_out}")


if __name__ == "__main__":
    main()




"""
import pandas as pd
import recordlinkage
from recordlinkage.index import Block

# 1. CARICAMENTO DATI
print("Caricamento dataset...")
df_records = pd.read_csv("dataset/ground_truth_records_no_vin.csv", low_memory=False)

# Pulizia e conversione tipi per risparmiare RAM
df_records['make'] = df_records['make'].astype(str).str.lower().str.strip()
df_records['model'] = df_records['model'].astype(str).str.lower().str.strip()
df_records['state'] = df_records['state'].astype(str).str.upper().str.strip()
# Convertiamo l'anno in intero per un matching più veloce
df_records['year'] = pd.to_numeric(df_records['year'], errors='coerce').fillna(0).astype(int)

df_craigslist = df_records[df_records['source'] == 'craigslist'].set_index('record_id')
df_us_used_cars = df_records[df_records['source'] == 'us_used_cars'].set_index('record_id')

del df_records # Libera memoria

# ==========================================
# 4.D: BLOCKING MULTI-LIVELLO (B1 e B2)
# ==========================================
print("\nGenerazione coppie candidate (Blocking B1 e B2)...")

indexer = recordlinkage.Index()

# Strategia B1: Molto specifica (Marca + Anno + Stato)
# Impedisce che auto uguali in stati diversi sovraccarichino la memoria
indexer.add(Block(['make', 'year', 'state']))

# Strategia B2: Specifica sul Modello (Marca + Modello + Anno)
# Rispetto a prima, aggiungiamo 'year' per evitare il crash sui modelli troppo comuni
indexer.add(Block(['make', 'model', 'year']))

print("Costruzione indice in corso...")
candidate_links = indexer.index(df_craigslist, df_us_used_cars)

print(f"Coppie candidate generate: {len(candidate_links)}")

# ==========================================
# 4.E: REGOLE DI COMPARAZIONE
# ==========================================
print("\nInizio comparazione...")
compare_cl = recordlinkage.Compare()

# Definiamo le regole basate sullo schema mediato
compare_cl.string('model', 'model', method='jarowinkler', threshold=0.85, label='model_score')
compare_cl.numeric('price', 'price', method='gauss', offset=500, label='price_score')
compare_cl.numeric('mileage', 'mileage', method='gauss', offset=5000, label='mileage_score')
compare_cl.exact('transmission', 'transmission', label='trans_exact')
compare_cl.exact('fuel_type', 'fuel_type', label='fuel_exact')

# Calcolo feature
print("Calcolo feature (questo richiederà tempo in base al numero di coppie)...")
features = compare_cl.compute(candidate_links, df_craigslist, df_us_used_cars)

print("\nProcesso completato!")
print(features.head())

# Salvataggio
features.to_csv("dataset/features_output.csv")
print("File salvato in: dataset/features_output.csv")
"""