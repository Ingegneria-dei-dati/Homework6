import argparse
import os
import time
import numpy as np
import pandas as pd

import recordlinkage
from recordlinkage.index import Block


# ============================================================
# FUNZIONI UTILI
# ============================================================

def detect_vin_column(df: pd.DataFrame) -> str:
    """
    Trova automaticamente il nome della colonna VIN.
    Gestisce casi comuni: 'Vin', 'VIN', 'vin'
    """
    for c in ["Vin", "VIN", "vin"]:
        if c in df.columns:
            return c
    raise ValueError("Non trovo la colonna VIN. Mi aspettavo 'Vin' oppure 'VIN' oppure 'vin'.")


def normalize_records_for_blocking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizzazione minima per rendere il blocking più stabile e ridurre rumore:
    - make/model: lowercase + strip
    - state: uppercase + strip
    - year: int (0 se mancante)
    """
    if "make" in df.columns:
        df["make"] = df["make"].astype(str).str.lower().str.strip()
    if "model" in df.columns:
        df["model"] = df["model"].astype(str).str.lower().str.strip()
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.upper().str.strip()

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    return df


def build_candidate_pairs(blocking: str, left: pd.DataFrame, right: pd.DataFrame):
    """
    Costruisce candidate pairs tra due sorgenti usando una strategia di blocking.
    """
    indexer = recordlinkage.Index()

    if blocking == "B1":
        # B1: marca + anno + stato (molto restrittivo)
        indexer.add(Block(["make", "year", "state"]))
    elif blocking == "B2":
        # B2: marca + modello + anno (meno restrittivo su state, più su model)
        indexer.add(Block(["make", "model", "year"]))
    else:
        raise ValueError("blocking deve essere 'B1' oppure 'B2'")

    return indexer.index(left, right)


def count_total_positive_pairs_by_vin(df: pd.DataFrame, vin_col: str) -> int:
    """
    Conta quanti match positivi (coppie) esistono in totale basandosi sui gruppi VIN.
    Questo serve per stimare quanti match teorici esistono (ground truth completa).
    NB: Qui contiamo solo cross-source? Per semplicità contiamo su tutto il dataset.
    """
    # Consideriamo solo VIN non nulli
    tmp = df[df[vin_col].notna()].copy()
    if tmp.empty:
        return 0

    # Per ogni VIN con k record, le coppie possibili sono k*(k-1)/2
    counts = tmp.groupby(vin_col).size().to_numpy()
    total_pairs = int(np.sum(counts * (counts - 1) // 2))
    return total_pairs


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Genera coppie candidate con label 0/1 per un blocking (B1 o B2) usando VIN SOLO per etichettare."
    )
    parser.add_argument("--blocking", choices=["B1", "B2"], required=True,
                        help="Strategia di blocking: B1 (make+year+state) oppure B2 (make+model+year).")
    parser.add_argument("--records", default="dataset/ground_truth_records.csv",
                        help="CSV record con VIN pulito (es: ground_truth_records.csv).")
    parser.add_argument("--out", required=True,
                        help="Output CSV: con colonne id_l,id_r,label (es: dataset/pairs_B1.csv).")
    parser.add_argument("--negative-ratio", type=int, default=10,
                        help="Quanti negativi (label=0) tenere per ogni positivo (label=1). Default 10.")
    parser.add_argument("--max-total-pairs", type=int, default=200000,
                        help="Limite massimo di coppie salvate (pos+neg). Default 200000.")
    parser.add_argument("--seed", type=int, default=42, help="Seed per campionamento riproducibile.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ------------------------------------------------------------
    # 1) CARICAMENTO DATI
    # ------------------------------------------------------------
    print("Caricamento dataset record...")
    df = pd.read_csv(args.records, low_memory=False)

    # Controlli colonne minime
    required = {"record_id", "source", "make", "model", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mancano colonne richieste nel file records: {missing}")

    vin_col = detect_vin_column(df)
    print(f"Colonna VIN rilevata: '{vin_col}'")

    # Normalizzazione (come fai tu)
    df = normalize_records_for_blocking(df)

    # ------------------------------------------------------------
    # 2) SPLIT PER SORGENTE (cross-source linkage)
    # ------------------------------------------------------------
    left = df[df["source"] == "craigslist"].set_index("record_id")
    right = df[df["source"] == "us_used_cars"].set_index("record_id")

    print(f"Record left (craigslist): {len(left)}")
    print(f"Record right (us_used_cars): {len(right)}")

    if left.empty or right.empty:
        raise ValueError("Una delle due sorgenti è vuota. Controlla la colonna 'source' nel CSV.")

    # ------------------------------------------------------------
    # 3) STATISTICHE GROUND-TRUTH (solo informative)
    # ------------------------------------------------------------
    total_pos_pairs_all = count_total_positive_pairs_by_vin(df, vin_col)
    print(f"[Info] Coppie positive teoriche (tutte le sorgenti, da VIN): {total_pos_pairs_all}")

    # ------------------------------------------------------------
    # 4) 4.D: BLOCKING -> CANDIDATE PAIRS
    # ------------------------------------------------------------
    print(f"\nGenerazione candidate pairs con blocking {args.blocking}...")
    t0 = time.perf_counter()
    candidate_links = build_candidate_pairs(args.blocking, left, right)
    t1 = time.perf_counter()

    print(f"Candidate pairs generate: {len(candidate_links)} in {t1 - t0:.2f}s")

    if len(candidate_links) == 0:
        raise ValueError("Blocking ha generato 0 coppie. È troppo restrittivo o mancano dati in campi chiave.")

    # ------------------------------------------------------------
    # 5) LABELING con VIN (solo per creare label 0/1)
    # ------------------------------------------------------------
    # Creiamo un DataFrame con id_l, id_r
    pairs_df = pd.DataFrame(candidate_links.tolist(), columns=["id_l", "id_r"])

    # Recuperiamo VIN dei due lati
    # (notare: left/right hanno indice record_id)
    left_vin = left[[vin_col]].rename(columns={vin_col: "vin_l"})
    right_vin = right[[vin_col]].rename(columns={vin_col: "vin_r"})

    pairs_df = pairs_df.merge(left_vin, left_on="id_l", right_index=True, how="left")
    pairs_df = pairs_df.merge(right_vin, left_on="id_r", right_index=True, how="left")

    # label = 1 se VIN uguale e non nullo
    pairs_df["label"] = (
        pairs_df["vin_l"].notna() &
        pairs_df["vin_r"].notna() &
        (pairs_df["vin_l"] == pairs_df["vin_r"])
    ).astype(int)

    n_pos = int(pairs_df["label"].sum())
    n_total = len(pairs_df)
    print(f"Positivi trovati tra le candidate pairs: {n_pos} su {n_total} (match%={n_pos / n_total * 100:.3f}%)")

    # Se non troviamo positivi, probabilmente blocking troppo stretto
    if n_pos == 0:
        print("\n[ATTENZIONE] Nessun positivo trovato con questo blocking.")
        print("Possibili cause:")
        print("- VIN pochi / quasi tutti nulli dopo pulizia")
        print("- Blocking troppo restrittivo (es. state o year discordanti tra sorgenti)")
        print("- make/model/anno rumorosi e non coincidono tra sorgenti")
        print("Consiglio: prova l'altro blocking e/o rilassa B1 (es. rimuovendo state).")

    # ------------------------------------------------------------
    # 6) CAMPIONAMENTO NEGATIVI per ratio e limite totale
    # ------------------------------------------------------------
    # Separiamo positivi e negativi
    pos_df = pairs_df[pairs_df["label"] == 1][["id_l", "id_r", "label"]]
    neg_df = pairs_df[pairs_df["label"] == 0][["id_l", "id_r", "label"]]

    # Numero negativi desiderato
    target_neg = min(len(neg_df), args.negative_ratio * max(1, len(pos_df)))

    # Se ho tantissimi negativi, campiono
    if len(neg_df) > target_neg:
        neg_df = neg_df.sample(n=target_neg, random_state=args.seed)

    out_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # Limite massimo di coppie salvate
    if len(out_df) > args.max_total_pairs:
        # Manteniamo TUTTI i positivi e campioniamo i negativi rimanenti
        pos_keep = out_df[out_df["label"] == 1]
        neg_keep = out_df[out_df["label"] == 0]

        remaining = max(0, args.max_total_pairs - len(pos_keep))
        if len(neg_keep) > remaining:
            neg_keep = neg_keep.sample(n=remaining, random_state=args.seed)

        out_df = pd.concat([pos_keep, neg_keep], ignore_index=True)

    # Shuffle finale (per non avere prima tutti 1 poi tutti 0)
    out_df = out_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # ------------------------------------------------------------
    # 7) STIMA "BLOCKING RECALL" (approssimata)
    # ------------------------------------------------------------
    # Nota: total_pos_pairs_all include anche coppie SAME-source.
    # Qui stiamo facendo cross-source (left vs right), quindi non è perfetto.
    # Però ti dà un’idea di quante coppie positive catturi.
    captured_pos = int(pos_df.shape[0])
    if total_pos_pairs_all > 0:
        approx_recall = captured_pos / total_pos_pairs_all
        print(f"[Info] Blocking recall (approssimato, rispetto a tutti i pos da VIN): {approx_recall*100:.2f}%")
    else:
        print("[Info] Non posso stimare recall: nessuna coppia positiva teorica (VIN non disponibili).")

    # ------------------------------------------------------------
    # 8) SALVATAGGIO
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print("\n=== SALVATO ===")
    print(f"File: {args.out}")
    print(f"Totale coppie salvate: {len(out_df)}")
    print(f"Positivi salvati:      {int(out_df['label'].sum())}")
    print(f"Negativi salvati:      {int((out_df['label']==0).sum())}")
    print(f"Match% finale:         {out_df['label'].mean()*100:.3f}%")


if __name__ == "__main__":
    main()


"""
import argparse
import time
import pandas as pd
import numpy as np
import recordlinkage
from recordlinkage.index import Block


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    # Normalizza campi usati nel blocking
    df = df.copy()
    df["make"] = df["make"].astype(str).str.lower().str.strip()
    df["model"] = df["model"].astype(str).str.lower().str.strip()
    df["state"] = df["state"].astype(str).str.upper().str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    # year spesso ha NaN: per bloccare conviene usare un intero
    df["year"] = df["year"].fillna(-1).astype(int)
    # VIN come stringa pulita (qui NON facciamo pulizia logica: quella sta in ground_truth.py)
    if "Vin" in df.columns:
        df["Vin"] = df["Vin"].astype(str).str.strip().str.upper()
        df.loc[df["Vin"].isin(["NAN", "NONE", ""]), "Vin"] = np.nan
    return df


def build_pairs(records_csv: str, out_csv: str, blocking: str, negative_ratio: int, max_total_pairs: int, seed: int):
    t0 = time.perf_counter()
    df = pd.read_csv(records_csv, low_memory=False)
    if "record_id" not in df.columns:
        raise ValueError("records_csv deve contenere la colonna 'record_id'. Usa dataset/ground_truth_records.csv.")

    if "source" not in df.columns:
        raise ValueError("records_csv deve contenere la colonna 'source' (craigslist / us_used_cars).")

    if "Vin" not in df.columns:
        raise ValueError("records_csv deve contenere la colonna 'Vin' (serve solo per creare le label).")

    df = _prep(df)

    left = df[df["source"] == "craigslist"].set_index("record_id")
    right = df[df["source"] == "us_used_cars"].set_index("record_id")

    if len(left) == 0 or len(right) == 0:
        raise ValueError("Non trovo entrambe le sorgenti. Controlla i valori in colonna 'source'.")

    indexer = recordlinkage.Index()
    if blocking.upper() == "B1":
        indexer.add(Block(["make", "year", "state"]))
    elif blocking.upper() == "B2":
        indexer.add(Block(["make", "model", "year"]))
    else:
        raise ValueError("blocking deve essere B1 o B2")

    print(f"\n== BUILD PAIRS {blocking.upper()} ==")
    print(f"Left(craigslist): {len(left):,} | Right(us_used_cars): {len(right):,}")

    t_index0 = time.perf_counter()
    candidate_links = indexer.index(left, right)
    t_index1 = time.perf_counter()
    print(f"Candidate pairs generati: {len(candidate_links):,} (tempo indicizzazione: {t_index1-t_index0:.2f}s)")

    # Convert MultiIndex -> DataFrame
    pairs_df = pd.DataFrame({"id_l": candidate_links.get_level_values(0),
                             "id_r": candidate_links.get_level_values(1)})

    # Label con VIN (solo per costruire la ground-truth)
    vin_left = left["Vin"]
    vin_right = right["Vin"]
    pairs_df["vin_l"] = pairs_df["id_l"].map(vin_left)
    pairs_df["vin_r"] = pairs_df["id_r"].map(vin_right)
    pairs_df["label"] = ((pairs_df["vin_l"].notna()) &
                         (pairs_df["vin_r"].notna()) &
                         (pairs_df["vin_l"] == pairs_df["vin_r"])).astype(int)

    # Campionamento: teniamo TUTTI i positivi, e solo una quota di negativi
    pos = pairs_df[pairs_df["label"] == 1]
    neg = pairs_df[pairs_df["label"] == 0]

    print(f"Positivi (match via VIN): {len(pos):,} | Negativi: {len(neg):,}")

    rng = np.random.default_rng(seed)

    if len(pos) == 0:
        print("ATTENZIONE: nessun positivo trovato con questo blocking. Prova ad allentare B1/B2 o controlla VIN.")
        # comunque salviamo un subset di negativi per debugging
        keep_neg = min(len(neg), max_total_pairs if max_total_pairs > 0 else len(neg))
        sampled = neg.sample(n=keep_neg, random_state=seed) if keep_neg > 0 else neg
    else:
        # target negativi = negative_ratio * positivi
        target_neg = negative_ratio * len(pos)
        if max_total_pairs and max_total_pairs > 0:
            # rispetta anche un limite massimo complessivo
            max_neg_allowed = max(0, max_total_pairs - len(pos))
            target_neg = min(target_neg, max_neg_allowed)

        target_neg = min(target_neg, len(neg))
        if target_neg > 0:
            sampled_neg_idx = rng.choice(neg.index.to_numpy(), size=target_neg, replace=False)
            sampled_neg = neg.loc[sampled_neg_idx]
            sampled = pd.concat([pos, sampled_neg], ignore_index=True)
        else:
            sampled = pos.copy()

    sampled = sampled[["id_l", "id_r", "label"]].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    sampled.to_csv(out_csv, index=False)

    t1 = time.perf_counter()
    print(f"Salvato: {out_csv}")
    print(f"Totale coppie salvate: {len(sampled):,} | %match: {sampled['label'].mean()*100:.2f}%")
    print(f"Tempo totale: {t1-t0:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="dataset/ground_truth_records.csv",
                        help="CSV con record_id, source, Vin (VIN serve solo per label).")
    parser.add_argument("--out", default=None, help="Output CSV (es. dataset/pairs_B1.csv).")
    parser.add_argument("--blocking", choices=["B1", "B2"], required=True)
    parser.add_argument("--negative-ratio", type=int, default=10,
                        help="Quanti negativi tenere per ogni positivo (es. 10 => 1:10).")
    parser.add_argument("--max-total-pairs", type=int, default=200000,
                        help="Limite massimo coppie salvate (0 = nessun limite).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = args.out
    if out is None:
        out = f"dataset/pairs_{args.blocking}.csv"

    build_pairs(args.records, out, args.blocking, args.negative_ratio, args.max_total_pairs, args.seed)
"""