import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Split stratificato train/val/test (70/15/15) per coppie con label 0/1")
    parser.add_argument("--pairs", required=True, help="CSV con colonne id_l,id_r,label")
    parser.add_argument("--out-prefix", required=True,
                        help="Prefisso output (es: dataset/B1 oppure dataset/B2). Verranno creati *_train.csv, *_val.csv, *_test.csv")
    parser.add_argument("--seed", type=int, default=42, help="Seed per ripetibilità")
    parser.add_argument("--train-size", type=float, default=0.70, help="Frazione train (default 0.70)")
    parser.add_argument("--val-size", type=float, default=0.15, help="Frazione validation (default 0.15)")
    parser.add_argument("--test-size", type=float, default=0.15, help="Frazione test (default 0.15)")
    args = parser.parse_args()

    # ------------------------------------------------------------
    # 1) CARICAMENTO COPPIE
    # ------------------------------------------------------------
    df = pd.read_csv(args.pairs)

    required = {'id_l', 'id_r', 'label'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.pairs} manca colonne richieste: {missing}")

    # Normalizza label in 0/1
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    df['label'] = df['label'].clip(0, 1)

    # Rimuovi eventuali duplicati di coppia
    df = df.drop_duplicates(subset=['id_l', 'id_r']).reset_index(drop=True)

    # Controllo: devono esistere sia 0 che 1
    vc = df['label'].value_counts(dropna=False).to_dict()
    print("Distribuzione label (prima dello split):", vc)
    if df['label'].nunique() < 2:
        raise ValueError("Nel file pairs c'è una sola classe (solo 0 o solo 1). Serve ground-truth con 0 e 1.")

    # ------------------------------------------------------------
    # 2) SPLIT STRATIFICATO: prima train vs temp, poi temp -> val/test
    # ------------------------------------------------------------
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size

    # Piccolo controllo di consistenza
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train-size + val-size + test-size deve fare 1.0")

    # Primo split: train e temp (val+test)
    df_train, df_temp = train_test_split(
        df,
        test_size=(1.0 - train_size),
        random_state=args.seed,
        shuffle=True,
        stratify=df['label']
    )

    # Secondo split: val e test dalla parte temp
    # proporzione relativa: val/(val+test)
    val_ratio_in_temp = val_size / (val_size + test_size)

    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1.0 - val_ratio_in_temp),
        random_state=args.seed,
        shuffle=True,
        stratify=df_temp['label']
    )

    # ------------------------------------------------------------
    # 3) SALVATAGGIO
    # ------------------------------------------------------------
    out_train = f"{args.out_prefix}_train.csv"
    out_val = f"{args.out_prefix}_val.csv"
    out_test = f"{args.out_prefix}_test.csv"

    os.makedirs(os.path.dirname(out_train) or ".", exist_ok=True)

    df_train.to_csv(out_train, index=False)
    df_val.to_csv(out_val, index=False)
    df_test.to_csv(out_test, index=False)

    # Report finale (utile per debug e relazione)
    def report(name, dfx):
        print(f"{name}: n={len(dfx)} match%={dfx['label'].mean()*100:.2f}%  counts={dfx['label'].value_counts().to_dict()}")

    print("\n=== SPLIT COMPLETATO ===")
    report("TRAIN", df_train)
    report("VAL", df_val)
    report("TEST", df_test)

    print("\nFile creati:")
    print(" -", out_train)
    print(" -", out_val)
    print(" -", out_test)


if __name__ == "__main__":
    main()


"""
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# 1. CARICAMENTO COPPIE (GROUND-TRUTH)
# ============================================================

pairs = pd.read_csv("dataset/ground_truth_pairs_no_vin.csv")

print(f"Coppie totali: {pairs.shape[0]}")

# ============================================================
# 2. SPLIT TRAIN / TEMP
# ============================================================

train_pairs, temp_pairs = train_test_split(
    pairs,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

# ============================================================
# 3. SPLIT VALIDATION / TEST
# ============================================================

val_pairs, test_pairs = train_test_split(
    temp_pairs,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

# ============================================================
# 4. SALVATAGGIO
# ============================================================

train_pairs.to_csv("dataset/train_pairs.csv", index=False)
val_pairs.to_csv("dataset/val_pairs.csv", index=False)
test_pairs.to_csv("dataset/test_pairs.csv", index=False)

print("\nSplit completato:")
print(f"Train: {train_pairs.shape[0]}")
print(f"Validation: {val_pairs.shape[0]}")
print(f"Test: {test_pairs.shape[0]}")
"""