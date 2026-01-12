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
