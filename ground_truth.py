import pandas as pd
import numpy as np
from itertools import combinations

# ============================================================
# 1. CARICAMENTO DATASET INTEGRATO
# ============================================================

df = pd.read_csv("dataset/integrated_cars.csv")
print(f"Dataset integrato caricato: {df.shape}")

# ============================================================
# 2. PULIZIA E NORMALIZZAZIONE VIN
# ============================================================

# 2.A. Uppercase e rimozione spazi
df["Vin"] = df["Vin"].astype(str).str.upper().str.strip()

# 2.B. VIN deve avere esattamente 17 caratteri
df.loc[df["Vin"].str.len() != 17, "Vin"] = np.nan

# 2.C. Rimozione VIN duplicati con valori inconsistenti
vin_groups = df.groupby("Vin")[["make", "model"]].nunique()
problematic_vins = vin_groups[(vin_groups["make"] > 1) | (vin_groups["model"] > 1)].index
df.loc[df["Vin"].isin(problematic_vins), "Vin"] = np.nan

# ============================================================
# 3. CREAZIONE GROUND-TRUTH
# ============================================================

# 3.A. Selezioniamo solo record con VIN valido
ground_truth = df.dropna(subset=["Vin"]).copy()
ground_truth.reset_index(drop=True, inplace=True)

# 3.B. Creiamo un record_id univoco
ground_truth["record_id"] = range(len(ground_truth))

print(f"Record validi per ground-truth: {ground_truth.shape[0]}")

# ============================================================
# 4. GENERAZIONE COPPIE DI MATCH
# ============================================================

pairs = []

# Raggruppiamo per VIN
vin_groups = ground_truth.groupby("Vin")
for vin, group in vin_groups:
    if len(group) > 1:
        # tutte le combinazioni di record con stesso VIN
        ids = group["record_id"].tolist()
        for a, b in combinations(ids, 2):
            pairs.append((a, b, 1))  # 1 = match

gt_pairs = pd.DataFrame(pairs, columns=["record_id_1", "record_id_2", "match"])
print(f"Coppie generate (match=1): {gt_pairs.shape[0]}")

# ============================================================
# 5. SALVATAGGIO FILE
# ============================================================

ground_truth.to_csv("dataset/ground_truth_records.csv", index=False)
gt_pairs.to_csv("dataset/ground_truth_pairs.csv", index=False)

print("\nGround-truth salvato in:")
print("- dataset/ground_truth_records.csv")
print("- dataset/ground_truth_pairs.csv")

# ============================================================
# 6. FACOLTATIVO: controlli
# ============================================================

# Controllo VIN duplicati residui
duplicates = ground_truth[ground_truth.duplicated(subset=["Vin"], keep=False)]
print(f"VIN duplicati residui: {duplicates.shape[0]}")
