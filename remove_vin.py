import pandas as pd

# ============================================================
# 1. CARICAMENTO GROUND-TRUTH
# ============================================================

records = pd.read_csv("dataset/ground_truth_records.csv")
pairs = pd.read_csv("dataset/ground_truth_pairs.csv")

print(f"Records: {records.shape}")
print(f"Pairs: {pairs.shape}")

# ============================================================
# 2. RIMOZIONE VIN
# ============================================================

if "Vin" in records.columns:
    records = records.drop(columns=["Vin"])
    print("Colonna VIN rimossa dai record.")

# le pairs NON contengono VIN (solo record_id)
# quindi non serve modificarle

# ============================================================
# 3. SALVATAGGIO
# ============================================================

records.to_csv("dataset/ground_truth_records_no_vin.csv", index=False)
pairs.to_csv("dataset/ground_truth_pairs_no_vin.csv", index=False)

print("\nFile salvati:")
print("- dataset/ground_truth_records_no_vin.csv")
print("- dataset/ground_truth_pairs_no_vin.csv")
