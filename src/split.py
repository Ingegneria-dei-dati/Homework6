import pandas as pd
import numpy as np
import json
import os
import random
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURAZIONE
# ============================================================
DATASET_PATH = "dataset/dataset_for_training.csv"
GT_PATH = "dataset/ground_truth_map.json"
OUTPUT_DIR = "dataset/splits"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Fase 4.C: Inizio Split delle Entità e Generazione Coppie...")

    # 1. Caricamento Dati
    if not os.path.exists(DATASET_PATH) or not os.path.exists(GT_PATH):
        print("Errore: Dataset o Ground Truth non trovati.")
        return

    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df['id'] = df['id'].astype(str) # Coerenza tipi
    
    with open(GT_PATH, "r") as f:
        gt_map = json.load(f)

    # 2. Split delle Entità (VIN) - Prevenzione Data Leakage
    # Dividiamo i VIN, non i singoli annunci
    all_vins = list(gt_map.keys())
    train_vins, temp_vins = train_test_split(all_vins, test_size=0.30, random_state=SEED)
    val_vins, test_vins = train_test_split(temp_vins, test_size=0.50, random_state=SEED)

    # Mappa inversa ID -> VIN per controllo rapido dei negativi
    id_to_vin = {str(idx): vin for vin, ids in gt_map.items() for idx in ids}
    all_ids = df['id'].tolist()

    # 3. Funzione Core per la generazione del dataset etichettato
    def create_labeled_pairs(vin_subset, subset_name):
        pos_pairs = []
        print(f"   Elaborazione {subset_name} ({len(vin_subset)} entità)...")

        # Generazione Coppie POSITIVE
        for vin in vin_subset:
            ids = gt_map[vin]
            n = len(ids)
            if n < 2: continue
            
            # Limite per evitare esplosione combinatoria su VIN molto frequenti
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    pos_pairs.append({'id1': ids[i], 'id2': ids[j], 'label': 1})
                    count += 1
                    if count >= 10: break
                if count >= 10: break

        df_pos = pd.DataFrame(pos_pairs)
        num_neg_needed = len(df_pos)

        # Generazione Coppie NEGATIVE 
        neg_pairs = []
        attempts = 0
        while len(neg_pairs) < num_neg_needed and attempts < num_neg_needed * 5:
            id1, id2 = random.sample(all_ids, 2)
            # Un negativo è valido se i due ID non condividono lo stesso VIN
            if id_to_vin.get(id1) != id_to_vin.get(id2):
                neg_pairs.append({'id1': id1, 'id2': id2, 'label': 0})
            attempts += 1

        combined = pd.concat([df_pos, pd.DataFrame(neg_pairs)], ignore_index=True)
        # Mescoliamo le righe
        return combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 4. Esecuzione e Salvataggio
    splits = [
        (train_vins, "train_pairs.csv"),
        (val_vins, "val_pairs.csv"),
        (test_vins, "test_pairs.csv")
    ]

    for vins, filename in splits:
        result_df = create_labeled_pairs(vins, filename)
        result_df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
        print(f"      Salvato {filename}: {len(result_df)} coppie.")

    print("\n" + "="*40)
    print("PUNTO 4.C COMPLETATO: Coppie pronte per il training.")
    print("="*40)

if __name__ == "__main__":
    main()