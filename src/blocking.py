import pandas as pd
import os
import json

def generate_blocks(df, group_cols, min_block_size=2):
    """Genera i blocchi raggruppando per le colonne specificate con normalizzazione."""
    df_temp = df.copy()
    
    for col in group_cols:
        # Normalizzazione e pulizia dell'anno
        # Trasformiamo in stringa, tutto minuscolo, rimuoviamo spazi bianchi ai bordi
        df_temp[col] = df_temp[col].fillna('unknown').astype(str).str.lower().str.strip()
        
        # Se la colonna è l'anno, rimuoviamo il ".0" (es. "2018.0" -> "2018")
        if col == 'year':
            df_temp[col] = df_temp[col].str.replace('.0', '', regex=False)
    
    grouped = df_temp.groupby(group_cols)['id'].apply(list).to_dict()
    return {str(k): v for k, v in grouped.items() if len(v) >= min_block_size}

def apply_and_save_blocking_split():
    input_path = os.path.join("dataset", "dataset_for_training.csv")
    splits_dir = os.path.join("dataset", "splits")
    blocks_dir = os.path.join("dataset", "blocks")
    
    if not os.path.exists(input_path):
        print(f"Errore: {input_path} non trovato.")
        return

    # 1. Caricamento dataset
    df = pd.read_csv(input_path, low_memory=False)
    df['id'] = df['id'].astype(str)

    # 2. Mappatura degli split (fondamentale per evitare Data Leakage)
    print("Mappatura degli split in corso...")
    df['split'] = 'exclude' 

    for s_name in ['train', 'val', 'test']:
        pair_path = os.path.join(splits_dir, f"{s_name}_pairs.csv")
        if os.path.exists(pair_path):
            pair_df = pd.read_csv(pair_path)
            ids_in_split = set(pair_df['id1'].astype(str)).union(set(pair_df['id2'].astype(str)))
            df.loc[df['id'].isin(ids_in_split), 'split'] = s_name

    os.makedirs(blocks_dir, exist_ok=True)

    # 3. Generazione blocchi indipendenti B1 e B2
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        if split_df.empty:
            continue

        # --- STRATEGIA B1: Broad (Marca + Anno) ---
        # Obiettivo: Massima Recall. Ora più robusta grazie alla pulizia stringhe.
        b1_blocks = generate_blocks(split_df, ['make', 'year'])
        b1_path = os.path.join(blocks_dir, f"blocking_B1_{split}.json")
        with open(b1_path, "w") as f:
            json.dump(b1_blocks, f)

        # --- STRATEGIA B2: Strict (Marca + Fuel + Transmission + Year) ---
        # Obiettivo: Efficienza. Ora più precisa eliminando discrepanze di formattazione.
        b2_blocks = generate_blocks(split_df, ['make', 'fuel_type', 'transmission', 'year'])
        b2_path = os.path.join(blocks_dir, f"blocking_B2_{split}.json")
        with open(b2_path, "w") as f:
            json.dump(b2_blocks, f)

        print(f"\nSplit: {split}")
        print(f"   B1 (Normalizzato): {len(b1_blocks)} blocchi.")
        print(f"   B2 (Normalizzato): {len(b2_blocks)} blocchi.")

    print("\nBlocking completato con normalizzazione applicata.")

if __name__ == "__main__":
    apply_and_save_blocking_split()