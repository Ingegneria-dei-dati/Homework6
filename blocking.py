import pandas as pd
import os
import json

def generate_blocks(df, group_cols, min_block_size=2):
    """Genera i blocchi raggruppando per le colonne specificate."""
    df_temp = df.copy()
    for col in group_cols:
        df_temp[col] = df_temp[col].fillna('unknown').astype(str)
    
    grouped = df_temp.groupby(group_cols)['id'].apply(list).to_dict()
    return {str(k): v for k, v in grouped.items() if len(v) >= min_block_size}

def apply_and_save_blocking_split():
    input_path = os.path.join("dataset", "dataset_for_training.csv")
    splits_dir = os.path.join("dataset", "splits")
    blocks_dir = os.path.join("dataset", "blocks")
    
    if not os.path.exists(input_path):
        print(f"Errore: {input_path} non trovato.")
        return

    # 1. Caricamento dataset principale
    df = pd.read_csv(input_path, low_memory=False)
    df['id'] = df['id'].astype(str)

    # 2. Assegnazione dello split a ogni riga del dataset
    # Leggiamo gli ID dai file generati nel punto 4.C
    print("Mappatura degli split in corso...")
    df['split'] = 'exclude' # Default: record che non partecipano a match o coppie

    for s_name in ['train', 'val', 'test']:
        pair_path = os.path.join(splits_dir, f"{s_name}_pairs.csv")
        if os.path.exists(pair_path):
            pair_df = pd.read_csv(pair_path)
            # Prendiamo tutti gli ID univoci presenti in questo split
            ids_in_split = set(pair_df['id1'].astype(str)).union(set(pair_df['id2'].astype(str)))
            df.loc[df['id'].isin(ids_in_split), 'split'] = s_name

    os.makedirs(blocks_dir, exist_ok=True)

    # 3. Generazione blocchi separati per ogni split
    # Consideriamo solo train, val e test (escludiamo i record non campionati se necessario)
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        if split_df.empty:
            continue

        # --- B1: Broad (Make + Year) ---
        b1_blocks = generate_blocks(split_df, ['make', 'year'])
        b1_path = os.path.join(blocks_dir, f"blocking_B1_{split}.json")
        with open(b1_path, "w") as f:
            json.dump(b1_blocks, f)

        # --- B2: Strict (Make + Fuel + Transmission + Year) ---
        b2_blocks = generate_blocks(split_df, ['make', 'fuel_type', 'transmission', 'year'])
        b2_path = os.path.join(blocks_dir, f"blocking_B2_{split}.json")
        with open(b2_path, "w") as f:
            json.dump(b2_blocks, f)

        print(f"\nSplit: {split}")
        print(f"   B1 ({b1_path}): {len(b1_blocks)} blocchi.")
        print(f"   B2 ({b2_path}): {len(b2_blocks)} blocchi.")

    print("\nBlocking completato con successo rispettando gli split.")

if __name__ == "__main__":
    apply_and_save_blocking_split()