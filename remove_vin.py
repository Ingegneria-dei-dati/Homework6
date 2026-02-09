import pandas as pd
import os
import json

def prepare_datasets_for_training():
    """
    Punto 4.B: Genera la Ground Truth basata sul VIN e prepara il dataset 
    per il training eliminando gli identificatori diretti.
    """
    # File generato dal punto 4.A (Cleaning VIN)
    input_path = os.path.join("dataset", "mediated_cleaned.csv")
    if not os.path.exists(input_path):
        print(f"Errore: {input_path} non trovato. Esegui prima il cleaning dei VIN.")
        return

    # Caricamento del dataset pulito
    df = pd.read_csv(input_path, low_memory=False)
    
    # 1. Creazione della Ground Truth (Mappa dei Match)
    # NOTA: Usiamo 'Vin' (maiuscolo) e 'listing_id' come da schema mediato precedente
    print("Generazione mappa Ground Truth...")
    
    # Raggruppiamo i listing_id che condividono lo stesso Vin
    vin_groups = df.groupby('Vin')['listing_id'].apply(list).to_dict()
    
    # Filtriamo: teniamo solo i VIN che compaiono in più di un annuncio (veri match)
    # Gli annunci singoli non sono utili per addestrare i "match", ma servono per i "non-match"
    gt_matches = {vin: ids for vin, ids in vin_groups.items() if len(ids) > 1}

    # 2. Rimozione degli attributi sensibili (Data Leakage)
    # Rimuoviamo 'Vin' e 'Vin_status' perché non devono essere usati per il confronto
    # Manteniamo 'listing_id' come chiave primaria rinominandola in 'id' per Dedupe
    df_no_vin = df.drop(columns=['Vin', 'Vin_status']).rename(columns={'listing_id': 'id'})

    # 3. Salvataggio output
    output_dataset = os.path.join("dataset", "dataset_for_training.csv")
    output_gt = os.path.join("dataset", "ground_truth_map.json")
    
    df_no_vin.to_csv(output_dataset, index=False)
    
    with open(output_gt, "w") as f:
        json.dump(gt_matches, f, indent=4)

    print(f"\n{'='*40}")
    print(f"PUNTO 4.B COMPLETATO")
    print(f"Dataset per training (senza VIN): {output_dataset}")
    print(f"Mappa Ground Truth salvata: {output_gt}")
    print(f"Cluster di match trovati: {len(gt_matches)}")
    print(f"{'='*40}")

if __name__ == "__main__":
    prepare_datasets_for_training()