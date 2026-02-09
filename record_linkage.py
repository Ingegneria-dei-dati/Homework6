import pandas as pd
import numpy as np
import recordlinkage
import os
import json
import time
from sklearn.metrics import precision_score, recall_score, f1_score

def run_complete_evaluation_suite():
    # 1. Caricamento Dati
    df_path = os.path.join("dataset", "dataset_for_training.csv")
    df = pd.read_csv(df_path, low_memory=False)
    df['id'] = df['id'].astype(str)
    df = df.set_index('id')
    df = df[~df.index.duplicated(keep='first')]
    
    with open(os.path.join("dataset", "ground_truth_map.json"), "r") as f:
        gt_map = json.load(f)
    id_to_vin = {str(idx): vin for vin, ids in gt_map.items() for idx in ids}
    
    results_dir = os.path.join("dataset", "results")
    os.makedirs(results_dir, exist_ok=True)

    weights = np.array([2.0, 4.0, 2.0, 1.0, 1.0])
    weight_sum = weights.sum()
    
    all_sensitivity_data = [] 
    global_results = []
    SOGLIA_SCELTA = 0.85

    splits = ['train', 'val', 'test']
    strategies = ['B1', 'B2']

    print("Inizio analisi completa (Sensibilità, Performance e Tempi)...")

    for split in splits:
        for strat in strategies:
            filename = f"blocking_{strat}_{split}.json"
            block_path = os.path.join("dataset", "blocks", filename)
            
            if not os.path.exists(block_path):
                continue
                
            with open(block_path, "r") as f:
                blocks = json.load(f)

            # --- FASE TRAINING (Euristico) ---
            # Nel record linkage manuale il training è il tempo di definizione dei pesi (quasi 0)
            start_train = time.time()
            # Simulazione caricamento logica/pesi
            _ = np.array([2.0, 4.0, 2.0, 1.0, 1.0]) 
            end_train = time.time()
            training_time = end_train - start_train

            # Generazione coppie candidate
            candidate_pairs = []
            df_index_set = set(df.index)
            for block_ids in blocks.values():
                if 1 < len(block_ids) < 200: 
                    valid_ids = [str(idx) for idx in block_ids if str(idx) in df_index_set]
                    if len(valid_ids) >= 2:
                        a = np.array(valid_ids)
                        res = np.array(np.meshgrid(a, a)).T.reshape(-1, 2)
                        res = res[res[:, 0] < res[:, 1]]
                        candidate_pairs.append(res)
            
            if not candidate_pairs: continue
            all_pairs = np.vstack(candidate_pairs)
            
            # --- FASE INFERENZA (Comparazione e Scoring) ---
            start_inf = time.time()
            
            compare = recordlinkage.Compare()
            compare.string('make', 'make', method='levenshtein', threshold=0.85)
            compare.string('model', 'model', method='jarowinkler', threshold=0.85)
            compare.exact('year', 'year')
            compare.numeric('price', 'price', method='lin', offset=0, scale=0.1) 
            compare.numeric('mileage', 'mileage', method='lin', offset=0, scale=0.1)

            links = pd.MultiIndex.from_arrays([all_pairs[:, 0], all_pairs[:, 1]])
            features = compare.compute(links, df)
            scores = (features.values * weights).sum(axis=1) / weight_sum
            
            end_inf = time.time()
            inference_time = end_inf - start_inf

            # Ground Truth per valutazione
            v1 = np.array([id_to_vin.get(str(p[0])) for p in all_pairs])
            v2 = np.array([id_to_vin.get(str(p[1])) for p in all_pairs])
            y_true = ((v1 == v2) & (v1 != None)).astype(int)

            # --- STUDIO SENSIBILITÀ ---
            for t in [0.70, 0.75, 0.80, 0.85, 0.90]:
                y_pred_t = (scores >= t).astype(int)
                p = precision_score(y_true, y_pred_t, zero_division=0)
                r = recall_score(y_true, y_pred_t, zero_division=0)
                f1 = f1_score(y_true, y_pred_t, zero_division=0)
                
                all_sensitivity_data.append({
                    'Split': split, 'Strategia': strat, 'Soglia': t, 
                    'Precision': round(p, 4), 'Recall': round(r, 4), 'F1': round(f1, 4)
                })

            # --- REPORT FINALE CON TEMPI ---
            y_pred_final = (scores >= SOGLIA_SCELTA).astype(int)
            global_results.append({
                'Split': split, 
                'Blocking': strat, 
                'Candidati': len(all_pairs),
                'Precision': round(precision_score(y_true, y_pred_final, zero_division=0), 4),
                'Recall': round(recall_score(y_true, y_pred_final, zero_division=0), 4),
                'F1': round(f1_score(y_true, y_pred_final, zero_division=0), 4),
                'Train_Time(s)': round(training_time, 6),
                'Inference_Time(s)': round(inference_time, 4)
            })

    # OUTPUT E SALVATAGGIO
    sensitivity_df = pd.DataFrame(all_sensitivity_data)
    global_df = pd.DataFrame(global_results)

    print("\n" + "="*85)
    print("1. STUDIO DELLA SENSIBILITÀ (TEST SET)")
    print("="*85)
    print(sensitivity_df[sensitivity_df['Split'] == 'test'].to_string(index=False))

    print("\n" + "="*105)
    print(f"2. REPORT GLOBALE E TEMPI (SOGLIA FISSA A {SOGLIA_SCELTA})")
    print("="*105)
    print(global_df.sort_values(by=['Split', 'Blocking']).to_string(index=False))

    sensitivity_df.to_csv(os.path.join(results_dir, "full_sensitivity_analysis.csv"), index=False)
    global_df.to_csv(os.path.join(results_dir, "global_evaluation_report_with_times.csv"), index=False)
    
    print(f"\nReport salvati in {results_dir}")

if __name__ == "__main__":
    run_complete_evaluation_suite()