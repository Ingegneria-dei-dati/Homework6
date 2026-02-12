import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score

def calcola_metriche(file_jsonl, nome_task):
    if not os.path.exists(file_jsonl):
        print(f"⚠️ Errore: Il file {file_jsonl} non esiste. Hai lanciato il matcher?")
        return

    y_true = []
    y_pred = []
    
    with open(file_jsonl, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                y_true.append(int(data['label']))
                # Ditto considera match se la confidenza è > 0.5
                y_pred.append(1 if data['match_confidence'] > 0.5 else 0)
            except Exception as e:
                continue
            
    if len(y_true) == 0:
        print(f"⚠️ Nessun dato trovato in {file_jsonl}")
        return

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"--- Risultati {nome_task} ---")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")

if __name__ == "__main__":
    # Assicurati di essere nella cartella dove Ditto ha generato i file .jsonl
    # Solitamente /content/FAIR-DA4ER/
    calcola_metriche("/content/FAIR-DA4ER/output_B1.jsonl", "Cars_B1")
    calcola_metriche("/content/FAIR-DA4ER/output_B2.jsonl", "Cars_B2")