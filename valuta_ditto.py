import json
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def analizza_esperimento_ditto(file_output, file_input, nome_pipeline, t_train, t_inf):
    """
    Analizza i risultati estratti da Ditto e calcola le metriche per la relazione.
    """
    if not os.path.exists(file_output):
        # Restituiamo comunque un record per mantenere la struttura della tabella
        return {
            "Pipeline": nome_pipeline,
            "Precision": "N/A",
            "Recall": "N/A",
            "F1-measure": "N/A",
            "Tempo Training": t_train,
            "Tempo Inferenza": t_inf,
            "Throughput (rec/s)": "N/A"
        }

    y_true = []
    y_pred = []
    
    # 1. Caricamento Ground Truth (Etichette reali)
    with open(file_input, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                y_true.append(int(parts[2]))

    # 2. Caricamento Predizioni Ditto
    with open(file_output, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                y_pred.append(int(data['match']))
            except: continue

    # Allineamento dei dati
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    if len(y_true) == 0: return None

    # Calcolo Metriche Statistiche
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calcolo Throughput (Efficienza)
    try:
        m, s = map(int, t_inf.replace('s', '').split('m '))
        sec_inf = m * 60 + s
        throughput = round(min_len / sec_inf, 2)
    except:
        throughput = 0
    
    return {
        "Pipeline": nome_pipeline, 
        "Precision": round(p, 4), 
        "Recall": round(r, 4), 
        "F1-measure": round(f1, 4), 
        "Tempo Training": t_train, 
        "Tempo Inferenza": t_inf, 
        "Throughput (rec/s)": throughput
    }

# --- CONFIGURAZIONE PERCORSI DRIVE ---
DRIVE_PATH = "/content/drive/MyDrive/FAIR_DATA_OUTPUT"

# Configuriamo le pipeline Ditto (Strategy 1 vs Strategy 2)
esperimenti_config = [
    {
        "nome": "B1-ditto",
        "input": f"{DRIVE_PATH}/cars_B1/test_small.txt", 
        "output": f"{DRIVE_PATH}/output_B1.jsonl",
        "t_train": "24m", 
        "t_inf": "3m 55s"
    },
    {
        "nome": "B2-ditto",
        "input": f"{DRIVE_PATH}/cars_B2/test_small.txt",
        "output": f"{DRIVE_PATH}/output_B2.jsonl",
        "t_train": "25m", 
        "t_inf": "4m 38s"
    }
]

# --- ESECUZIONE ANALISI ---
report_data = []
for config in esperimenti_config:
    risultato = analizza_esperimento_ditto(
        config["output"], 
        config["input"], 
        config["nome"], 
        config["t_train"], 
        config["t_inf"]
    )
    if risultato:
        report_data.append(risultato)

# --- SALVATAGGIO E VISUALIZZAZIONE ---
if report_data:
    df_relazione = pd.DataFrame(report_data)
    
    # Salvataggio su Drive in formato CSV
    file_relazione = f"{DRIVE_PATH}/Analisi_Prestazioni_Ditto.csv"
    df_relazione.to_csv(file_relazione, index=False)
    
    print(f"✅ Report tecnico generato: {file_relazione}")
    print("\n" + "="*100)
    print(df_relazione.to_string(index=False))
    print("="*100)
else:
    print("❌ Nessun dato trovato per generare la relazione.")