import pandas as pd
import os

def analizza_dataset_completo(file_path, nome_sorgente):
    print(f"\n{'='*40}")
    print(f"ANALISI SORGENTE: {nome_sorgente}")
    print(f"{'='*40}")

    # Parametri per la gestione della memoria
    chunk_size = 100000
    total_rows = 0
    null_counts = None
    unique_trackers = {}

    # Verifichiamo se il file esiste
    if not os.path.exists(file_path):
        print(f"Errore: Il file '{file_path}' non è nella cartella!")
        return

    # Inizio lettura a pezzi
    reader = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)

    for i, chunk in enumerate(reader):
        # Inizializzazione al primo pezzo
        if i == 0:
            null_counts = pd.Series(0, index=chunk.columns)
            for col in chunk.columns:
                unique_trackers[col] = set()

        total_rows += len(chunk)

        # 1. Conteggio Nulli
        null_counts += chunk.isnull().sum()

        # 2. Conteggio Unici (usiamo i set per non duplicare i valori tra i chunk)
        for col in chunk.columns:
            # Aggiungiamo i valori unici del chunk al set (escludendo i nulli)
            unique_trackers[col].update(chunk[col].dropna().unique())

        print(f"Righe elaborate: {total_rows}...", end='\r')

    # Creazione del report finale
    report = []
    for col in null_counts.index:
        n_unici = len(unique_trackers[col])
        report.append({
            'Attributo': col,
            'Nulli (%)': round((null_counts[col] / total_rows) * 100, 2),
            'Unici (%)': round((n_unici / total_rows) * 100, 4),
            'Conteggio Unici': n_unici
        })

    df_report = pd.DataFrame(report)
    # Mostra tutti i risultati ordinati per percentuale di nulli
    print("\n")
    print(df_report.sort_values(by='Nulli (%)', ascending=False).to_string(index=False))
    return df_report

# --- ESECUZIONE ---

# 1. Analisi del dataset Craigslist
# Assicurati che il nome del file sia esattamente questo
report_craigslist = analizza_dataset_completo('dataset/craigslist_vehicles.csv', "CRAIGSLIST")

# 2. Analisi del dataset US Used Cars
# Assicurati che il nome del file sia esattamente questo
report_us_cars = analizza_dataset_completo('dataset/used_cars_data.csv', "US USED CARS")