'''import pandas as pd
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
report_us_cars = analizza_dataset_completo('dataset/used_cars_data.csv', "US USED CARS")'''


import pandas as pd
import os
import matplotlib.pyplot as plt

def profiling_completo(file_path, nome_sorgente):
    print(f"\n" + "="*70)
    print(f" PROFILING ATTRIBUTI: {nome_sorgente} ".center(70, "="))
    print("="*70)

    if not os.path.exists(file_path):
        print(f"Errore: Il file {file_path} non è presente.")
        return

    # 1. ELABORAZIONE DATI (Memory Efficient)
    chunk_size = 100000
    total_rows = 0
    null_counts = None
    unique_trackers = {}

    reader = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)

    for i, chunk in enumerate(reader):
        if i == 0:
            null_counts = pd.Series(0, index=chunk.columns)
            for col in chunk.columns: unique_trackers[col] = set()
        
        total_rows += len(chunk)
        null_counts += chunk.isnull().sum()
        for col in chunk.columns:
            unique_trackers[col].update(chunk[col].dropna().unique())
        
        print(f"Righe elaborate: {total_rows}...", end='\r')

    # 2. CALCOLO PERCENTUALI
    report = []
    for col in null_counts.index:
        n_unici = len(unique_trackers[col])
        null_pct = (null_counts[col] / total_rows) * 100
        unique_pct = (n_unici / total_rows) * 100
        
        report.append({
            'Attributo': col,
            'Nulli (%)': round(null_pct, 2),
            'Unici (%)': round(unique_pct, 4),
            'Conteggio Unici': n_unici
        })

    # Ordiniamo per % di nulli per una migliore visualizzazione
    df_report = pd.DataFrame(report).sort_values(by='Nulli (%)', ascending=False)

    # 3. STAMPA A VIDEO
    print("\n")
    print(df_report.to_string(index=False))

    # 4. LOGICA DI RIMOZIONE (>70% Missing)
    to_drop = df_report[df_report['Nulli (%)'] > 70]['Attributo'].tolist()
    print(f"\n[DECISIONE] Colonne suggerite per rimozione (>70% nulli):")
    print(f"-> {to_drop if to_drop else 'Nessuna colonna sopra soglia'}")

    # 5. SALVATAGGIO CSV
    results_dir = "dataset/results/eda"
    os.makedirs(results_dir, exist_ok=True)
    file_csv = os.path.join(results_dir, f"report_{nome_sorgente.lower()}.csv")
    df_report.to_csv(file_csv, index=False)

    # 6. GENERAZIONE GRAFICO VISIVO
    # Calcolo altezza dinamica: 0.3 pollici per ogni attributo (minimo 8 pollici)
    altezza_dinamica = max(8, len(df_report) * 0.2)
    
    plt.figure(figsize=(12, altezza_dinamica))
    
    # height=0.7 riduce lo spessore delle barre lasciando più spazio tra loro
    bars = plt.barh(df_report['Attributo'], df_report['Nulli (%)'], 
                    color='skyblue', edgecolor='navy', height=0.7)
    
    # Linea rossa per la soglia del 70%
    plt.axvline(x=70, color='red', linestyle='--', linewidth=2, label='Soglia 70% (Rimozione)')
    
    plt.xlabel('Percentuale di Valori Nulli (%)', fontsize=12)
    plt.ylabel('Attributi', fontsize=12)
    plt.title(f'Distribuzione Valori Nulli - {nome_sorgente}', fontsize=14, pad=20)
    
    # Ottimizzazione etichette asse Y
    plt.yticks(fontsize=10) 
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle=':', alpha=0.7)
    plt.legend(loc='upper right')
    

    # Salvataggio immagine con bbox_inches='tight' per non tagliare le scritte lunghe
    file_img = os.path.join(results_dir, f"plot_nulli_{nome_sorgente.lower()}.png")
    plt.tight_layout()
    plt.savefig(file_img, bbox_inches='tight', dpi=150)
    plt.close() # Chiude la figura per liberare memoria
    
    print(f"[OK] Report CSV: {file_csv}")
    print(f"[OK] Grafico PNG: {file_img}")
    
    return df_report

# --- ESECUZIONE ---
#report_cl = profiling_completo('dataset/craigslist_vehicles.csv', "CRAIGSLIST")
report_uc = profiling_completo('dataset/used_cars_data.csv', "USED_CARS")