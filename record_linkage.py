import pandas as pd
import recordlinkage
from recordlinkage.index import Block

# 1. CARICAMENTO DATI
print("Caricamento dataset...")
df_records = pd.read_csv("dataset/ground_truth_records_no_vin.csv", low_memory=False)

# Pulizia e conversione tipi per risparmiare RAM
df_records['make'] = df_records['make'].astype(str).str.lower().str.strip()
df_records['model'] = df_records['model'].astype(str).str.lower().str.strip()
df_records['state'] = df_records['state'].astype(str).str.upper().str.strip()
# Convertiamo l'anno in intero per un matching più veloce
df_records['year'] = pd.to_numeric(df_records['year'], errors='coerce').fillna(0).astype(int)

df_craigslist = df_records[df_records['source'] == 'craigslist'].set_index('record_id')
df_us_used_cars = df_records[df_records['source'] == 'us_used_cars'].set_index('record_id')

del df_records # Libera memoria

# ==========================================
# 4.D: BLOCKING MULTI-LIVELLO (B1 e B2)
# ==========================================
print("\nGenerazione coppie candidate (Blocking B1 e B2)...")

indexer = recordlinkage.Index()

# Strategia B1: Molto specifica (Marca + Anno + Stato)
# Impedisce che auto uguali in stati diversi sovraccarichino la memoria
indexer.add(Block(['make', 'year', 'state']))

# Strategia B2: Specifica sul Modello (Marca + Modello + Anno)
# Rispetto a prima, aggiungiamo 'year' per evitare il crash sui modelli troppo comuni
indexer.add(Block(['make', 'model', 'year']))

print("Costruzione indice in corso...")
candidate_links = indexer.index(df_craigslist, df_us_used_cars)

print(f"Coppie candidate generate: {len(candidate_links)}")

# ==========================================
# 4.E: REGOLE DI COMPARAZIONE
# ==========================================
print("\nInizio comparazione...")
compare_cl = recordlinkage.Compare()

# Definiamo le regole basate sullo schema mediato
compare_cl.string('model', 'model', method='jarowinkler', threshold=0.85, label='model_score')
compare_cl.numeric('price', 'price', method='gauss', offset=500, label='price_score')
compare_cl.numeric('mileage', 'mileage', method='gauss', offset=5000, label='mileage_score')
compare_cl.exact('transmission', 'transmission', label='trans_exact')
compare_cl.exact('fuel_type', 'fuel_type', label='fuel_exact')

# Calcolo feature
print("Calcolo feature (questo richiederà tempo in base al numero di coppie)...")
features = compare_cl.compute(candidate_links, df_craigslist, df_us_used_cars)

print("\nProcesso completato!")
print(features.head())

# Salvataggio
features.to_csv("dataset/features_output.csv")
print("File salvato in: dataset/features_output.csv")