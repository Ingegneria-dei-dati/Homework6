import pandas as pd
import os

# --- PUNTO 2: DEFINIZIONE SCHEMA MEDIATO ---
def definisci_schema_mediato():
    """
    Definisce i mapping basati sull'analisi reale: 
    - VIN al 100% per US_CARS.
    - Rimozione di 'condition' da US_CARS per evitare errori.
    """
    map_craigslist = {
        'id': 'id_univoco', 'VIN': 'vin', 'manufacturer': 'marca', 'model': 'modello',
        'year': 'anno', 'price': 'prezzo', 'odometer': 'miglia', 'fuel': 'carburante',
        'transmission': 'cambio', 'drive': 'trazione', 'cylinders': 'cilindri',
        'type': 'carrozzeria', 'paint_color': 'colore_est', 'condition': 'condizione',
        'title_status': 'titolo', 'lat': 'latitudine', 'long': 'longitudine',
        'state': 'stato_usa', 'posting_date': 'data_post'
    }

    map_us_cars = {
        'listing_id': 'id_univoco', 'vin': 'vin', 'make_name': 'marca', 'model_name': 'modello',
        'year': 'anno', 'price': 'prezzo', 'mileage': 'miglia', 'fuel_type': 'carburante',
        'transmission': 'cambio', 'wheel_system': 'trazione', 'engine_cylinders': 'cilindri',
        'body_type': 'carrozzeria', 'exterior_color': 'colore_est',
        'salvage': 'titolo', 'latitude': 'latitudine', 'longitude': 'longitudine',
        'dealer_zip': 'stato_usa', 'listed_date': 'data_post'
    }
    return map_craigslist, map_us_cars

# --- PUNTO 3: ALLINEAMENTO FISICO ---
def esegui_integrazione_allineata():
    print("--- INIZIO PUNTO 3: INTEGRAZIONE ALLINEATA ---")
    cartella = "dataset"
    file_uscita = os.path.join(cartella, "auto_integrate_finale.csv")
    
    # Pulizia file precedente
    if os.path.exists(file_uscita):
        os.remove(file_uscita)

    # Otteniamo i mapping dal Punto 2
    map_c, map_u = definisci_schema_mediato()
    
    config = [
        {'path': "craigslist_vehicles.csv", 'label': "CRAIGSLIST", 'map': map_c},
        {'path': "used_cars_data.csv", 'label': "US_CARS", 'map': map_u}
    ]

    scrivi_header = True
    for c in config:
        full_path = os.path.join(cartella, c['path'])
        if not os.path.exists(full_path):
            print(f"Salto {c['label']}: file non trovato in {full_path}")
            continue

        print(f"Elaborazione {c['label']}...")
        
        # Identifichiamo quali colonne del mapping esistono davvero nel file CSV
        cols_nel_file = pd.read_csv(full_path, nrows=0).columns.tolist()
        map_valido = {k: v for k, v in c['map'].items() if k in cols_nel_file}
        
        # Lettura a blocchi (chunking) per gestire i 3 milioni di righe
        reader = pd.read_csv(full_path, usecols=list(map_valido.keys()), chunksize=150000, low_memory=False)
        
        count = 0
        for chunk in reader:
            # 1. Rinominiamo secondo lo schema mediato
            chunk = chunk.rename(columns=map_valido)
            
            # 2. Forziamo l'etichetta provenienza per ogni singola riga
            chunk['provenienza'] = c['label']
            
            # 3. Standardizzazione testo (tutto minuscolo e pulito)
            cols_testo = ['marca', 'modello', 'carburante', 'cambio', 'trazione']
            for col in cols_testo:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype(str).str.lower().str.strip()
            
            # 4. Scrittura incrementale
            chunk.to_csv(file_uscita, mode='a', index=False, header=scrivi_header)
            scrivi_header = False
            
            count += len(chunk)
            print(f"Righe accumulate per {c['label']}: {count}", end='\r')
            
        print(f"\n{c['label']} completato ({count} righe).")

    print(f"\n--- PUNTO 3 COMPLETATO CON SUCCESSO ---")
    print(f"File integrato salvato in: {file_uscita}")

if __name__ == "__main__":
    esegui_integrazione_allineata()