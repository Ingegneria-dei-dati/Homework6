import pandas as pd
import os

def genera_ground_truth_vin():
    print("--- SVOLGIMENTO PUNTO 4.A: GENERAZIONE GROUND-TRUTH ---")
    
    cartella = "dataset"
    path_integrato = os.path.join(cartella, "auto_integrate_finale.csv")
    path_gt = os.path.join(cartella, "ground_truth.csv")
    
    # Carichiamo solo le colonne necessarie per risparmiare RAM
    df = pd.read_csv(path_integrato, usecols=['id_univoco', 'vin', 'provenienza'], low_memory=False)

    # Strategia di pulizia ad-hoc per dati rumorosi
    def pulisci_vin_strict(v):
        if pd.isna(v) or str(v).lower() in ['nan', 'none', 'n/a', '0']: 
            return None
        # Rimuove tutto ciò che non è lettera o numero e rende maiuscolo
        v_clean = ''.join(e for e in str(v) if e.isalnum()).upper()
        # Teniamo solo VIN significativi (min 11 caratteri)
        return v_clean if len(v_clean) >= 11 else None

    print("Pulizia dei VIN rumorosi in corso...")
    df['vin_clean'] = df['vin'].apply(pulisci_vin_strict)

    # Separazione delle sorgenti
    c_df = df[df['provenienza'] == 'CRAIGSLIST'].dropna(subset=['vin_clean'])
    u_df = df[df['provenienza'] == 'US_CARS'].dropna(subset=['vin_clean'])

    # Identificazione dei match (Intersezione VIN)
    vins_comuni = set(c_df['vin_clean']).intersection(set(u_df['vin_clean']))
    print(f"VIN comuni trovati: {len(vins_comuni)}")

    coppie = []
    
    # 1. Generazione dei MATCH (Label 1)
    # Creiamo coppie di ID che hanno lo stesso VIN
    for v in list(vins_comuni)[:2000]: # Limite per efficienza addestramento
        id_c = c_df[c_df['vin_clean'] == v]['id_univoco'].iloc[0]
        id_u = u_df[u_df['vin_clean'] == v]['id_univoco'].iloc[0]
        coppie.append({'id_1': id_c, 'id_2': id_u, 'label': 1})

    # 2. Generazione dei NON-MATCH (Label 0)
    # Selezioniamo coppie casuali (probabilità di match accidentale è quasi zero)
    if len(coppie) > 0:
        c_rand = c_df.sample(len(coppie), replace=True)
        u_rand = u_df.sample(len(coppie), replace=True)
        for i in range(len(coppie)):
            coppie.append({
                'id_1': c_rand.iloc[i]['id_univoco'], 
                'id_2': u_rand.iloc[i]['id_univoco'], 
                'label': 0
            })

    # Salvataggio del file
    df_gt = pd.DataFrame(coppie)
    df_gt.to_csv(path_gt, index=False)
    print(f"Ground-Truth salvato con {len(df_gt)} coppie in: {path_gt}")

if __name__ == "__main__":
    genera_ground_truth_vin()