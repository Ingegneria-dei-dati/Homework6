import pandas as pd
import numpy as np
import os
import re

def validate_vin_checksum(vin):
    """Calcola e valida il check digit del VIN in posizione 9 (Standard ISO 3779)."""
    if len(vin) != 17: return False
    
    # Mappatura caratteri-valori e pesi standard ISO
    trans = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'J':1, 'K':2, 'L':3, 'M':4, 
             'N':5, 'P':7, 'R':9, 'S':2, 'T':3, 'U':4, 'V':5, 'W':6, 'X':7, 'Y':8, 'Z':9}
    weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
    
    try:
        # Trasformazione caratteri in valori numerici
        values = [(trans[c] if c in trans else int(c)) for c in vin]
        total = sum(v * w for v, w in zip(values, weights))
        check_digit = total % 11
        check_digit = 'X' if check_digit == 10 else str(check_digit)
        return vin[8] == check_digit
    except:
        return False

def clean_and_repair_vin_pipeline():
    """
    Pipeline per la pulizia del VIN e generazione Ground Truth.
    Allineata ai nomi colonne dello Schema Mediato.
    """
    # Usiamo il file generato dallo script di allineamento
    input_path = os.path.join("dataset", "integrated_cars.csv")
    if not os.path.exists(input_path):
        print(f"Errore: {input_path} non trovato. Esegui prima lo script di allineamento.")
        return

    # Caricamento chunked non necessario qui se la RAM lo permette, 
    # ma usiamo low_memory per sicurezza
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Dataset integrato caricato: {len(df)} record.")

    # 1. Normalizzazione stringa e rimozione placeholder
    # Nota: La colonna nello script di allineamento era "Vin"
    df['Vin'] = df['Vin'].astype(str).str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
    placeholders = ['NONE', 'NAN', 'NULL', 'UNKNOWN', 'EMPTY', '0', 'X', 'ANY']
    df.loc[df['Vin'].isin(placeholders), 'Vin'] = np.nan

    # 2. Euristiche per pattern implausibili
    def is_invalid_pattern(v):
        if pd.isna(v) or v == 'NAN' or v == '': return True
        if len(set(v)) < 4: return True # Es. '1111111111111'
        if 'XXX' in v or v.startswith('000'): return True
        if v.isdigit() or v.isalpha(): return True # VIN reali sono alfanumerici misti
        return False

    df.loc[df['Vin'].apply(is_invalid_pattern), 'Vin'] = np.nan

    # 3. Gestione storici VIN (< 1981)
    # Isoliamo i record prodotti prima della standardizzazione ISO del 1981
    legacy_mask = (df['year'] < 1981) & (df['Vin'].notna())
    df_legacy = df[legacy_mask].copy()
    df = df[~legacy_mask].copy()

    # 4. Creazione Whitelist per riparazioni
    def is_whitelist_candidate(v):
        if pd.isna(v) or len(v) != 17: return False
        if any(c in v for c in 'IOQ'): return False
        return validate_vin_checksum(v)

    whitelist = set(df['Vin'][df['Vin'].apply(is_whitelist_candidate)].unique())

    # 5. Logica di Riparazione e Classificazione
    def repair_logic(v):
        if pd.isna(v): return v, 'dropped'
        
        # Correzione errori comuni di OCR/Digitazione (I->1, O->0, Q->0)
        if len(v) == 17 and any(c in v for c in 'IOQ'):
            v_rep = v.replace('I', '1').replace('O', '0').replace('Q', '0')
            if v_rep in whitelist or validate_vin_checksum(v_rep): 
                return v_rep, 'repaired'
            return v, 'dropped'

        if len(v) == 17:
            if validate_vin_checksum(v):
                return v, 'valid'
            else:
                return v, 'dropped' # Checksum fallito
        
        return v, 'ambiguous'

    print("Validazione Checksum e riparazione caratteri ambigui...")
    results = df['Vin'].apply(repair_logic)
    df['Vin'] = [r[0] for r in results]
    df['Vin_status'] = [r[1] for r in results]

    # 6. Filtro finale per la Ground Truth (solo Valid e Repaired)
    # Questi record costituiranno la base certa per valutare Dedupe e Ditto
    df_cleaned = df[df['Vin_status'].isin(['valid', 'repaired'])].copy()
    
    # Salvataggio output coerente con i nomi della relazione
    df_cleaned.to_csv(os.path.join("dataset", "mediated_cleaned.csv"), index=False)
    df_legacy.to_csv(os.path.join("dataset", "vin_legacy.csv"), index=False)
    
    print("\n" + "="*40)
    print("PUNTO 4.A COMPLETATO")
    print(f"Record validi/riparati (mediated_cleaned.csv): {len(df_cleaned)}")
    print(f"Record legacy isolati (vin_legacy.csv): {len(df_legacy)}")
    print("="*40)

if __name__ == "__main__":
    clean_and_repair_vin_pipeline()