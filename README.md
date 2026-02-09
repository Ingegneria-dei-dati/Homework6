# Homework 6 – End-to-end project (B1/B2 + RecordLinkage + Dedupe + Ditto + optional LLM eval)

## 0) Prerequisiti
- Python 3.10+ consigliato
- File originali dentro `dataset/`:
  - `craigslist_vehicles.csv`
  - `used_cars_data.csv`

## 1) Setup (VS Code)
Apri la cartella `Homework6_project` in VS Code, poi nel terminale:

### Windows PowerShell
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows CMD
```bat
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Esecuzione (ordine consigliato)
1) Analisi sorgenti (punto 1)
```bash
python scripts/01_analyze_sources.py
```

2) Integrazione chunked (punto 2–3) → crea `dataset/integrated_cars.csv`
```bash
python scripts/02_integrate_chunked.py
```

3) Ground-truth con VIN (punto 4.A) → crea `dataset/ground_truth_records.csv`
```bash
python scripts/03_build_ground_truth.py
```

4) Rimozione VIN (punto 4.B) → crea `dataset/ground_truth_records_no_vin.csv`
```bash
python scripts/04_remove_vin.py
```

5) Pairs etichettate 0/1 (punto 4.D + base 4.C)
```bash
python scripts/05_build_pairs_safe.py --blocking B1
python scripts/05_build_pairs_safe.py --blocking B2
```

6) Split train/val/test (punto 4.C)
```bash
python scripts/06_split_pairs.py --pairs dataset/pairs_B1.csv --out-prefix dataset/B1
python scripts/06_split_pairs.py --pairs dataset/pairs_B2.csv --out-prefix dataset/B2
```

7) RecordLinkage (punto 4.E)
```bash
python scripts/07_recordlinkage_pipeline.py --blocking B1 --train dataset/B1_train.csv --test dataset/B1_test.csv --out results/rl_B1_metrics.json
python scripts/07_recordlinkage_pipeline.py --blocking B2 --train dataset/B2_train.csv --test dataset/B2_test.csv --out results/rl_B2_metrics.json
```

8) Dedupe (punto 4.F)
```bash
python scripts/08_dedupe_pipeline.py --train dataset/B1_train.csv --test dataset/B1_test.csv --out results/dedupe_B1.json
python scripts/08_dedupe_pipeline.py --train dataset/B2_train.csv --test dataset/B2_test.csv --out results/dedupe_B2.json
```

9) Ditto (punto 4.G)
- Export formato Ditto:
```bash
python scripts/09_export_ditto.py --pairs dataset/B1_train.csv --out ditto_data/B1_train.txt
python scripts/09_export_ditto.py --pairs dataset/B1_val.csv   --out ditto_data/B1_val.txt
python scripts/09_export_ditto.py --pairs dataset/B1_test.csv  --out ditto_data/B1_test.txt
python scripts/09_export_ditto.py --pairs dataset/B2_train.csv --out ditto_data/B2_train.txt
python scripts/09_export_ditto.py --pairs dataset/B2_val.csv   --out ditto_data/B2_val.txt
python scripts/09_export_ditto.py --pairs dataset/B2_test.csv  --out ditto_data/B2_test.txt
```
- Poi segui `scripts/09_run_ditto_optional.txt` (serve repo FAIR-DA4ER).

10) Tabella finale (punto 4.H)
```bash
python scripts/10_collect_results.py
```
Output: `results/final_comparison.csv`

## 3) Blocking usati (importante per relazione)
- Il dataset ha `state` quasi sempre nullo (~92%): blocchi con `state` danno 0 candidate pairs.
- Per evitare la "pair explosion" di `make+year` su milioni di record, usiamo feature derivate dal modello:
  - `model_tok`: primo token del modello (es. "civic lx" → "civic")
  - `model_pfx2`: prime 2 lettere del token (es. "civic" → "ci")
- B1: `make + year + model_pfx2` (più largo ma sicuro)
- B2: `make + year + model_tok` (più stretto)

## 4) Optional: LLM evaluation
Vedi `scripts/11_llm_eval_optional.py`.
