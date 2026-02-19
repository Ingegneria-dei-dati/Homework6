# Progetto di Integrazione e Record Linkage: Mercato Automobilistico US

## 📌 Panoramica del Progetto
Il progetto affronta la sfida dell'**Entity Resolution**, integrando due sorgenti di dati eterogenee relative al mercato automobilistico statunitense:

- **Craigslist**: inserzioni informali, alto livello di rumore testuale.
- **US Used Cars**: dataset strutturato con attributi regolari e presenza sistematica del codice VIN.

L'obiettivo è progettare e confrontare diverse pipeline di **record linkage** per identificare record semanticamente equivalenti, valutando i trade-off tra accuratezza (Precision, Recall, F1) e sostenibilità computazionale.

## 🛠️ Architettura del Sistema

### 1. Schema Mediato e Normalizzazione
È stato definito uno schema comune basato sulla densità informativa (esclusione di campi con oltre il 70% di valori nulli) e stabilità semantica. Le fasi di pulizia includono:

- **Tipizzazione numerica:** conversione standard di anno, prezzo, chilometraggio e coordinate.
- **Normalizzazione categorica:** canonical mapping per campi come `fuel_type`, `drive` e `transmission`.
- **Standardizzazione testuale:** case folding e rimozione spazi per `make` e `model`.

### 2. Ground Truth
La **Ground Truth** è stata costruita utilizzando il **VIN** (Vehicle Identification Number) come identificatore univoco. La pipeline include la validazione del checksum (ISO 3779) e la correzione di errori comuni di digitazione o OCR.

Il dataset è stato suddiviso a livello di VIN (70% Train, 15% Val, 15% Test) per prevenire rigorosamente il **data leakage**.

### 3. Strategie di Blocking
Per ottimizzare i confronti, sono state implementate due strategie:

- **B1 (Broad):** basata su `make` e `year`, mirata a massimizzare la Recall.
- **B2 (Strict):** include anche `fuel_type` e `transmission` per ridurre drasticamente la dimensione dei blocchi.

## 🧪 Paradigmi a Confronto
Sono stati analizzati tre approcci metodologici:

**A. Record Linkage Manuale (Rule-based)**  
- Utilizza regole deterministiche e punteggi di similarità pesati tra attributi.  
- Soglia ottimale di classificazione: 0.85.

**B. Dedupe (Machine Learning Supervisionato)**  
- Classificatore logistico addestrato su coppie di esempi.  
- **Similarità:** fuzzy matching per stringhe, scala logaritmica per prezzi.  
- **Pipeline testate:**  
  - `P1_full`: tutti gli attributi  
  - `P2_minimal`: esclusione di `price` e `mileage` per ridurre il rumore

**C. Ditto (Deep Learning)**  
- Framework basato su Transformer (RoBERTa-base) che serializza i record in sequenze testuali.  
- Gestisce relazioni semantiche complesse, superando limiti dei confronti puramente sintattici.

## 📈 Risultati e Conclusioni

### Performance Finali (F1-Score su Test Set)

| Metodo                  | Blocking | F1-Score |
|-------------------------|-----------|---------|
| Record Linkage          | B2      | 0.87     |
| Dedupe (P2_minimal)     | B1      | 0.9114   |
| Ditto (RoBERTa)         | B1      | 0.9860   |


### Conclusioni
- **Dedupe** rappresenta il compromesso ottimale tra efficienza e precisione, risultando la soluzione più equilibrata per dataset tabellari già normalizzati.  
- **Ditto** massimizza la **Recall** (fino al 99.95%) ed è preferibile quando l’accuratezza è la priorità assoluta, a fronte di costi computazionali maggiori.

