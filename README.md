# adabi-ml-project
## 1. Il Problema
La domanda centrale del progetto è: **Può il machine learning prevedere la compatibilità tra due persone?**
Il dataset simula profili di app di incontri per classificare le coppie come "compatibili" o "non compatibili" basandosi su 26 variabili di input (13 per ogni individuo).

## 2. Il Dataset
- **Fonte:** [Cupid's Algorithm su Kaggle](https://www.kaggle.com/datasets/likithagedipudi/cupids-algorithm).
- **Dimensioni:** 100.000 righe e 30 colonne.
- **Bilanciamento:** 43% compatibili / 57% non compatibili.
- **Variabili chiave:** Tratti Big Five (apertura, estroversione, ecc.), cronotipo, linguaggio dell'amore e ambizione professionale.

## 3. Metodologia e Preprocessing
Il workflow segue un approccio **End-to-End** per garantire la generalizzazione del modello ed evitare il *data leakage*:

1. **Split dei dati:** Suddivisione 80% training / 20% test.
2. **Preprocessing (ColumnTransformer):**
   - **Variabili Numeriche:** Standardizzazione tramite `StandardScaler` (età, tratti della personalità, cronotipo).
   - **Variabili Categoriali:** Codifica `OneHotEncoder` (posizione geografica, campo lavorativo, linguaggio dell'amore).
3. **Pipeline:** Integrazione del preprocessing e del modello per un'esecuzione atomica e coerente.

## 4. Modelli e Risultati

### Random Forest Classifier 
Il modello ha mostrato una maggiore capacità di gestire il rumore intrinseco dei dati (rumore gaussiano aggiunto nel dataset per realismo) [3, 17].

| Metrica | Punteggio (Cross-Validation) |
| :--- | :---: |
| **Precision** | 0.6340 |
| **Recall** | 0.3259 |
| **F1-Score** | **0.4305** |

### Matrice di Confusione
[[39229  6455]
 [23133 11183]]

## 5. Interpretazione e Conclusioni
Il modello raggiunge una **precisione del 63%**, il che significa che quando il sistema suggerisce una compatibilità, è significativamente più affidabile del caso (43%). Tuttavia, il basso valore di **recall (32%)** indica che molte coppie potenzialmente compatibili vengono scartate. Questo è giustificato dalla natura complessa e "rumorosa" del comportamento umano simulato nel dataset.

## 6. Sviluppi Futuri
