# Cupid's Algorithm 💘 — Previsione della Compatibilità di Coppia con Machine Learning

Progetto di Machine Learning basato sul dataset **Cupid's Algorithm**, che affronta
**due problemi complementari** a partire dagli stessi dati anagrafici e tratti di
personalità di coppia:

- **Classificazione binaria**: prevedere se due persone sono **compatibili o meno**
  (`compatible`, 0/1).
- **Regressione**:prevedere il **punteggio numerico di compatibilità**
  (`compatibility_score`).

Per ciascun problema vengono confrontati due modelli: una **Rete Neurale (MLP)** e
un modello ad albero (**Random Forest** per la classificazione, **Decision Tree** per
la regressione). Seguendo lo stesso flusso di lavoro: esplorazione dei dati, feature
engineering, preprocessing, addestramento, ottimizzazione degli iperparametri e
valutazione finale su un Test Set mai visto dai modelli.

---

## 📋 Indice

- [Dataset](#-dataset)
- [Struttura del repository](#-struttura-del-repository)
- [Requisiti](#-requisiti)
- [Come eseguire il notebook](#-come-eseguire-il-notebook)
- [Metodologia](#-metodologia)
- [Risultati](#-risultati)
- [Confronto finale tra i modelli](#-confronto-finale-tra-i-modelli)
- [Limiti e possibili miglioramenti](#-limiti-e-possibili-miglioramenti)
- [Autore](#-autore)
- [Licenza](#-licenza)

---

## 📊 Dataset

Il dataset utilizzato è **[Cupid's Algorithm](https://www.kaggle.com/datasets/likithagedipudi/cupids-algorithm)**,
disponibile su Kaggle e scaricato automaticamente tramite `kagglehub`.

Contiene 30 colonne per ogni coppia, tra cui:

- **Dati demografici**: età, livello di istruzione, luogo di residenza, settore
  professionale;
- **Tratti psicologici**: apertura mentale, estroversione, gradevolezza,
  coscienziosità, cronotipo, spontaneità, espressività emotiva, linguaggio dell'amore;
- **Variabili target**: `compatible` (0/1), `compatibility_score`,
  `relationship_longevity_months`.

> Nota: il dataset contiene rumore aggiunto intenzionalmente dall'autore originale,
> per simulare l'imprevedibilità reale delle relazioni umane.

---

## 📁 Struttura del repository

```
├── Progetto_ML_Classificazione.ipynb    # Notebook 1: classificazione binaria (compatible)
├── Progetto_ML_Regressione.ipynb        # Notebook 2: regressione (compatibility_score)
├── README.md                             # Questo file
└── Images/
├── classificazione/
│   ├── 01_istogrammi_variabili.png
│   ├── 02_matrice_correlazione.png
│   ├── 03_matrice_confusione_finale.png
│   ├── 04_curva_precision_recall.png
│   └── 05_ablation_study_feature_coppia.png
└── regressione/
├── 01_istogrammi_variabili.png
├── 02_correlazione_pearson_spearman.png
├── 03_loss_curve_mlp.png
├── 04_decision_tree_struttura.png
└── 05_confronto_finale_mlp_vs_tree.png
```
---

## ⚙️ Requisiti

- Python 3.10+
- Jupyter Notebook o Google Colab
- Le seguenti librerie:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib kagglehub
```

---

## ▶️ Come eseguire il notebook

1. Clona questo repository:
   ```bash
   git clone https://github.com/<tuo-utente>/<tuo-repo>.git
   cd <tuo-repo>
   ```
2. Installa le dipendenze (vedi sopra).
3. Apri il notebook `Cupids_Algorithm_ML_corretto.ipynb` con Jupyter o caricalo su
   Google Colab.
4. Esegui le celle in ordine dall'alto verso il basso: la prima cella scarica
   automaticamente il dataset da Kaggle tramite `kagglehub`.

---

## 🎯 Progetto 1 — Classificazione: coppia compatibile o no?

### Metodologia

1. **Importazione dei dati** — download automatico da Kaggle.
2. **Osservazione del dataset** — prima occhiata alla struttura dei dati.
3. **Training/Test split e Feature Engineering** — separazione 80/20 eseguita
   prima di ogni analisi approfondita, per evitare data leakage; creazione di
   variabili di coppia (`age_diff`, `ambition_diff`).
4. **Data Exploration** — controllo dei valori mancanti, distribuzioni e matrice
   di correlazione, calcolata solo sul Training Set.
5. **Verifica sperimentale delle feature (ablation study)** — confronto tra sole
   variabili originali, sole variabili di coppia, ed entrambe insieme.
6. **Preprocessing** — standardizzazione delle variabili numeriche e codifica
   delle variabili categoriche (One-Hot Encoding).
7. **Addestramento** — modello iniziale Random Forest.
8. **Validazione** — Cross Validation e F1-Score.
9. **Analisi degli errori** — matrice di confusione, precisione e recall.
10. **Ottimizzazione degli iperparametri** — `GridSearchCV` su un campione ridotto
    del Training Set per velocità; il modello finale viene poi riallenato
    sull'intero Training Set con la configurazione migliore trovata.
11. **Curve Precision/Recall e soglia di decisione** — soglia scelta nel punto
    di incrocio tra precisione e recall, calcolato tramite cross-validation.
12. **Confronto con una Rete Neurale (MLP)** — ottimizzata anch'essa con
    `GridSearchCV`, con la stessa procedura di soglia.
13. **Valutazione finale comparativa** — Accuracy, F1-Score e ROC-AUC su Test Set.


---

## 📈 Risultati

### Distribuzione delle variabili

![Istogrammi delle variabili](Images/01_istogrammi_variabili.png)

Le variabili numeriche (età, tratti di personalità) mostrano distribuzioni
abbastanza uniformi o a campana, senza valori mancanti o anomalie evidenti.

### Matrice di correlazione

![Matrice di correlazione](Images/02_matrice_correlazione.png)

Le correlazioni tra le variabili sono generalmente basse, il che esclude
correlazioni spurie e conferma che i dati seguono una logica psicologica
plausibile.

### Matrice di confusione del modello ottimizzato (Random Forest)

![Matrice di confusione finale](Images/03_matrice_confusione_finale.png)

Dopo l'ottimizzazione degli iperparametri, il Random Forest raggiunge un
**F1-Score di 0.56** sul Test Set.

### Curva Precision/Recall

![Curva Precision-Recall](Images/04_curva_precision_recall.png)

Il grafico mostra chiaramente il compromesso tra precisione e recall al variare
della soglia di decisione: aumentando la soglia il modello diventa più preciso ma
individua meno coppie realmente compatibili.

### Confronto Random Forest vs MLP in fase di addestramento

![Confronto RF vs MLP in training](Images/05_confronto_rf_vs_mlp_training.png)

---

## 🏆 Confronto finale tra i modelli

Valutazione sul Test Set (dati mai visti durante l'addestramento):

![Confronto finale RF vs MLP](Images/06_confronto_finale_rf_vs_mlp.png)

| Modello | Precisione | Recall | F1-Score |
|---|---|---|---|
| Random Forest (soglia 0.55) | 0.60 | 0.40 | 0.48 |
| **MLP / Rete Neurale (soglia 0.54)** | **0.68** | **0.56** | **0.62** |

La rete neurale (MLP) risulta il modello più efficace, riuscendo a individuare più
coppie realmente compatibili mantenendo comunque una precisione più alta rispetto al
Random Forest.

## 📈 Progetto 2 — Regressione: punteggio di compatibilità

### Metodologia

1. **Importazione dei dati** — download automatico da Kaggle.
2. **Osservazione del dataset** — quick look (head/info/describe/istogrammi).
3. **Train/test split** — eseguito subito dopo il quick look, prima di ogni
   analisi che metta in relazione le feature con il target.
4. **Data Exploration e creazione di nuove variabili** — feature di coppia
   (differenze tra tratti: `age_diff`, `openness_diff`, ecc.; variabili di
   somiglianza: `same_love_language`, `same_location`, `same_career`), con
   analisi di correlazione (Pearson e Spearman) calcolata solo sul Training Set.
5. **Data preprocessing e Pipeline** — standardizzazione delle variabili
   numeriche e codifica delle variabili categoriche.
6. **Definizione e Addestramento del modello** — Rete Neurale (`MLPRegressor`).
7. **Learning curve e validazione interna** — analisi della loss e dello score
   di validazione durante il training.
8. **Valutazione sul Test Set** — RMSE.
9. **Confronto con un Decision Tree Regressor** — ottimizzazione degli
   iperparametri con `GridSearchCV` (profondità, numero di foglie).
10. **Visualizzazione della struttura dell'albero** — analisi delle feature più
    rilevanti per le prime divisioni.
11. **Valutazione finale comparativa** — RMSE, MAE e R² tra i due modelli.

### Risultati

### Confronto finale

| Modello | RMSE | MAE | R² |
|---|---|---|---|
| Decision Tree | _da aggiornare_ | _da aggiornare_ | _da aggiornare_ |
| **Rete Neurale (MLP)** | _da aggiornare_ | _da aggiornare_ | _da aggiornare_ |

---

## ⚠️ Limiti e possibili miglioramenti

- L'MLP non è stato ottimizzato con una ricerca sistematica degli iperparametri
  (a differenza del Random Forest, ottimizzato con `GridSearchCV`).
- Le soglie di decisione finali (0.55 e 0.54) sono state scelte osservando i grafici,
  non con una procedura sistematica identica per entrambi i modelli.
- Si potrebbero esplorare altri algoritmi (es. Gradient Boosting) o tecniche di
  bilanciamento delle classi più sofisticate per migliorare ulteriormente il recall
  senza sacrificare la precisione.
- Il dataset contiene rumore aggiunto intenzionalmente, quindi i punteggi ottenuti
  vanno interpretati come un limite intrinseco dei dati più che del modello.

---

## 👤 Autore

Rivera, Lina, Bortoni, Vanina

---

## 📄 Licenza

Questo progetto è distribuito con licenza [MIT](https://opensource.org/licenses/MIT).
Il dataset utilizzato appartiene ai rispettivi autori su Kaggle.

