# Analisi-Consumi-Elettrici
Analisi di serie storiche ed elaborazione di modelli di Machine Learning predittivi (Random Forest) per il consumo energetico.


# ⚡ Energy Consumption Analysis & Predictive Modeling

Questo progetto di Data Science si concentra sull'analisi approfondita di serie storiche energetiche pluriennali e sullo sviluppo di modelli di Machine Learning predittivi per stimare il dispendio energetico attivo (*Global Active Power*).

## 🎯 Obiettivo del Progetto
Estrarre pattern comportamentali e tendenze temporali da oltre 2 milioni di record di consumi domestici, costruendo una pipeline di regressione capace di prevedere efficacemente il carico energetico futuro.

## 🛠️ Tecnologie Utilizzate
* **Linguaggio:** Python
* **Librerie:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Joblib
* **Competenze:** Data Analysis, Feature Engineering, Regression Models, Model Persistence

## 📊 Pipeline del Progetto
1. **Data Preprocessing & Wrangling:** Gestione dei valori mancanti, formattazione e conversione delle serie temporali in oggetti `DateTime` per consentire analisi e slicing mirati.
2. **Feature Engineering:** Estrazione di variabili temporali chiave (ora del giorno, giorno della settimana, mese) per catturare la stagionalità e la ciclicità dei consumi.
3. **Analisi Esplorativa (EDA):** Studio delle distribuzioni statistiche e delle correlazioni tramite istogrammi e matrici di dispersione.
4. **Modellazione Predittiva:** Addestramento e confronto di modelli di regressione supervisionata (Regressione Lineare e Random Forest Regressor) previa normalizzazione delle feature tramite `StandardScaler`.
5. **Persistenza del Modello:** Salvataggio della Random Forest addestrata e dello scaler tramite `joblib` (`rf_model.pkl`, `scaler.pkl`) per l'utilizzo immediato in fase di inferenza.

---

## ⚙ Guida all'Utilizzo e Istruzioni d'Uso

> **IMPORTANTE (Prerequisito fondamentale):** 
> * **Posizione dei file:** Per il corretto funzionamento degli script, assicurarsi che il file del dataset, il notebook (`consumo_elettrico.ipynb`) e i file dei modelli salvati si trovino **tutti nella stessa cartella (nella stessa directory di lavoro)**.
> * **Download del Dataset:** Il file di testo originale dei consumi supera i limiti di dimensione consentiti da GitHub (25 MB). Per poter eseguire il codice, scaricare il dataset originale (*Individual Household Electric Power Consumption*) da **Kaggle** o dalla **UCI Machine Learning Repository** e posizionarlo nella cartella di lavoro principale.

### 1. Esecuzione del Notebook
* Aprire ed eseguire il notebook Jupyter (`consumo_elettrico.ipynb`) in sequenza per replicare la pulizia dei dati, l'addestramento dei modelli e la visualizzazione dei grafici.

### 2. Utilizzo dei Modelli Salvati
* I file `.pkl` inclusi nella repository (`rf_model.pkl` e `scaler.pkl`) permettono di caricare direttamente il modello predittivo e lo scaler in Python senza dover rieseguire l'intero addestramento:
  ```python
  import joblib
  model = joblib.load('rf_model.pkl')
  scaler = joblib.load('scaler.pkl')
