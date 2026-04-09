# GreenIoT-MA

GreenIoT-MA est un projet de pilotage energetique pour data center qui combine simulation IoT, lakehouse Delta, machine learning et dashboard Streamlit.

Le projet couvre quatre besoins complementaires :

- observer l'etat energetique et thermique des serveurs
- structurer les donnees dans une architecture Bronze / Silver / Gold
- predire la charge IT et detecter les anomalies
- optimiser le decalage de taches batch vers les meilleures plages solaires

## Vue d'ensemble

Le pipeline suit une chaine de bout en bout :

```text
Simulation / Datasets
    -> Kafka / PySpark
    -> Delta Lake sur MinIO
    -> Features Gold
    -> Modeles ML
    -> Dashboard Streamlit
```

Le dashboard expose aujourd'hui trois pages :

- `Monitoring` : telemetrie temps reel et signaux de sante
- `Predictions` : comparaison LSTM / XGBoost et vues anomalies
- `Optimization` : load shifting journalier avec historique sur les 5 derniers jours

## Fonctionnalites principales

### 1. Simulation et donnees

- simulation de capteurs `solar`, `servers` et `cooling`
- generation d'un dataset statique local pour la demo
- integration possible de datasets de reference comme UCI

### 2. Ingestion et lakehouse

- ingestion temps reel via Kafka
- consumer Python et pipeline PySpark
- stockage Delta / Parquet sur MinIO
- transformations Medallion :
  - Bronze : donnees brutes
  - Silver : nettoyage et enrichissement
  - Gold : features pour prediction, anomalies et optimisation

### 3. Machine learning

- prediction de consommation IT avec `XGBoost` et `LSTM`
- detection d'anomalies par `XGBoost supervise` sur labels heuristiques enrichis
- calibration du seuil de decision pour la detection d'anomalies
- suivi des experiences via MLflow

### 4. Optimisation energetique

- construction d'un profil solaire journalier en slots de 15 minutes
- recherche d'une fenetre solaire optimale
- planification de taches batch selon :
  - priorite
  - duree
  - puissance requise
  - couverture solaire disponible
- affichage du `load shifting` et du `CO2 potentiel`

## Architecture technique

| Couche | Role | Outils |
|---|---|---|
| Simulation | Generation des flux IoT | Python |
| Ingestion | Streaming et collecte | Kafka, PySpark |
| Stockage | Lakehouse Medallion | Delta Lake, MinIO, Parquet |
| ML | Prevision et anomalies | scikit-learn, XGBoost, PyTorch, MLflow |
| Visualisation | Pilotage et restitution | Streamlit, Plotly |

## Structure du depot

```text
greeniot_ma/
|-- 01_simulation/
|-- 02_ingestion/
|-- 03_lakehouse/
|-- 04_ml/
|-- 05_dashboard/
|-- data/
|-- models/
|-- Rapport/
|-- tests/
|-- docker-compose.yml
|-- pytest.ini
|-- requirements.txt
`-- README.md
```

### Dossiers importants

- `01_simulation/`
  - `sensor_simulator.py`
  - `kafka_producer.py`
  - `generate_static_dataset.py`
- `02_ingestion/`
  - `spark_streaming.py`
  - `kafka_consumer.py`
- `03_lakehouse/`
  - `bronze_to_silver.py`
  - `silver_to_gold.py`
- `04_ml/`
  - `train_prediction.py`
  - `train_anomaly.py`
  - `optimize_load.py`
- `05_dashboard/`
  - `app.py`
  - `pages/monitoring.py`
  - `pages/predictions.py`
  - `pages/optimization.py`

## Installation

### Prerequis

- Python 3.10 ou plus
- Docker Desktop
- 8 Go de RAM minimum recommandes

### Installation locale

```bash
git clone <repo-url>
cd greeniot_ma
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Demarrage rapide

### 1. Infrastructure

```bash
docker-compose up -d
```

Services attendus :

- Kafka sur `localhost:9092`
- MinIO API sur `http://localhost:9000`
- MinIO Console sur `http://localhost:9001`
- MLflow sur `http://localhost:5000`

Le `docker-compose.yml` initialise aussi le bucket MinIO necessaire au projet.

### 2. Mode demo locale

Ce mode est le plus simple pour tester le projet sans streaming live.

```bash
python 01_simulation/generate_static_dataset.py
python 03_lakehouse/bronze_to_silver.py
python 03_lakehouse/silver_to_gold.py
python 04_ml/train_prediction.py
python 04_ml/train_anomaly.py
streamlit run 05_dashboard/app.py
```

### 3. Mode streaming

```bash
python 01_simulation/kafka_producer.py
python 02_ingestion/spark_streaming.py
python 03_lakehouse/bronze_to_silver.py
python 03_lakehouse/silver_to_gold.py
```

## Dashboard

### Monitoring

- telemetrie temps reel des serveurs
- signaux critiques CPU / temperature / puissance
- etat general de l'infrastructure

### Predictions

- comparaison des performances `LSTM` et `XGBoost`
- affichage des previsions si les artefacts ML sont disponibles
- lecture des anomalies detectees et de leur contexte

### Optimization

- vue strictement journaliere
- historique selectable sur les `5 derniers jours` disponibles
- calcul coherent de :
  - `Pic de production`
  - `Moyenne diurne`
  - `Energie du jour`
  - `CO2 potentiel du jour`
- planification des taches batch sur la meilleure capacite solaire

## Modeles et logique metier

### Prediction

Le projet entraine deux familles de modeles :

- `XGBoost` pour une baseline solide et interpretable
- `LSTM` pour capter la dynamique temporelle

Les metriques suivies sont :

- `MAE`
- `RMSE`
- `R2`
- `MAPE`

### Anomalies

La detection d'anomalies n'utilise plus `Isolation Forest` comme description principale du projet.
La version actuelle repose sur :

- des labels heuristiques construits en Silver
- un entrainement supervise `XGBoost`
- un seuil de decision calibre sur validation

L'objectif est d'obtenir une detection plus realiste et plus defendable academiquement.

### Load shifting

Le module `04_ml/optimize_load.py` deplace des taches batch vers les meilleurs creneaux solaires en fonction :

- de la priorite
- de la duree
- de la puissance requise
- de la disponibilite solaire par slot

Le dashboard affiche ensuite :

- la fenetre optimale
- la charge couverte par le solaire
- la part solaire moyenne par tache
- l'energie solaire et l'energie reseau du planning

## Tests

```bash
pytest -q
```

Tests presents :

- `tests/test_dataloader.py`
- `tests/test_minio.py`
- `tests/test_minio_read.py`
- `tests/test_minio_exact_error.py`

## Limites actuelles

- les transformations Bronze / Silver / Gold restent principalement en mode `overwrite`
- la page `Optimization` repose sur des taches batch parametrees dans le dashboard
- certaines parties live dependent de la disponibilite complete de Delta / MinIO / Kafka

## Pistes d'amelioration

- passage a un pipeline incrementiel avec `merge`
- integration d'un vrai ordonnanceur de taches batch
- ajout d'un cout horaire de l'electricite pour une optimisation cout + carbone
- industrialisation CI / smoke tests dashboard / validation de schema

## Resume

GreenIoT-MA est un demonstrateur complet de data engineering et d'IA appliquee a la gestion energetique. Le projet ne se limite pas a visualiser des donnees : il cherche aussi a recommander quand executer les charges flexibles pour mieux exploiter le solaire.
