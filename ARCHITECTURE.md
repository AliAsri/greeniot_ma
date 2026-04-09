# Architecture et fonctionnement de GreenIoT-MA

Ce document decrit l'architecture technique du projet GreenIoT-MA et le role de chaque couche du pipeline.

## 1. Objectif du projet

GreenIoT-MA vise a observer, structurer, predire et optimiser la consommation energetique d'une infrastructure numerique alimentee en partie par le solaire.

Le projet combine :

- simulation ou ingestion de donnees IoT
- architecture lakehouse Medallion
- modeles de prediction et de detection d'anomalies
- optimisation de taches batch par `load shifting`
- restitution dans un dashboard Streamlit

## 2. Vue d'ensemble du pipeline

```text
Simulation / Datasets
    -> Kafka / PySpark
    -> Bronze
    -> Silver
    -> Gold
    -> Machine Learning
    -> Dashboard Streamlit
```

Le projet peut fonctionner dans deux logiques principales :

- `mode demo locale` : generation de donnees locales et execution hors streaming
- `mode streaming` : ingestion Kafka puis transformations lakehouse

## 3. Couche simulation et acquisition

Le dossier `01_simulation/` produit les donnees du projet.

Les grandes familles de donnees sont :

- `servers` : charge IT, CPU, RAM, temperature, puissance
- `solar` : production photovoltaque
- `cooling` : signaux de refroidissement et efficacite
- `battery` : donnees de stockage energie si la simulation locale les genere

Les scripts les plus importants sont :

- `sensor_simulator.py`
- `kafka_producer.py`
- `generate_static_dataset.py`
- `fetch_uci_household.py`
- `fetch_ashrae.py`

`generate_static_dataset.py` est le point d'entree principal pour une demo locale rapide.

## 4. Ingestion temps reel

Le dossier `02_ingestion/` gere la collecte et la persistance initiale.

Scripts principaux :

- `spark_streaming.py`
- `kafka_consumer.py`

Le projet supporte :

- une ingestion PySpark vers Bronze
- un consumer Python complementaire

Kafka sert de bus de messages, puis les flux sont persistes dans le stockage lakehouse.

## 5. Lakehouse Medallion

GreenIoT-MA suit une architecture Bronze / Silver / Gold.

### Bronze

La couche Bronze conserve les donnees brutes avec leur structure la plus proche de la source.

Role :

- centraliser l'historique brut
- servir de source de verite initiale
- alimenter les traitements suivants

### Silver

La couche Silver nettoie et enrichit les donnees.

Role :

- normaliser les types
- dedoublonner
- calculer des variables intermediaires
- produire un label heuristique d'anomalie pour les serveurs

Script principal :

- `03_lakehouse/bronze_to_silver.py`

### Gold

La couche Gold prepare les donnees pour le machine learning et le dashboard.

Role :

- produire des moyennes glissantes
- generer des lags et ecarts-types
- encoder le temps de facon cyclique
- fournir des tables directement exploitables

Script principal :

- `03_lakehouse/silver_to_gold.py`

## 6. Stockage et execution

Le projet s'appuie sur :

- `MinIO` pour le stockage objet compatible S3
- `Delta Lake` pour la structuration des tables
- `Parquet` pour les fichiers locaux de demo

Le `docker-compose.yml` fournit l'infrastructure principale du projet, notamment :

- Kafka
- Zookeeper
- MinIO
- MLflow

Le bucket MinIO necessaire au projet est initialise par la stack Docker.

## 7. Machine learning

Le dossier `04_ml/` contient trois briques principales.

### Prediction de consommation

Script :

- `train_prediction.py`

Le projet compare deux familles de modeles :

- `XGBoost`
- `LSTM`

Les metriques de comparaison visibles dans le dashboard sont :

- `MAE`
- `RMSE`
- `R2`
- `MAPE`

### Detection d'anomalies

Script :

- `train_anomaly.py`

La logique actuelle repose sur :

- des labels heuristiques construits en Silver
- un modele `XGBoost supervise`
- une calibration de seuil sur validation

Le projet ne presente plus la detection d'anomalies comme une simple demonstration `Isolation Forest`.

### Optimisation energetique

Script :

- `optimize_load.py`

Ce module implemente une logique de `load shifting` :

- calcul du profil solaire journalier
- recherche d'une fenetre solaire optimale
- placement de taches batch selon priorite, duree et puissance requise
- estimation de la couverture solaire, de l'energie reseau et du CO2 potentiel

## 8. Dashboard Streamlit

Le dossier `05_dashboard/` contient l'interface utilisateur.

La navigation actuelle comporte trois pages :

- `Monitoring`
- `Predictions`
- `Optimization`

### Monitoring

Affiche la telemetrie recente et les signaux de sante de l'infrastructure.

### Predictions

Affiche :

- les performances des modeles
- les previsions si les artefacts sont disponibles
- les anomalies detectees

### Optimization

Affiche une vue journaliere coherente de l'optimisation :

- selection d'un jour parmi les 5 derniers disponibles
- KPI solaires du jour
- courbe solaire de cette journee
- fenetre optimale
- planning de taches batch

## 9. Organisation des dossiers

```text
01_simulation/   -> generation et adaptation des donnees
02_ingestion/    -> streaming et collecte
03_lakehouse/    -> transformations Bronze / Silver / Gold
04_ml/           -> prediction, anomalies, optimisation
05_dashboard/    -> interface Streamlit
data/            -> jeux de donnees locaux et exports
models/          -> artefacts d'entrainement
tests/           -> verification du chargement et de MinIO
```

## 10. Limites actuelles

Les principales limites connues sont :

- transformations lakehouse encore majoritairement en `overwrite`
- taches d'optimisation encore parametrees dans l'interface
- qualite du mode live dependante de Kafka, MinIO et Delta

## 11. Resume

GreenIoT-MA est un projet qui relie data engineering, machine learning et pilotage energetique. Sa valeur principale vient du fait qu'il ne se contente pas de visualiser les donnees : il produit aussi des recommandations d'execution pour mieux utiliser la production solaire.
