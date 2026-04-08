# 🌿 GreenIoT-MA

**Pipeline IoT · Lakehouse · Machine Learning · Green Data Centers**

> Data Engineering & ML
> Contexte : Stratégie Maroc Digital 2030 · Défi Green IT · Neutralité carbone 2035

---

## 📋 Description

GreenIoT-MA est un pipeline end-to-end de monitoring intelligent pour les Green Data Centers marocains. Le projet combine :

- **Collecte IoT temps réel** via capteurs simulés (solaire, serveurs, refroidissement)
- **Stockage Lakehouse souverain** avec architecture Medallion (Bronze → Silver → Gold)
- **Machine Learning prédictif** : LSTM/XGBoost pour la consommation, Isolation Forest pour les anomalies
- **Optimisation de charge** : décalage des tâches batch vers les pics de production solaire
- **Dashboard interactif** Streamlit pour le monitoring en temps réel

## 🏗️ Architecture

```
Sources IoT → Kafka (Ingestion) → Delta Lake (Medallion) → ML Models → Streamlit Dashboard
     ↓              ↓                    ↓                    ↓              ↓
  Simulateurs   Streaming         Bronze/Silver/Gold      MLflow        Monitoring
  Python        PySpark           MinIO (S3)              PyTorch       Plotly
```

| Couche | Rôle | Technologies |
|--------|------|-------------|
| 1 — Sources | Simulation capteurs IoT | Python, psutil, faker |
| 2 — Ingestion | Streaming temps réel | Apache Kafka, PySpark |
| 3 — Stockage | Lakehouse Medallion | Delta Lake, MinIO, Parquet |
| 4 — ML | Prédiction & anomalies | scikit-learn, PyTorch, MLflow |
| 5 — Viz | Dashboard monitoring | Streamlit, Plotly |

## 📁 Structure du projet

```
greeniot_ma/
├── docker-compose.yml          # Kafka + Zookeeper + MinIO + MLflow
├── requirements.txt            # Dépendances Python
├── pytest.ini                  # Configuration des tests
├── README.md                   # Ce fichier
├── .env                        # Variables d'environnement
├── .env.example                # Exemple de configuration locale
│
├── 01_simulation/
│   ├── sensor_simulator.py     # Générateur de capteurs IoT
│   ├── kafka_producer.py       # Envoi vers Kafka
│   ├── generate_static_dataset.py  # Données statiques (7 jours)
│   └── datasets/               # Données open source (UCI, Google PUE)
│
├── 02_ingestion/
│   ├── kafka_consumer.py       # Consommateur Kafka → Bronze
│   └── spark_streaming.py      # PySpark Structured Streaming
│
├── 03_lakehouse/
│   ├── schema.py               # Schémas Delta Lake
│   ├── bronze_to_silver.py     # Nettoyage Bronze → Silver
│   └── silver_to_gold.py       # Feature engineering → Gold
│
├── 04_ml/
│   ├── train_prediction.py     # LSTM + XGBoost (prédiction conso)
│   ├── train_anomaly.py        # Isolation Forest (anomalies)
│   ├── optimize_load.py        # Décalage charge solaire
│   ├── inspect_preds.py        # Inspection rapide des sorties modèles
│   └── mlflow_tracking.py      # Suivi expériences MLflow
│
├── 05_dashboard/
│   ├── app.py                  # Application Streamlit principale
│   ├── pages/
│   │   ├── monitoring.py       # Monitoring temps réel
│   │   ├── predictions.py      # Prédictions ML
│   │   └── optimization.py     # Optimisation charge
│   └── utils/
│       ├── data_loader.py      # Chargement données
│       └── ui_blocks.py        # Blocs UI réutilisables
│
├── tests/
│   ├── conftest.py             # Bootstrap commun des tests
│   ├── test_dataloader.py      # Tests unitaires data loader
│   └── test_minio*.py          # Tests d'intégration MinIO (optionnels)
│
└── Rapport/
    └── figures/                # Graphiques pour le rapport
```

## 🚀 Installation et lancement

### Prérequis

- Python 3.10+
- Docker & Docker Compose
- 8 Go RAM minimum

### 1. Cloner et installer

```bash
git clone <repo-url>
cd greeniot_ma
pip install -r requirements.txt
copy .env.example .env
```

Puis adapter `.env` si nécessaire (paths datasets, mode de données, endpoints MinIO/MLflow).

### 2. Démarrer l'infrastructure

```bash
docker-compose up -d
```

Cela lance :
- **Kafka** : `localhost:9092`
- **MinIO Console** : `http://localhost:9001` (user: `greeniot` / pass: `greeniot2030`)
- **MLflow UI** : `http://localhost:5000`

### 3. Générer les données statiques

```bash
python 01_simulation/generate_static_dataset.py
```

### 4. Lancer le pipeline (avec Docker)

```bash
# Terminal 1 — Producer Kafka
python 01_simulation/kafka_producer.py

# Terminal 2 — Consumer PySpark
python 02_ingestion/spark_streaming.py

# Terminal 3 — Transformations Lakehouse
python 03_lakehouse/bronze_to_silver.py
python 03_lakehouse/silver_to_gold.py
```

### 5. Entraîner les modèles ML

```bash
python 04_ml/train_prediction.py
python 04_ml/train_anomaly.py
python 04_ml/optimize_load.py
```

### 6. Lancer le dashboard

```bash
streamlit run 05_dashboard/app.py
```

Accès : `http://localhost:8501`

## ✅ Vérification rapide

```bash
pytest -q
```

Les tests MinIO sont désactivés par défaut pour éviter les faux échecs hors environnement live.  
Pour les activer :

```bash
set RUN_MINIO_TESTS=true
pytest -q
```

## 📊 Datasets utilisés

| Dataset | Source | Usage |
|---------|--------|-------|
| UCI Individual Household Electric | [archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/235) | Baseline prédiction, pré-entraînement LSTM |

## 📈 Métriques cibles

| Modèle | Métriques | Objectif |
|--------|-----------|----------|
| Prédiction conso (LSTM/XGBoost) | MAE, RMSE, R², MAPE | MAE < 5 kW, MAPE < 8% |
| Détection anomalies (Isolation Forest) | Précision, Rappel, F1 | F1 > 0.80, FP < 5% |
| Optimiseur décalage | kWh décalés/j, CO2 économisé | ≥ 20% charge, ≥ 50 kg CO2/j |

## 🇲🇦 Contexte Maroc Digital 2030

- **Neutralité carbone** : 100% des data centers d'ici 2035
- **Méga-campus Igoudar Dajla** : 500 MW, 100% énergies renouvelables
- **Conformité réglementaire** : loi 09-08, décret 2-24-921
