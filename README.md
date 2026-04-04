# 🌿 GreenIoT-MA

**Pipeline IoT · Lakehouse · Machine Learning · Green Data Centers**

> Data Engineering & ML
> Contexte : Stratégie Maroc Digital 2030 · Défi Green IT · Neutralité carbone 2035

---

## 📋 Description

GreenIoT-MA est un pipeline end-to-end de monitoring intelligent pour les Green Data Centers marocains. Le projet combine :

- **Collecte IoT temps réel** via capteurs simulés (solaire, serveurs, refroidissement)
- **Stockage Lakehouse souverain** avec architecture Medallion (Bronze → Silver → Gold)
- **Machine Learning prédictif** : LSTM/XGBoost pour la consommation, XGBoost Classifier pour les anomalies
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
├── README.md                   # Ce fichier
├── .env                        # Variables d'environnement
│
├── 01_simulation/
│   ├── sensor_simulator.py          # Générateur de capteurs IoT (temps réel)
│   ├── kafka_producer.py            # Envoi vers Kafka (4 topics)
│   ├── generate_static_dataset.py  # Générateur de données statiques (7 jours)
│   ├── fetch_uci_household.py       # [WIP] Téléchargement dataset UCI (non intégré)
│   └── datasets/                    # Dossier pour datasets locaux optionnels
│
├── 02_ingestion/
│   ├── kafka_consumer.py       # Consommateur Kafka basique
│   └── spark_streaming.py      # PySpark Structured Streaming (4 topics)
│
├── 03_lakehouse/
│   ├── schema.py               # Schémas Delta Lake (Bronze/Silver/Gold × servers+solar+battery)
│   ├── bronze_to_silver.py     # Nettoyage Bronze → Silver (servers + solar)
│   └── silver_to_gold.py       # Feature engineering Silver → Gold (servers + solar)
│
├── 04_ml/
│   ├── train_prediction.py     # LSTM + XGBoost (prédiction consommation)
│   ├── train_anomaly.py        # XGBoost Supervisé ou Isolation Forest (auto)
│   ├── optimize_load.py        # Décalage charge solaire
│   └── mlflow_tracking.py      # Suivi expériences MLflow
│
├── 05_dashboard/
│   ├── app.py                  # Application Streamlit principale
│   ├── pages/
│   │   ├── monitoring.py       # Monitoring temps réel
│   │   ├── predictions.py      # Prédictions ML (LSTM + XGBoost)
│   │   └── optimization.py     # Optimisation charge solaire
│   └── utils/
│       └── data_loader.py      # Chargement données (Mode Démo inclus)
│
└── 06_rapport/
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
```

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

## 📊 Sources de données

### Mode `DATA_MODE=real` (défaut — datasets réels intégrés)

Le pipeline utilise de vraies données pour les serveurs et le refroidissement,
couplées à une synthèse physique pour le solaire et la batterie.

| Capteur | Source réelle | Chemin local |
|---------|--------------|-------------|
| **Serveurs** | [UCI Household Electric](https://archive.ics.uci.edu/dataset/235) — 2M mesures 1 min | `UCI_DATASET` dans `.env` |
| **Refroidissement** | [ASHRAE Energy Prediction](https://www.kaggle.com/c/ashrae-energy-prediction) — 1 an, 1k bâtiments | `ASHRAE_TRAIN_DATASET` dans `.env` |
| **Solaire** | Synthèse physique (modèle irradiance Dakhla, `sin(π*(h-6)/12)`) | — |
| **Batterie** | Synthèse physique (modèle SOC évolutif, charge 10h–16h) | — |

**Mapping des données réelles :**
- UCI `Global_active_power` (kW) → `power_kw` des racks (re-scalé 30–120 kW)
- UCI `Global_intensity` (A) → proxy `cpu_pct` normalisé
- ASHRAE `meter_reading` (kWh) → `it_load_kw` refroidissement
- ASHRAE `weather_train.air_temperature` → `PUE = 1.20 + 0.012 × max(0, T-18°C)`

### Mode `DATA_MODE=synthetic` (fallback)

Toutes les données sont générées mathématiquement (modèles physiques) sans
aucun fichier externe requis.

| Capteur | Modèle physique |
|---------|----------------|
| **Solaire** | `sin(π*(h-6)/12)` + bruit `gauss(σ=3%)` |
| **Serveurs** | Pattern circadien (8h–20h pic), weekday vs weekend |
| **Refroidissement** | PUE = f(T_ext marocaine) |
| **Batterie** | SOC évolutif borné [10%, 100%] |

> [!NOTE]
> Le mode `real` est activé automatiquement si les fichiers UCI et ASHRAE sont
> accessibles aux chemins définis dans `.env`. En cas d'erreur de lecture,
> la génération bascule silencieusement en mode synthétique.

## 🏗️ Architecture Medallion — État réel

| Flux | Couche Bronze | Couche Silver | Couche Gold |
|------|--------------|--------------|-------------|
| **Servers** | ✅ Ingérée (Kafka) | ✅ Complète (rolling, anomaly_flag) | ✅ ML-ready (lags, cyclique) |
| **Solar** | ✅ Ingérée (Kafka) | ✅ Complète (rolling, anomaly_solar) | ✅ ML-ready (lags, cyclique) |
| **Cooling** | ✅ Ingérée (Kafka) | ❗ Non transformée (monitoring uniquement) | ❗ Non applicable |
| **Battery** | ✅ Ingérée (Kafka) | ❗ Non transformée (monitoring uniquement) | ❗ Non applicable |

## 🤖 Détection d'anomalies — Stratégie adaptative

Le script `train_anomaly.py` choisit automatiquement son algorithme selon la disponibilité des labels :

- **XGBoost Supervisé** : si la colonne `anomaly_flag` est présente dans les données Gold (labels générés par la couche Silver via z-score > 3)
- **Isolation Forest** : si aucun label n'est disponible (détection non-supervisée, `contamination=0.05`)

Les deux modèles sont tracés dans MLflow avec leurs métriques respectives.

## 🎭 Mode Démonstration

Le dashboard intègre un Mode Démo qui se déclenche automatiquement si MinIO (Delta Lake) est inaccessible. Il génère des données synthétiques localement pour permettre une démonstration sans infrastructure.

```bash
# Force le mode démo dans .env
DEMO_MODE=true
```

La page Prédictions propose également un toggle **"🪄 Lisser le signal"** qui applique un filtre rolling(4) sur les valeurs réelles pour atténuer le bruit blanc des capteurs lors d'une soutenance.

| Modèle | Métriques | Objectif |
|--------|-----------|----------|
| Prédiction conso (LSTM/XGBoost) | MAE, RMSE, R², MAPE | MAE < 5 kW, MAPE < 8% |
| Détection anomalies (XGBoost Classifier) | Précision, Rappel, F1 | F1 > 0.80, FP < 5% |
| Optimiseur décalage | kWh décalés/j, CO2 économisé | ≥ 20% charge, ≥ 50 kg CO2/j |

## 🇲🇦 Contexte Maroc Digital 2030

- **Neutralité carbone** : 100% des data centers d'ici 2035
- **Méga-campus Igoudar Dajla** : 500 MW, 100% énergies renouvelables
- **Conformité réglementaire** : loi 09-08, décret 2-24-921
