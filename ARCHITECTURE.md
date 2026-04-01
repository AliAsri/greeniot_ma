# 🏗️ Architecture et Fonctionnement de GreenIoT-MA

Ce document explique en détail l'architecture technique, le pipeline de données, et le cycle de vie du projet **GreenIoT-MA**. Conçu pour optimiser et monitorer la consommation énergétique des Green Data Centers, ce projet s'appuie sur une stack Data Engineering et Machine Learning de pointe.

---

## 1. 🔄 Le Cycle de Vie de la Donnée (Pipeline)

Le projet suit un flux de données continu et résilient, de la génération capteur jusqu'à la visualisation utilisateur :

1. **Génération (Simulation)** : Des capteurs virtuels lisent des datasets réels massifs (UCI, ASHRAE, Open-Meteo) et injectent ces relevés (Température, CPU, Consommation, Ensoleillement).
2. **Ingestion (Temps Réel)** : Ces données sont envoyées sous forme de messages vers un *Message Broker* (Kafka), capable d'encaisser des millions de requêtes par seconde sans perturber le système.
3. **Consommation & Stockage (Datalake)** : Un moteur de Streaming (Apache Spark) lit Kafka au fil de l'eau et sauvegarde les données brutes dans le stockage objet local (MinIO).
4. **Raffinage (Lakehouse/Medallion)** : Le framework DeltaLake convertit et nettoie ces données par couches successives (Bronze, Silver, Gold).
5. **Intelligence Stratégique (Machine Learning)** : Des modèles d'IA (LSTM PyTorch, XGBoost) consomment la couche "Gold" ultra-propre pour prédire, détecter et optimiser.
6. **Restitution (Dashboard UX)** : Streamlit charge ces métriques et prédictions pour l'utilisateur final.

---

## 2. 🏅 L'Architecture "Medallion" (Delta Lake)

La gestion de la donnée repose sur le standard de l'industrie : l'Architecture Médaillon, stockée physiquement sur le cluster **MinIO** (format Parquet / Delta).

- **🥉 Couche Bronze (Données Brutes)** :
  - **Concept :** Les données arrivent telles quelles (avec leurs éventuelles erreurs, blancs ou bugs). C'est la source de vérité absolue historique (*immutable append-only*).
  - **Outil :** `spark_streaming.py` ou `generate_static_dataset.py` enregistrent vers MinIO.

- **🥈 Couche Silver (Données Nettoyées & Filtrées)** :
  - **Concept :** Le Lakehouse filtre les aberrations, supprime les lignes corrompues (NaNs) et mappe le PUE manquant de façon statique si besoin. C'est l'étape de conformité.
  - **Outil :** `03_lakehouse/bronze_to_silver.py`.

- **🥇 Couche Gold (Données Affinées & ML-Ready)** :
  - **Concept :** La donnée prend une valeur métier inestimable (*Feature Engineering*). On calcule ici les moyennes mobiles, l'encodage du temps de façon cyclique (Sinus/Cosinus pour le ML), et les *lag features* (différence thermique H contre H-1). 
  - **Outil :** `03_lakehouse/silver_to_gold.py`. C'est cette source que consommeront l'IA et le Dashboard avancé.

---

## 3. 🛠️ Stack Technologique Employée

### Data Engineering & Ingestion
- **Python / Faker / Psutil** : Pour la restitution simulée des métriques IoT des serveurs matériels.
- **Apache Kafka (Docker)** : Bus de messages à haute performance (Kafka Producer/Consumer).
- **Apache Spark (PySpark)** : Moteur de traitement de données ultra-massif (Structured Streaming).

### Storage & Lakehouse
- **MinIO** : Object Storage privé et souverain 100% compatible AWS S3.
- **Delta Lake (Linux Foundation)** : Format de table transactionnel garantissant la consistance ACID sur du Parquet brut, permettant le *Time Travel* (historique) et le requêtage massif ultra-rapide côté Streamlit (via *Predicate Pushdown* avec PyArrow).

### Intelligence Artificielle (Machine Learning)
- **PyTorch (Deep Learning)** : Exploité pour construire des réseaux de Neurones Récurrents (LSTM) qui excellent dans la prédiction des contextes horaires à long terme.
- **XGBoost (Ensemble Learning)** : Utilisé comme *baseline* haute-performance pour des prédictions brutes très fiables, et l'extraction de *Feature Importance*.
- **XGBoost Classifier (Supervisé)** : Algorithme déployé récemment en remplacement d'Isolation Forest pour la détection d'anomalies d'infrastructure avec un F1-Score ultra précis (Serveur qui surchauffe silencieusement).
- **MLflow** : Plateforme de MLOps de suivi (tracking) du cycle de vie des modèles, pour enregistrer les paramètres, métriques (MAE, R2, RMSE) et centraliser le registre des modèles.

### Frontend (Application Utilisateur)
- **Streamlit** : Framework de création de portails Data interactifs ultra-réactifs.
- **Plotly Express / Graph Objects** : Génération de la télémétrie visuelle, des cartes thermiques et diagrammes de Gantt paramétriques.

---

## 4. 📂 Explication détaillé de l'arborescence

### `01_simulation/`
Ce dossier est l'étincelle initiale de tout le projet.
- **`kafka_producer.py`** : Script simulant un nœud IoT qui stream des données capteurs à l'infini vers le serveur Kafka (port 9092) sous format JSON.
- **`sensor_simulator.py`** : La logique interne des capteurs. Injecte le bruit IoT et combine de la donnée aléatoire probabiliste avec de véritables datasets IT (UCI) ou thermiques (ASHRAE) pour le réalisme.
- **`generate_static_dataset.py`** : Permet de générer instantanément l'historique complet des derniers jours, générant mathématiquement la base "Bronze, Silver, Gold" en quelques secondes.

### `02_ingestion/`
- **`spark_streaming.py`** : Écoute le topic `greeniot_servers` sur Kafka, structure le texte brut en colonnes PySpark valides, et l'ajoute (*appendstream*) vers la racine de la couche Delta Bronze de MinIO (`s3a://greeniot/bronze/servers`).

### `03_lakehouse/`
Ici, les pipelines transforment le Lakehouse sous format transactionnel Delta.
- **`bronze_to_silver.py`** : Charge la totalité des données brutes, déduit les types, compense les manques, et écrase l'anomalie de sonde en données propres.
- **`silver_to_gold.py`** : Ingénierie des caractéristiques temporelles essentielles au bon entraînement des modèles d'intelligence artificielle de la couche `04`.

### `04_ml/`
Le cerveau mathématique de la plateforme. Tous les entraînements utilisent **MLflow** en arrière-plan (qui tourne sur `localhost:5000`).
- **`train_prediction.py`** : Script lourd créant deux IA (LSTM et XGBoost). Compilera les prédictions H+3 (15 minutes), vérifiera leurs métriques, et les expédiera vers un fichier "Pickle" global (`models/`).
- **`train_anomaly.py`** : Entraîne et valide un puissant classifieur XGBoost ("XGBoost Supervisé") capable de flagger automatiquement un relevé IoT s'il présente une dimension anormale entre Energie et Température.
- **`optimize_load.py`** : Un ordonnanceur customisé qui croise la météo Open-Meteo (Irradiation solaire locale) et la durée prévue des "Tâches Batch" pour décaler ces dernières lors des pics de production électrique sans carbone (Green Shift).

### `05_dashboard/`
Au cœur de l'expérience utilisateur, l'Interface Streamlit. Ce dossier contient 3 pages métier interconnectées par `app.py`.
- **`app.py`** : Le socle d'entrée, gérant le thème CSS, le menu de la *Sidebar* et les variables de routage.
- **La magie de `utils/data_loader.py`** : Ce script est crucial ! Plutôt que de télécharger d'énormes jeux de données depuis MinIO vers la RAM de Streamlit, ses fonctions (ex: `_load_bronze_filtered`) mandatent le moteur d'exécution (via PyArrow) de **filtrer la donnée temporellement sur le disque distant** avant de l'importer (*Predicate Pushdown*), garantissant des temps de chargement minimes.
- Les fichiers dans `pages/` (Monitoring, Prédiction, Optimisation) importent ce Data Loader pour présenter la métrologie via le rendu visuel haut de gamme de Plotly.

---
*Ce document sert de spécification technique principale à la plateforme. Tout contributeur externe ou évaluateur technique doit se référer en priorité à ces paragraphes pour appréhender la complexité orchestrale de **GreenIoT-MA**.*
