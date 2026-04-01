# 📊 Datasets et Sources de Données (GreenIoT-MA)

Ce document décrit les sources de données réelles et les API publiques exploitées par la plateforme **GreenIoT-MA** pour simuler avec un haut degré de réalisme la consommation énergétique (IT et Refroidissement) ainsi que la production d'énergie solaire d'un Green Data Center.

> [!NOTE]
> Pour des raisons de respect de la propriété intellectuelle et des limites d'espace de stockage GitHub (`.gitignore`), **les fichiers bruts des datasets ne sont pas inclus dans ce dépôt**. Les liens ci-dessous vous permettront de les télécharger localement.

---

## 1. UC Irvine — Individual Household Electric Power Consumption
*Utilisé pour la simulation de la charge informatique (IT Load / Serveurs).*

- **Source :** [UCI Machine Learning Repository - Dataset 235](https://archive.ics.uci.edu/dataset/235)
- **Taille :** ~2 millions de relevés (2006-2010), ~130 Mo.
- **Rôle dans le projet :** 
  Plutôt que d'utiliser des générateurs aléatoires cycliques (`sin`/`cos`) pour la puissance `power_kw`, nous mappons les relevés massifs de consommation active (Global Active Power) de ce dataset à la consommation individuelle de nos **racks de serveurs**. Cela permet aux modèles de Machine Learning (LSTM/XGBoost) de s'entraîner sur des "pics" et des "bruits" temporels réels.
- **Variables extraites :** `Date`, `Time`, `Global_active_power`, `Global_reactive_power`.

---

## 2. ASHRAE Energy Prediction (Kaggle)
*Utilisé pour la simulation de la charge thermique (Système de Refroidissement & PUE).*

- **Source :** [Kaggle ASHRAE Great Energy Predictor III](https://www.kaggle.com/c/ashrae-energy-prediction/data)
- **Rôle dans le projet :**
  L'efficacité énergétique (PUE - Power Usage Effectiveness) d'un centre de données est drastiquement impactée par le système de refroidissement (Chillers / CRAH). Nous utilisons le fichier d'entraînement d'ASHRAE pour les séries temporelles de "chilled_water" (eau glacée). 
- **Variables extraites :** `meter_reading` (converti en charge thermique frigorifique et électricité de refroidissement), `timestamp`.

---

## 3. Open-Meteo API
*Utilisé pour la production Solaire (Photovoltaïque) et la température de l'air.*

- **Source :** [Open-Meteo Open Source Weather API](https://open-meteo.com/)
- **Rôle dans le projet :**
  - **Irradiation (W/m²) :** L'API `Historical Weather API` ainsi que `Forecast API` nous délivrent l'irradiation solaire globale (`direct_radiation_instant`) aux coordonnées géographiques du centre de données (ex: Dakhla, Maroc). C'est le cœur de notre module d'**Optimisation de charge**, nous permettant de décaler les "Batch Jobs" au moment du pic solaire.
  - **Température extérieure (°C) :** Utilisée pour simuler les rendements de refroidissement (free cooling) en calculant le PUE de manière dynamique.
- **Variables extraites :** `temperature_2m`, `direct_radiation`, `shortwave_radiation`.

---

## ⚙️ Comment utiliser ces données ?

Si vous exécutez le projet localement :
1. Téléchargez les fichiers de Kaggle et UCI.
2. Placez-les dans un répertoire non suivi par git (par exemple, créez un dossier `data/raw/` à la racine de votre projet local).
3. Modifiez votre fichier `.env` avec les chemins absolus vers vos fichiers de données (ex: `POWER_CONSUMPTION_DATASET`, `ASHRAE_TRAIN_DATASET`).
4. Lancez le script de simulation spatio-temporelle : `python 01_simulation/generate_static_dataset.py`.
