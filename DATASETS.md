# Datasets et sources de donnees

Ce document recense les sources de donnees supportees par GreenIoT-MA et explique comment elles sont utilisees dans le projet.

## 1. Principe general

GreenIoT-MA peut fonctionner de deux manieres :

- avec des datasets reels optionnels pour enrichir la simulation
- avec une generation completement synthetique pour la demo locale

Les fichiers bruts volumineux ne sont pas versionnes dans ce depot.

## 2. Sources supportees

### UCI Individual Household Electric Power Consumption

Usage principal :

- simulation de la charge `servers`

Script associe :

- `01_simulation/fetch_uci_household.py`

Role dans le projet :

- fournir une serie temporelle reelle de consommation electrique
- servir de base a la generation de charges IT plus credibles
- alimenter la creation de donnees `servers` en mode `real`

Variables exploitees :

- `Date`
- `Time`
- `Global_active_power`
- variables electriques associees selon le fichier disponible

Chemin attendu :

- variable d'environnement `UCI_DATASET`

### ASHRAE Energy Prediction

Usage principal :

- simulation de la partie `cooling`

Script associe :

- `01_simulation/fetch_ashrae.py`

Role dans le projet :

- produire des profils thermiques et energetiques plus credibles
- enrichir la simulation de refroidissement
- servir de base au mode `real` pour les donnees `cooling`

Variables couramment utilisees :

- `meter_reading`
- `timestamp`
- `air_temperature`
- metadonnees batiment selon disponibilite

Chemins attendus :

- `ASHRAE_TRAIN_DATASET`
- `ASHRAE_BUILDING_META`
- `ASHRAE_WEATHER_TRAIN`

## 3. Donnees synthetiques generees localement

Le script `01_simulation/generate_static_dataset.py` peut generer tout le socle de donnees necessaire au projet sans dependre d'une infrastructure live.

Il produit notamment :

- des donnees `servers`
- des donnees `solar`
- des donnees `cooling`
- des donnees `battery`

Le solaire est genere a partir d'un modele journalier physique simplifie avec variabilite meteorologique simulee. Il ne depend pas d'une API externe obligatoire pour la demo locale.

## 4. Modes de generation

Le projet distingue deux logiques de generation :

### Mode `real`

Utilise les datasets optionnels lorsqu'ils sont disponibles :

- `UCI` pour `servers`
- `ASHRAE` pour `cooling`
- solaire genere localement par modele physique

### Mode `synthetic`

Tout est genere localement sans fichiers externes.

Ce mode est utile pour :

- tester vite le pipeline
- lancer le dashboard sans preparation lourde
- travailler hors connexion aux datasets reels

## 5. Fichiers produits localement

La generation locale cree des fichiers dans le dossier `data/`.

Exemples importants :

- `data/raw_servers.parquet`
- `data/raw_solar.parquet`
- `data/raw_cooling.parquet`
- `data/raw_battery.parquet`
- `data/silver_servers_latest.parquet`
- `data/silver_solar_latest.parquet`
- `data/gold_servers.parquet`
- `data/gold_solar.parquet`

Ces fichiers servent ensuite a :

- entrainer les modeles
- alimenter le dashboard en mode local
- verifier la coherence des transformations

## 6. Utilisation pratique

### Cas le plus simple

Pour une demo locale rapide :

```bash
python 01_simulation/generate_static_dataset.py
python 03_lakehouse/bronze_to_silver.py
python 03_lakehouse/silver_to_gold.py
streamlit run 05_dashboard/app.py
```

### Cas avec datasets reels

1. telecharger les jeux de donnees UCI et ASHRAE
2. definir les variables d'environnement correspondantes
3. lancer `generate_static_dataset.py`
4. poursuivre le pipeline lakehouse et ML

## 7. Ce qui n'est pas inclus dans le depot

Ne sont pas commits ici :

- gros fichiers bruts UCI
- gros fichiers bruts ASHRAE
- exports lourds de travail local

Le but est de garder le depot versionnable et reproductible sans stocker de donnees proprietaires ou trop volumineuses.

## 8. Resume

GreenIoT-MA supporte a la fois :

- des datasets reels optionnels pour gagner en realisme
- des donnees synthetiques pour la demo et le developpement rapide

Cette combinaison rend le projet plus pratique a presenter, plus simple a rejouer et plus credible pour l'entrainement des modeles.
