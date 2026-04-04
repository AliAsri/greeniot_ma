"""
GreenIoT-MA — Générateur de dataset statique
===============================================
Produit 7 jours de données en CSV et Parquet pour le développement
et le test hors-ligne du pipeline Medallion.

DEUX MODES selon la variable DATA_MODE dans .env :

  DATA_MODE=real (défaut si datasets disponibles) :
    - Serveurs  → UCI Individual Household Electric (real patterns circadiens)
    - Cooling   → ASHRAE Energy Prediction (vraies charges thermiques)
    - Solaire   → Synthèse physique (modèle irradiance Dakhla)
    - Batterie  → Synthèse physique (modèle SOC évolutif)

  DATA_MODE=synthetic :
    - Tout synthétique (à base physique, aucun fichier requis)

Datasets produits :
- data/raw_servers.parquet        → Bronze servers
- data/raw_solar.parquet          → Bronze solar
- data/raw_cooling.parquet        → Bronze cooling
- data/raw_battery.parquet        → Bronze battery
- data/silver_servers_latest.parquet → Silver servers (nettoyé + features)
- data/silver_solar_latest.parquet   → Silver solar  (nettoyé + features)
- data/gold_servers.parquet       → Gold servers (ML-ready)
- data/gold_solar.parquet         → Gold solar   (ML-ready)
"""

import os
import sys
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Paramètres
DAYS             = int(os.getenv("SIMULATION_DAYS", "7"))
INTERVAL_MINUTES = 5
DATA_MODE        = os.getenv("DATA_MODE", "real").lower()
OUTPUT_DIR       = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Imports optionnels (datasets réels)
_uci_available    = False
_ashrae_available = False
try:
    from fetch_uci_household import load_uci_as_servers
    _uci_available = True
except ImportError:
    pass
try:
    from fetch_ashrae import load_ashrae_as_cooling
    _ashrae_available = True
except ImportError:
    pass


def generate_solar_data(days: int, interval_min: int) -> pd.DataFrame:
    """Génère des données de panneaux solaires réalistes."""
    records = []
    sensors = [
        ("solar_dakhla_01", 500),
        ("solar_dakhla_02", 480),
    ]
    start = datetime.now() - timedelta(days=days)
    n_points = int(days * 24 * 60 / interval_min)

    for i in range(n_points):
        ts = start + timedelta(minutes=i * interval_min)
        hour = ts.hour + ts.minute / 60

        for sensor_id, peak_kw in sensors:
            irradiance = max(0, math.sin(math.pi * (hour - 6) / 12))
            # Ajout de variabilité journalière (météo)
            weather_factor = 0.7 + 0.3 * random.random()
            production = peak_kw * irradiance * weather_factor * random.gauss(1.0, 0.03)

            records.append({
                "sensor_id": sensor_id,
                "type": "solar",
                "timestamp": ts.isoformat(),
                "production_kw": round(max(0, production), 2),
                "irradiance_wm2": round(max(0, irradiance * 1000 * random.gauss(1.0, 0.02)), 1),
                "panel_temp_c": round(25 + irradiance * 20 + random.gauss(0, 1.5), 1),
            })

    return pd.DataFrame(records)


def generate_server_data(days: int, interval_min: int) -> pd.DataFrame:
    """Génère des données de serveurs réalistes avec patterns circadiens."""
    records = []
    sensors = [
        ("rack_A1", 80, 0.45),
        ("rack_A2", 75, 0.50),
        ("rack_B1", 90, 0.40),
    ]
    start = datetime.now() - timedelta(days=days)
    n_points = int(days * 24 * 60 / interval_min)

    for i in range(n_points):
        ts = start + timedelta(minutes=i * interval_min)
        hour = ts.hour + ts.minute / 60
        is_weekday = ts.weekday() < 5

        for sensor_id, max_kw, base_load in sensors:
            # Pattern journalier
            if 8 <= hour <= 20 and is_weekday:
                day_factor = 0.5 + 0.4 * math.sin(math.pi * max(0, hour - 8) / 10)
            elif 8 <= hour <= 18:
                day_factor = 0.3 + 0.2 * math.sin(math.pi * max(0, hour - 8) / 10)
            else:
                day_factor = 0.25 + random.gauss(0, 0.05)

            cpu = base_load * day_factor * 100 + random.gauss(0, 3)
            ram = 40 + day_factor * 40 + random.gauss(0, 2)
            kw = max_kw * (cpu / 100) * random.gauss(1.0, 0.02)

            # Injection d'anomalies (~2% des points)
            if random.random() < 0.02:
                cpu = min(100, cpu + random.uniform(30, 50))
                kw = kw * random.uniform(1.5, 2.5)

            records.append({
                "sensor_id": sensor_id,
                "type": "server",
                "timestamp": ts.isoformat(),
                "cpu_pct": round(min(100, max(0, cpu)), 1),
                "ram_pct": round(min(100, max(0, ram)), 1),
                "power_kw": round(max(0, kw), 2),
                "temp_c": round(35 + (cpu / 100) * 25 + random.gauss(0, 1), 1),
            })

    return pd.DataFrame(records)


def generate_cooling_data(days: int, interval_min: int) -> pd.DataFrame:
    """Génère des données de refroidissement réalistes."""
    records = []
    start = datetime.now() - timedelta(days=days)
    n_points = int(days * 24 * 60 / interval_min)

    for i in range(n_points):
        ts = start + timedelta(minutes=i * interval_min)
        hour = ts.hour + ts.minute / 60

        # Température extérieure variant avec l'heure (climat marocain)
        outdoor_temp = 25 + 10 * math.sin(math.pi * (hour - 6) / 12) + random.gauss(0, 2)
        it_load = random.uniform(200, 600)
        pue = 1.3 + 0.15 * (outdoor_temp - 20) / 20 + random.gauss(0, 0.05)
        pue = max(1.1, min(2.0, pue))

        records.append({
            "sensor_id": "cooling_unit_01",
            "type": "cooling",
            "timestamp": ts.isoformat(),
            "inlet_temp_c": round(random.gauss(22 + outdoor_temp * 0.1, 1.5), 1),
            "outlet_temp_c": round(random.gauss(35 + outdoor_temp * 0.05, 2), 1),
            "it_load_kw": round(it_load, 1),
            "total_power_kw": round(it_load * pue, 1),
            "pue": round(pue, 3),
        })

    return pd.DataFrame(records)


def generate_battery_data(days: int, interval_min: int) -> pd.DataFrame:
    """Génère des données de batterie de stockage réalistes.

    Modèle physique :
    - Charge pendant les heures solaires (10h–16h)
    - Décharge la nuit pour alimenter le data center
    - SOC (State of Charge) évolutif et borné entre 10% et 100%
    """
    records = []
    sensors = [
        ("battery_01", 1000.0),  # 1 MWh
    ]
    start = datetime.now() - timedelta(days=days)
    n_points = int(days * 24 * 60 / interval_min)

    for sensor_id, capacity_kwh in sensors:
        soc = random.uniform(0.4, 0.8)  # SOC initial
        for i in range(n_points):
            ts = start + timedelta(minutes=i * interval_min)
            hour = ts.hour + ts.minute / 60

            # Charge solaire diurne, décharge nocturne
            if 10 <= hour <= 16:
                charge_rate = random.uniform(20, 80)  # kW charge
                soc = min(1.0, soc + charge_rate / capacity_kwh * (interval_min / 60))
            else:
                charge_rate = -random.uniform(10, 50)  # kW décharge
                soc = max(0.1, soc + charge_rate / capacity_kwh * (interval_min / 60))

            records.append({
                "sensor_id": sensor_id,
                "type": "battery",
                "timestamp": ts.isoformat(),
                "soc_pct": round(soc * 100, 1),
                "charge_rate_kw": round(charge_rate, 2),
                "voltage_v": round(48 * (0.9 + 0.1 * soc) + random.gauss(0, 0.5), 1),
                "temp_c": round(25 + abs(charge_rate) * 0.05 + random.gauss(0, 1), 1),
            })

    return pd.DataFrame(records)


def create_silver_data(server_df: pd.DataFrame) -> pd.DataFrame:
    """Crée une version Silver (nettoyée) des données serveurs."""
    df = server_df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sensor_id", "ts"])
    df = df.drop_duplicates(subset=["sensor_id", "timestamp"])

    # Filtrage des valeurs aberrantes
    df = df[df["power_kw"].notna() & (df["power_kw"] > 0) & (df["power_kw"] < 200)]

    # Rolling features (fenêtre 5 mesures)
    for sid in df["sensor_id"].unique():
        mask = df["sensor_id"] == sid
        df.loc[mask, "power_kw_avg5"] = df.loc[mask, "power_kw"].rolling(5, min_periods=1).mean()
        df.loc[mask, "power_kw_std5"] = df.loc[mask, "power_kw"].rolling(5, min_periods=1).std().fillna(0)
        df.loc[mask, "cpu_avg5"] = df.loc[mask, "cpu_pct"].rolling(5, min_periods=1).mean()
        df.loc[mask, "power_delta"] = df.loc[mask, "power_kw"].diff().fillna(0)

    # Flag anomalie (z-score > 3)
    df["anomaly_flag"] = (abs(df["power_delta"]) > 3 * df["power_kw_std5"]).astype(int)

    # PUE simulé pour le dashboard (reproductible avec seed)
    rng = np.random.default_rng(42)
    df["pue"] = rng.uniform(1.3, 1.8, len(df))

    return df


def create_gold_data(silver_df: pd.DataFrame) -> pd.DataFrame:
    """Crée une version Gold (ML-ready) des données serveurs."""
    df = silver_df.copy()
    df["ts"] = pd.to_datetime(df["ts"]) if "ts" in df.columns else pd.to_datetime(df["timestamp"])

    # Features temporelles encodées
    df["hour"] = df["ts"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week"] = df["ts"].dt.dayofweek
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_business_hours"] = ((df["hour"] >= 8) & (df["hour"] <= 18)).astype(int)

    # Lag features
    for sid in df["sensor_id"].unique():
        mask = df["sensor_id"] == sid
        for lag in [1, 3, 6, 12]:
            df.loc[mask, f"power_kw_lag{lag}"] = df.loc[mask, "power_kw"].shift(lag)

    df = df.dropna()
    return df


def create_silver_solar(solar_df: pd.DataFrame) -> pd.DataFrame:
    """Crée la couche Silver des données solaires (Bronze → Silver).

    Étape 1 de la transformation Medallion pour le solar :
    - Nettoyage et dédoublonnage
    - Rolling features : production_avg5, production_std5
    - Dérivée première : production_delta
    - Flag anomalie solaire (chute soudaine d'irradiance)
    """
    df = solar_df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sensor_id", "ts"])
    df = df.drop_duplicates(subset=["sensor_id", "timestamp"])

    # Filtrage des valeurs physiquement impossibles
    df = df[df["production_kw"].notna() & (df["production_kw"] >= 0) & (df["production_kw"] < 1000)]

    # Rolling features (fenêtre 5 mesures)
    for sid in df["sensor_id"].unique():
        mask = df["sensor_id"] == sid
        df.loc[mask, "production_avg5"] = df.loc[mask, "production_kw"].rolling(5, min_periods=1).mean()
        df.loc[mask, "production_std5"] = df.loc[mask, "production_kw"].rolling(5, min_periods=1).std().fillna(0)
        df.loc[mask, "production_delta"] = df.loc[mask, "production_kw"].diff().fillna(0)

    # Flag anomalie solaire (chute soudaine : z-score > 3)
    df["anomaly_solar"] = (
        abs(df["production_delta"]) > 3 * df["production_std5"]
    ).astype(int)

    return df


def create_gold_solar(silver_solar_df: pd.DataFrame) -> pd.DataFrame:
    """Crée la couche Gold des données solaires (Silver → Gold).

    Étape 2 de la transformation Medallion pour le solar :
    Prend en entrée le DataFrame Silver (déjà nettoyé) et ajoute
    les features temporelles cycliques et les lags pour le ML.
    """
    df = silver_solar_df.copy()
    df["ts"] = pd.to_datetime(df["ts"]) if "ts" in df.columns else pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sensor_id", "ts"])

    df["hour"] = df["ts"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week"] = df["ts"].dt.dayofweek
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lag features solaires
    for sid in df["sensor_id"].unique():
        mask = df["sensor_id"] == sid
        for lag in [1, 3, 6]:
            df.loc[mask, f"production_lag{lag}"] = df.loc[mask, "production_kw"].shift(lag)

    df = df.dropna()
    return df


def main():
    """Point d'entrée principal — génère tous les datasets."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mode_actif = DATA_MODE if DATA_MODE in ("real", "synthetic") else "synthetic"
    print(f"🌿 GreenIoT-MA — Génération de dataset statique [MODE={mode_actif.upper()}]")
    print(f"   Durée : {DAYS} jours | Intervalle : {INTERVAL_MINUTES} min")
    print(f"   Output : {OUTPUT_DIR}\n")

    # ── 1. Données SOLAIRES (toujours synthétique — modèle Dakhla) ───────
    print("📡 [Synthèse] Génération des données solaires (irradiance Dakhla)...")
    solar_df = generate_solar_data(DAYS, INTERVAL_MINUTES)
    solar_df.to_parquet(os.path.join(OUTPUT_DIR, "raw_solar.parquet"), index=False)
    print(f"   ✅ {len(solar_df)} enregistrements solaires")

    # ── 2. Données SERVEURS (UCI si disponible, sinon synthèse) ────────
    server_df = pd.DataFrame()
    if mode_actif == "real" and _uci_available:
        try:
            print("🖥️  [UCI REAL] Chargement UCI Household Power Consumption...")
            server_df = load_uci_as_servers(n_racks=3, sample_days=DAYS)
            if not server_df.empty:
                print(f"   ✅ {len(server_df):,} enregistrements serveurs (source : UCI réel)")
        except Exception as e:
            print(f"   ⚠️  UCI échec : {e} — Fallback synthèse")
            server_df = pd.DataFrame()

    if server_df.empty:
        print("🖥️  [Synthèse] Génération des données serveurs...")
        server_df = generate_server_data(DAYS, INTERVAL_MINUTES)
        print(f"   ✅ {len(server_df)} enregistrements serveurs (source : synthèse physique)")

    server_df.to_parquet(os.path.join(OUTPUT_DIR, "raw_servers.parquet"), index=False)

    # ── 3. Données COOLING (ASHRAE si disponible, sinon synthèse) ──────
    cooling_df = pd.DataFrame()
    if mode_actif == "real" and _ashrae_available:
        try:
            print("❄️  [ASHRAE REAL] Chargement ASHRAE Energy Prediction...")
            cooling_df = load_ashrae_as_cooling(sample_days=DAYS)
            if not cooling_df.empty:
                print(f"   ✅ {len(cooling_df):,} enregistrements cooling (source : ASHRAE réel)")
        except Exception as e:
            print(f"   ⚠️  ASHRAE échec : {e} — Fallback synthèse")
            cooling_df = pd.DataFrame()

    if cooling_df.empty:
        print("❄️  [Synthèse] Génération des données refroidissement...")
        cooling_df = generate_cooling_data(DAYS, INTERVAL_MINUTES)
        print(f"   ✅ {len(cooling_df)} enregistrements refroidissement (source : synthèse physique)")

    cooling_df.to_parquet(os.path.join(OUTPUT_DIR, "raw_cooling.parquet"), index=False)

    # ── 4. Données BATTERIE (toujours synthétique — modèle SOC) ───────
    print("🔋 [Synthèse] Génération des données batterie (modèle SOC)...")
    battery_df = generate_battery_data(DAYS, INTERVAL_MINUTES)
    battery_df.to_parquet(os.path.join(OUTPUT_DIR, "raw_battery.parquet"), index=False)
    print(f"   ✅ {len(battery_df)} enregistrements batterie")

    # 2. Couche Silver
    print("\n🔄 Transformation Bronze → Silver...")
    silver_servers = create_silver_data(server_df)
    silver_servers.to_parquet(os.path.join(OUTPUT_DIR, "silver_servers_latest.parquet"), index=False)
    print(f"   ✅ {len(silver_servers)} enregistrements Silver (serveurs)")

    silver_solar = create_silver_solar(solar_df)
    silver_solar.to_parquet(os.path.join(OUTPUT_DIR, "silver_solar_latest.parquet"), index=False)
    print(f"   ✅ {len(silver_solar)} enregistrements Silver (solaire)")

    # 3. Couche Gold (ML-ready)
    print("🏅 Transformation Silver → Gold...")
    gold_servers = create_gold_data(silver_servers)
    gold_servers.to_parquet(os.path.join(OUTPUT_DIR, "gold_servers.parquet"), index=False)
    print(f"   ✅ {len(gold_servers)} enregistrements Gold (serveurs)")

    gold_solar = create_gold_solar(silver_solar)  # Silver Solar → Gold Solar
    gold_solar.to_parquet(os.path.join(OUTPUT_DIR, "gold_solar.parquet"), index=False)
    print(f"   ✅ {len(gold_solar)} enregistrements Gold (solaire)")

    # Résumé
    print(f"\n{'='*60}")
    print(f"📊 Résumé des données générées :")
    print(f"   Bronze serveurs : {len(server_df):>8} lignes")
    print(f"   Bronze solaire  : {len(solar_df):>8} lignes")
    print(f"   Bronze cooling  : {len(cooling_df):>8} lignes")
    print(f"   Bronze batterie : {len(battery_df):>8} lignes")
    print(f"   Silver serveurs : {len(silver_servers):>8} lignes")
    print(f"   Silver solaire  : {len(silver_solar):>8} lignes")
    print(f"   Gold serveurs   : {len(gold_servers):>8} lignes")
    print(f"   Gold solaire    : {len(gold_solar):>8} lignes")
    print(f"\n   📂 Fichiers dans : {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
