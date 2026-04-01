"""
GreenIoT-MA — Générateur de dataset statique
===============================================
Produit 7 jours de données simulées en CSV et Parquet
pour le développement et le test hors-ligne du pipeline.
Génère des fichiers dans le dossier data/.

Datasets produits :
- data/raw_servers.parquet    → Données serveurs brutes (Bronze)
- data/raw_solar.parquet      → Données solaires brutes (Bronze)
- data/raw_cooling.parquet    → Données refroidissement brutes (Bronze)
- data/silver_servers_latest.parquet → Données nettoyées (Silver)
- data/gold_servers.parquet   → Données ML-ready (Gold)
- data/gold_solar.parquet     → Données solaires ML-ready (Gold)
"""

import os
import sys
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Paramètres
DAYS = 7
INTERVAL_MINUTES = 5
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


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
        hour = ts.hour
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
        hour = ts.hour

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

    # PUE simulé pour le dashboard
    df["pue"] = np.random.uniform(1.3, 1.8, len(df))

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


def create_gold_solar(solar_df: pd.DataFrame) -> pd.DataFrame:
    """Crée une version Gold des données solaires."""
    df = solar_df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sensor_id", "ts"])

    df["hour"] = df["ts"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Rolling production
    for sid in df["sensor_id"].unique():
        mask = df["sensor_id"] == sid
        df.loc[mask, "production_avg5"] = df.loc[mask, "production_kw"].rolling(5, min_periods=1).mean()

    return df


def main():
    """Point d'entrée principal — génère tous les datasets."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🌿 GreenIoT-MA — Génération de dataset statique")
    print(f"   Durée : {DAYS} jours | Intervalle : {INTERVAL_MINUTES} min")
    print(f"   Output : {OUTPUT_DIR}\n")

    # 1. Données brutes (Bronze)
    print("📡 Génération des données solaires...")
    solar_df = generate_solar_data(DAYS, INTERVAL_MINUTES)
    solar_df.to_parquet(os.path.join(OUTPUT_DIR, "raw_solar.parquet"), index=False)
    print(f"   ✅ {len(solar_df)} enregistrements solaires")

    print("🖥️  Génération des données serveurs...")
    server_df = generate_server_data(DAYS, INTERVAL_MINUTES)
    server_df.to_parquet(os.path.join(OUTPUT_DIR, "raw_servers.parquet"), index=False)
    print(f"   ✅ {len(server_df)} enregistrements serveurs")

    print("❄️  Génération des données refroidissement...")
    cooling_df = generate_cooling_data(DAYS, INTERVAL_MINUTES)
    cooling_df.to_parquet(os.path.join(OUTPUT_DIR, "raw_cooling.parquet"), index=False)
    print(f"   ✅ {len(cooling_df)} enregistrements refroidissement")

    # 2. Données Silver
    print("\n🔄 Transformation Bronze → Silver...")
    silver_df = create_silver_data(server_df)
    silver_df.to_parquet(os.path.join(OUTPUT_DIR, "silver_servers_latest.parquet"), index=False)
    print(f"   ✅ {len(silver_df)} enregistrements Silver")

    # 3. Données Gold (ML-ready)
    print("🏅 Transformation Silver → Gold...")
    gold_servers = create_gold_data(silver_df)
    gold_servers.to_parquet(os.path.join(OUTPUT_DIR, "gold_servers.parquet"), index=False)
    print(f"   ✅ {len(gold_servers)} enregistrements Gold (serveurs)")

    gold_solar = create_gold_solar(solar_df)
    gold_solar.to_parquet(os.path.join(OUTPUT_DIR, "gold_solar.parquet"), index=False)
    print(f"   ✅ {len(gold_solar)} enregistrements Gold (solaire)")

    # Résumé
    print(f"\n{'='*60}")
    print(f"📊 Résumé des données générées :")
    print(f"   Bronze serveurs : {len(server_df):>8} lignes")
    print(f"   Bronze solaire  : {len(solar_df):>8} lignes")
    print(f"   Bronze cooling  : {len(cooling_df):>8} lignes")
    print(f"   Silver serveurs : {len(silver_df):>8} lignes")
    print(f"   Gold serveurs   : {len(gold_servers):>8} lignes")
    print(f"   Gold solaire    : {len(gold_solar):>8} lignes")
    print(f"\n   📂 Fichiers dans : {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
