"""
GreenIoT-MA — ASHRAE Energy Prediction Loader
===============================================
Charge le dataset ASHRAE Energy Prediction depuis le dossier
défini dans .env (variables ASHRAE_TRAIN_DATASET, ASHRAE_BUILDING_META,
ASHRAE_WEATHER_TRAIN).

Structure du dataset ASHRAE :
  train.csv           : building_id, meter, timestamp, meter_reading (kWh)
  building_metadata.csv : site_id, building_id, primary_use, floor_count_feet, year_built
  weather_train.csv   : site_id, timestamp, air_temperature, dew_temperature,
                        cloud_coverage, precip_depth_1_hr, sea_level_pressure,
                        wind_direction, wind_speed

Mapping vers GreenIoT-MA (format Bronze cooling) :
  meter_reading (kWh) → it_load_kw   (charge IT data center)
  air_temperature (°C) → outdoor température pour PUE
  meter_reading + temp → pue         (Power Usage Effectiveness)

On filtre les bâtiments de type "Office" ou "Education" (typiques data centers)
avec meter=0 (électricité principale, pas chaud/froid/vapeur).

Usage :
    python fetch_ashrae.py                       # stats + sauvegarde
    from fetch_ashrae import load_ashrae_as_cooling  # intégration pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── Chemins ───────────────────────────────────────────────────────────────────
ASHRAE_TRAIN   = os.getenv(
    "ASHRAE_TRAIN_DATASET",
    r"C:\Users\ALI\Downloads\ashrae-energy-prediction\train.csv"
)
ASHRAE_META    = os.getenv(
    "ASHRAE_BUILDING_META",
    r"C:\Users\ALI\Downloads\ashrae-energy-prediction\building_metadata.csv"
)
ASHRAE_WEATHER = os.getenv(
    "ASHRAE_WEATHER_TRAIN",
    r"C:\Users\ALI\Downloads\ashrae-energy-prediction\weather_train.csv"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


# ══════════════════════════════════════════════════════════════════════════════
# Chargement brut (avec optimisation mémoire)
# ══════════════════════════════════════════════════════════════════════════════

def load_ashrae_raw(
    building_ids: list = None,
    max_buildings: int = 5,
    meter_type: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge les fichiers ASHRAE en mémoire optimisée.

    Args:
        building_ids: Liste d'IDs de bâtiments spécifiques (None = auto-sélection)
        max_buildings: Nombre max de bâtiments à charger (si building_ids=None)
        meter_type: 0=Electricity, 1=Chilled Water, 2=Steam, 3=Hot Water

    Returns:
        (df_train, df_meta, df_weather) — DataFrames bruts filtrés
    """
    for path, name in [(ASHRAE_TRAIN, "train.csv"), (ASHRAE_META, "building_metadata.csv")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset ASHRAE introuvable : {path}\n"
                f"Vérifiez la variable correspondante dans .env"
            )

    print(f"   📂 Chargement ASHRAE depuis : {os.path.dirname(ASHRAE_TRAIN)}")

    # 1. Métadonnées (petit fichier)
    print("   📃 Lecture building_metadata.csv...")
    df_meta = pd.read_csv(ASHRAE_META)
    print(f"      {len(df_meta)} bâtiments totaux")

    # 2. Sélectionner les bâtiments data-center like (Office ou Education)
    dc_like = df_meta[df_meta["primary_use"].isin(["Office", "Education", "Technology/science"])]
    if dc_like.empty:
        dc_like = df_meta

    if building_ids is None:
        # Prendre les bâtiments les plus grands (floor_count_feet)
        if "square_feet" in dc_like.columns:
            dc_like = dc_like.nlargest(max_buildings, "square_feet")
        else:
            dc_like = dc_like.head(max_buildings)
        building_ids = dc_like["building_id"].tolist()

    print(f"      Bâtiments sélectionnés : {building_ids}")

    # 3. Train (gros fichier — lecture par chunks filtrée)
    print(f"   📊 Lecture train.csv (peut prendre 30-60 sec, fichier ~650 Mo)...")
    file_size_mb = os.path.getsize(ASHRAE_TRAIN) / 1e6

    chunks = []
    chunk_size = 200_000
    for chunk in pd.read_csv(
        ASHRAE_TRAIN,
        chunksize=chunk_size,
        dtype={"building_id": np.int16, "meter": np.int8, "meter_reading": np.float32},
    ):
        filtered = chunk[
            (chunk["building_id"].isin(building_ids)) &
            (chunk["meter"] == meter_type)
        ]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        print(f"   ❌ Aucune donnée pour les bâtiments {building_ids} avec meter={meter_type}")
        return pd.DataFrame(), df_meta, pd.DataFrame()

    df_train = pd.concat(chunks, ignore_index=True)
    df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])
    df_train = df_train.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    print(f"      ✅ {len(df_train):,} mesures chargées")
    print(f"         Période : {df_train['timestamp'].min()} → {df_train['timestamp'].max()}")
    print(f"         Lecture moy. : {df_train['meter_reading'].mean():.2f} kWh")

    # 4. Météo (si disponible)
    df_weather = pd.DataFrame()
    if os.path.exists(ASHRAE_WEATHER):
        print("   🌤  Lecture weather_train.csv...")
        df_weather = pd.read_csv(ASHRAE_WEATHER)
        df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"])
        # Filtrer sur les sites des bâtiments sélectionnés
        site_ids = dc_like[dc_like["building_id"].isin(building_ids)]["site_id"].unique().tolist()
        df_weather = df_weather[df_weather["site_id"].isin(site_ids)]
        print(f"      ✅ {len(df_weather):,} mesures météo")

    return df_train, df_meta, df_weather


# ══════════════════════════════════════════════════════════════════════════════
# Transformation → format GreenIoT Bronze (cooling)
# ══════════════════════════════════════════════════════════════════════════════

def load_ashrae_as_cooling(
    sample_days: int = 7,
    building_ids: list = None,
) -> pd.DataFrame:
    """
    Charge ASHRAE et le transforme en données de refroidissement GreenIoT-MA.

    Stratégie de mapping :
    - meter_reading (kWh/h) → it_load_kw (charge IT mesurée)
    - Température extérieure  → calcul PUE réaliste (chaud = PUE élevé)
    - PUE = 1.20 + 0.012 × max(0, T_ext - 18°C)

    Args:
        sample_days: Nombre de jours à extraire
        building_ids: IDs spécifiques (None = auto-sélection)

    Returns:
        DataFrame format Bronze cooling compatible avec GreenIoT-MA
    """
    df_train, df_meta, df_weather = load_ashrae_raw(
        building_ids=building_ids,
        max_buildings=3,
    )

    if df_train.empty:
        return pd.DataFrame()

    # Prendre uniquement les N derniers jours du dataset
    cutoff = df_train["timestamp"].max() - timedelta(days=sample_days)
    df_train = df_train[df_train["timestamp"] >= cutoff].copy()

    # Joindre avec la météo si disponible
    if not df_weather.empty and "air_temperature" in df_weather.columns:
        # Joindre par site_id + timestamp (arrondi à l'heure)
        df_meta_sel = df_meta[df_meta["building_id"].isin(df_train["building_id"].unique())]
        df_train = df_train.merge(
            df_meta_sel[["building_id", "site_id"]], on="building_id", how="left"
        )
        df_weather["hour_ts"] = df_weather["timestamp"].dt.floor("h")
        df_train["hour_ts"]   = df_train["timestamp"].dt.floor("h")
        df_train = df_train.merge(
            df_weather[["site_id", "hour_ts", "air_temperature"]].drop_duplicates(),
            on=["site_id", "hour_ts"], how="left"
        )
        df_train["air_temperature"] = df_train["air_temperature"].fillna(25.0)
    else:
        df_train["air_temperature"] = 25.0  # fallback

    rng = np.random.default_rng(42)
    records = []

    for bid in df_train["building_id"].unique():
        sub = df_train[df_train["building_id"] == bid].copy()

        # meter_reading (kWh) → it_load_kw
        it_load = sub["meter_reading"].clip(0, 2000).values

        # PUE réaliste basé sur la température extérieure
        t_ext = sub["air_temperature"].values
        pue = (1.20 + 0.012 * np.maximum(0, t_ext - 18) + rng.normal(0, 0.03, len(sub))).clip(1.08, 2.1).round(3)

        total_power = (it_load * pue).round(1)

        # inlet/outlet température (delta ~13°C typical CRAC)
        inlet  = (t_ext * 0.3 + 15 + rng.normal(0, 1, len(sub))).clip(12, 28).round(1)
        outlet = (inlet + 13 + rng.normal(0, 1.5, len(sub))).clip(20, 55).round(1)

        cooling_df = pd.DataFrame({
            "sensor_id":      f"cooling_unit_{bid:02d}",
            "type":           "cooling",
            "timestamp":      sub["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").values,
            "inlet_temp_c":   inlet,
            "outlet_temp_c":  outlet,
            "it_load_kw":     it_load.round(1),
            "total_power_kw": total_power,
            "pue":            pue,
        })
        records.append(cooling_df)

    result = pd.concat(records, ignore_index=True)
    print(f"\n   ❄️  Données cooling générées : {len(result):,} lignes ({result['pue'].mean():.3f} PUE moy.)")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée standalone
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Charge ASHRAE, affiche les stats et sauvegarde les fichiers."""
    print("🌿 GreenIoT-MA — Intégration du dataset ASHRAE Energy Prediction\n")

    df_cooling = load_ashrae_as_cooling(sample_days=7)
    if df_cooling.empty:
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "raw_cooling.parquet")
    df_cooling.to_parquet(out_path, index=False)
    print(f"\n   💾 Bronze cooling → {out_path}")

    print(f"\n{'═' * 60}")
    print("   ✅ Dataset ASHRAE intégré avec succès !")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
