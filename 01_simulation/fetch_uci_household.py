"""
GreenIoT-MA — UCI Household Power Consumption Loader
=======================================================
Charge et prépare le dataset UCI Individual Household Electric
Power Consumption pour enrichir le pipeline GreenIoT-MA.

Source : https://archive.ics.uci.edu/dataset/235
Fichier attendu : datasets/household_power_consumption/household_power_consumption.txt
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime


DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
UCI_FILE = os.path.join(
    DATASET_DIR, "household_power_consumption", "household_power_consumption.txt"
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_uci_dataset() -> pd.DataFrame:
    """
    Charge le dataset UCI Household Electric Power Consumption.

    Colonnes du dataset :
    - Date, Time
    - Global_active_power (kW)
    - Global_reactive_power (kW)
    - Voltage (V)
    - Global_intensity (A)
    - Sub_metering_1, Sub_metering_2, Sub_metering_3 (Wh)

    Returns:
        DataFrame nettoyé et prêt pour l'intégration
    """
    if not os.path.exists(UCI_FILE):
        print(f"   ⚠️  Fichier UCI non trouvé : {UCI_FILE}")
        print("   💡 Téléchargez-le depuis : https://archive.ics.uci.edu/dataset/235")
        return pd.DataFrame()

    print(f"   📂 Chargement : {UCI_FILE}")
    df = pd.read_csv(
        UCI_FILE,
        sep=";",
        low_memory=False,
        na_values=["?"],
    )

    # Parser la date et l'heure
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"])
    df = df.drop(columns=["Date", "Time"])

    # Convertir les colonnes numériques
    numeric_cols = [
        "Global_active_power", "Global_reactive_power",
        "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Supprimer les lignes avec trop de NaN
    df = df.dropna(thresh=5)

    # Trier par timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"   ✅ {len(df)} enregistrements chargés ({df['timestamp'].min()} → {df['timestamp'].max()})")
    return df


def prepare_for_greeniot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme les données UCI pour qu'elles soient compatibles
    avec le pipeline GreenIoT-MA (format serveur).

    Mapping :
    - Global_active_power → power_kw
    - Global_intensity → cpu_pct (normalisé)
    - Voltage → temp_c (mappé sur plage réaliste)
    """
    if df.empty:
        return df

    result = pd.DataFrame()
    result["sensor_id"] = "uci_household"
    result["type"] = "server"
    result["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Mapper les métriques vers le format GreenIoT
    result["power_kw"] = df["Global_active_power"].clip(0, 200)
    result["cpu_pct"] = (df["Global_intensity"] / df["Global_intensity"].max() * 100).clip(0, 100).round(1)
    result["ram_pct"] = (df["Sub_metering_1"].fillna(0) / df["Sub_metering_1"].max() * 80 + 20).clip(0, 100).round(1)
    result["temp_c"] = (35 + df["Global_active_power"] * 3 + np.random.randn(len(df)) * 1).clip(25, 80).round(1)

    result = result.dropna()
    return result


def main():
    """Charge le dataset UCI et le prépare pour GreenIoT-MA."""
    print("🌿 GreenIoT-MA — Intégration du dataset UCI\n")

    # Charger
    df = load_uci_dataset()
    if df.empty:
        return

    # Statistiques
    print(f"\n   📊 Statistiques UCI :")
    print(f"      Période        : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"      Enregistrements: {len(df):,}")
    print(f"      Puissance moy. : {df['Global_active_power'].mean():.2f} kW")
    print(f"      Puissance max  : {df['Global_active_power'].max():.2f} kW")

    # Préparer pour GreenIoT
    greeniot_df = prepare_for_greeniot(df)
    print(f"\n   🔄 Conversion vers format GreenIoT : {len(greeniot_df)} lignes")

    # Sauvegarder
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Dataset UCI brut (pour référence)
    uci_path = os.path.join(OUTPUT_DIR, "uci_household_raw.parquet")
    df.to_parquet(uci_path, index=False)
    print(f"   💾 UCI brut → {uci_path}")

    # Dataset au format GreenIoT (Bronze)
    greeniot_path = os.path.join(OUTPUT_DIR, "uci_as_bronze.parquet")
    greeniot_df.to_parquet(greeniot_path, index=False)
    print(f"   💾 UCI→GreenIoT → {greeniot_path}")

    # Échantillon pour vérification
    print(f"\n   🔍 Échantillon (5 premières lignes) :")
    print(greeniot_df.head().to_string(index=False))

    print(f"\n{'═' * 50}")
    print("   ✅ Dataset UCI intégré avec succès !")
    print(f"{'═' * 50}")


if __name__ == "__main__":
    main()
