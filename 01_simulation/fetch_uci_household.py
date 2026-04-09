"""
GreenIoT-MA — UCI Household Power Consumption Loader
=======================================================
Charge le dataset UCI Individual Household Electric Power Consumption
depuis le chemin défini dans .env (variable UCI_DATASET).

Colonnes du dataset UCI (séparateur ';') :
  Date;Time;Global_active_power;Global_reactive_power;
  Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3

Mapping vers GreenIoT-MA (format Bronze servers) :
  Global_active_power (kW)   → power_kw   (consommation rack)
  Global_intensity (A)       → cpu_pct    (charge CPU normalisée)
  Sub_metering_1 (Wh)        → ram_pct    (charge RAM proxy)
  Voltage (V) + power        → temp_c     (température estimée)

Usage :
    python fetch_uci_household.py                  # affiche stats + sauvegarde
    from fetch_uci_household import load_uci_as_servers  # intégration pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── Chemins ───────────────────────────────────────────────────────────────────
UCI_FILE   = os.getenv(
    "UCI_DATASET",
    r"C:\Users\ALI\Downloads\individual+household+electric+power+consumption\household_power_consumption.txt"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


# ══════════════════════════════════════════════════════════════════════════════
# Chargement brut
# ══════════════════════════════════════════════════════════════════════════════

def load_uci_raw(sample_days: int = None) -> pd.DataFrame:
    """
    Charge le dataset UCI brut.

    Args:
        sample_days: Si spécifié, ne charge que les N derniers jours du dataset
                     (utile pour les tests rapides — le fichier complet = 2M lignes).

    Returns:
        DataFrame avec colonnes UCI originales + colonne 'timestamp'.
    """
    if not os.path.exists(UCI_FILE):
        raise FileNotFoundError(
            f"Dataset UCI introuvable : {UCI_FILE}\n"
            f"Vérifiez la variable UCI_DATASET dans .env"
        )

    print(f"   📂 Chargement UCI depuis : {UCI_FILE}")
    print(f"      Taille fichier : {os.path.getsize(UCI_FILE) / 1e6:.1f} Mo")

    df = pd.read_csv(
        UCI_FILE,
        sep=";",
        low_memory=False,
        na_values=["?"],
        header=None,   # La première ligne est déjà les données (pas d'entête propre)
    )

    # Le fichier UCI n'a pas d'entête dans certaines versions — on gère les deux cas
    if df.iloc[0, 0] == "Date":
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
    else:
        df.columns = [
            "Date", "Time",
            "Global_active_power", "Global_reactive_power",
            "Voltage", "Global_intensity",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
        ]

    # Parser timestamp
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"])
    df = df.drop(columns=["Date", "Time"], errors="ignore")

    # Conversion numérique
    num_cols = [
        "Global_active_power", "Global_reactive_power",
        "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Supprimer les lignes avec trop de valeurs manquantes
    df = df.dropna(thresh=5).sort_values("timestamp").reset_index(drop=True)

    # Filtrage optionnel sur N jours
    if sample_days is not None:
        cutoff = df["timestamp"].max() - timedelta(days=sample_days)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    print(f"   ✅ {len(df):,} lignes chargées")
    print(f"      Période : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"      Puissance moy. : {df['Global_active_power'].mean():.3f} kW")
    print(f"      Puissance max  : {df['Global_active_power'].max():.3f} kW")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Transformation → format GreenIoT Bronze (servers)
# ══════════════════════════════════════════════════════════════════════════════

def load_uci_as_servers(
    n_racks: int = 3,
    sample_days: int = 7,
    target_timestamps: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    """
    Charge UCI et le transforme en données de serveurs GreenIoT-MA.

    Stratégie de mapping :
    - La consommation UCI (maison) est re-scalée pour représenter
      un rack de serveurs (plage : 30–120 kW).
    - Chaque rack is une interpolation/variation légère de la série UCI.
    - Le pattern horaire UCI (circadien résidentiel) est conservé car
      il est corrélé au trafic web (forte charge le soir).

    Args:
        n_racks: Nombre de racks à simuler depuis la même source UCI
        sample_days: Jours de données à charger
        target_timestamps: Si fourni, rééchantillonne aux timestamps voulus

    Returns:
        DataFrame format Bronze servers compatible avec le pipeline GreenIoT
    """
    df = load_uci_raw(sample_days=sample_days)
    if df.empty:
        return pd.DataFrame()

    # Normaliser la puissance UCI (max ~14kW maison → racks serveurs 30-120kW)
    p = df["Global_active_power"].fillna(df["Global_active_power"].median())
    p_norm = (p - p.min()) / (p.max() - p.min() + 1e-9)  # 0–1

    rng = np.random.default_rng(42)
    records = []

    rack_ids = [f"rack_A{i+1}" for i in range(n_racks)]
    for i, rack_id in enumerate(rack_ids):
        # Variation légère par rack : décalage de phase + bruit différent
        noise = rng.normal(0, 0.02, len(df))
        scale_factor = rng.uniform(0.85, 1.15)  # ±15% entre racks
        power_rack = (30 + p_norm * 90 * scale_factor + noise * 5).clip(20, 130).round(2)

        # CPU corrélé à la puissance (plus de charge = plus de CPU)
        cpu = (p_norm * 70 + 15 + rng.normal(0, 3, len(df))).clip(5, 98).round(1)

        # RAM légèrement corrélée au CPU avec inertie
        ram_base = pd.Series(cpu).rolling(10, min_periods=1).mean().values
        ram = (ram_base * 0.6 + 30 + rng.normal(0, 2, len(df))).clip(10, 95).round(1)

        # Température : dépend de la puissance et du CPU
        temp = (35 + power_rack * 0.15 + cpu * 0.08 + rng.normal(0, 0.8, len(df))).clip(28, 75).round(1)

        rack_df = pd.DataFrame({
            "sensor_id": rack_id,
            "type": "server",
            "timestamp": df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "ts": df["timestamp"],
            "power_kw": power_rack,
            "cpu_pct": cpu,
            "ram_pct": ram,
            "temp_c": temp,
        })
        records.append(rack_df)

    result = pd.concat(records, ignore_index=True)

    # Rééchantillonnage optionnel (pour aligner sur interval_min)
    if target_timestamps is not None:
        aligned_frames = []
        for rack_id, rack_df in result.groupby("sensor_id", sort=False):
            aligned = (
                rack_df.sort_values("ts")
                .set_index("ts")
                .reindex(target_timestamps, method="nearest", tolerance="5min")
                .reset_index()
                .rename(columns={"index": "ts"})
            )
            aligned["sensor_id"] = rack_id
            aligned["timestamp"] = aligned["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S")
            aligned_frames.append(aligned)

        result = pd.concat(aligned_frames, ignore_index=True)
        result = result.dropna(subset=["power_kw"])

    print(f"\n   🖥️  Racks générés : {n_racks} × {len(df):,} points = {len(result):,} lignes total")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée standalone
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Charge UCI, affiche les stats et sauvegarde les fichiers."""
    print("🌿 GreenIoT-MA — Intégration du dataset UCI Household\n")

    df_raw = load_uci_raw(sample_days=7)
    if df_raw.empty:
        sys.exit(1)

    df_servers = load_uci_as_servers(n_racks=3, sample_days=7)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sauvegardes
    raw_path     = os.path.join(OUTPUT_DIR, "uci_household_raw.parquet")
    bronze_path  = os.path.join(OUTPUT_DIR, "raw_servers.parquet")

    df_raw.to_parquet(raw_path, index=False)
    print(f"\n   💾 UCI brut     → {raw_path}")

    df_servers.to_parquet(bronze_path, index=False)
    print(f"   💾 Bronze srv   → {bronze_path}")

    print(f"\n{'═' * 60}")
    print("   ✅ Dataset UCI intégré avec succès !")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
