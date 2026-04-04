"""
GreenIoT-MA — Optimiseur de décalage de charge vers les pics solaires
========================================================================
Calcule la fenêtre optimale pour décaler les tâches batch
(sauvegardes, entraînements ML, rapports) vers les heures
de production solaire maximale.

Objectifs :
- ≥ 20% de charge déplacée vers les pics solaires
- ≥ 50 kg CO2 économisé par jour
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from deltalake import DeltaTable

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GOLD_SOLAR_PATH = os.getenv("DELTA_GOLD", "s3a://greeniot/gold").replace("s3a://", "s3://") + "/solar"

STORAGE_OPTIONS = {
    "AWS_ACCESS_KEY_ID":        os.getenv("MINIO_ACCESS_KEY", "greeniot"),
    "AWS_SECRET_ACCESS_KEY":    os.getenv("MINIO_SECRET_KEY", "greeniot2030"),
    "AWS_ENDPOINT_URL":         os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
    "AWS_REGION":               "us-east-1",
    "AWS_ALLOW_HTTP":           "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}


def find_solar_peaks(solar_df: pd.DataFrame, window_hours: int = 2) -> dict:
    """
    Trouve les fenêtres de production solaire maximale.

    Args:
        solar_df: DataFrame avec colonnes 'timestamp' et 'production_kw'
        window_hours: Durée de la fenêtre en heures

    Returns:
        dict avec optimal_start, optimal_end, expected_kw
    """
    solar_df = solar_df.copy()
    solar_df["ts"] = pd.to_datetime(solar_df["timestamp"], format="mixed", errors="coerce")
    solar_df = solar_df.dropna(subset=["ts"])
    solar_df = solar_df.set_index("ts").sort_index()

    # CORRECTION : Ne regarder que les TOUTES DERNIÈRES 24 HEURES de données.
    # Sinon, idxmax() reste toujours bloqué sur le plus gros pic historique de toute l'année !
    if not solar_df.empty:
        last_24h = solar_df.index.max() - timedelta(hours=24)
        solar_df = solar_df[solar_df.index >= last_24h]

    # Rolling sum pour trouver les meilleures fenêtres
    window_str = f"{window_hours * 60}min"
    solar_df["window_sum"] = solar_df["production_kw"].rolling(window_str).sum()

    best_idx = solar_df["window_sum"].idxmax()
    
    # Si le dataframe est vide (problème source), fallback sécurisé
    if pd.isna(best_idx):
        best_idx = datetime.now()
        
    best_start = best_idx - timedelta(hours=window_hours)
    best_end = best_idx

    return {
        "optimal_date": best_idx.strftime("%Y-%m-%d"),
        "optimal_start": best_start.strftime("%H:%M"),
        "optimal_end": best_end.strftime("%H:%M"),
        "expected_kw": round(solar_df["window_sum"].max() / max(1, (window_hours * 12)), 1) if not solar_df.empty else 0,
        "peak_timestamp": best_idx.isoformat(),
    }


def schedule_deferred_tasks(
    tasks: list[dict],
    solar_df: pd.DataFrame,
    window_hours: int = 4
) -> pd.DataFrame:
    """
    Associe chaque tâche batch à une fenêtre solaire optimale.

    Args:
        tasks: Liste de dicts avec 'name', 'duration_min', 'priority'
        solar_df: DataFrame solaire
        window_hours: Fenêtre de planification

    Returns:
        DataFrame avec le planning optimisé
    """
    peaks = find_solar_peaks(solar_df, window_hours)

    # Facteur d'émission CO2 pour le Maroc (kg CO2 / kWh)
    CO2_FACTOR = 0.7  # Réseau électrique marocain

    scheduled = []
    current_offset = 0

    # Trier par priorité (low en premier car plus flexible)
    priority_order = {"low": 0, "medium": 1, "high": 2}
    sorted_tasks = sorted(tasks, key=lambda x: priority_order.get(x.get("priority", "low"), 0))

    for task in sorted_tasks:
        start = datetime.strptime(peaks["optimal_start"], "%H:%M") \
            + timedelta(minutes=current_offset)
        end = start + timedelta(minutes=task["duration_min"])

        # CO2 économisé = kWh consommés × facteur d'émission évité
        co2_saved = round(task["duration_min"] / 60 * peaks["expected_kw"] * CO2_FACTOR, 2)

        scheduled.append({
            "task": task["name"],
            "priority": task.get("priority", "low"),
            "duration_min": task["duration_min"],
            "scheduled_start": start.strftime("%H:%M"),
            "scheduled_end": end.strftime("%H:%M"),
            "estimated_solar_kw": peaks["expected_kw"],
            "co2_saved_kg": co2_saved,
            "energy_source": "solar" if 10 <= start.hour <= 16 else "mixed",
        })
        current_offset += task["duration_min"] + 5  # 5 min buffer entre tâches

    return pd.DataFrame(scheduled)


def generate_daily_report(solar_df: pd.DataFrame, tasks: list[dict]) -> dict:
    """
    Génère un rapport quotidien d'optimisation.

    Returns:
        dict avec métriques d'optimisation
    """
    peaks = find_solar_peaks(solar_df)
    schedule = schedule_deferred_tasks(tasks, solar_df)

    total_duration = schedule["duration_min"].sum()
    total_co2 = schedule["co2_saved_kg"].sum()
    solar_tasks = schedule[schedule["energy_source"] == "solar"]
    pct_solar = len(solar_tasks) / max(1, len(schedule)) * 100

    return {
        "date": peaks.get("optimal_date", datetime.now().strftime("%Y-%m-%d")),
        "optimal_window": f"{peaks['optimal_start']} — {peaks['optimal_end']}",
        "expected_solar_kw": peaks["expected_kw"],
        "tasks_scheduled": len(schedule),
        "total_duration_min": total_duration,
        "total_co2_saved_kg": round(total_co2, 2),
        "pct_tasks_solar": round(pct_solar, 1),
        "schedule": schedule.to_dict(orient="records"),
    }


# ══════════════════════════════════════════════════════════════
# Exécution principale
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🌿 GreenIoT-MA — Optimiseur de décalage de charge\n")

    # Charger les données solaires
    DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in ("true", "1", "yes")
    solar_path = os.path.join(DATA_DIR, "gold_solar.parquet")

    if DEMO_MODE:
        print(f"   📡 [DEMO_MODE] Lecture locale depuis : {solar_path}")
        if not os.path.exists(solar_path):
            print(f"   ❌ Fichier local introuvable.")
            print("   💡 Lancez d'abord : python 01_simulation/generate_static_dataset.py")
            exit(1)
        solar_df = pd.read_parquet(solar_path)
        if "ts" in solar_df.columns and "timestamp" not in solar_df.columns:
            solar_df["timestamp"] = solar_df["ts"]
    else:
        print(f"   📡 Lecture Gold Delta depuis : {GOLD_SOLAR_PATH}")
        try:
            dt = DeltaTable(GOLD_SOLAR_PATH, storage_options=STORAGE_OPTIONS)
            solar_df = dt.to_pandas()
            if "ts" in solar_df.columns and "timestamp" not in solar_df.columns:
                solar_df["timestamp"] = solar_df["ts"]
                
        except Exception as e:
            print(f"   ⚠️  Erreur Delta ({e}).")
            print("   🔄 Fallback transparent sur le fichier local...")
            if not os.path.exists(solar_path):
                exit(1)
            solar_df = pd.read_parquet(solar_path)
            if "ts" in solar_df.columns and "timestamp" not in solar_df.columns:
                solar_df["timestamp"] = solar_df["ts"]
            
    print(f"   📊 Données solaires: {len(solar_df)} enregistrements\n")

    # Tâches batch à planifier
    tasks = [
        {"name": "Backup Delta Lake", "duration_min": 45, "priority": "low"},
        {"name": "Entraînement ML nuit", "duration_min": 120, "priority": "low"},
        {"name": "Export rapports BI", "duration_min": 20, "priority": "medium"},
        {"name": "Compression Bronze → archive", "duration_min": 30, "priority": "low"},
        {"name": "Synchronisation MinIO", "duration_min": 15, "priority": "medium"},
    ]

    # Fenêtres solaires optimales
    peaks = find_solar_peaks(solar_df)
    print("☀️  Fenêtre solaire optimale :")
    print(f"   Début    : {peaks['optimal_start']}")
    print(f"   Fin      : {peaks['optimal_end']}")
    print(f"   Puissance: {peaks['expected_kw']} kW\n")

    # Planning
    schedule = schedule_deferred_tasks(tasks, solar_df)
    print("📅 Planning de décalage de charge :")
    print(schedule.to_string(index=False))

    total_co2 = schedule["co2_saved_kg"].sum()
    print(f"\n🌍 CO2 total économisé : {total_co2:.2f} kg/jour")

    # Export CSV
    output_path = os.path.join(DATA_DIR, "load_schedule.csv")
    schedule.to_csv(output_path, index=False)
    print(f"📂 Planning exporté : {output_path}")

    # Rapport
    report = generate_daily_report(solar_df, tasks)
    print(f"\n{'═' * 50}")
    print(f"📊 Rapport quotidien — {report['date']}")
    print(f"   Fenêtre optimale    : {report['optimal_window']}")
    print(f"   Tâches planifiées   : {report['tasks_scheduled']}")
    print(f"   Durée totale        : {report['total_duration_min']} min")
    print(f"   CO2 économisé       : {report['total_co2_saved_kg']} kg")
    print(f"   % tâches solaires   : {report['pct_tasks_solar']:.0f}%")
    print(f"{'═' * 50}")
