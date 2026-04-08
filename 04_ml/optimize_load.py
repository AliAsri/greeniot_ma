"""
GreenIoT-MA - Solar load shifting optimizer
===========================================
Schedules deferred batch workloads into the best solar production slots.

The optimizer now works on real time slots with full timestamps instead of
only HH:MM strings. This makes the planning more consistent for both the
dashboard and exported reports.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv

try:
    from deltalake import DeltaTable
except ImportError:  # pragma: no cover - optional at runtime for local/demo mode
    DeltaTable = None

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GOLD_SOLAR_PATH = os.getenv("DELTA_GOLD", "s3a://greeniot/gold").replace("s3a://", "s3://") + "/solar"

STORAGE_OPTIONS = {
    "AWS_ACCESS_KEY_ID": os.getenv("MINIO_ACCESS_KEY", "greeniot"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("MINIO_SECRET_KEY", "greeniot2030"),
    "AWS_ENDPOINT_URL": os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
    "AWS_REGION": "us-east-1",
    "AWS_ALLOW_HTTP": "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

CO2_FACTOR = 0.7
DEFAULT_SLOT_MINUTES = 15
DEFAULT_WINDOW_HOURS = 4
DEFAULT_TASK_POWER_KW = {
    "low": 35.0,
    "medium": 55.0,
    "high": 75.0,
}
PRIORITY_WEIGHT = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


def _normalize_solar_input(solar_df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean dataframe with ts and production_kw columns."""
    if solar_df is None or solar_df.empty:
        return pd.DataFrame(columns=["ts", "production_kw"])

    df = solar_df.copy()
    if "ts" not in df.columns:
        if "timestamp" in df.columns:
            df["ts"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
        else:
            return pd.DataFrame(columns=["ts", "production_kw"])
    else:
        df["ts"] = pd.to_datetime(df["ts"], format="mixed", errors="coerce")

    if "production_kw" not in df.columns:
        return pd.DataFrame(columns=["ts", "production_kw"])

    df = df.dropna(subset=["ts", "production_kw"]).sort_values("ts")
    if df.empty:
        return df[["ts", "production_kw"]]

    cutoff = df["ts"].max() - pd.Timedelta(hours=24)
    df = df[df["ts"] >= cutoff]
    return df[["ts", "production_kw"]]


def build_solar_profile(solar_df: pd.DataFrame, slot_minutes: int = DEFAULT_SLOT_MINUTES) -> pd.DataFrame:
    """Aggregate solar production per time slot across all sensors."""
    df = _normalize_solar_input(solar_df)
    if df.empty:
        return pd.DataFrame(columns=["slot_ts", "solar_kw", "available_kw", "available_kwh"])

    profile = (
        df.set_index("ts")
        .resample(f"{slot_minutes}min")["production_kw"]
        .sum()
        .fillna(0.0)
        .reset_index()
        .rename(columns={"ts": "slot_ts", "production_kw": "solar_kw"})
    )
    slot_hours = slot_minutes / 60
    profile["available_kw"] = profile["solar_kw"].astype(float)
    profile["available_kwh"] = profile["available_kw"] * slot_hours
    return profile


def find_solar_peaks(solar_df: pd.DataFrame, window_hours: int = 2, slot_minutes: int = DEFAULT_SLOT_MINUTES) -> dict:
    """Find the best solar window on the last 24h profile."""
    profile = build_solar_profile(solar_df, slot_minutes=slot_minutes)
    if profile.empty:
        now = pd.Timestamp.now().floor(f"{slot_minutes}min")
        return {
            "optimal_date": now.strftime("%Y-%m-%d"),
            "optimal_start": now.strftime("%H:%M"),
            "optimal_end": (now + pd.Timedelta(hours=window_hours)).strftime("%H:%M"),
            "expected_kw": 0.0,
            "peak_timestamp": now.isoformat(),
            "optimal_start_ts": now.isoformat(),
            "optimal_end_ts": (now + pd.Timedelta(hours=window_hours)).isoformat(),
            "window_energy_kwh": 0.0,
        }

    window_slots = max(1, int(window_hours * 60 / slot_minutes))
    rolling_energy = profile["available_kwh"].rolling(window_slots, min_periods=window_slots).sum()

    if rolling_energy.notna().any():
        best_end_idx = int(rolling_energy.idxmax())
        best_start_idx = max(0, best_end_idx - window_slots + 1)
    else:
        best_start_idx = 0
        best_end_idx = min(len(profile) - 1, window_slots - 1)

    start_ts = profile.iloc[best_start_idx]["slot_ts"]
    end_ts = profile.iloc[best_end_idx]["slot_ts"] + pd.Timedelta(minutes=slot_minutes)
    window_energy = profile.iloc[best_start_idx : best_end_idx + 1]["available_kwh"].sum()
    duration_hours = max(window_hours, slot_minutes / 60)
    expected_kw = round(window_energy / duration_hours, 1)

    return {
        "optimal_date": start_ts.strftime("%Y-%m-%d"),
        "optimal_start": start_ts.strftime("%H:%M"),
        "optimal_end": end_ts.strftime("%H:%M"),
        "expected_kw": expected_kw,
        "peak_timestamp": end_ts.isoformat(),
        "optimal_start_ts": start_ts.isoformat(),
        "optimal_end_ts": end_ts.isoformat(),
        "window_energy_kwh": round(float(window_energy), 2),
    }


def _normalize_tasks(tasks: list[dict]) -> list[dict]:
    normalized = []
    for task in tasks:
        priority = str(task.get("priority", "low")).lower()
        if priority not in PRIORITY_WEIGHT:
            priority = "low"

        duration_min = max(5, int(task.get("duration_min", 30)))
        power_kw = float(task.get("power_kw_required", DEFAULT_TASK_POWER_KW[priority]))

        normalized.append(
            {
                "name": task.get("name", "Task"),
                "priority": priority,
                "duration_min": duration_min,
                "power_kw_required": power_kw,
            }
        )

    normalized.sort(
        key=lambda item: (
            -PRIORITY_WEIGHT[item["priority"]],
            -item["power_kw_required"],
            -item["duration_min"],
        )
    )
    return normalized


def _score_window(window: pd.DataFrame, task_power_kw: float, slot_hours: float, priority: str, peak_center: pd.Timestamp) -> tuple[float, dict]:
    available_kw = window["available_kw"].to_numpy(dtype=float)
    solar_used_kw = available_kw.clip(max=task_power_kw)
    grid_used_kw = (task_power_kw - solar_used_kw).clip(min=0.0)

    solar_kwh = float(solar_used_kw.sum() * slot_hours)
    grid_kwh = float(grid_used_kw.sum() * slot_hours)
    total_kwh = float(task_power_kw * slot_hours * len(window))
    center_ts = window.iloc[0]["slot_ts"] + (window.iloc[-1]["slot_ts"] - window.iloc[0]["slot_ts"]) / 2
    distance_penalty = abs((center_ts - peak_center).total_seconds()) / 3600.0

    score = (
        grid_kwh * 1000
        - solar_kwh * 10 * PRIORITY_WEIGHT[priority]
        + distance_penalty
    )

    return score, {
        "solar_kwh": solar_kwh,
        "grid_kwh": grid_kwh,
        "total_kwh": total_kwh,
    }


def optimize_schedule(
    tasks: list[dict],
    solar_df: pd.DataFrame,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    slot_minutes: int = DEFAULT_SLOT_MINUTES,
) -> dict:
    """Return a unified optimization result consumed by the dashboard."""
    profile = build_solar_profile(solar_df, slot_minutes=slot_minutes)
    peaks = find_solar_peaks(solar_df, window_hours=window_hours, slot_minutes=slot_minutes)

    if profile.empty:
        empty_schedule = pd.DataFrame(
            columns=[
                "task",
                "priority",
                "duration_min",
                "power_kw_required",
                "scheduled_start",
                "scheduled_end",
                "scheduled_start_ts",
                "scheduled_end_ts",
                "estimated_solar_kw",
                "solar_energy_kwh",
                "grid_energy_kwh",
                "co2_saved_kg",
                "energy_source",
                "solar_share_pct",
            ]
        )
        return {
            "peaks": peaks,
            "schedule": empty_schedule,
            "summary": {
                "tasks_scheduled": 0,
                "total_duration_min": 0,
                "total_solar_kwh": 0.0,
                "total_grid_kwh": 0.0,
                "total_co2_saved_kg": 0.0,
                "pct_tasks_solar": 0.0,
                "load_shift_pct": 0.0,
                "avg_solar_share_pct": 0.0,
                "window_energy_kwh": peaks["window_energy_kwh"],
            },
            "profile": profile,
        }

    normalized_tasks = _normalize_tasks(tasks)
    schedule_rows = []
    profile_work = profile.copy()
    slot_hours = slot_minutes / 60
    peak_center = pd.to_datetime(peaks["optimal_start_ts"]) + pd.Timedelta(hours=window_hours / 2)

    for task in normalized_tasks:
        required_slots = max(1, math.ceil(task["duration_min"] / slot_minutes))
        best_choice = None

        for start_idx in range(0, len(profile_work) - required_slots + 1):
            window = profile_work.iloc[start_idx : start_idx + required_slots].copy()
            score, metrics = _score_window(
                window,
                task_power_kw=task["power_kw_required"],
                slot_hours=slot_hours,
                priority=task["priority"],
                peak_center=peak_center,
            )

            candidate = {
                "score": score,
                "start_idx": start_idx,
                "window": window,
                "metrics": metrics,
            }

            if best_choice is None or candidate["score"] < best_choice["score"]:
                best_choice = candidate

        if best_choice is None:
            continue

        start_idx = best_choice["start_idx"]
        window = best_choice["window"]
        metrics = best_choice["metrics"]

        solar_used_kw = window["available_kw"].to_numpy(dtype=float).clip(max=task["power_kw_required"])
        profile_work.loc[window.index, "available_kw"] = (
            profile_work.loc[window.index, "available_kw"].to_numpy(dtype=float) - solar_used_kw
        ).clip(min=0.0)
        profile_work.loc[window.index, "available_kwh"] = profile_work.loc[window.index, "available_kw"] * slot_hours

        start_ts = window.iloc[0]["slot_ts"]
        end_ts = window.iloc[-1]["slot_ts"] + pd.Timedelta(minutes=slot_minutes)
        solar_share_pct = (metrics["solar_kwh"] / max(metrics["total_kwh"], 1e-9)) * 100

        if solar_share_pct >= 80:
            energy_source = "solar"
        elif solar_share_pct >= 35:
            energy_source = "hybrid"
        else:
            energy_source = "mixed"

        schedule_rows.append(
            {
                "task": task["name"],
                "priority": task["priority"],
                "duration_min": task["duration_min"],
                "power_kw_required": task["power_kw_required"],
                "scheduled_start": start_ts.strftime("%H:%M"),
                "scheduled_end": end_ts.strftime("%H:%M"),
                "scheduled_start_ts": start_ts.isoformat(),
                "scheduled_end_ts": end_ts.isoformat(),
                "estimated_solar_kw": round(metrics["solar_kwh"] / max(task["duration_min"] / 60, 1e-9), 1),
                "solar_energy_kwh": round(metrics["solar_kwh"], 2),
                "grid_energy_kwh": round(metrics["grid_kwh"], 2),
                "co2_saved_kg": round(metrics["solar_kwh"] * CO2_FACTOR, 2),
                "energy_source": energy_source,
                "solar_share_pct": round(solar_share_pct, 1),
            }
        )

    schedule_df = pd.DataFrame(schedule_rows).sort_values("scheduled_start_ts").reset_index(drop=True)

    total_duration = int(schedule_df["duration_min"].sum()) if not schedule_df.empty else 0
    total_solar = float(schedule_df["solar_energy_kwh"].sum()) if not schedule_df.empty else 0.0
    total_grid = float(schedule_df["grid_energy_kwh"].sum()) if not schedule_df.empty else 0.0
    total_energy = total_solar + total_grid
    solar_tasks = int((schedule_df["energy_source"] == "solar").sum()) if not schedule_df.empty else 0

    summary = {
        "tasks_scheduled": int(len(schedule_df)),
        "total_duration_min": total_duration,
        "total_solar_kwh": round(total_solar, 2),
        "total_grid_kwh": round(total_grid, 2),
        "total_co2_saved_kg": round(total_solar * CO2_FACTOR, 2),
        "pct_tasks_solar": round((solar_tasks / max(len(schedule_df), 1)) * 100, 1),
        "load_shift_pct": round((total_solar / max(total_energy, 1e-9)) * 100, 1),
        "avg_solar_share_pct": round(schedule_df["solar_share_pct"].mean(), 1) if not schedule_df.empty else 0.0,
        "window_energy_kwh": peaks["window_energy_kwh"],
    }

    return {
        "peaks": peaks,
        "schedule": schedule_df,
        "summary": summary,
        "profile": profile_work,
    }


def schedule_deferred_tasks(
    tasks: list[dict],
    solar_df: pd.DataFrame,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    slot_minutes: int = DEFAULT_SLOT_MINUTES,
) -> pd.DataFrame:
    """Backward-compatible wrapper returning only the schedule dataframe."""
    return optimize_schedule(
        tasks=tasks,
        solar_df=solar_df,
        window_hours=window_hours,
        slot_minutes=slot_minutes,
    )["schedule"]


def generate_daily_report(solar_df: pd.DataFrame, tasks: list[dict]) -> dict:
    """Generate a daily optimization report."""
    result = optimize_schedule(tasks, solar_df)
    peaks = result["peaks"]
    summary = result["summary"]
    schedule = result["schedule"]

    return {
        "date": peaks.get("optimal_date", datetime.now().strftime("%Y-%m-%d")),
        "optimal_window": f"{peaks['optimal_start']} - {peaks['optimal_end']}",
        "expected_solar_kw": peaks["expected_kw"],
        "tasks_scheduled": summary["tasks_scheduled"],
        "total_duration_min": summary["total_duration_min"],
        "total_co2_saved_kg": summary["total_co2_saved_kg"],
        "pct_tasks_solar": summary["pct_tasks_solar"],
        "load_shift_pct": summary["load_shift_pct"],
        "schedule": schedule.to_dict(orient="records"),
    }


if __name__ == "__main__":
    print("GreenIoT-MA - Optimiseur de decalage de charge\n")

    demo_mode = os.getenv("DEMO_MODE", "true").lower() in ("true", "1", "yes")
    solar_path = os.path.join(DATA_DIR, "gold_solar.parquet")

    if demo_mode:
        print(f"   [DEMO_MODE] Lecture locale depuis : {solar_path}")
        if not os.path.exists(solar_path):
            print("   Fichier local introuvable.")
            raise SystemExit(1)
        solar_df = pd.read_parquet(solar_path)
        if "ts" in solar_df.columns and "timestamp" not in solar_df.columns:
            solar_df["timestamp"] = solar_df["ts"]
    else:
        print(f"   Lecture Gold Delta depuis : {GOLD_SOLAR_PATH}")
        try:
            if DeltaTable is None:
                raise RuntimeError("deltalake is not installed")
            dt = DeltaTable(GOLD_SOLAR_PATH, storage_options=STORAGE_OPTIONS)
            solar_df = dt.to_pandas()
            if "ts" in solar_df.columns and "timestamp" not in solar_df.columns:
                solar_df["timestamp"] = solar_df["ts"]
        except Exception as exc:
            print(f"   Erreur Delta ({exc}).")
            if not os.path.exists(solar_path):
                raise SystemExit(1)
            solar_df = pd.read_parquet(solar_path)
            if "ts" in solar_df.columns and "timestamp" not in solar_df.columns:
                solar_df["timestamp"] = solar_df["ts"]

    tasks = [
        {"name": "Backup Delta Lake", "duration_min": 45, "priority": "medium", "power_kw_required": 40},
        {"name": "Entrainement ML", "duration_min": 120, "priority": "high", "power_kw_required": 70},
        {"name": "Export rapports BI", "duration_min": 20, "priority": "medium", "power_kw_required": 25},
        {"name": "Compression Bronze", "duration_min": 30, "priority": "low", "power_kw_required": 30},
        {"name": "Synchronisation MinIO", "duration_min": 15, "priority": "medium", "power_kw_required": 35},
    ]

    result = optimize_schedule(tasks, solar_df)
    peaks = result["peaks"]
    schedule = result["schedule"]
    summary = result["summary"]

    print("Fenetre solaire optimale :")
    print(f"   Debut    : {peaks['optimal_start_ts']}")
    print(f"   Fin      : {peaks['optimal_end_ts']}")
    print(f"   Puissance moyenne attendue: {peaks['expected_kw']} kW\n")

    print("Planning de decalage de charge :")
    if schedule.empty:
        print("   Aucun planning genere.")
    else:
        print(schedule.to_string(index=False))

    print(f"\nCO2 total economise : {summary['total_co2_saved_kg']:.2f} kg/jour")

    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "load_schedule.csv")
    schedule.to_csv(output_path, index=False)
    print(f"Planning exporte : {output_path}")
