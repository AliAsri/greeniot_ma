"""
IoT Energy Analytics — Data Loader Utilities
======================================
Chargement des données Delta Lake / Parquet pour le dashboard.
Optimisé pour la performance : filtrage temporel côté stockage,
sélection de colonnes, et cache TTL adapté.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from deltalake import DeltaTable

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data"
)

# Colonnes nécessaires pour le dashboard (évite de charger tout)
_BRONZE_COLUMNS = [
    "timestamp", "sensor_id", "power_kw", "cpu_pct", "ram_pct", "temp_c", "pue"
]

def get_storage_options():
    return {
        "AWS_ACCESS_KEY_ID": "greeniot",
        "AWS_SECRET_ACCESS_KEY": "greeniot2030",
        "AWS_ENDPOINT_URL": "http://localhost:9000",
        "AWS_REGION": "us-east-1",
        "AWS_ALLOW_HTTP": "true",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
        "AWS_S3_FORCE_PATH_STYLE": "true", # Obligatoire pour MinIO en local
    }

def _get_s3_path(env_var, default_path, suffix):
    path = os.getenv(env_var, default_path) + suffix
    # deltalake natively expects s3:// instead of s3a://
    return path.replace("s3a://", "s3://")

def _load_bronze_filtered(hours_back=2):
    """Charge uniquement les N dernières heures depuis Bronze avec filtrage PyArrow.
    
    Utilise le predicate pushdown de DeltaTable pour éviter de charger
    l'intégralité de la table Bronze dans la mémoire.
    """
    path = _get_s3_path("DELTA_BRONZE", "s3a://greeniot/bronze", "/servers")
    dt = DeltaTable(path, storage_options=get_storage_options())
    
    cutoff = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
    
    # Sélectionner uniquement les colonnes qui existent dans la table
    # Note: .to_pyarrow() removed in newer deltalake — use .fields instead
    available_cols = [f.name for f in dt.schema().fields]
    columns = [c for c in _BRONZE_COLUMNS if c in available_cols]
    
    # Filtrage côté stockage (predicate pushdown) — ne charge que les données récentes
    try:
        arrow_table = dt.to_pyarrow_table(
            columns=columns,
            filters=[("timestamp", ">=", cutoff)]
        )
    except Exception:
        # Fallback si le filtre ne fonctionne pas (schéma différent, etc.)
        arrow_table = dt.to_pyarrow_table(columns=columns)
    
    df = arrow_table.to_pandas()
    return df

def _enrich_bronze(df):
    """Enrichissement inline des données Bronze (rolling, anomalies, PUE)."""
    df["ts"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values(["sensor_id", "ts"])
    
    # Rolling moyen sur 5 points par capteur
    df["power_kw_avg5"] = df.groupby("sensor_id")["power_kw"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df["power_delta"] = df.groupby("sensor_id")["power_kw"].diff()
    std5 = df.groupby("sensor_id")["power_kw"].transform(
        lambda x: x.rolling(5, min_periods=1).std()
    )
    df["anomaly_flag"] = (df["power_delta"].abs() > 3 * std5).astype(int).fillna(0)
    
    # PUE estimé seulement si absent de la source
    if "pue" not in df.columns:
        base_pue = 1.2
        pwr_range = df["power_kw"].max() - df["power_kw"].min()
        power_factor = (df["power_kw"] - df["power_kw"].min()) / max(1, pwr_range) * 0.3
        temp_factor = (df["temp_c"] - 20).clip(0, 40) / 100 if "temp_c" in df.columns else 0.1
        df["pue"] = (base_pue + power_factor + temp_factor).round(3)
    
    return df

@st.cache_data(ttl=30)
def load_bronze_servers():
    """Charge les 2 dernières heures de données Bronze et enrichit à la volée.
    
    Optimisations vs la version précédente :
    - Filtrage temporel côté stockage (predicate pushdown) au lieu de to_pandas() complet
    - Sélection de colonnes pour réduire la mémoire
    - TTL cache de 60s au lieu de 10s
    """
    try:
        df = _load_bronze_filtered(hours_back=2)
        if df.empty:
            st.info("ℹ️ Aucune donnée Bronze récente (< 2h). Chargement des 1000 dernières lignes...")
            # Fallback : charger les dernières lignes si pas de données récentes
            dt = DeltaTable(
                _get_s3_path("DELTA_BRONZE", "s3a://greeniot/bronze", "/servers"),
                storage_options=get_storage_options()
            )
            available_cols = [f.name for f in dt.schema().fields]
            columns = [c for c in _BRONZE_COLUMNS if c in available_cols]
            df = dt.to_pyarrow_table(columns=columns).to_pandas().tail(1000)
        
        return _enrich_bronze(df)
    except Exception as e:
        import traceback
        st.error(f"Erreur MinIO/Delta : {str(e)}")
        path = os.path.join(DATA_DIR, "raw_servers.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            # Need to format timestamp properly for _enrich_bronze
            # We take the tail because Bronze can be huge
            df = df.tail(1000).copy()
            if "timestamp" in df.columns:
                df["timestamp"] = df["timestamp"].astype(str)
            return _enrich_bronze(df)
        
        return _generate_demo_server_data()

@st.cache_data(ttl=300)
def load_silver_servers():
    """Charge les données Silver des serveurs."""
    path = _get_s3_path("DELTA_SILVER", "s3a://greeniot/silver", "/servers")
    try:
        df = DeltaTable(path, storage_options=get_storage_options()).to_pandas()
        df["ts"] = pd.to_datetime(df["ts"]) if "ts" in df.columns else pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        path = os.path.join(DATA_DIR, "silver_servers_latest.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df["ts"] = pd.to_datetime(df["ts"]) if "ts" in df.columns else pd.to_datetime(df["timestamp"])
            return df
        return _generate_demo_server_data()

@st.cache_data(ttl=30)
def load_bronze_solar():
    """Charge les données Bronze solaires (24 dernières heures pour l'optimiseur)."""
    path = _get_s3_path("DELTA_BRONZE", "s3a://greeniot/bronze", "/solar")
    try:
        dt = DeltaTable(path, storage_options=get_storage_options())
        # L'optimiseur a besoin de voir la journée entière (24h) pour trouver le pic d'ensoleillement
        cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        
        try:
            # Filtrage côté stockage (predicate pushdown)
            arrow_table = dt.to_pyarrow_table(filters=[("timestamp", ">=", cutoff)])
        except Exception:
            arrow_table = dt.to_pyarrow_table()
            
        df = arrow_table.to_pandas()
        
        if df.empty:
            # Fallback si pas de données récentes
            df = dt.to_pyarrow_table().to_pandas().tail(1000)
            
        df["ts"] = pd.to_datetime(df["timestamp"], format="mixed")
        return df
    except Exception as e:
        path = os.path.join(DATA_DIR, "raw_solar.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df["ts"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
            return df.tail(1000)
        return _generate_demo_solar_data()

@st.cache_data(ttl=900)
def load_gold_servers():
    """Charge les données Gold des serveurs."""
    path = _get_s3_path("DELTA_GOLD", "s3a://greeniot/gold", "/servers")
    try:
        df = DeltaTable(path, storage_options=get_storage_options()).to_pandas()
        df["ts"] = pd.to_datetime(df["ts"]) if "ts" in df.columns else pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        path = os.path.join(DATA_DIR, "gold_servers.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df["ts"] = pd.to_datetime(df["ts"]) if "ts" in df.columns else pd.to_datetime(df["timestamp"])
            return df
        return _generate_demo_server_data()

@st.cache_data(ttl=30)
def load_gold_solar():
    """Charge les données Gold solaires."""
    path = _get_s3_path("DELTA_GOLD", "s3a://greeniot/gold", "/solar")
    try:
        df = DeltaTable(path, storage_options=get_storage_options()).to_pandas()
        df["ts"] = pd.to_datetime(df["ts"]) if "ts" in df.columns else pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        path = os.path.join(DATA_DIR, "gold_solar.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df["ts"] = pd.to_datetime(df["ts"]) if "ts" in df.columns else pd.to_datetime(df["timestamp"])
            return df
        return _generate_demo_solar_data()

@st.cache_data(ttl=300)
def load_anomalies():
    """Charge les anomalies détectées."""
    path = os.path.join(DATA_DIR, "anomalies_detected.csv")
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_schedule():
    """Charge le planning de décalage de charge."""
    path = os.path.join(DATA_DIR, "load_schedule.csv")
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _generate_demo_server_data(n: int = 500):
    """Fallback : données simulées pour la démo."""
    ts = pd.date_range(end=datetime.now(), periods=n, freq="30s")
    np.random.seed(42)

    return pd.DataFrame({
        "ts": ts,
        "sensor_id": np.random.choice(["rack_A1", "rack_A2", "rack_B1"], n),
        "power_kw": 40 + 20 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n) * 3,
        "cpu_pct": 50 + 20 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n) * 5,
        "ram_pct": 60 + 10 * np.sin(np.linspace(0, 3 * np.pi, n)) + np.random.randn(n) * 3,
        "temp_c": 45 + 10 * np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.randn(n) * 2,
        "anomaly_flag": np.random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], n),
        "pue": np.random.uniform(1.3, 1.8, n),
        "power_kw_avg5": 40 + 18 * np.sin(np.linspace(0, 4 * np.pi, n)),
        "power_delta": np.random.randn(n) * 2,
    })

def _generate_demo_solar_data(n: int = 500):
    """Fallback : données solaires simulées."""
    ts = pd.date_range(end=datetime.now(), periods=n, freq="5min")
    np.random.seed(42)
    hours = ts.hour + ts.minute / 60
    irradiance = np.maximum(0, np.sin(np.pi * (hours - 6) / 12))

    return pd.DataFrame({
        "ts": ts,
        "timestamp": ts.astype(str),
        "sensor_id": np.random.choice(["solar_panel_01", "solar_panel_02"], n),
        "production_kw": (500 * irradiance * np.random.uniform(0.85, 1.0, n)).round(2),
        "irradiance_wm2": (irradiance * 1000).round(1),
        "panel_temp_c": (25 + irradiance * 20 + np.random.randn(n) * 1.5).round(1),
    })
