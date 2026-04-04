"""
GreenIoT-MA — Schémas Delta Lake
==================================
Définition des schémas pour les trois couches Medallion :
- Bronze (Raw) : Données IoT brutes, aucune transformation
- Silver (Clean) : Données nettoyées, déduplication, features rolling
- Gold (ML-Ready) : Features encodées, fenêtres temporelles, prêt pour ML
"""

from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    TimestampType, IntegerType, LongType
)


# ══════════════════════════════════════════════════════════════
# BRONZE — Données brutes
# ══════════════════════════════════════════════════════════════

BRONZE_SERVER_SCHEMA = StructType([
    StructField("sensor_id", StringType(), nullable=False),
    StructField("type", StringType(), nullable=False),
    StructField("timestamp", StringType(), nullable=False),
    StructField("cpu_pct", DoubleType()),
    StructField("ram_pct", DoubleType()),
    StructField("power_kw", DoubleType()),
    StructField("temp_c", DoubleType()),
    StructField("ingestion_ts", TimestampType()),
])

BRONZE_SOLAR_SCHEMA = StructType([
    StructField("sensor_id", StringType(), nullable=False),
    StructField("type", StringType(), nullable=False),
    StructField("timestamp", StringType(), nullable=False),
    StructField("production_kw", DoubleType()),
    StructField("irradiance_wm2", DoubleType()),
    StructField("panel_temp_c", DoubleType()),
    StructField("ingestion_ts", TimestampType()),
])

BRONZE_COOLING_SCHEMA = StructType([
    StructField("sensor_id", StringType(), nullable=False),
    StructField("type", StringType(), nullable=False),
    StructField("timestamp", StringType(), nullable=False),
    StructField("inlet_temp_c", DoubleType()),
    StructField("outlet_temp_c", DoubleType()),
    StructField("it_load_kw", DoubleType()),
    StructField("total_power_kw", DoubleType()),
    StructField("pue", DoubleType()),
    StructField("ingestion_ts", TimestampType()),
])

BRONZE_BATTERY_SCHEMA = StructType([
    StructField("sensor_id", StringType(), nullable=False),
    StructField("type", StringType(), nullable=False),
    StructField("timestamp", StringType(), nullable=False),
    StructField("soc_pct", DoubleType()),        # State of Charge (%)
    StructField("charge_rate_kw", DoubleType()),  # >0 charge, <0 décharge
    StructField("voltage_v", DoubleType()),
    StructField("temp_c", DoubleType()),
    StructField("ingestion_ts", TimestampType()),
])


# ══════════════════════════════════════════════════════════════
# SILVER — Données nettoyées + features de base
# ══════════════════════════════════════════════════════════════

SILVER_SERVER_SCHEMA = StructType([
    StructField("sensor_id", StringType(), nullable=False),
    StructField("type", StringType(), nullable=False),
    StructField("timestamp", StringType(), nullable=False),
    StructField("ts", TimestampType(), nullable=False),
    StructField("cpu_pct", DoubleType()),
    StructField("ram_pct", DoubleType()),
    StructField("power_kw", DoubleType()),
    StructField("temp_c", DoubleType()),
    # Features rolling (fenêtre 5 mesures)
    StructField("power_kw_avg5", DoubleType()),
    StructField("power_kw_std5", DoubleType()),
    StructField("cpu_avg5", DoubleType()),
    # Delta consommation
    StructField("power_delta", DoubleType()),
    # Flag anomalie
    StructField("anomaly_flag", IntegerType()),
    StructField("ingestion_ts", TimestampType()),
])

SILVER_SOLAR_SCHEMA = StructType([
    StructField("sensor_id", StringType(), nullable=False),
    StructField("type", StringType(), nullable=False),
    StructField("timestamp", StringType(), nullable=False),
    StructField("ts", TimestampType(), nullable=False),
    StructField("production_kw", DoubleType()),
    StructField("irradiance_wm2", DoubleType()),
    StructField("panel_temp_c", DoubleType()),
    # Features rolling (fenêtre 5 mesures)
    StructField("production_avg5", DoubleType()),
    StructField("production_std5", DoubleType()),
    StructField("production_delta", DoubleType()),
    # Flag anomalie solaire (chute soudaine d'irradiance)
    StructField("anomaly_solar", IntegerType()),
    StructField("ingestion_ts", TimestampType()),
])


# ══════════════════════════════════════════════════════════════
# GOLD — ML-Ready
# ══════════════════════════════════════════════════════════════

GOLD_SERVER_SCHEMA = StructType([
    StructField("sensor_id", StringType(), nullable=False),
    StructField("type", StringType(), nullable=False),
    StructField("timestamp", StringType(), nullable=False),
    StructField("ts", TimestampType(), nullable=False),
    StructField("cpu_pct", DoubleType()),
    StructField("ram_pct", DoubleType()),
    StructField("power_kw", DoubleType()),
    StructField("temp_c", DoubleType()),
    StructField("power_kw_avg5", DoubleType()),
    StructField("power_kw_std5", DoubleType()),
    StructField("cpu_avg5", DoubleType()),
    StructField("power_delta", DoubleType()),
    StructField("anomaly_flag", IntegerType()),
    # Features temporelles
    StructField("hour", IntegerType()),
    StructField("hour_sin", DoubleType()),
    StructField("hour_cos", DoubleType()),
    StructField("day_of_week", IntegerType()),
    StructField("day_sin", DoubleType()),
    StructField("day_cos", DoubleType()),
    StructField("is_weekend", IntegerType()),
    StructField("is_business_hours", IntegerType()),
    # Lag features
    StructField("power_kw_lag1", DoubleType()),
    StructField("power_kw_lag3", DoubleType()),
    StructField("power_kw_lag6", DoubleType()),
    StructField("power_kw_lag12", DoubleType()),
])


GOLD_SOLAR_SCHEMA = StructType([
    StructField("sensor_id", StringType(), nullable=False),
    StructField("type", StringType(), nullable=False),
    StructField("timestamp", StringType(), nullable=False),
    StructField("ts", TimestampType(), nullable=False),
    StructField("production_kw", DoubleType()),
    StructField("irradiance_wm2", DoubleType()),
    StructField("panel_temp_c", DoubleType()),
    StructField("production_avg5", DoubleType()),
    StructField("production_std5", DoubleType()),
    StructField("production_delta", DoubleType()),
    StructField("anomaly_solar", IntegerType()),
    # Features temporelles cycliques
    StructField("hour", IntegerType()),
    StructField("hour_sin", DoubleType()),
    StructField("hour_cos", DoubleType()),
    StructField("day_of_week", IntegerType()),
    StructField("day_sin", DoubleType()),
    StructField("day_cos", DoubleType()),
    # Lag features solaires
    StructField("production_lag1", DoubleType()),
    StructField("production_lag3", DoubleType()),
    StructField("production_lag6", DoubleType()),
])


# ── Mapping des schémas par couche et type ────────────────────
SCHEMAS = {
    "bronze": {
        "server":  BRONZE_SERVER_SCHEMA,
        "solar":   BRONZE_SOLAR_SCHEMA,
        "cooling": BRONZE_COOLING_SCHEMA,
        "battery": BRONZE_BATTERY_SCHEMA,
    },
    "silver": {
        "server": SILVER_SERVER_SCHEMA,
        "solar":  SILVER_SOLAR_SCHEMA,
    },
    "gold": {
        "server": GOLD_SERVER_SCHEMA,
        "solar":  GOLD_SOLAR_SCHEMA,
    },
}


def get_schema(layer: str, sensor_type: str) -> StructType:
    """Récupère le schéma Spark pour une couche et un type donné."""
    if layer not in SCHEMAS:
        raise ValueError(f"Couche inconnue: {layer}. Valeurs possibles: {list(SCHEMAS.keys())}")
    if sensor_type not in SCHEMAS[layer]:
        raise ValueError(
            f"Type inconnu '{sensor_type}' pour la couche '{layer}'. "
            f"Valeurs possibles: {list(SCHEMAS[layer].keys())}"
        )
    return SCHEMAS[layer][sensor_type]
