"""
GreenIoT-MA — Bronze to Silver Transformation
================================================
Nettoyage et enrichissement des données brutes :
1. Cast timestamp et suppression nulls
2. Déduplication par (sensor_id, timestamp)
3. Filtrage des valeurs aberrantes
4. Features rolling (fenêtre 5 mesures)
5. Delta consommation (dérivée première)
6. Flag anomalie (z-score > 3)
"""

from pyspark.sql.functions import (
    col, to_timestamp, avg, stddev, lag, when, abs as spark_abs, isnan
)
from pyspark.sql.window import Window
import os
from dotenv import load_dotenv
from spark_utils import get_spark

load_dotenv()

BRONZE = os.getenv("DELTA_BRONZE", "s3a://greeniot/bronze") + "/servers"
SILVER = os.getenv("DELTA_SILVER", "s3a://greeniot/silver") + "/servers"

# ── Initialisation Spark (config Windows dans spark_utils) ────
spark = get_spark("GreenIoT-Bronze-to-Silver")

print("🌿 GreenIoT-MA — Transformation Bronze → Silver")
print(f"   Source : {BRONZE}")
print(f"   Cible  : {SILVER}\n")

# ── Lecture Bronze ────────────────────────────────────────────
df = spark.read.format("delta").load(BRONZE)
print(f"   📥 Bronze lu : {df.count()} lignes")

# ── 1. Cast timestamp et suppression nulls ────────────────────
df = df.withColumn("ts", to_timestamp("timestamp")) \
    .dropDuplicates(["sensor_id", "timestamp"]) \
    .filter(col("power_kw").isNotNull() & ~isnan("power_kw")) \
    .filter((col("power_kw") > 0) & (col("power_kw") < 200))

# ── 2. Features rolling (fenêtre 5 mesures) ──────────────────
w5 = Window.partitionBy("sensor_id").orderBy("ts").rowsBetween(-5, 0)
df = df.withColumn("power_kw_avg5", avg("power_kw").over(w5)) \
    .withColumn("power_kw_std5", stddev("power_kw").over(w5)) \
    .withColumn("cpu_avg5", avg("cpu_pct").over(w5))

# ── 3. Delta consommation (dérivée première) ─────────────────
w_lag = Window.partitionBy("sensor_id").orderBy("ts")
df = df.withColumn("power_delta", col("power_kw") - lag("power_kw", 1).over(w_lag))

# ── 4. Flag anomalie simple (z-score > 3) ────────────────────
df = df.withColumn(
    "anomaly_flag",
    when(spark_abs(col("power_delta")) > 3 * col("power_kw_std5"), 1).otherwise(0)
)

# ── Écriture Silver (ACID, merge-on-read optimisé) ────────────
df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(SILVER)

count = df.count()
anomalies = df.filter(col("anomaly_flag") == 1).count()

print(f"\n   ✅ Silver écrit : {count} lignes")
print(f"   ⚠️  Anomalies flaggées : {anomalies} ({anomalies/max(1,count)*100:.1f}%)")
print(f"   📂 Cible : {SILVER}")

spark.stop()
