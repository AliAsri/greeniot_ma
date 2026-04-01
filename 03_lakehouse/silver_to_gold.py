"""
GreenIoT-MA — Silver to Gold Transformation
===============================================
Feature engineering avancé pour préparer les données ML :
1. Encodage temporel cyclique (hour sin/cos, day sin/cos)
2. Flags binaires (weekend, heures de bureau)
3. Lag features (1, 3, 6, 12 pas)
4. Rolling windows étendues
5. Fenêtres temporelles pour séquences LSTM
"""

from pyspark.sql.functions import (
    col, hour, dayofweek, sin, cos, lit, lag, avg, stddev, max as spark_max
)
from pyspark.sql.window import Window
import math
import os
from dotenv import load_dotenv
from spark_utils import get_spark

load_dotenv()

SILVER = os.getenv("DELTA_SILVER", "s3a://greeniot/silver") + "/servers"
GOLD   = os.getenv("DELTA_GOLD",   "s3a://greeniot/gold")   + "/servers"

# ── Initialisation Spark (config Windows dans spark_utils) ────
spark = get_spark("GreenIoT-Silver-to-Gold")

print("🌿 GreenIoT-MA — Transformation Silver → Gold")
print(f"   Source : {SILVER}")
print(f"   Cible  : {GOLD}\n")

# ── Lecture Silver ────────────────────────────────────────────
df = spark.read.format("delta").load(SILVER)
print(f"   📥 Silver lu : {df.count()} lignes")

# ── 1. Features temporelles cycliques ─────────────────────────
PI2 = 2 * math.pi
df = df.withColumn("hour", hour("ts")) \
    .withColumn("hour_sin", sin(col("hour").cast("double") * lit(PI2 / 24))) \
    .withColumn("hour_cos", cos(col("hour").cast("double") * lit(PI2 / 24))) \
    .withColumn("day_of_week", dayofweek("ts")) \
    .withColumn("day_sin", sin(col("day_of_week").cast("double") * lit(PI2 / 7))) \
    .withColumn("day_cos", cos(col("day_of_week").cast("double") * lit(PI2 / 7)))

# ── 2. Flags binaires ────────────────────────────────────────
df = df.withColumn(
    "is_weekend",
    (col("day_of_week").isin([1, 7])).cast("integer")
).withColumn(
    "is_business_hours",
    ((col("hour") >= 8) & (col("hour") <= 18)).cast("integer")
)

# ── 3. Lag features ──────────────────────────────────────────
w = Window.partitionBy("sensor_id").orderBy("ts")
for n_lag in [1, 3, 6, 12]:
    df = df.withColumn(f"power_kw_lag{n_lag}", lag("power_kw", n_lag).over(w))

# ── 4. Rolling windows étendues ──────────────────────────────
w12 = Window.partitionBy("sensor_id").orderBy("ts").rowsBetween(-12, 0)
w24 = Window.partitionBy("sensor_id").orderBy("ts").rowsBetween(-24, 0)

df = df.withColumn("power_kw_avg12", avg("power_kw").over(w12)) \
    .withColumn("power_kw_std12", stddev("power_kw").over(w12)) \
    .withColumn("power_kw_max12", spark_max("power_kw").over(w12)) \
    .withColumn("power_kw_avg24", avg("power_kw").over(w24))

# ── 5. Suppression des lignes avec NaN (dues aux lags) ───────
df = df.dropna()

# ── Écriture Gold (ACID) ─────────────────────────────────────
df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(GOLD)

count      = df.count()
n_features = len(df.columns)

print(f"\n   ✅ Gold écrit : {count} lignes")
print(f"   📊 Nombre de features : {n_features}")
print(f"   📋 Colonnes : {', '.join(df.columns)}")
print(f"   📂 Cible : {GOLD}")

spark.stop()
