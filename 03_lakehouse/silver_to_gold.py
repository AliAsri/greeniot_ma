"""
GreenIoT-MA — Silver to Gold Transformation
===============================================
Feature engineering avancé pour préparer les données ML :
1. Encodage temporel cyclique (hour sin/cos, day sin/cos)
2. Flags binaires (weekend, heures de bureau)
3. Lag features (1, 3, 6, 12 pas)
4. Rolling windows étendues
5. Fenêtres temporelles pour séquences LSTM

Applicable aux streams : servers + solar
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

SILVER_SERVERS = os.getenv("DELTA_SILVER", "s3a://greeniot/silver") + "/servers"
GOLD_SERVERS   = os.getenv("DELTA_GOLD",   "s3a://greeniot/gold")   + "/servers"
SILVER_SOLAR   = os.getenv("DELTA_SILVER", "s3a://greeniot/silver") + "/solar"
GOLD_SOLAR     = os.getenv("DELTA_GOLD",   "s3a://greeniot/gold")   + "/solar"

# ── Initialisation Spark (config Windows dans spark_utils) ────
spark = get_spark("GreenIoT-Silver-to-Gold")

print("🌿 GreenIoT-MA — Transformation Silver → Gold")
print(f"   [Servers] {SILVER_SERVERS} → {GOLD_SERVERS}")
print(f"   [Solar]   {SILVER_SOLAR} → {GOLD_SOLAR}\n")

PI2 = 2 * math.pi

# ════════════════════════════════════════════════════════════════
# SERVERS : Silver → Gold
# ════════════════════════════════════════════════════════════════
print("🖥️  Transformation Servers...")
df_srv = spark.read.format("delta").load(SILVER_SERVERS)
print(f"   📥 Silver lu : {df_srv.count()} lignes")

# 1. Features temporelles cycliques
df_srv = df_srv.withColumn("hour", hour("ts")) \
    .withColumn("hour_sin", sin(col("hour").cast("double") * lit(PI2 / 24))) \
    .withColumn("hour_cos", cos(col("hour").cast("double") * lit(PI2 / 24))) \
    .withColumn("day_of_week", dayofweek("ts")) \
    .withColumn("day_sin", sin(col("day_of_week").cast("double") * lit(PI2 / 7))) \
    .withColumn("day_cos", cos(col("day_of_week").cast("double") * lit(PI2 / 7)))

# 2. Flags binaires
df_srv = df_srv.withColumn(
    "is_weekend",
    (col("day_of_week").isin([1, 7])).cast("integer")
).withColumn(
    "is_business_hours",
    ((col("hour") >= 8) & (col("hour") <= 18)).cast("integer")
)

# 3. Lag features
w_srv = Window.partitionBy("sensor_id").orderBy("ts")
for n_lag in [1, 3, 6, 12]:
    df_srv = df_srv.withColumn(f"power_kw_lag{n_lag}", lag("power_kw", n_lag).over(w_srv))

# 4. Rolling windows étendues
w12_srv = Window.partitionBy("sensor_id").orderBy("ts").rowsBetween(-12, 0)
w24_srv = Window.partitionBy("sensor_id").orderBy("ts").rowsBetween(-24, 0)
df_srv = df_srv.withColumn("power_kw_avg12", avg("power_kw").over(w12_srv)) \
    .withColumn("power_kw_std12", stddev("power_kw").over(w12_srv)) \
    .withColumn("power_kw_max12", spark_max("power_kw").over(w12_srv)) \
    .withColumn("power_kw_avg24", avg("power_kw").over(w24_srv))

# 5. Suppression des lignes avec NaN (dues aux lags)
df_srv = df_srv.dropna()

# Écriture Gold Servers
df_srv.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(GOLD_SERVERS)

count_srv  = df_srv.count()
feats_srv  = len(df_srv.columns)
print(f"   ✅ Gold Servers écrit : {count_srv} lignes, {feats_srv} features")

# ════════════════════════════════════════════════════════════════
# SOLAR : Silver → Gold
# ════════════════════════════════════════════════════════════════
print("\n☀️  Transformation Solar...")
df_sol = spark.read.format("delta").load(SILVER_SOLAR)
print(f"   📥 Silver lu : {df_sol.count()} lignes")

# 1. Features temporelles cycliques
df_sol = df_sol.withColumn("hour", hour("ts")) \
    .withColumn("hour_sin", sin(col("hour").cast("double") * lit(PI2 / 24))) \
    .withColumn("hour_cos", cos(col("hour").cast("double") * lit(PI2 / 24))) \
    .withColumn("day_of_week", dayofweek("ts")) \
    .withColumn("day_sin", sin(col("day_of_week").cast("double") * lit(PI2 / 7))) \
    .withColumn("day_cos", cos(col("day_of_week").cast("double") * lit(PI2 / 7)))

# 2. Lag features solaires (production précédente)
w_sol = Window.partitionBy("sensor_id").orderBy("ts")
for n_lag in [1, 3, 6]:
    df_sol = df_sol.withColumn(f"production_lag{n_lag}", lag("production_kw", n_lag).over(w_sol))

# 3. Suppression des lignes avec NaN (dues aux lags)
df_sol = df_sol.dropna()

# Écriture Gold Solar
df_sol.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(GOLD_SOLAR)

count_sol = df_sol.count()
feats_sol = len(df_sol.columns)
print(f"   ✅ Gold Solar écrit  : {count_sol} lignes, {feats_sol} features")

print(f"\n   📂 Gold Servers : {GOLD_SERVERS}")
print(f"   📂 Gold Solar   : {GOLD_SOLAR}")

spark.stop()
