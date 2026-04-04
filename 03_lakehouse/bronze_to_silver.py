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

Applicable aux streams : servers + solar
"""

from pyspark.sql.functions import (
    col, to_timestamp, avg, stddev, lag, when, abs as spark_abs, isnan
)
from pyspark.sql.window import Window
import os
from dotenv import load_dotenv
from spark_utils import get_spark

load_dotenv()

BRONZE_SERVERS = os.getenv("DELTA_BRONZE", "s3a://greeniot/bronze") + "/servers"
SILVER_SERVERS = os.getenv("DELTA_SILVER", "s3a://greeniot/silver") + "/servers"
BRONZE_SOLAR   = os.getenv("DELTA_BRONZE", "s3a://greeniot/bronze") + "/solar"
SILVER_SOLAR   = os.getenv("DELTA_SILVER", "s3a://greeniot/silver") + "/solar"

# ── Initialisation Spark (config Windows dans spark_utils) ────
spark = get_spark("GreenIoT-Bronze-to-Silver")

print("🌿 GreenIoT-MA — Transformation Bronze → Silver")
print(f"   [Servers] {BRONZE_SERVERS} → {SILVER_SERVERS}")
print(f"   [Solar]   {BRONZE_SOLAR} → {SILVER_SOLAR}\n")

# ════════════════════════════════════════════════════════════════
# SERVERS : Bronze → Silver
# ════════════════════════════════════════════════════════════════
print("🖥️  Transformation Servers...")
df_srv = spark.read.format("delta").load(BRONZE_SERVERS)
print(f"   📥 Bronze lu : {df_srv.count()} lignes")

# 1. Cast timestamp et suppression nulls
df_srv = df_srv.withColumn("ts", to_timestamp("timestamp")) \
    .dropDuplicates(["sensor_id", "timestamp"]) \
    .filter(col("power_kw").isNotNull() & ~isnan("power_kw")) \
    .filter((col("power_kw") > 0) & (col("power_kw") < 200))

# 2. Features rolling (fenêtre 5 mesures)
w5_srv = Window.partitionBy("sensor_id").orderBy("ts").rowsBetween(-5, 0)
df_srv = df_srv.withColumn("power_kw_avg5", avg("power_kw").over(w5_srv)) \
    .withColumn("power_kw_std5", stddev("power_kw").over(w5_srv)) \
    .withColumn("cpu_avg5", avg("cpu_pct").over(w5_srv))

# 3. Delta consommation (dérivée première)
w_lag_srv = Window.partitionBy("sensor_id").orderBy("ts")
df_srv = df_srv.withColumn("power_delta", col("power_kw") - lag("power_kw", 1).over(w_lag_srv))

# 4. Flag anomalie simple (z-score > 3)
df_srv = df_srv.withColumn(
    "anomaly_flag",
    when(spark_abs(col("power_delta")) > 3 * col("power_kw_std5"), 1).otherwise(0)
)

# Écriture Silver Servers
df_srv.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(SILVER_SERVERS)

count_srv = df_srv.count()
anomalies_srv = df_srv.filter(col("anomaly_flag") == 1).count()
print(f"   ✅ Silver Servers écrit : {count_srv} lignes ({anomalies_srv} anomalies)")

# ════════════════════════════════════════════════════════════════
# SOLAR : Bronze → Silver
# ════════════════════════════════════════════════════════════════
print("\n☀️  Transformation Solar...")
df_sol = spark.read.format("delta").load(BRONZE_SOLAR)
print(f"   📥 Bronze lu : {df_sol.count()} lignes")

# 1. Cast timestamp, suppression nulls et valeurs physiquement impossibles
df_sol = df_sol.withColumn("ts", to_timestamp("timestamp")) \
    .dropDuplicates(["sensor_id", "timestamp"]) \
    .filter(col("production_kw").isNotNull() & ~isnan("production_kw")) \
    .filter((col("production_kw") >= 0) & (col("production_kw") < 1000))

# 2. Features rolling (fenêtre 5 mesures)
w5_sol = Window.partitionBy("sensor_id").orderBy("ts").rowsBetween(-5, 0)
df_sol = df_sol.withColumn("production_avg5", avg("production_kw").over(w5_sol)) \
    .withColumn("production_std5", stddev("production_kw").over(w5_sol))

# 3. Delta production
w_lag_sol = Window.partitionBy("sensor_id").orderBy("ts")
df_sol = df_sol.withColumn(
    "production_delta",
    col("production_kw") - lag("production_kw", 1).over(w_lag_sol)
)

# 4. Flag anomalie solaire (chute soudaine d'irradiance : z-score > 3)
df_sol = df_sol.withColumn(
    "anomaly_solar",
    when(spark_abs(col("production_delta")) > 3 * col("production_std5"), 1).otherwise(0)
)

# Écriture Silver Solar
df_sol.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(SILVER_SOLAR)

count_sol  = df_sol.count()
anomalies_sol = df_sol.filter(col("anomaly_solar") == 1).count()
print(f"   ✅ Silver Solar écrit  : {count_sol} lignes ({anomalies_sol} anomalies)")

print(f"\n   📂 Silver Servers : {SILVER_SERVERS}")
print(f"   📂 Silver Solar   : {SILVER_SOLAR}")

spark.stop()
