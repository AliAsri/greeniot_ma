"""
GreenIoT-MA - Bronze to Silver Transformation
=============================================
Nettoyage et enrichissement des donnees brutes :
1. Cast timestamp et suppression des nulls
2. Deduplication par (sensor_id, timestamp)
3. Filtrage des valeurs aberrantes
4. Features rolling
5. Delta de consommation / production
6. Flags d'anomalie simples
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pyspark.sql.functions import abs as spark_abs
from pyspark.sql.functions import avg, col, isnan, lag, stddev, to_timestamp, when
from pyspark.sql.window import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_utils import get_spark

load_dotenv()

BRONZE_SERVERS = os.getenv("DELTA_BRONZE", "s3a://greeniot/bronze") + "/servers"
SILVER_SERVERS = os.getenv("DELTA_SILVER", "s3a://greeniot/silver") + "/servers"
BRONZE_SOLAR = os.getenv("DELTA_BRONZE", "s3a://greeniot/bronze") + "/solar"
SILVER_SOLAR = os.getenv("DELTA_SILVER", "s3a://greeniot/silver") + "/solar"


def transform_servers(spark):
    print("Transformation Servers...")
    df_srv = spark.read.format("delta").load(BRONZE_SERVERS)
    print(f"  Bronze lu : {df_srv.count()} lignes")

    df_srv = (
        df_srv.withColumn("ts", to_timestamp("timestamp"))
        .dropDuplicates(["sensor_id", "timestamp"])
        .filter(col("power_kw").isNotNull() & ~isnan("power_kw"))
        .filter((col("power_kw") > 0) & (col("power_kw") < 200))
    )

    w5_srv = Window.partitionBy("sensor_id").orderBy("ts").rowsBetween(-5, 0)
    df_srv = (
        df_srv.withColumn("power_kw_avg5", avg("power_kw").over(w5_srv))
        .withColumn("power_kw_std5", stddev("power_kw").over(w5_srv))
        .withColumn("cpu_avg5", avg("cpu_pct").over(w5_srv))
        .withColumn("temp_avg5", avg("temp_c").over(w5_srv))
    )

    w_lag_srv = Window.partitionBy("sensor_id").orderBy("ts")
    df_srv = df_srv.withColumn("power_delta", col("power_kw") - lag("power_kw", 1).over(w_lag_srv))

    std_guard = when(col("power_kw_std5").isNull() | (col("power_kw_std5") < 2.5), 2.5).otherwise(col("power_kw_std5"))
    thermal_alert = (col("temp_c") >= 60) | (col("temp_c") > col("temp_avg5") + 6)
    cpu_alert = col("cpu_pct") >= 92
    unstable_power = (spark_abs(col("power_delta")) > 2.4 * std_guard) & (spark_abs(col("power_delta")) > 8)
    sustained_drift = (
        (col("power_kw") > col("power_kw_avg5") * 1.20)
        & (col("cpu_pct") > col("cpu_avg5") * 1.10)
        & (col("temp_c") > col("temp_avg5") + 3)
    )

    df_srv = df_srv.withColumn(
        "anomaly_flag",
        when(thermal_alert | cpu_alert | unstable_power | sustained_drift, 1).otherwise(0),
    )

    df_srv.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(SILVER_SERVERS)

    count_srv = df_srv.count()
    anomalies_srv = df_srv.filter(col("anomaly_flag") == 1).count()
    print(f"  Silver Servers ecrit : {count_srv} lignes ({anomalies_srv} anomalies)")


def transform_solar(spark):
    print("Transformation Solar...")
    df_sol = spark.read.format("delta").load(BRONZE_SOLAR)
    print(f"  Bronze lu : {df_sol.count()} lignes")

    df_sol = (
        df_sol.withColumn("ts", to_timestamp("timestamp"))
        .dropDuplicates(["sensor_id", "timestamp"])
        .filter(col("production_kw").isNotNull() & ~isnan("production_kw"))
        .filter((col("production_kw") >= 0) & (col("production_kw") < 1000))
    )

    w5_sol = Window.partitionBy("sensor_id").orderBy("ts").rowsBetween(-5, 0)
    df_sol = (
        df_sol.withColumn("production_avg5", avg("production_kw").over(w5_sol))
        .withColumn("production_std5", stddev("production_kw").over(w5_sol))
    )

    w_lag_sol = Window.partitionBy("sensor_id").orderBy("ts")
    df_sol = df_sol.withColumn(
        "production_delta",
        col("production_kw") - lag("production_kw", 1).over(w_lag_sol),
    )

    df_sol = df_sol.withColumn(
        "anomaly_solar",
        when(spark_abs(col("production_delta")) > 3 * col("production_std5"), 1).otherwise(0),
    )

    df_sol.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(SILVER_SOLAR)

    count_sol = df_sol.count()
    anomalies_sol = df_sol.filter(col("anomaly_solar") == 1).count()
    print(f"  Silver Solar ecrit : {count_sol} lignes ({anomalies_sol} anomalies)")


def main():
    spark = get_spark("GreenIoT-Bronze-to-Silver")
    print("GreenIoT-MA - Transformation Bronze -> Silver")
    print(f"  [Servers] {BRONZE_SERVERS} -> {SILVER_SERVERS}")
    print(f"  [Solar]   {BRONZE_SOLAR} -> {SILVER_SOLAR}\n")

    try:
        transform_servers(spark)
        print()
        transform_solar(spark)
        print(f"\n  Silver Servers : {SILVER_SERVERS}")
        print(f"  Silver Solar   : {SILVER_SOLAR}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
