"""
GreenIoT-MA - Silver to Gold Transformation
===========================================
Feature engineering avance pour preparer les donnees ML :
1. Encodage temporel cyclique
2. Flags binaires
3. Lag features
4. Rolling windows etendues
5. Suppression des lignes inexploitables
"""

import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pyspark.sql.functions import avg, col, cos, dayofweek, hour, lag, lit, max as spark_max, sin, stddev
from pyspark.sql.window import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_utils import get_spark

load_dotenv()

SILVER_SERVERS = os.getenv("DELTA_SILVER", "s3a://greeniot/silver") + "/servers"
GOLD_SERVERS = os.getenv("DELTA_GOLD", "s3a://greeniot/gold") + "/servers"
SILVER_SOLAR = os.getenv("DELTA_SILVER", "s3a://greeniot/silver") + "/solar"
GOLD_SOLAR = os.getenv("DELTA_GOLD", "s3a://greeniot/gold") + "/solar"

PI2 = 2 * math.pi


def transform_servers(spark):
    print("Transformation Servers...")
    df_srv = spark.read.format("delta").load(SILVER_SERVERS)
    print(f"  Silver lu : {df_srv.count()} lignes")

    df_srv = (
        df_srv.withColumn("hour", hour("ts"))
        .withColumn("hour_sin", sin(col("hour").cast("double") * lit(PI2 / 24)))
        .withColumn("hour_cos", cos(col("hour").cast("double") * lit(PI2 / 24)))
        .withColumn("day_of_week", dayofweek("ts"))
        .withColumn("day_sin", sin(col("day_of_week").cast("double") * lit(PI2 / 7)))
        .withColumn("day_cos", cos(col("day_of_week").cast("double") * lit(PI2 / 7)))
        .withColumn("is_weekend", (col("day_of_week").isin([1, 7])).cast("integer"))
        .withColumn("is_business_hours", ((col("hour") >= 8) & (col("hour") <= 18)).cast("integer"))
    )

    w_srv = Window.partitionBy("sensor_id").orderBy("ts")
    for n_lag in [1, 3, 6, 12]:
        df_srv = df_srv.withColumn(f"power_kw_lag{n_lag}", lag("power_kw", n_lag).over(w_srv))

    w12_srv = Window.partitionBy("sensor_id").orderBy("ts").rowsBetween(-12, 0)
    w24_srv = Window.partitionBy("sensor_id").orderBy("ts").rowsBetween(-24, 0)
    df_srv = (
        df_srv.withColumn("power_kw_avg12", avg("power_kw").over(w12_srv))
        .withColumn("power_kw_std12", stddev("power_kw").over(w12_srv))
        .withColumn("power_kw_max12", spark_max("power_kw").over(w12_srv))
        .withColumn("power_kw_avg24", avg("power_kw").over(w24_srv))
        .dropna()
    )

    df_srv.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(GOLD_SERVERS)

    count_srv = df_srv.count()
    feats_srv = len(df_srv.columns)
    print(f"  Gold Servers ecrit : {count_srv} lignes, {feats_srv} features")


def transform_solar(spark):
    print("Transformation Solar...")
    df_sol = spark.read.format("delta").load(SILVER_SOLAR)
    print(f"  Silver lu : {df_sol.count()} lignes")

    df_sol = (
        df_sol.withColumn("hour", hour("ts"))
        .withColumn("hour_sin", sin(col("hour").cast("double") * lit(PI2 / 24)))
        .withColumn("hour_cos", cos(col("hour").cast("double") * lit(PI2 / 24)))
        .withColumn("day_of_week", dayofweek("ts"))
        .withColumn("day_sin", sin(col("day_of_week").cast("double") * lit(PI2 / 7)))
        .withColumn("day_cos", cos(col("day_of_week").cast("double") * lit(PI2 / 7)))
    )

    w_sol = Window.partitionBy("sensor_id").orderBy("ts")
    for n_lag in [1, 3, 6]:
        df_sol = df_sol.withColumn(f"production_lag{n_lag}", lag("production_kw", n_lag).over(w_sol))

    df_sol = df_sol.dropna()

    df_sol.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(GOLD_SOLAR)

    count_sol = df_sol.count()
    feats_sol = len(df_sol.columns)
    print(f"  Gold Solar ecrit : {count_sol} lignes, {feats_sol} features")


def main():
    spark = get_spark("GreenIoT-Silver-to-Gold")
    print("GreenIoT-MA - Transformation Silver -> Gold")
    print(f"  [Servers] {SILVER_SERVERS} -> {GOLD_SERVERS}")
    print(f"  [Solar]   {SILVER_SOLAR} -> {GOLD_SOLAR}\n")

    try:
        transform_servers(spark)
        print()
        transform_solar(spark)
        print(f"\n  Gold Servers : {GOLD_SERVERS}")
        print(f"  Gold Solar   : {GOLD_SOLAR}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
