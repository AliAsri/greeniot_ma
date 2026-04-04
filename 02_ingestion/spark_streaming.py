"""
GreenIoT-MA — PySpark Structured Streaming
=============================================
Consumer PySpark : Kafka → Bronze Delta Lake
Lit les topics Kafka et écrit en format Delta avec checkpoints.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType
)
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

KAFKA_SERVERS = os.getenv("KAFKA_BROKERS", "localhost:9092")
DELTA_PATH = os.getenv("DELTA_BRONZE", "s3a://greeniot/bronze")

# ── Initialisation Spark ──────────────────────────────────────
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("GreenIoT-Bronze-Ingestion") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.ui.port", "0") \
    .config("spark.jars.packages",
            "io.delta:delta-spark_2.12:3.0.0,"
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "org.apache.hadoop:hadoop-aws:3.3.4") \
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "greeniot") \
    .config("spark.hadoop.fs.s3a.secret.key", "greeniot2030") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

# ── Schémas des capteurs ──────────────────────────────────────
schema_server = StructType([
    StructField("sensor_id", StringType()),
    StructField("type", StringType()),
    StructField("timestamp", StringType()),
    StructField("cpu_pct", DoubleType()),
    StructField("ram_pct", DoubleType()),
    StructField("power_kw", DoubleType()),
    StructField("temp_c", DoubleType()),
])

schema_solar = StructType([
    StructField("sensor_id", StringType()),
    StructField("type", StringType()),
    StructField("timestamp", StringType()),
    StructField("production_kw", DoubleType()),
    StructField("irradiance_wm2", DoubleType()),
    StructField("panel_temp_c", DoubleType()),
])

schema_cooling = StructType([
    StructField("sensor_id", StringType()),
    StructField("type", StringType()),
    StructField("timestamp", StringType()),
    StructField("inlet_temp_c", DoubleType()),
    StructField("outlet_temp_c", DoubleType()),
    StructField("it_load_kw", DoubleType()),
    StructField("total_power_kw", DoubleType()),
    StructField("pue", DoubleType()),
])

schema_battery = StructType([
    StructField("sensor_id", StringType()),
    StructField("type", StringType()),
    StructField("timestamp", StringType()),
    StructField("soc_pct", DoubleType()),
    StructField("charge_rate_kw", DoubleType()),
    StructField("voltage_v", DoubleType()),
    StructField("temp_c", DoubleType()),
])


def write_bronze(df, topic: str, schema: StructType, path_suffix: str):
    """Parse les messages Kafka et écrit en Delta Lake (Bronze)."""
    parsed = df \
        .filter(col("topic") == topic) \
        .select(from_json(col("value").cast("string"), schema).alias("d")) \
        .select("d.*") \
        .withColumn("ingestion_ts", current_timestamp())

    chk_base = os.path.join(tempfile.gettempdir(), "greeniot_chk")

    return parsed.writeStream \
        .format("delta") \
        .outputMode("append") \
        .option("checkpointLocation", os.path.join(chk_base, path_suffix)) \
        .start(f"{DELTA_PATH}/{path_suffix}")


# ── Lecture streaming depuis Kafka ────────────────────────────
print("🌿 GreenIoT-MA — Spark Structured Streaming démarré")
print(f"   Kafka: {KAFKA_SERVERS}")
print(f"   Delta: {DELTA_PATH}\n")

raw = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_SERVERS) \
    .option("subscribe", "greeniot.servers,greeniot.solar,greeniot.cooling,greeniot.battery") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

# ── Écriture Bronze par type de capteur ───────────────────────
q_servers = write_bronze(raw, "greeniot.servers", schema_server, "servers")
q_solar   = write_bronze(raw, "greeniot.solar",   schema_solar,  "solar")
q_cooling = write_bronze(raw, "greeniot.cooling", schema_cooling, "cooling")
q_battery = write_bronze(raw, "greeniot.battery", schema_battery, "battery")

print("   ✅ Queries streaming démarrées :")
print("      → greeniot.servers → Bronze/servers")
print("      → greeniot.solar   → Bronze/solar")
print("      → greeniot.cooling → Bronze/cooling")
print("      → greeniot.battery → Bronze/battery")
print("\n   Ctrl+C pour stopper.\n")

spark.streams.awaitAnyTermination()
