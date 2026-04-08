"""
GreenIoT-MA - PySpark Structured Streaming
==========================================
Consumer PySpark : Kafka -> Bronze Delta Lake
Lit les topics Kafka et ecrit en format Delta avec checkpoints.
"""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, from_json
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

load_dotenv()

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

KAFKA_SERVERS = os.getenv("KAFKA_BROKERS", "localhost:9092")
DELTA_PATH = os.getenv("DELTA_BRONZE", "s3a://greeniot/bronze")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "greeniot")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "greeniot2030")
MINIO_PATH_STYLE = os.getenv("AWS_S3_FORCE_PATH_STYLE", "true")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "data" / "checkpoints" / "streaming"
STREAMING_CHECKPOINT_ROOT = Path(
    os.getenv("STREAMING_CHECKPOINT_ROOT") or DEFAULT_CHECKPOINT_ROOT
)


def _env_flag(name: str, default: str = "false") -> bool:
    """Interpret common truthy environment variable values."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


STREAMING_RESET_CHECKPOINTS = _env_flag("STREAMING_RESET_CHECKPOINTS")

TOPICS = {
    "greeniot.servers": "servers",
    "greeniot.solar": "solar",
    "greeniot.cooling": "cooling",
    "greeniot.battery": "battery",
}

schema_server = StructType(
    [
        StructField("sensor_id", StringType()),
        StructField("type", StringType()),
        StructField("timestamp", StringType()),
        StructField("cpu_pct", DoubleType()),
        StructField("ram_pct", DoubleType()),
        StructField("power_kw", DoubleType()),
        StructField("temp_c", DoubleType()),
    ]
)

schema_solar = StructType(
    [
        StructField("sensor_id", StringType()),
        StructField("type", StringType()),
        StructField("timestamp", StringType()),
        StructField("production_kw", DoubleType()),
        StructField("irradiance_wm2", DoubleType()),
        StructField("panel_temp_c", DoubleType()),
    ]
)

schema_cooling = StructType(
    [
        StructField("sensor_id", StringType()),
        StructField("type", StringType()),
        StructField("timestamp", StringType()),
        StructField("inlet_temp_c", DoubleType()),
        StructField("outlet_temp_c", DoubleType()),
        StructField("it_load_kw", DoubleType()),
        StructField("total_power_kw", DoubleType()),
        StructField("pue", DoubleType()),
    ]
)

schema_battery = StructType(
    [
        StructField("sensor_id", StringType()),
        StructField("type", StringType()),
        StructField("timestamp", StringType()),
        StructField("soc_pct", DoubleType()),
        StructField("charge_rate_kw", DoubleType()),
        StructField("voltage_v", DoubleType()),
        StructField("temp_c", DoubleType()),
    ]
)

TOPIC_SCHEMAS = {
    "greeniot.servers": schema_server,
    "greeniot.solar": schema_solar,
    "greeniot.cooling": schema_cooling,
    "greeniot.battery": schema_battery,
}


def build_spark() -> SparkSession:
    """Create a Spark session configured for Kafka + Delta + MinIO."""
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("GreenIoT-Bronze-Ingestion")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.port", "0")
        .config(
            "spark.jars.packages",
            "io.delta:delta-spark_2.12:3.0.0,"
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "org.apache.hadoop:hadoop-aws:3.3.4",
        )
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", MINIO_PATH_STYLE)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .getOrCreate()
    )


def prepare_checkpoints() -> None:
    """Prepare or reset streaming checkpoints before Spark starts."""
    if STREAMING_RESET_CHECKPOINTS and STREAMING_CHECKPOINT_ROOT.exists():
        shutil.rmtree(STREAMING_CHECKPOINT_ROOT)
        print(f"  Checkpoints reinitialises: {STREAMING_CHECKPOINT_ROOT}")

    STREAMING_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)


def write_bronze(stream_df, topic: str, schema: StructType, path_suffix: str):
    """Parse Kafka messages and append them to a Delta Bronze table."""
    parsed = (
        stream_df.filter(col("topic") == topic)
        .select(from_json(col("value").cast("string"), schema).alias("d"))
        .select("d.*")
        .withColumn("ingestion_ts", current_timestamp())
    )

    checkpoint_dir = STREAMING_CHECKPOINT_ROOT / path_suffix

    return (
        parsed.writeStream.format("delta")
        .outputMode("append")
        .queryName(f"bronze_{path_suffix}")
        .option("checkpointLocation", str(checkpoint_dir))
        .start(f"{DELTA_PATH}/{path_suffix}")
    )


def main():
    prepare_checkpoints()
    spark = build_spark()

    print("GreenIoT-MA - Spark Structured Streaming demarre")
    print(f"  Kafka: {KAFKA_SERVERS}")
    print(f"  Delta: {DELTA_PATH}")
    print(f"  MinIO: {MINIO_ENDPOINT}")
    print(f"  Checkpoints: {STREAMING_CHECKPOINT_ROOT}\n")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVERS)
        .option("subscribe", ",".join(TOPICS.keys()))
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    queries = []
    for topic, path_suffix in TOPICS.items():
        queries.append(write_bronze(raw, topic, TOPIC_SCHEMAS[topic], path_suffix))

    print("  Queries streaming demarrees:")
    for topic, path_suffix in TOPICS.items():
        print(f"    {topic} -> Bronze/{path_suffix}")
    print("\n  Ctrl+C pour stopper.\n")

    try:
        spark.streams.awaitAnyTermination()
    finally:
        for query in queries:
            if query.isActive:
                query.stop()
        spark.stop()


if __name__ == "__main__":
    main()
