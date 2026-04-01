"""
GreenIoT-MA — Shared Spark Session Builder
===========================================
Centralise la configuration Spark pour tous les scripts lakehouse.
Inclut les workarounds pour les problèmes connus sous Windows :
  - BlockManagerId NPE (heartbeater race condition)     → driver.host + heartbeatInterval
  - SparkUI port conflict                               → spark.ui.enabled=false
  - Temp dir cleanup failure (wildfly-openssl JAR lock) → log4j2.properties logger silencieux
"""

import os
from pathlib import Path
from pyspark.sql import SparkSession
from dotenv import load_dotenv

load_dotenv()

# Force loopback — évite le BlockManagerId NPE sous Windows
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

# Chemin absolu vers le fichier log4j2 — dans le même répertoire que ce module
_LOG4J2 = Path(__file__).parent / "log4j2.properties"


def get_spark(app_name: str) -> SparkSession:
    """
    Crée et retourne une SparkSession configurée pour GreenIoT-MA.

    Workarounds Windows inclus :
    - spark.driver.host / bindAddress = 127.0.0.1   → règle le BlockManagerId NPE
    - spark.executor.heartbeatInterval élevé         → réduit les logs heartbeater
    - spark.ui.enabled = false                       → évite le conflit de port 4040
    - log4j2.properties custom via extraJavaOptions  → silence ShutdownHookManager
                                                       et SparkEnv (JAR lock Windows)
    """
    log4j_opt = f"-Dlog4j2.configurationFile=file:///{_LOG4J2.as_posix()}"

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        # ── Réseau / Windows workarounds ──────────────────────
        .config("spark.driver.host",                "127.0.0.1")
        .config("spark.driver.bindAddress",         "127.0.0.1")
        .config("spark.executor.heartbeatInterval", "20s")
        .config("spark.network.timeout",            "120s")
        # ── UI désactivée (évite conflit port 4040/4041) ──────
        .config("spark.ui.enabled",                 "false")
        # ── log4j2 custom : silence ShutdownHookManager/SparkEnv
        .config("spark.driver.extraJavaOptions",    log4j_opt)
        # ── Delta Lake ────────────────────────────────────────
        .config("spark.jars.packages",
                "io.delta:delta-spark_2.12:3.0.0,"
                "org.apache.hadoop:hadoop-aws:3.3.4")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # ── MinIO / S3A ───────────────────────────────────────
        .config("spark.hadoop.fs.s3a.endpoint",           "http://localhost:9000")
        .config("spark.hadoop.fs.s3a.access.key",         os.getenv("MINIO_ACCESS_KEY", "greeniot"))
        .config("spark.hadoop.fs.s3a.secret.key",         os.getenv("MINIO_SECRET_KEY", "greeniot2030"))
        .config("spark.hadoop.fs.s3a.path.style.access",  "true")
        .config("spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .getOrCreate()
    )

    # Supprime les logs verbeux Spark/Hadoop en cours d'exécution (garde WARN+)
    spark.sparkContext.setLogLevel("WARN")

    return spark
