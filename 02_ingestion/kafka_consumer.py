"""
GreenIoT-MA — Kafka Consumer
==============================
Consomme les messages Kafka et les écrit en Bronze (Parquet/Delta).
Utilisable pour le développement sans PySpark.
"""

import json
import os
import pandas as pd
from datetime import datetime
from kafka import KafkaConsumer
from dotenv import load_dotenv

load_dotenv()

KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "localhost:9092")
TOPICS = ["greeniot.solar", "greeniot.servers", "greeniot.cooling", "greeniot.battery"]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bronze")


def consume_to_bronze(batch_size: int = 100, flush_interval_sec: int = 60):
    """
    Consomme les messages Kafka et les écrit en Parquet par batch.

    Args:
        batch_size: Nombre de messages avant écriture
        flush_interval_sec: Intervalle max entre écritures (secondes)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    consumer = KafkaConsumer(
        *TOPICS,
        bootstrap_servers=KAFKA_BROKERS,
        api_version=(3, 5, 0),
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        group_id="greeniot-bronze-consumer",
        enable_auto_commit=False,
    )

    print(f"🌿 Consumer GreenIoT-MA démarré — Kafka: {KAFKA_BROKERS}")
    print(f"   Topics: {', '.join(TOPICS)}")
    print(f"   Batch size: {batch_size} | Flush interval: {flush_interval_sec}s")
    print("   Ctrl+C pour stopper.\n")

    buffers = {topic: [] for topic in TOPICS}
    last_flush = datetime.now()

    try:
        for message in consumer:
            topic = message.topic
            record = message.value
            record["_kafka_topic"] = topic
            record["_kafka_offset"] = message.offset
            record["_kafka_partition"] = message.partition
            record["_ingestion_ts"] = datetime.utcnow().isoformat()

            buffers[topic].append(record)
            print(f"  [{record['timestamp'][:19]}] {topic} ← {record.get('sensor_id', 'unknown')}")

            # Flush synchronise puis commit seulement apres persistance reussie.
            elapsed = (datetime.now() - last_flush).seconds
            should_flush = any(len(buffers[t]) >= batch_size for t in TOPICS) or (
                elapsed >= flush_interval_sec and any(buffers[t] for t in TOPICS)
            )
            if should_flush:
                _flush_all_buffers(buffers)
                consumer.commit()
                last_flush = datetime.now()

    except KeyboardInterrupt:
        print("\n\n🛑 Flush final en cours...")
        if any(buffers[t] for t in TOPICS):
            _flush_all_buffers(buffers)
            consumer.commit()
        print("   Consumer arrêté proprement.")
    finally:
        consumer.close()


def _flush_all_buffers(buffers: dict):
    """Persist every non-empty topic buffer before Kafka offsets are committed."""
    for topic, records in buffers.items():
        if records:
            _flush_buffer(topic, records)
            buffers[topic] = []


def _flush_buffer(topic: str, records: list):
    """Écrit un batch de records en Parquet."""
    if not records:
        return

    topic_dir = os.path.join(OUTPUT_DIR, topic.replace(".", "_"))
    os.makedirs(topic_dir, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{len(records)}.parquet"
    filepath = os.path.join(topic_dir, filename)

    df = pd.DataFrame(records)
    df.to_parquet(filepath, index=False, engine="pyarrow")
    print(f"  💾 Flush: {topic} → {filename} ({len(records)} records)")


if __name__ == "__main__":
    consume_to_bronze()
