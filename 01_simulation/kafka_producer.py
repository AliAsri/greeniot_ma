"""
GreenIoT-MA — Kafka Producer
==============================
Envoie les mesures IoT simulées vers des topics Kafka typés.
Topics : greeniot.solar, greeniot.servers, greeniot.cooling, greeniot.battery
"""

from kafka import KafkaProducer
from sensor_simulator import generate_stream
import json
import os
from dotenv import load_dotenv

load_dotenv()

KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "localhost:9092")

TOPIC_MAP = {
    "solar": "greeniot.solar",
    "server": "greeniot.servers",
    "cooling": "greeniot.cooling",
    "battery": "greeniot.battery",
}

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKERS,
    api_version=(3, 5, 0),
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    acks="all",           # Durabilité maximale
    retries=3,
    linger_ms=10,         # Batching léger pour performance
    compression_type="gzip",
)

print(f"🌿 Producer GreenIoT-MA démarré — Kafka: {KAFKA_BROKERS}")
print("   Ctrl+C pour stopper.\n")

try:
    for record in generate_stream(interval_sec=2.0):
        topic = TOPIC_MAP.get(record["type"], "greeniot.misc")
        future = producer.send(topic, value=record)
        print(f"  [{record['timestamp'][:19]}] {topic} → {record['sensor_id']}")
except KeyboardInterrupt:
    print("\n\n🛑 Producer arrêté proprement.")
finally:
    producer.flush()
    producer.close()
