"""
=============================================================
 Kafka Producer — Cold Chain Sensor Stream
=============================================================
 Reads augmented_cold_chain_dataset.csv row-by-row and
 publishes each as a JSON event to Kafka topic: sensor-raw

 Every event contains all 18 fields your Spark Streaming
 job expects — including pre-computed ML features.

 Usage   : python kafka_producer.py
 Install : pip install kafka-python pandas
 Kafka   : must be running (docker compose up)
=============================================================
"""

import pandas as pd
import json
import time
import sys
import os
from kafka import KafkaProducer
from datetime import datetime

# ── Config ────────────────────────────────────────────────
# KAFKA_BROKER defaults to localhost:9092 for running on your host
# machine. Set the KAFKA_BROKER env var to kafka:9093 when running
# this script from inside a Docker container on the same network
# (e.g. via Airflow's ingest_dag.py).
KAFKA_BROKER   = os.environ.get("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC    = "sensor-raw"
CSV_FILE       = "augmented_cold_chain_dataset.csv"
DELAY_SECONDS  = 0.1   # 0.1s = ~10 events/sec  (change to 0.5 for slower)
PRINT_EVERY    = 500   # print progress every N events

# ── Connect ────────────────────────────────────────────────
print(f"Connecting to Kafka at {KAFKA_BROKER}...")
try:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        acks="all",              # wait for all brokers to confirm
        retries=3,
    )
    print("✅ Connected to Kafka")
except Exception as e:
    print(f"❌ Kafka connection failed: {e}")
    print("   Make sure Kafka is running: docker compose up -d")
    sys.exit(1)

# ── Load CSV ───────────────────────────────────────────────
print(f"Loading {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)
print(f"✅ Loaded {len(df):,} rows | Publishing to topic: {KAFKA_TOPIC}")
print(f"   Speed: {1/DELAY_SECONDS:.0f} events/sec | Ctrl+C to stop\n")

sent = 0
errors = 0
start = datetime.now()

for idx, row in df.iterrows():
    event = {
        "event_id":             row["event_id"],
        "truck_id":             row["truck_id"],
        "timestamp":            str(row["timestamp"]),
        "gps_lat":              float(row["gps_lat"]),
        "gps_lon":              float(row["gps_lon"]),
        "food_name":            row["Food_Name"],
        "temperature":          float(row["Temperature"]),
        "humidity":             float(row["Humidity"]),
        "methane":              float(row["Methane"]),
        "co2":                  float(row["CO2"]),
        "storage_days":         float(row["Storage_Days"]),
        "temp_humidity_index":  float(row["temp_humidity_index"]),
        "gas_spoilage_score":   float(row["gas_spoilage_score"]),
        "methane_co2_ratio":    float(row["methane_co2_ratio"]),
        "temp_deviation":       float(row["temp_deviation"]),
        "storage_risk_score":   float(row["storage_risk_score"]),
        "spoiled":              int(row["Spoiled"]),
        "source":               row["source"],
    }

    try:
        producer.send(KAFKA_TOPIC, value=event)
        sent += 1
    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"⚠️  Send error at row {idx}: {e}")

    if sent % PRINT_EVERY == 0:
        elapsed = (datetime.now() - start).seconds or 1
        rate    = sent / elapsed
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"Sent: {sent:,} | "
              f"Truck: {event['truck_id']} | "
              f"Food: {event['food_name']:<12} | "
              f"Temp: {event['temperature']:5.1f}°C | "
              f"Spoiled: {event['spoiled']} | "
              f"Rate: {rate:.0f} ev/s")
        sent = 0

    time.sleep(DELAY_SECONDS)
    if(sent != 0):
        try:
                producer.send(KAFKA_TOPIC, value=event)
                sent += 1
        except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"⚠️  Send error at row {idx}: {e}")
producer.flush()
elapsed = (datetime.now() - start).seconds or 1
print(f"\n✅ Done — {sent:,} events sent in {elapsed}s "
      f"({sent/elapsed:.0f} ev/s) | Errors: {errors}")
 