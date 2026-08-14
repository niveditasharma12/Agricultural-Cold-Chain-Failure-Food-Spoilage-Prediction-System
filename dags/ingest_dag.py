"""
=============================================================
  Cold Chain Project — Ingest DAG
=============================================================
  What it does:
    Triggers kafka_producer.py every 5 minutes, which reads
    augmented_cold_chain_dataset.csv and publishes sensor
    events to the Kafka topic: sensor-raw.

  Requires:
    - kafka_producer.py in ./scripts (mounted to
      /opt/airflow/scripts in the Airflow containers)
    - augmented_cold_chain_dataset.csv in ./data (mounted to
      /opt/airflow/data)
    - kafka_producer.py reads KAFKA_BROKER from the environment
      if set, defaulting to localhost:9092 for host runs. Here
      we set it to kafka:9093 since Airflow runs inside Docker.

  NOTE: kafka_producer.py currently replays the WHOLE CSV file
  on every run (with a small delay between rows). Running it
  every 5 minutes means the entire dataset gets re-sent each
  time. That's fine for demo/dev purposes, but worth revisiting
  before treating this as a "real" continuous stream.
=============================================================
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "cold-chain-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="ingest_dag",
    description="Triggers the Kafka producer to publish sensor events every 5 minutes",
    default_args=default_args,
    schedule_interval="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["cold-chain", "ingest", "kafka"],
) as dag:

    run_kafka_producer = BashOperator(
        task_id="run_kafka_producer",
        bash_command=(
            "cd /opt/airflow/data && "
            "pip install --quiet kafka-python pandas && "
            "KAFKA_BROKER=kafka:9093 python /opt/airflow/scripts/kafka_producer.py"
        ),
    )
