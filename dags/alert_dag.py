"""
=============================================================
  Cold Chain Project — Alert & Fleet Status DAG
=============================================================
  What it does, every 15 minutes:
    1. score_and_alert — runs alert_to_postgres.py:
         - Reads recent CRITICAL / temp-breach rows already
           written by the streaming job to HDFS /alerts
         - Deduplicates against what's already stored
         - Inserts new rows into cold_chain_alerts in Postgres
           (read by Grafana / the Streamlit dashboard's
           Alert History tab)
    2. sync_fleet_status — runs fleet_status_to_postgres.py:
         - Reads recent readings from HDFS /processed
         - Upserts the latest reading per truck into
           cold_chain_fleet_status (Fleet Monitor tab)
         - Appends new rows to cold_chain_sensor_history and
           prunes old ones (Sensor Charts tab)

  Both tasks are independent (no ordering between them) and
  requires alert_to_postgres.py + fleet_status_to_postgres.py
  to be present in ./scripts (mounted to /opt/spark-apps in
  the Spark containers).
=============================================================
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "cold-chain-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="alert_dag",
    description="Fires new alerts and syncs fleet/sensor-history to Postgres every 15 minutes",
    default_args=default_args,
    schedule_interval="*/15 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["cold-chain", "alerts", "fleet-status", "spark", "postgres"],
) as dag:

    score_and_alert = BashOperator(
        task_id="score_and_alert",
        bash_command=(
            "docker exec spark-master /opt/spark/bin/spark-submit "
            "--master spark://spark-master:7077 "
            "--packages org.postgresql:postgresql:42.7.3 "
            "--total-executor-cores 1 "
            "--executor-memory 1g "
            "/opt/spark-apps/alert_to_postgres.py"
        ),
    )

    sync_fleet_status = BashOperator(
        task_id="sync_fleet_status",
        bash_command=(
            # fleet_status_to_postgres.py needs psycopg2 for the UPSERT/DELETE
            # it issues (Spark's JDBC writer can't do either). Installed into
            # an isolated --target dir + scoped PYTHONPATH, same fix already
            # used for the ML deps in train_model_dag.py — a plain global
            # `pip install` here previously broke the Airflow scheduler
            # (see train_model_dag.py's docstring for that incident).
            "docker exec spark-master bash -c "
            "\"pip install --target=/tmp/ml_deps psycopg2-binary --quiet && "
            "PYTHONPATH=/tmp/ml_deps /opt/spark/bin/spark-submit "
            "--master spark://spark-master:7077 "
            "--packages org.postgresql:postgresql:42.7.3 "
            "--total-executor-cores 1 "
            "--executor-memory 1g "
            "/opt/spark-apps/fleet_status_to_postgres.py\""
        ),
    )

    # Independent — no dependency needed, but declaring them lets
    # Airflow show both clearly on the same graph.
    [score_and_alert, sync_fleet_status]