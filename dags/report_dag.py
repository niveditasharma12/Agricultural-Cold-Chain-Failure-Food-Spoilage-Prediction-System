"""
=============================================================
  Cold Chain Project — Report DAG
=============================================================
  What it does:
    Runs daily_report.py at 6 AM, which reads the feature table
    written by the previous night's feature_dag run and logs a
    summary (row counts, spoilage label distribution, risk
    level distribution), plus records the run in pipeline_runs.

  Requires daily_report.py to be present in ./scripts (mounted
  to /opt/spark-apps in the Spark containers).
=============================================================
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "cold-chain-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="report_dag",
    description="Generates the daily summary report at 6 AM",
    default_args=default_args,
    schedule_interval="0 6 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["cold-chain", "reports", "spark", "daily"],
) as dag:

    generate_daily_report = BashOperator(
        task_id="generate_daily_report",
        bash_command=(
            "docker exec spark-master /opt/spark/bin/spark-submit "
            "--master spark://spark-master:7077 "
            "--packages org.postgresql:postgresql:42.7.3 "
            "--total-executor-cores 1 "
            "--executor-memory 1g "
            "/opt/spark-apps/daily_report.py"
        ),
    )