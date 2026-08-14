"""
=============================================================
  Cold Chain Project — Feature DAG
=============================================================
  What it does:
    Runs spark_batch_etl.py nightly at 1 AM with a full 24-hour
    lookback window, rebuilding the complete feature table at
    HDFS /features (overwrite mode, partitioned by
    year/month/day/hour). This is the authoritative nightly
    pass — transform_dag's 15-minute runs are lighter, rolling
    updates on top of the same output path.
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
    dag_id="feature_dag",
    description="Nightly full feature engineering pass at 1 AM",
    default_args=default_args,
    schedule_interval="0 1 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["cold-chain", "features", "spark", "nightly"],
) as dag:

    run_nightly_feature_engineering = BashOperator(
        task_id="run_nightly_feature_engineering",
        bash_command=(
            "docker exec spark-master /opt/spark/bin/spark-submit "
            "--master spark://spark-master:7077 "
            "--total-executor-cores 2 "
            "--executor-memory 2g "
            "/opt/spark-apps/spark_batch_etl.py 24"
        ),
    )