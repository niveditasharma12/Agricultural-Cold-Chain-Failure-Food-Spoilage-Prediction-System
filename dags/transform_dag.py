"""
=============================================================
  Cold Chain Project — Transform DAG
=============================================================
  What it does:
    Every 15 minutes:
      1. Checks that the always-on Spark Streaming job
         (spark-streaming-job, a separate long-running Docker
         Compose service) is still alive by hitting its Spark
         UI on port 4040.
      2. Runs spark_batch_etl.py with a 1-hour lookback window,
         to roll up whatever the streaming job has written to
         HDFS /processed since the last run.

  NOTE: the streaming job itself runs continuously as its own
  docker-compose service (spark-streaming-job) — this DAG does
  NOT start/stop it, it only verifies it's alive and then runs
  the batch rollup on top of what it has produced so far.
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
    dag_id="transform_dag",
    description="Checks streaming health and runs batch ETL every 15 minutes",
    default_args=default_args,
    schedule_interval="*/15 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["cold-chain", "transform", "spark"],
) as dag:

    check_streaming_job = BashOperator(
        task_id="check_streaming_job",
        bash_command=(
            "docker exec spark-streaming-job curl -sf http://localhost:4040 "
            "|| echo 'WARNING: streaming job UI unreachable — check spark-streaming-job container'"
        ),
    )

    run_batch_etl = BashOperator(
        task_id="run_batch_etl",
        bash_command=(
            "docker exec spark-master /opt/spark/bin/spark-submit "
            "--master spark://spark-master:7077 "
            "--total-executor-cores 1 "
            "--executor-memory 1g "
            "/opt/spark-apps/spark_batch_etl.py 1"
        ),
    )

    check_streaming_job >> run_batch_etl