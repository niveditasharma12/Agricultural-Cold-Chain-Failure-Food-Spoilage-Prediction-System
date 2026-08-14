"""
=============================================================
  Cold Chain Project — Daily Summary Report
=============================================================
  What this script does:
    1. Reads the current HDFS /features table (written nightly
       by feature_dag / spark_batch_etl.py)
    2. Logs a summary: total rows, spoilage label distribution,
       risk level distribution
    3. Writes one row to the pipeline_runs Postgres table so
       report history can be tracked over time

  Run manually:
    docker exec spark-master spark-submit \
      --master spark://spark-master:7077 \
      --packages org.postgresql:postgresql:42.7.3 \
      --total-executor-cores 1 \
      --executor-memory 1g \
      /opt/spark-apps/daily_report.py
=============================================================
"""

from pyspark.sql import SparkSession
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("ColdChainDailyReport")

HDFS_FEATURES = "hdfs://namenode:9000/features"
PG_URL        = "jdbc:postgresql://postgres:5432/airflow"
PG_PROPS      = {"user": "airflow", "password": "airflow", "driver": "org.postgresql.Driver"}

spark = (
    SparkSession.builder
    .appName("ColdChain-DailyReport")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

run_start = datetime.now()
log.info(f"Generating daily summary report for {run_start.date()}...")

status = "success"
row_count = 0

try:
    features_df = spark.read.parquet(HDFS_FEATURES)
    row_count = features_df.count()

    label_dist = features_df.groupBy("spoilage_label_name").count()
    risk_dist  = features_df.groupBy("risk_level").count()

    log.info("=" * 55)
    log.info(f"  DAILY SUMMARY REPORT — {run_start.date()}")
    log.info("=" * 55)
    log.info(f"  Total feature rows: {row_count:,}")
    log.info("  Spoilage label distribution:")
    label_dist.show(truncate=False)
    log.info("  Risk level distribution:")
    risk_dist.show(truncate=False)

except Exception as e:
    log.warning(f"Could not read features table (has feature_dag run yet?): {e}")
    status = "failed"

duration_secs = int((datetime.now() - run_start).total_seconds())

# ── Log this run into pipeline_runs so history is tracked ─────
run_log_df = spark.createDataFrame(
    [(
        "report_dag",
        f"report_{run_start.strftime('%Y%m%d_%H%M%S')}",
        status,
        row_count,
        duration_secs,
    )],
    ["dag_id", "run_id", "status", "rows_processed", "duration_secs"],
)

run_log_df.write.jdbc(url=PG_URL, table="pipeline_runs", mode="append", properties=PG_PROPS)

log.info(f"✅ Daily report complete (status={status}) — logged to pipeline_runs.")
spark.stop()
