"""
=============================================================
  Cold Chain Project — Alert Scoring / Postgres Sync
=============================================================
  What this script does:
    1. Reads recent rows from HDFS /alerts — this is the
       CRITICAL / temp-breach stream that spark_streaming.py
       already filters and writes continuously.
    2. Filters to the last 2 hours (keeps the scan fast).
    3. Compares against event_ids already stored in the
       cold_chain_alerts Postgres table, so re-running this
       every 15 minutes never inserts duplicates.
    4. Inserts only the new rows.

  Run manually:
    docker exec spark-master spark-submit \
      --master spark://spark-master:7077 \
      --packages org.postgresql:postgresql:42.7.3 \
      --total-executor-cores 1 \
      --executor-memory 1g \
      /opt/spark-apps/alert_to_postgres.py
=============================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("ColdChainAlertScoring")

HDFS_ALERTS = "hdfs://namenode:9000/alerts"
PG_URL      = "jdbc:postgresql://postgres:5432/airflow"
PG_PROPS    = {"user": "airflow", "password": "airflow", "driver": "org.postgresql.Driver"}
PG_TABLE    = "cold_chain_alerts"
LOOKBACK_HOURS = 2

spark = (
    SparkSession.builder
    .appName("ColdChain-AlertScoring")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
log.info("Spark session ready.")


# ═══════════════════════════════════════════════════════════════
#  STEP 1 — READ RECENT ALERT ROWS FROM HDFS
# ═══════════════════════════════════════════════════════════════
log.info(f"Reading alert stream from {HDFS_ALERTS} ...")

try:
    alerts_df = spark.read.option("basePath", HDFS_ALERTS).parquet(HDFS_ALERTS)
except Exception as e:
    log.warning(f"No alert data found yet (streaming job may not have produced any): {e}")
    spark.stop()
    raise SystemExit(0)

cutoff = datetime.now() - timedelta(hours=LOOKBACK_HOURS)
recent_df = alerts_df.filter(
    (F.col("year") == cutoff.year) &
    (F.col("month") == cutoff.month) &
    (F.col("day") >= cutoff.day)
)

total_recent = recent_df.count()
log.info(f"Recent alert rows found (last {LOOKBACK_HOURS}h window): {total_recent:,}")

if total_recent == 0:
    log.info("No recent alerts to process. Exiting cleanly.")
    spark.stop()
    raise SystemExit(0)


# ═══════════════════════════════════════════════════════════════
#  STEP 2 — DEDUPLICATE AGAINST WHAT'S ALREADY IN POSTGRES
# ═══════════════════════════════════════════════════════════════
log.info("Fetching existing event_ids from Postgres to avoid duplicate inserts...")

existing_ids_df = spark.read.jdbc(
    url=PG_URL,
    table="(SELECT DISTINCT event_id FROM cold_chain_alerts WHERE event_id IS NOT NULL) AS ids",
    properties=PG_PROPS,
)

new_alerts_df = recent_df.join(existing_ids_df, on="event_id", how="left_anti")
new_count = new_alerts_df.count()
log.info(f"New alerts to insert: {new_count:,}")


# ═══════════════════════════════════════════════════════════════
#  STEP 3 — WRITE NEW ALERTS TO POSTGRES
# ═══════════════════════════════════════════════════════════════
if new_count > 0:
    output_df = new_alerts_df.select(
        F.col("event_id"),
        F.col("truck_id"),
        F.col("food_name"),
        F.col("temperature"),
        F.col("methane"),
        F.col("co2"),
        F.col("storage_days"),
        F.col("storage_risk_score").alias("spoilage_prob"),
        F.when(F.col("risk_level") == "CRITICAL", F.lit(2))
         .when(F.col("risk_level") == "WARNING", F.lit(1))
         .otherwise(F.lit(0)).alias("predicted_class"),
        F.col("risk_level"),
        F.col("alert_message"),
        F.lit(None).cast("string").alias("nearest_city"),
        F.col("gps_lat"),
        F.col("gps_lon"),
    )

    output_df.write.jdbc(url=PG_URL, table=PG_TABLE, mode="append", properties=PG_PROPS)
    log.info(f"✅ Inserted {new_count:,} new alerts into Postgres table '{PG_TABLE}'.")
else:
    log.info("No new alerts — everything in this window is already recorded.")

spark.stop()
log.info("Alert scoring run complete.")
