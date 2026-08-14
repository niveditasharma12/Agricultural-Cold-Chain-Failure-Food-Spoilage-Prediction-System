"""
=============================================================
  Cold Chain Project — Fleet Status / Sensor History Sync
=============================================================
  Why this exists:
    spark_streaming.py writes every clean reading to HDFS
    /processed, and alert_to_postgres.py already syncs the
    CRITICAL/breach subset to Postgres for the alert history
    tab. But nothing ever synced the FULL reading stream
    anywhere queryable — so the dashboard's fleet map and
    24h sensor-history charts had no real source to read from
    and were simulated instead.

  What this script does, every run:
    1. Reads the last LOOKBACK_HOURS of HDFS /processed data.
    2. Computes the single latest reading per truck_id and
       UPSERTs it into cold_chain_fleet_status (one row per
       truck — powers the fleet map + truck table).
    3. Inserts any new (deduplicated by event_id) readings
       from the lookback window into cold_chain_sensor_history
       (many rows per truck — powers the 24h charts tab).
    4. Prunes sensor_history rows older than HISTORY_RETAIN_HOURS
       so the table — and the dashboard's queries — stay fast.

  Run manually (needs psycopg2 for the UPSERT/DELETE — installed
  isolated to avoid the global pip-install/Airflow scheduler
  conflict documented in train_model_dag.py):
    docker exec spark-master bash -c "
      pip install --target=/tmp/ml_deps psycopg2-binary --quiet &&
      PYTHONPATH=/tmp/ml_deps /opt/spark/bin/spark-submit \
        --master spark://spark-master:7077 \
        --packages org.postgresql:postgresql:42.7.3 \
        --total-executor-cores 1 \
        --executor-memory 1g \
        /opt/spark-apps/fleet_status_to_postgres.py"
=============================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime, timedelta
import logging
import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("ColdChainFleetStatus")

HDFS_PROCESSED       = "hdfs://namenode:9000/processed"
PG_URL               = "jdbc:postgresql://postgres:5432/airflow"
PG_PROPS             = {"user": "airflow", "password": "airflow", "driver": "org.postgresql.Driver"}
PG_CONN_STR          = "host=postgres port=5432 dbname=airflow user=airflow password=airflow"
FLEET_TABLE          = "cold_chain_fleet_status"
HISTORY_TABLE        = "cold_chain_sensor_history"
LOOKBACK_HOURS       = 3     # how far back to read from HDFS each run
HISTORY_RETAIN_HOURS = 48    # prune sensor_history older than this

spark = (
    SparkSession.builder
    .appName("ColdChain-FleetStatusSync")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
log.info("Spark session ready.")


# ═══════════════════════════════════════════════════════════════
#  STEP 0 — MAKE SURE THE TARGET TABLES EXIST
#  (config/init_db.sql only runs on a fresh Postgres volume, so
#   create these idempotently here too for existing deployments)
# ═══════════════════════════════════════════════════════════════
def ensure_tables():
    conn = psycopg2.connect(PG_CONN_STR)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cold_chain_fleet_status (
                truck_id            VARCHAR(10)  PRIMARY KEY,
                food_name           VARCHAR(50),
                event_time          TIMESTAMP,
                temperature         FLOAT,
                humidity            FLOAT,
                methane             FLOAT,
                co2                 FLOAT,
                storage_days        FLOAT,
                temp_deviation      FLOAT,
                gas_spoilage_score  FLOAT,
                storage_risk_score  FLOAT,
                risk_level          VARCHAR(10),
                gps_lat             FLOAT,
                gps_lon             FLOAT,
                updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cold_chain_sensor_history (
                id              SERIAL PRIMARY KEY,
                event_id        VARCHAR(20),
                truck_id        VARCHAR(10)  NOT NULL,
                event_time      TIMESTAMP    NOT NULL,
                temperature     FLOAT,
                humidity        FLOAT,
                methane         FLOAT,
                co2             FLOAT,
                UNIQUE (event_id)
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_sensor_history_truck_time
                ON cold_chain_sensor_history (truck_id, event_time);
        """)
    conn.close()
    log.info("Fleet/history tables verified.")

ensure_tables()


# ═══════════════════════════════════════════════════════════════
#  STEP 1 — READ RECENT PROCESSED DATA FROM HDFS
# ═══════════════════════════════════════════════════════════════
log.info(f"Reading processed stream from {HDFS_PROCESSED} ...")

try:
    processed_df = spark.read.option("basePath", HDFS_PROCESSED).parquet(HDFS_PROCESSED)
except Exception as e:
    log.warning(f"No processed data found yet (streaming job may not have produced any): {e}")
    spark.stop()
    raise SystemExit(0)

cutoff = datetime.now() - timedelta(hours=LOOKBACK_HOURS)
recent_df = processed_df.filter(
    (F.col("year") == cutoff.year) &
    (F.col("month") == cutoff.month) &
    (F.col("day") >= cutoff.day)
).cache()

total_recent = recent_df.count()
log.info(f"Recent processed rows found (last {LOOKBACK_HOURS}h window): {total_recent:,}")

if total_recent == 0:
    log.info("No recent processed rows to sync. Exiting cleanly.")
    spark.stop()
    raise SystemExit(0)


# ═══════════════════════════════════════════════════════════════
#  STEP 2 — LATEST READING PER TRUCK → cold_chain_fleet_status
#  Written to a staging table, then upserted into the real
#  table with a single SQL statement (Spark's JDBC writer can't
#  do ON CONFLICT natively).
# ═══════════════════════════════════════════════════════════════
log.info("Computing latest reading per truck...")

w = Window.partitionBy("truck_id").orderBy(F.col("event_time").desc())
latest_df = (
    recent_df
    .withColumn("rn", F.row_number().over(w))
    .filter(F.col("rn") == 1)
    .select(
        "truck_id", "food_name", "event_time", "temperature", "humidity",
        "methane", "co2", "storage_days", "temp_deviation",
        "gas_spoilage_score", "storage_risk_score", "risk_level",
        "gps_lat", "gps_lon",
    )
)
n_trucks = latest_df.count()
log.info(f"Latest snapshot computed for {n_trucks} trucks.")

STAGING_TABLE = "cold_chain_fleet_status_staging"
(latest_df.write
    .jdbc(url=PG_URL, table=STAGING_TABLE, mode="overwrite", properties=PG_PROPS))

upsert_sql = f"""
    INSERT INTO {FLEET_TABLE} (
        truck_id, food_name, event_time, temperature, humidity, methane, co2,
        storage_days, temp_deviation, gas_spoilage_score, storage_risk_score,
        risk_level, gps_lat, gps_lon, updated_at
    )
    SELECT truck_id, food_name, event_time, temperature, humidity, methane, co2,
           storage_days, temp_deviation, gas_spoilage_score, storage_risk_score,
           risk_level, gps_lat, gps_lon, CURRENT_TIMESTAMP
    FROM {STAGING_TABLE}
    ON CONFLICT (truck_id) DO UPDATE SET
        food_name           = EXCLUDED.food_name,
        event_time          = EXCLUDED.event_time,
        temperature         = EXCLUDED.temperature,
        humidity            = EXCLUDED.humidity,
        methane             = EXCLUDED.methane,
        co2                 = EXCLUDED.co2,
        storage_days        = EXCLUDED.storage_days,
        temp_deviation      = EXCLUDED.temp_deviation,
        gas_spoilage_score  = EXCLUDED.gas_spoilage_score,
        storage_risk_score  = EXCLUDED.storage_risk_score,
        risk_level          = EXCLUDED.risk_level,
        gps_lat             = EXCLUDED.gps_lat,
        gps_lon             = EXCLUDED.gps_lon,
        updated_at          = CURRENT_TIMESTAMP
    WHERE {FLEET_TABLE}.event_time IS DISTINCT FROM EXCLUDED.event_time;
"""

conn = psycopg2.connect(PG_CONN_STR)
conn.autocommit = True
with conn.cursor() as cur:
    cur.execute(upsert_sql)
    cur.execute(f"DROP TABLE IF EXISTS {STAGING_TABLE};")
conn.close()
log.info(f"✅ Upserted {n_trucks} rows into '{FLEET_TABLE}'.")


# ═══════════════════════════════════════════════════════════════
#  STEP 3 — APPEND NEW ROWS TO cold_chain_sensor_history
#  Same dedup-by-event_id pattern as alert_to_postgres.py
# ═══════════════════════════════════════════════════════════════
log.info("Syncing sensor history (dedup against existing event_ids)...")

existing_ids_df = spark.read.jdbc(
    url=PG_URL,
    table=f"(SELECT DISTINCT event_id FROM {HISTORY_TABLE} WHERE event_id IS NOT NULL) AS ids",
    properties=PG_PROPS,
)

history_candidates_df = recent_df.select(
    "event_id", "truck_id", "event_time", "temperature", "humidity", "methane", "co2"
)
new_history_df = history_candidates_df.join(existing_ids_df, on="event_id", how="left_anti")
new_count = new_history_df.count()
log.info(f"New sensor-history rows to insert: {new_count:,}")

if new_count > 0:
    new_history_df.write.jdbc(url=PG_URL, table=HISTORY_TABLE, mode="append", properties=PG_PROPS)
    log.info(f"✅ Inserted {new_count:,} new rows into '{HISTORY_TABLE}'.")
else:
    log.info("No new history rows — everything in this window is already recorded.")


# ═══════════════════════════════════════════════════════════════
#  STEP 4 — PRUNE OLD HISTORY ROWS
#  Keeps the table (and the dashboard's charts) fast
# ═══════════════════════════════════════════════════════════════
prune_cutoff = datetime.now() - timedelta(hours=HISTORY_RETAIN_HOURS)
conn = psycopg2.connect(PG_CONN_STR)
conn.autocommit = True
with conn.cursor() as cur:
    cur.execute(f"DELETE FROM {HISTORY_TABLE} WHERE event_time < %s;", (prune_cutoff,))
    log.info(f"Pruned sensor_history rows older than {HISTORY_RETAIN_HOURS}h ({cur.rowcount} rows).")
conn.close()

recent_df.unpersist()
spark.stop()
log.info("Fleet status sync run complete.")
