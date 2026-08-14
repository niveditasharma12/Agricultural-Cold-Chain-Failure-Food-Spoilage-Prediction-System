"""
=============================================================
  Cold Chain Project — PySpark Structured Streaming
=============================================================
  What this script does:
    1. Reads JSON events from Kafka topic: sensor-raw
    2. Parses and validates the schema
    3. Cleans dirty data (nulls, outliers, duplicates)
    4. Joins with food-type safe temperature thresholds
    5. Engineers 5 ML features
    6. Writes clean Parquet to HDFS /processed (partitioned)
    7. Writes a separate alert stream to HDFS /alerts
       for any temperature breach detected in real time

  Run command (inside Spark container):
    spark-submit \
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 \
      --master spark://spark-master:7077 \
      /opt/spark-apps/spark_streaming.py

  Or from your machine (if spark-submit is local):
    spark-submit \
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 \
      spark_streaming.py

  Data ranges from your real dataset:
    Temperature  : -1.28 to 30.62 °C
    Humidity     : 32.0  to 100.0  %
    Methane      : 0.0   to 175.97 ppm
    CO2          : 304.68 to 2366.57 ppm
    Storage_Days : 0.0   to 15.0   days
    Missing vals : ~500 per column → imputed with food-type median
=============================================================
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, FloatType, DoubleType,
    IntegerType, TimestampType
)
import logging

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("ColdChainStreaming")

# ═══════════════════════════════════════════════════════════════
#  CONFIG — change these if your ports differ
# ═══════════════════════════════════════════════════════════════
KAFKA_BROKER    = "kafka:9093"          # inside Docker network
KAFKA_TOPIC     = "sensor-raw"
HDFS_PROCESSED  = "hdfs://namenode:9000/processed"
HDFS_ALERTS     = "hdfs://namenode:9000/alerts"
HDFS_RAW        = "hdfs://namenode:9000/raw"
CHECKPOINT_PROC = "hdfs://namenode:9000/checkpoints/processed"
CHECKPOINT_ALRT = "hdfs://namenode:9000/checkpoints/alerts"
CHECKPOINT_RAW  = "hdfs://namenode:9000/checkpoints/raw"
TRIGGER_SECS    = 30           # micro-batch every 30 seconds
WATERMARK_MINS  = "5 minutes"  # tolerate 5 min late data

# ── Safe temperature max per food (°C) ────────────────────────
# Based on food science standards + your dataset analysis
FOOD_SAFE_TEMP = {
    "Chicken":   4.0,
    "Beef":      4.0,
    "Fish":      3.0,
    "Milk":      5.0,
    "Yogurt":    5.0,
    "Cheese":    6.0,
    "Eggs":      6.0,
    "Spinach":   5.0,
    "Mushroom":  6.0,
    "Bread":    23.0,
    "Potato":   10.0,
    "Tomato":   10.0,
    "Apple":     7.0,
    "Orange":    8.0,
    "Strawberry": 5.0,
}

# ── Median imputation values per food type (from real dataset) ─
# Pre-computed to avoid joining inside streaming — keep it fast
FOOD_TEMP_MEDIANS = {
    "Chicken": 11.0, "Beef": 11.2, "Fish": 10.8, "Milk": 10.5,
    "Yogurt": 10.7, "Cheese": 10.6, "Eggs": 11.0, "Spinach": 11.1,
    "Mushroom": 11.3, "Bread": 21.0, "Potato": 13.5, "Tomato": 13.8,
    "Apple": 12.0, "Orange": 12.4, "Strawberry": 10.9,
}
FOOD_HUMIDITY_MEDIAN  = 63.0   # global median
FOOD_METHANE_MEDIAN   = 38.0   # global median
FOOD_CO2_MEDIAN       = 870.0  # global median


# ═══════════════════════════════════════════════════════════════
#  SPARK SESSION
# ═══════════════════════════════════════════════════════════════
log.info("Initialising Spark session...")

spark = (
    SparkSession.builder
    .appName("ColdChain-SensorStreaming")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.streaming.stopGracefullyOnShutdown", "true")
    # Hadoop config (inside Docker these are auto-discovered)
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    # Kafka max offset per trigger (controls throughput)
    .config("spark.sql.streaming.kafka.useDeprecatedOffsetFetching", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
log.info("Spark session ready.")


# ═══════════════════════════════════════════════════════════════
#  STEP 1 — READ FROM KAFKA
# ═══════════════════════════════════════════════════════════════
log.info(f"Connecting to Kafka broker: {KAFKA_BROKER}, topic: {KAFKA_TOPIC}")

kafka_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BROKER)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "latest")       # only new messages
    .option("maxOffsetsPerTrigger", 5000)      # 5k events per micro-batch
    .option("failOnDataLoss", "false")         # tolerate Kafka retention gaps
    .load()
)

# Kafka gives us: key, value (bytes), topic, partition, offset, timestamp
# We only need value (our JSON payload)
log.info("Kafka stream connected. Extracting JSON value field...")

raw_value_df = kafka_df.selectExpr(
    "CAST(value AS STRING) AS json_str",
    "timestamp AS kafka_timestamp",            # Kafka ingestion timestamp
    "partition AS kafka_partition",
    "offset AS kafka_offset",
)


# ═══════════════════════════════════════════════════════════════
#  STEP 2 — DEFINE JSON SCHEMA
#  Must exactly match kafka_producer.py event fields
# ═══════════════════════════════════════════════════════════════
sensor_schema = StructType([
    StructField("event_id",            StringType(),  True),
    StructField("truck_id",            StringType(),  True),
    StructField("timestamp",           StringType(),  True),  # parse later
    StructField("gps_lat",             DoubleType(),  True),
    StructField("gps_lon",             DoubleType(),  True),
    StructField("food_name",           StringType(),  True),
    StructField("temperature",         FloatType(),   True),
    StructField("humidity",            FloatType(),   True),
    StructField("methane",             FloatType(),   True),
    StructField("co2",                 FloatType(),   True),
    StructField("storage_days",        FloatType(),   True),
    # Pre-computed features from producer (we recompute to ensure accuracy)
    StructField("temp_humidity_index", FloatType(),   True),
    StructField("gas_spoilage_score",  FloatType(),   True),
    StructField("methane_co2_ratio",   FloatType(),   True),
    StructField("temp_deviation",      FloatType(),   True),
    StructField("storage_risk_score",  FloatType(),   True),
    StructField("spoiled",             IntegerType(), True),
    StructField("source",              StringType(),  True),
])

# Parse JSON
parsed_df = raw_value_df.select(
    F.from_json(F.col("json_str"), sensor_schema).alias("data"),
    F.col("kafka_timestamp"),
    F.col("kafka_partition"),
    F.col("kafka_offset"),
)

# Flatten the nested struct
flat_df = parsed_df.select(
    "data.*",
    "kafka_timestamp",
    "kafka_partition",
    "kafka_offset",
)

log.info("JSON schema applied. Schema:")


# ═══════════════════════════════════════════════════════════════
#  STEP 3 — PARSE TIMESTAMP + WATERMARK
#  Watermark handles late-arriving data gracefully
# ═══════════════════════════════════════════════════════════════
timestamped_df = flat_df.withColumn(
    "event_time",
    F.to_timestamp(F.col("timestamp"), "yyyy-MM-dd HH:mm:ss")
).withWatermark("event_time", WATERMARK_MINS)


# ═══════════════════════════════════════════════════════════════
#  STEP 4 — VALIDATE: DROP COMPLETELY BAD ROWS
#  A row is invalid if truck_id or food_name is missing
#  OR if ALL sensor readings are null (zombie row)
# ═══════════════════════════════════════════════════════════════
log.info("Applying row-level validation filters...")

validated_df = timestamped_df.filter(
    F.col("truck_id").isNotNull() &
    F.col("food_name").isNotNull() &
    F.col("event_time").isNotNull() &
    # At least one sensor reading must be present
    (
        F.col("temperature").isNotNull() |
        F.col("methane").isNotNull()     |
        F.col("co2").isNotNull()
    )
)


# ═══════════════════════════════════════════════════════════════
#  STEP 5 — DEDUPLICATE
#  Drop duplicate events within the watermark window
#  (producer may retry, Kafka may redeliver)
# ═══════════════════════════════════════════════════════════════
log.info("Applying deduplication on event_id + event_time...")

deduped_df = validated_df.dropDuplicates(["event_id", "event_time"])

# Fill null food_name with UNKNOWN to prevent downstream null crashes
deduped_df = deduped_df.withColumn(
    "food_name",
    F.coalesce(F.col("food_name"), F.lit("Chicken"))  # default to Chicken if null
)


# ═══════════════════════════════════════════════════════════════
#  STEP 6 — IMPUTE MISSING SENSOR VALUES
#  Strategy: per-food-type median for temperature
#            global median for humidity, methane, CO2
#  From your real dataset: ~500 missing per column
# ═══════════════════════════════════════════════════════════════
log.info("Imputing missing sensor values with food-type medians...")

# Build a map expression for food → median temperature
temp_median_map = F.create_map(
    *[val for pair in
      [(F.lit(food), F.lit(median)) for food, median in FOOD_TEMP_MEDIANS.items()]
      for val in pair]
)

imputed_df = deduped_df \
    .withColumn(
        "temperature",
        F.when(
            F.col("temperature").isNull(),
            F.coalesce(
                temp_median_map[F.col("food_name")],
                F.lit(11.0)              # global fallback
            )
        ).otherwise(F.col("temperature"))
    ) \
    .withColumn(
        "humidity",
        F.when(F.col("humidity").isNull(), F.lit(FOOD_HUMIDITY_MEDIAN))
         .otherwise(F.col("humidity"))
    ) \
    .withColumn(
        "methane",
        F.when(F.col("methane").isNull(), F.lit(FOOD_METHANE_MEDIAN))
         .otherwise(F.col("methane"))
    ) \
    .withColumn(
        "co2",
        F.when(F.col("co2").isNull(), F.lit(FOOD_CO2_MEDIAN))
         .otherwise(F.col("co2"))
    ) \
    .withColumn(
        "storage_days",
        F.when(F.col("storage_days").isNull(), F.lit(3.0))
         .otherwise(F.col("storage_days"))
    )


# ═══════════════════════════════════════════════════════════════
#  STEP 7 — CLIP OUTLIERS TO VALID PHYSICAL RANGES
#  Based on actual data ranges from your dataset:
#    Temperature  : -5 to 35 °C
#    Humidity     : 20 to 100 %
#    Methane      : 0  to 180 ppm
#    CO2          : 300 to 2400 ppm
#    Storage_Days : 0  to 15 days
# ═══════════════════════════════════════════════════════════════
log.info("Clipping sensor readings to valid physical ranges...")

clipped_df = imputed_df \
    .withColumn("temperature",
        F.greatest(F.lit(-5.0),  F.least(F.lit(35.0),  F.col("temperature")))
    ) \
    .withColumn("humidity",
        F.greatest(F.lit(20.0),  F.least(F.lit(100.0), F.col("humidity")))
    ) \
    .withColumn("methane",
        F.greatest(F.lit(0.0),   F.least(F.lit(180.0), F.col("methane")))
    ) \
    .withColumn("co2",
        F.greatest(F.lit(300.0), F.least(F.lit(2400.0),F.col("co2")))
    ) \
    .withColumn("storage_days",
        F.greatest(F.lit(0.0),   F.least(F.lit(15.0),  F.col("storage_days")))
    )



# ═══════════════════════════════════════════════════════════════
#  ML MODEL UDF — Load trained Random Forest from MLflow
#  and register as a Spark UDF so every row in the stream
#  gets a REAL model prediction instead of a hardcoded threshold.
#
#  WHY UDF APPROACH (not HTTP call):
#    - Model loads ONCE into Spark driver memory at startup
#    - No network call per row — 100x faster than HTTP
#    - Works entirely inside Docker network — no port issues
#    - If MLflow is unreachable, gracefully falls back to
#      rule-based scoring (no pipeline crash)
# ═══════════════════════════════════════════════════════════════
log.info("Loading trained Random Forest from MLflow for UDF inference...")

# ── Shared model state (loaded once, used by all workers) ──────
_rf_model       = None   # sklearn RandomForestClassifier
_model_loaded   = False  # flag for fallback logic

MLFLOW_TRACKING_URI = "http://mlflow:5000"
RF_MODEL_NAME       = "cold-chain-rf-classifier"

def _load_rf_model():
    """
    Load the Production Random Forest model from MLflow registry.
    Called ONCE at Spark driver startup — not per row.
    Returns (model, True) on success, (None, False) on failure.
    """
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.sklearn.load_model(
            f"models:/{RF_MODEL_NAME}/Production"
        )
        log.info(f"✅ RF model loaded from MLflow: models:/{RF_MODEL_NAME}/Production")
        return model, True
    except Exception as e:
        log.warning(f"⚠️  Could not load RF model from MLflow: {e}")
        log.warning("   Falling back to rule-based scoring (storage_risk_score thresholds)")
        return None, False

# Load model at driver startup
_rf_model, _model_loaded = _load_rf_model()

# Feature column order — MUST match train_random_forest.py FEATURE_COLS exactly
RF_FEATURE_COLS = [
    "temperature", "humidity", "methane", "co2", "storage_days",
    "temp_humidity_index", "gas_spoilage_score", "methane_co2_ratio",
    "temp_deviation", "storage_risk_score",
]

# One-hot food encoding order — must match training
ALL_FOODS_ORDERED = [
    "apple","beef","bread","cheese","chicken","eggs","fish",
    "milk","mushroom","orange","potato","spinach","strawberry","tomato","yogurt"
]

def _rule_based_predict(storage_risk_score):
    """
    Fallback when ML model is not loaded.
    Uses the same thresholds as before — returns (class, prob_fresh, prob_at_risk, prob_spoiled).
    """
    if storage_risk_score >= 0.65:
        return 2, 0.05, 0.15, 0.80   # Spoiled
    elif storage_risk_score >= 0.40:
        return 1, 0.15, 0.65, 0.20   # At Risk
    elif storage_risk_score >= 0.20:
        return 1, 0.40, 0.45, 0.15   # Watch
    else:
        return 0, 0.85, 0.10, 0.05   # Fresh

def _ml_predict_row(food_name, temperature, humidity, methane, co2,
                    storage_days, temp_humidity_index, gas_spoilage_score,
                    methane_co2_ratio, temp_deviation, storage_risk_score):
    """
    Core prediction function called by Spark UDF for EACH row.

    If RF model is loaded → uses real model probabilities.
    If not loaded → falls back to rule-based thresholds.

    Returns: predicted_class (int 0/1/2)
    """
    global _rf_model, _model_loaded

    if not _model_loaded or _rf_model is None:
        cls, _, _, _ = _rule_based_predict(storage_risk_score or 0.0)
        return cls

    try:
        import pandas as pd
        # Build feature row in exact order the model was trained on
        row = {col: 0.0 for col in RF_FEATURE_COLS}
        row["temperature"]          = float(temperature or 0)
        row["humidity"]             = float(humidity or 63)
        row["methane"]              = float(methane or 0)
        row["co2"]                  = float(co2 or 400)
        row["storage_days"]         = float(storage_days or 0)
        row["temp_humidity_index"]  = float(temp_humidity_index or 0)
        row["gas_spoilage_score"]   = float(gas_spoilage_score or 0)
        row["methane_co2_ratio"]    = float(methane_co2_ratio or 0)
        row["temp_deviation"]       = float(temp_deviation or 0)
        row["storage_risk_score"]   = float(storage_risk_score or 0)

        # One-hot encode food_name
        food_key = (food_name or "chicken").lower()
        for food in ALL_FOODS_ORDERED:
            row[f"food_{food}"] = 1 if food == food_key else 0

        X = pd.DataFrame([row])
        pred = int(_rf_model.predict(X)[0])
        return pred

    except Exception as e:
        # Never crash the stream — fall back gracefully
        cls, _, _, _ = _rule_based_predict(storage_risk_score or 0.0)
        return cls


def _ml_prob_spoiled(food_name, temperature, humidity, methane, co2,
                     storage_days, temp_humidity_index, gas_spoilage_score,
                     methane_co2_ratio, temp_deviation, storage_risk_score):
    """Returns probability of Spoiled class (float 0-1) for alert threshold."""
    global _rf_model, _model_loaded

    if not _model_loaded or _rf_model is None:
        _, _, _, prob_s = _rule_based_predict(storage_risk_score or 0.0)
        return prob_s

    try:
        import pandas as pd
        row = {col: 0.0 for col in RF_FEATURE_COLS}
        row["temperature"]          = float(temperature or 0)
        row["humidity"]             = float(humidity or 63)
        row["methane"]              = float(methane or 0)
        row["co2"]                  = float(co2 or 400)
        row["storage_days"]         = float(storage_days or 0)
        row["temp_humidity_index"]  = float(temp_humidity_index or 0)
        row["gas_spoilage_score"]   = float(gas_spoilage_score or 0)
        row["methane_co2_ratio"]    = float(methane_co2_ratio or 0)
        row["temp_deviation"]       = float(temp_deviation or 0)
        row["storage_risk_score"]   = float(storage_risk_score or 0)
        food_key = (food_name or "chicken").lower()
        for food in ALL_FOODS_ORDERED:
            row[f"food_{food}"] = 1 if food == food_key else 0
        X = pd.DataFrame([row])
        probs = _rf_model.predict_proba(X)[0]
        return float(probs[2])   # index 2 = Spoiled class
    except:
        _, _, _, prob_s = _rule_based_predict(storage_risk_score or 0.0)
        return prob_s


# Register as Spark UDFs
from pyspark.sql.types import IntegerType, FloatType as FT

predict_class_udf = F.udf(_ml_predict_row, IntegerType())
predict_prob_udf  = F.udf(_ml_prob_spoiled, FT())

log.info(f"   ML UDFs registered | Model loaded: {_model_loaded}")
log.info(f"   Source: {'MLflow RF model' if _model_loaded else 'Rule-based fallback'}")

# ═══════════════════════════════════════════════════════════════
#  STEP 8 — ENGINEER ML FEATURES (recomputed from clean data)
#  These override the pre-computed values from the producer
#  to ensure they are based on cleaned sensor readings
# ═══════════════════════════════════════════════════════════════
log.info("Engineering ML features from clean sensor readings...")

# Safe temp lookup map (for temp_deviation)
safe_temp_map = F.create_map(
    *[val for pair in
      [(F.lit(food), F.lit(temp)) for food, temp in FOOD_SAFE_TEMP.items()]
      for val in pair]
)

featured_df = clipped_df \
    .withColumn(
        # Feature 1: temperature × humidity interaction
        "temp_humidity_index",
        F.round(F.col("temperature") * F.col("humidity") / F.lit(100.0), 4)
    ) \
    .withColumn(
        # Feature 2: normalised gas spoilage signal (0–1)
        "gas_spoilage_score",
        F.round(
            F.least(
                F.lit(1.0),
                (F.col("methane") / F.lit(175.0) + F.col("co2") / F.lit(2400.0)) / F.lit(2.0)
            ), 4
        )
    ) \
    .withColumn(
        # Feature 3: methane to CO2 ratio (bacteria vs respiration signal)
        "methane_co2_ratio",
        F.round(
            F.when(F.col("co2") > 0,
                   F.col("methane") / F.col("co2"))
             .otherwise(F.lit(0.0)),
            6
        )
    ) \
    .withColumn(
        # Feature 4: degrees above food-specific safe temperature
        "temp_deviation",
        F.round(
            F.greatest(
                F.lit(0.0),
                F.col("temperature") - F.coalesce(
                    safe_temp_map[F.col("food_name")],
                    F.lit(6.0)     # default if food not in map
                )
            ), 2
        )
    ) \
    .withColumn(
        # Feature 5: weighted composite risk score (0–1)
        # Weights: temp_deviation 40%, gas_score 35%, storage 25%
        "storage_risk_score",
        F.round(
            F.lit(0.40) * F.least(F.lit(1.0), F.col("temp_deviation") / F.lit(25.0)) +
            F.lit(0.35) * F.col("gas_spoilage_score") +
            F.lit(0.25) * F.least(F.lit(1.0), F.col("storage_days") / F.lit(15.0)),
            4
        )
    ) \
    .withColumn(
        # Breach flag: 1 if temperature exceeds safe max for this food
        "temp_breach_flag",
        F.when(
            F.col("temperature") > F.coalesce(
                safe_temp_map[F.col("food_name").cast("string")], F.lit(6.0)
            ),
            F.lit(1)
        ).otherwise(F.lit(0))
    ) \
    .withColumn(
        # ── REAL ML PREDICTION (replaces hardcoded thresholds) ──────
        # predict_class_udf calls the trained Random Forest from MLflow.
        # Returns: 0 = Fresh, 1 = At Risk, 2 = Spoiled
        # Falls back to rule-based automatically if model not loaded.
        "ml_predicted_class",
        predict_class_udf(
            F.col("food_name"),      # categorical — UDF one-hot encodes it
            F.col("temperature"),    # raw sensor °C
            F.col("humidity"),       # raw sensor %
            F.col("methane"),        # raw sensor ppm
            F.col("co2"),            # raw sensor ppm
            F.col("storage_days"),   # how long in transit
            F.col("temp_humidity_index"),   # Feature 1
            F.col("gas_spoilage_score"),    # Feature 2
            F.col("methane_co2_ratio"),     # Feature 3
            F.col("temp_deviation"),        # Feature 4
            F.col("storage_risk_score"),    # Feature 5
        )
    ) \
    .withColumn(
        # Probability of Spoiled class from RF model.predict_proba()
        # Alert fires if prob_spoiled > 0.70 (configurable threshold)
        "prob_spoiled",
        predict_prob_udf(
            F.col("food_name"),
            F.col("temperature"),
            F.col("humidity"),
            F.col("methane"),
            F.col("co2"),
            F.col("storage_days"),
            F.col("temp_humidity_index"),
            F.col("gas_spoilage_score"),
            F.col("methane_co2_ratio"),
            F.col("temp_deviation"),
            F.col("storage_risk_score"),
        )
    ) \
    .withColumn(
        # Human-readable label derived from ML prediction (NOT hardcoded score)
        "risk_level",
        F.when(F.col("ml_predicted_class") == F.lit(2), F.lit("CRITICAL"))
         .when(F.col("ml_predicted_class") == F.lit(1), F.lit("WARNING"))
         .otherwise(F.lit("SAFE"))
    )



# ═══════════════════════════════════════════════════════════════
#  STEP 9 — ADD PARTITION COLUMNS
#  Parquet files partitioned by year/month/day/food_name
#  Enables fast Spark SQL queries like:
#    WHERE year=2025 AND month=11 AND food_name='Chicken'
# ═══════════════════════════════════════════════════════════════
log.info("Adding partition columns (year / month / day / hour)...")

final_df = featured_df \
    .withColumn("ingested_at", F.current_timestamp()) \
    .withColumn(
        "_safe_ts",
        F.coalesce(F.col("event_time"), F.col("kafka_timestamp"), F.current_timestamp())
    ) \
    .withColumn("year",  F.coalesce(F.year(F.col("_safe_ts")),  F.lit(2026))) \
    .withColumn("month", F.coalesce(F.month(F.col("_safe_ts")), F.lit(1))) \
    .withColumn("day",   F.coalesce(F.dayofmonth(F.col("_safe_ts")), F.lit(1))) \
    .withColumn("hour",  F.coalesce(F.hour(F.col("_safe_ts")),  F.lit(0))) \
    .drop("_safe_ts")

# Select final column order (clean, no duplicates)
PROCESSED_COLS = [
    # Identifiers
    "event_id", "truck_id", "event_time", "ingested_at",
    # Location
    "gps_lat", "gps_lon",
    # Raw sensor readings (cleaned)
    "food_name", "temperature", "humidity", "methane", "co2", "storage_days",
    # Engineered features
    "temp_humidity_index", "gas_spoilage_score", "methane_co2_ratio",
    "temp_deviation", "storage_risk_score",
    # Derived flags
    "temp_breach_flag", "ml_predicted_class", "prob_spoiled", "risk_level",
    # Target variable
    "spoiled",
    # Kafka metadata (useful for debugging)
    "kafka_partition", "kafka_offset",
    # Partition columns (last — Spark convention)
    "year", "month", "day", "hour",
]

output_df = final_df.select(PROCESSED_COLS)


# ═══════════════════════════════════════════════════════════════
#  STEP 10 — ALERT STREAM
#  Separate stream: only rows where risk_level is CRITICAL
#  Sink: HDFS /alerts  (also Parquet, read by Airflow alert DAG)
# ═══════════════════════════════════════════════════════════════
log.info("Preparing alert stream (CRITICAL risk events only)...")

# Alert threshold: fire when ML model says prob_spoiled > 0.70
# OR when temperature physically breaches the safe max for that food
ALERT_THRESHOLD = 0.70

alert_df = output_df.filter(
    (F.col("prob_spoiled") >= ALERT_THRESHOLD) |   # ML model confident
    (F.col("ml_predicted_class") == F.lit(2))  |   # ML says Spoiled
    (F.col("temp_breach_flag") == 1)               # physical breach
).withColumn(
    "alert_type",
    F.when(F.col("ml_predicted_class") == F.lit(2), F.lit("ML_SPOILED"))
     .when(F.col("prob_spoiled") >= ALERT_THRESHOLD,  F.lit("ML_HIGH_RISK"))
     .otherwise(F.lit("TEMP_BREACH"))
).withColumn(
    "alert_message",
    F.concat(
        F.lit("ALERT: "), F.col("truck_id"),
        F.lit(" | "), F.col("food_name"),
        F.lit(" | Temp="), F.col("temperature").cast("string"),
        F.lit("°C | ML_class="), F.col("ml_predicted_class").cast("string"),
        F.lit(" | Prob_Spoiled="), F.col("prob_spoiled").cast("string"),
        F.lit(" | Level="), F.col("risk_level")
    )
)


# ═══════════════════════════════════════════════════════════════
#  STEP 11 — WRITE RAW STREAM TO HDFS (before cleaning)
#  Good practice: keep raw data for reprocessing if model improves
# ═══════════════════════════════════════════════════════════════
log.info(f"Starting raw sink → {HDFS_RAW}")

raw_query = (
    kafka_df
    .selectExpr("CAST(value AS STRING) AS raw_json",
                "timestamp AS kafka_timestamp",
                "partition", "offset")
    .writeStream
    .format("parquet")
    .option("path",            HDFS_RAW)
    .option("checkpointLocation", CHECKPOINT_RAW)
    .partitionBy("partition")
    .trigger(processingTime=f"{TRIGGER_SECS} seconds")
    .outputMode("append")
    .start()
)
log.info("Raw sink started.")


# ═══════════════════════════════════════════════════════════════
#  STEP 12 — WRITE PROCESSED STREAM TO HDFS
#  Parquet, partitioned by year/month/day/hour
#  This is what Spark SQL batch jobs and Hive will query
# ═══════════════════════════════════════════════════════════════
log.info(f"Starting processed sink → {HDFS_PROCESSED}")

processed_query = (
    output_df
    .writeStream
    .format("parquet")
    .option("path",            HDFS_PROCESSED)
    .option("checkpointLocation", CHECKPOINT_PROC)
    .partitionBy("year", "month", "day")
    .trigger(processingTime=f"{TRIGGER_SECS} seconds")
    .outputMode("append")
    .start()
)
log.info("Processed sink started.")


# ═══════════════════════════════════════════════════════════════
#  STEP 13 — WRITE ALERT STREAM TO HDFS
#  Only CRITICAL events — read by Airflow alert DAG every 15 min
# ═══════════════════════════════════════════════════════════════
log.info(f"Starting alert sink → {HDFS_ALERTS}")

alert_query = (
    alert_df
    .writeStream
    .format("parquet")
    .option("path",            HDFS_ALERTS)
    .option("checkpointLocation", CHECKPOINT_ALRT)
    .partitionBy("year", "month", "day")
    .trigger(processingTime=f"{TRIGGER_SECS} seconds")
    .outputMode("append")
    .start()
)
log.info("Alert sink started.")


# ═══════════════════════════════════════════════════════════════
#  STEP 14 — CONSOLE SINK (progress monitoring)
#  Shows a sample of processed rows every batch
#  Remove in production — only for dev/debugging
# ═══════════════════════════════════════════════════════════════
console_query = (
    output_df.select(
        "truck_id", "food_name", "temperature",
        "methane", "storage_risk_score", "risk_level",
        "temp_breach_flag", "event_time"
    )
    .writeStream
    .format("console")
    .option("truncate", "false")
    .option("numRows", "5")
    .trigger(processingTime=f"{TRIGGER_SECS} seconds")
    .outputMode("append")
    .start()
)


# ═══════════════════════════════════════════════════════════════
#  STEP 15 — AWAIT TERMINATION
#  Keeps all queries running until manually stopped (Ctrl+C)
#  or Airflow sends a stop signal
# ═══════════════════════════════════════════════════════════════
log.info("=" * 55)
log.info("  All streaming queries running. Pipeline is LIVE.")
log.info(f"  Raw sink      → {HDFS_RAW}")
log.info(f"  Processed sink→ {HDFS_PROCESSED}")
log.info(f"  Alert sink    → {HDFS_ALERTS}")
log.info(f"  Trigger every : {TRIGGER_SECS} seconds")
log.info(f"  Watermark     : {WATERMARK_MINS}")
log.info("  Press Ctrl+C to stop gracefully.")
log.info("=" * 55)

# Wait for all queries — stops only when all fail or are stopped
spark.streams.awaitAnyTermination()