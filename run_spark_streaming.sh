#!/bin/bash
# =============================================================
#  Cold Chain — Spark Streaming Job Launcher
#  Run this from inside cold-chain-project/ folder
#
#  Option A: Submit via Docker exec (recommended)
#  Option B: Submit directly if Spark is installed locally
# =============================================================

echo "=================================================="
echo "  Cold Chain — Spark Streaming Job"
echo "=================================================="

# ── Option A: Run INSIDE the Spark master container ──────────
# This is the correct way when using Docker Compose
echo ""
echo "Submitting job to Spark cluster inside Docker..."
echo ""

docker exec spark-master spark-submit master spark://spark-master:7077 packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 conf spark.sql.shuffle.partitions=8 conf spark.streaming.stopGracefullyOnShutdown=true conf spark.executor.memory=1g conf spark.driver.memory=1g num-executors 1 executor-cores 2 /opt/spark-apps/spark_streaming.py

# ── Option B: Run locally (if Spark installed on your machine) ─
# Uncomment lines below and comment out Option A above
#
# spark-submit \
#   --master local[2] \
#   --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 \
#   --conf spark.sql.shuffle.partitions=4 \
#   spark_streaming.py
