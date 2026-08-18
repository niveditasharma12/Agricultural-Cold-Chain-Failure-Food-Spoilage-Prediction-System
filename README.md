# Cold Chain Food Spoilage Prediction

A real-time IoT streaming and machine-learning pipeline that monitors refrigerated trucks in transit and predicts food spoilage before it happens — end to end, from a Kafka sensor stream to a live operator dashboard.

Sensors on each truck report temperature, humidity, methane, CO2, and GPS. Spark Structured Streaming cleans and engineers features from that stream in real time; a Random Forest classifier and an LSTM regressor (retrained nightly, tracked in MLflow) predict spoilage risk and remaining shelf life; critical breaches are alerted through Postgres, Grafana, and a Streamlit dashboard.

## Contents
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Getting started](#getting-started)
- [Services & ports](#services--ports)
- [Airflow DAGs](#airflow-dags)
- [Dashboard](#dashboard)
- [Known limitations](#known-limitations)

## Architecture

```
Truck Sensors ──▶ Kafka ──▶ Spark Structured Streaming ──▶ HDFS Data Lake
(temp/humidity/                topic: sensor-raw          /raw  /processed  /alerts
 methane/CO2/GPS)                     │
                                       ▼
                              Spark Batch ETL ──▶ HDFS /features ──▶ ML Training ──▶ MLflow Registry
                              (15-min rollup +                       (Random Forest        │
                               nightly rebuild)                       + LSTM, nightly)      ▼
                                                                                    FastAPI Model API
                                                                              /predict/spoilage
                                                                              /predict/shelf_life
                              HDFS /alerts ──▶ PostgreSQL ──▶ Grafana + Streamlit dashboard
```

Orchestrated end-to-end by **Apache Airflow**:

| DAG | Schedule | Does |
|---|---|---|
| `ingest_dag` | every 5 min | Kicks off `kafka_producer.py` |
| `transform_dag` | every 15 min | Health-checks the streaming job / rollups |
| `feature_dag` | nightly, 1 AM | Full batch feature rebuild |
| `train_model_dag` | nightly, 2 AM | Retrains RF + LSTM, logs to MLflow, auto-promotes the better model |
| `alert_dag` | every 15 min | Syncs new alerts + live fleet/sensor-history data to Postgres |
| `report_dag` | daily, 6 AM | Generates a daily summary report |

## Tech stack

| Technology | Used for |
|---|---|
| **Apache Kafka** | Distributed event bus for live sensor readings (`sensor-raw` topic) |
| **Apache Spark** (Structured Streaming + batch) | Cleans/validates/engineers features in real time; nightly batch rebuilds |
| **Hadoop HDFS** | Distributed storage for `/raw`, `/processed`, `/features`, `/alerts` |
| **Apache Airflow** | Orchestrates all 6 scheduled pipelines with retries |
| **MLflow** | Experiment tracking + model registry with metric-gated auto-promotion |
| **scikit-learn / TensorFlow (Keras)** | Random Forest spoilage classifier / LSTM shelf-life regressor |
| **FastAPI** | Serves live predictions from whichever model is currently `Production` in MLflow |
| **PostgreSQL** | Alerts, live fleet status, sensor history, model metrics |
| **Grafana + Streamlit** | Live dashboards for operators |
| **Docker Compose** | Runs all 16 services together |

## Project structure

```
.
├── docker-compose.yml          # all 16 services
├── augmented_cold_chain_dataset.csv   # 100,000-row training dataset
├── config/
│   ├── init_db.sql             # Postgres schema (alerts, fleet status, sensor history, model metrics)
│   └── grafana_datasources.yml
├── dags/                       # Airflow DAGs (see table above)
├── scripts/
│   ├── kafka_producer.py           # replays the dataset onto Kafka as a live sensor stream
│   ├── spark_streaming.py          # Structured Streaming: clean → feature-engineer → HDFS
│   ├── spark_batch_etl.py          # nightly batch rollup / feature rebuild
│   ├── train_random_forest.py      # spoilage classifier, logged to MLflow
│   ├── train_lstm.py               # shelf-life regressor, logged to MLflow
│   ├── promote_best_model.py       # metric-gated auto-promotion to MLflow "Production"
│   ├── food_shelf_life.py          # derives the shelf-life training target
│   ├── alert_to_postgres.py        # syncs HDFS /alerts → cold_chain_alerts table
│   ├── fleet_status_to_postgres.py # syncs HDFS /processed → live fleet status + sensor history
│   ├── serve_api.py                # FastAPI model-serving app
│   ├── dashboard.py                # Streamlit operator dashboard
│   └── daily_report.py
├── data/
│   └── mlruns/                 # MLflow local artifact/tracking store
└── notebooks/                  # exploratory analysis
```

## Dataset

`augmented_cold_chain_dataset.csv` — **100,000 rows × 18 columns**, no missing values.

| Column(s) | Description |
|---|---|
| `event_id`, `truck_id`, `timestamp` | Event identity, which truck, when |
| `gps_lat`, `gps_lon` | Truck location at the time of the reading |
| `Food_Name` | 15 cargo types |
| `Temperature`, `Humidity`, `Methane`, `CO2` | Raw sensor readings |
| `Storage_Days` | Days since the cargo was loaded |
| `temp_humidity_index`, `gas_spoilage_score`, `methane_co2_ratio`, `temp_deviation`, `storage_risk_score` | 5 engineered features used by both ML models |
| `Spoiled` | Target label: 0 = Fresh, 1 = At Risk, 2 = Spoiled |
| `source` | synthetic vs. real-sourced row |

25 trucks, 15 food types, roughly balanced across the 3 classes (40% / 35% / 25%).

## Getting started

**Requirements:** Docker + Docker Compose, ~8 GB RAM free for the stack.

```bash
# 1. Bring up every service
docker compose up -d

# 2. Start the live sensor stream (replays the dataset onto Kafka)
python scripts/kafka_producer.py

# 3. Streaming job is started automatically by the spark-streaming-job
#    container, but if you need to (re)run it manually:
docker compose up -d spark-streaming-job

# 4. Kick off a batch ETL run manually (otherwise runs nightly via Airflow)
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark-apps/spark_batch_etl.py

# 5. Confirm everything is up
docker ps -a
```

Then open:
- **Streamlit dashboard** → http://localhost:8501
- **Airflow** → http://localhost:8085 — trigger `alert_dag` at least once to populate live fleet/alert data
- **MLflow** → http://localhost:5000 — trigger `train_model_dag` at least once to get a `Production` model
- **Model API docs** → http://localhost:8000/docs

## Services & ports

| Service | Container | URL / Port | Purpose |
|---|---|---|---|
| Kafka broker | `kafka` | `localhost:9092` | Sensor event bus (`sensor-raw`, `gps-stream`, `weather-feed`) |
| Zookeeper | `zookeeper` | `localhost:2181` | Kafka coordination |
| HDFS NameNode UI | `namenode` | `localhost:9870` | Browse the data lake |
| HDFS DataNode UI | `datanode` | `localhost:9864` | — |
| Spark Master UI | `spark-master` | `localhost:8080` | Job monitoring |
| Spark Worker UI | `spark-worker` | `localhost:8081` | — |
| Postgres | `postgres` | `localhost:5432` | db `airflow`, user/pass `airflow`/`airflow` |
| Airflow Webserver | `airflow-webserver` | `localhost:8085` | DAG runs, schedules, retries, logs |
| MLflow UI | `mlflow` | `localhost:5000` | Experiments, metrics, model registry |
| Grafana | `grafana` | `localhost:3000` | admin/admin — live dashboards |
| Streamlit dashboard | `streamlit` | `localhost:8501` | Operator-facing UI |
| Model API (FastAPI) | `model-api` | `localhost:8000/docs` | Live predictions + Swagger UI |

## Airflow DAGs

See the [architecture](#architecture) table above for schedule and purpose of each DAG. All DAGs use `BashOperator` to `docker exec` into `spark-master` and run the relevant script from `/opt/spark-apps` (mounted from `./scripts`).

## Dashboard

The Streamlit dashboard (`scripts/dashboard.py`) has 5 tabs:

- **Fleet monitor** — live map + status table of all trucks, sourced from `cold_chain_fleet_status` in Postgres
- **Sensor charts** — 24h reading history per truck, from `cold_chain_sensor_history`
- **Predict spoilage** — calls the live Model API directly
- **Alert history** — real alerts from `cold_chain_alerts`
- **Analytics** — model performance from `model_metrics`, plus fleet-wide breakdowns

All data comes from Postgres tables that the pipeline itself writes (`fleet_status_to_postgres.py` and `alert_to_postgres.py`, both run every 15 minutes by `alert_dag`) — nothing in the dashboard is simulated or hardcoded. If a table is empty, the dashboard says so rather than fabricating data.

