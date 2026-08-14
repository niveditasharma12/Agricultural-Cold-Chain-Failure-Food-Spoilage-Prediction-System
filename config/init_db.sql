-- =============================================================
--  Cold Chain Project — Postgres Init Script
--  Runs automatically when Postgres container starts
--  Creates the alerts table used by Airflow alert DAG
-- =============================================================

-- Alerts table (stores every spoilage alert fired)
CREATE TABLE IF NOT EXISTS cold_chain_alerts (
    id              SERIAL PRIMARY KEY,
    event_id        VARCHAR(20),
    truck_id        VARCHAR(10)     NOT NULL,
    food_name       VARCHAR(50),
    temperature     FLOAT,
    methane         FLOAT,
    co2             FLOAT,
    storage_days    FLOAT,
    spoilage_prob   FLOAT,          -- ML model output (0.0 to 1.0)
    predicted_class INT,            -- 0=Fresh 1=AtRisk 2=Spoiled
    risk_level      VARCHAR(10),    -- LOW / MEDIUM / HIGH / CRITICAL
    alert_message   TEXT,
    nearest_city    VARCHAR(100),   -- rerouting suggestion
    gps_lat         FLOAT,
    gps_lon         FLOAT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model metrics table (tracks ML model performance over time)
CREATE TABLE IF NOT EXISTS model_metrics (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(100),
    model_name      VARCHAR(100),
    f1_score        FLOAT,
    accuracy        FLOAT,
    precision_val   FLOAT,
    recall_val      FLOAT,
    trained_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pipeline run log (one row per Airflow DAG run)
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              SERIAL PRIMARY KEY,
    dag_id          VARCHAR(100),
    run_id          VARCHAR(200),
    status          VARCHAR(20),    -- success / failed / running
    rows_processed  INT,
    duration_secs   INT,
    run_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fleet status table (ONE row per truck — latest known reading).
-- Upserted every run of fleet_status_to_postgres.py from HDFS
-- /processed, so this is always the most recent real reading
-- Spark has processed for that truck — not simulated data.
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
    risk_level          VARCHAR(10),   -- SAFE / WATCH / WARNING / CRITICAL
    gps_lat             FLOAT,
    gps_lon             FLOAT,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sensor history table (rolling window, many rows per truck).
-- Populated from the same HDFS /processed read as fleet_status,
-- pruned to the last ~48h so it stays fast for interactive charts.
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
CREATE INDEX IF NOT EXISTS idx_sensor_history_truck_time
    ON cold_chain_sensor_history (truck_id, event_time);

-- Insert test row to verify connection works
INSERT INTO cold_chain_alerts (truck_id, food_name, temperature, risk_level, alert_message)
VALUES ('TRK-000', 'TEST', 0.0, 'LOW', 'DB initialised successfully — cold chain project ready');
