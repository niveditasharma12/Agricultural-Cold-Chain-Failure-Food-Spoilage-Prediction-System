import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
import time
import requests
import os
import psycopg2

# ── Helpers ───────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
    """Convert a '#rrggbb' hex color string into a valid 'rgba(r,g,b,a)' string."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Cold Chain Monitor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding: 1rem 1.5rem 1rem; }
  .metric-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
  }
  .metric-val { font-size: 28px; font-weight: 600; margin: 0; }
  .metric-lbl { font-size: 12px; color: #6c757d; margin: 0; }
  .truck-card {
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    background: white;
  }
  .risk-fresh   { border-left: 4px solid #28a745; }
  .risk-atrisk  { border-left: 4px solid #fd7e14; }
  .risk-spoiled { border-left: 4px solid #dc3545; }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
  }
  .badge-fresh   { background: #d4edda; color: #155724; }
  .badge-atrisk  { background: #fff3cd; color: #856404; }
  .badge-spoiled { background: #f8d7da; color: #721c24; }
  div[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 12px 16px;
  }
  .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 500; }
  .alert-critical { background: #fff5f5; border-left: 4px solid #e53e3e; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
  .alert-warning  { background: #fffaf0; border-left: 4px solid #ed8936; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
  .stButton > button { border-radius: 8px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  REFERENCE DATA (static lookups, not sensor data)
# ═══════════════════════════════════════════════════════════════

FOOD_TYPES = ['Chicken','Spinach','Beef','Cheese','Mushroom','Yogurt',
              'Bread','Eggs','Potato','Tomato','Fish','Orange',
              'Strawberry','Milk','Apple']

# Per-food safe temp max (°C)
SAFE_TEMP = {
    'Chicken':4,'Fish':3,'Beef':4,'Milk':5,'Yogurt':5,
    'Cheese':6,'Eggs':6,'Spinach':5,'Mushroom':6,'Bread':23,
    'Potato':10,'Tomato':10,'Apple':7,'Orange':8,'Strawberry':5
}

# Maps the real-time risk_level spark_streaming.py computes for
# every reading onto the 0/1/2 classes the rest of this dashboard
# displays as Fresh / At Risk / Spoiled.
RISK_LEVEL_TO_CLASS = {"SAFE": 0, "WATCH": 0, "WARNING": 1, "CRITICAL": 2}


# ═══════════════════════════════════════════════════════════════
#  DATABASE CONNECTION
#  Reads straight from the Postgres tables the pipeline writes:
#    cold_chain_fleet_status    ← fleet_status_to_postgres.py (Spark, every 15 min)
#    cold_chain_sensor_history  ← fleet_status_to_postgres.py (Spark, every 15 min)
#    cold_chain_alerts          ← alert_to_postgres.py        (Spark, every 15 min)
#    model_metrics              ← train_random_forest.py      (Airflow, nightly)
#  Override host/db/user/password via env vars if your deployment differs.
# ═══════════════════════════════════════════════════════════════
PG_CONFIG = {
    "host":     os.environ.get("PG_HOST", "postgres"),
    "port":     os.environ.get("PG_PORT", "5432"),
    "dbname":   os.environ.get("PG_DB", "airflow"),
    "user":     os.environ.get("PG_USER", "airflow"),
    "password": os.environ.get("PG_PASSWORD", "airflow"),
}

def _run_query(sql, params=None):
    """Runs a read-only query and returns a DataFrame, or None if Postgres is unreachable."""
    conn = None
    try:
        conn = psycopg2.connect(**PG_CONFIG, connect_timeout=3)
        return pd.read_sql(sql, conn, params=params)
    except Exception as e:
        st.session_state["_db_error"] = str(e)
        return None
    finally:
        if conn is not None:
            conn.close()


@st.cache_data(ttl=10)
def get_fleet_data():
    """
    Latest known reading per truck, straight from
    cold_chain_fleet_status — upserted by fleet_status_to_postgres.py
    from HDFS /processed. Empty (not fabricated) if the pipeline
    hasn't synced anything yet.
    """
    df = _run_query("""
        SELECT truck_id, food_name, event_time, temperature, humidity,
               methane, co2, storage_days, temp_deviation,
               gas_spoilage_score, storage_risk_score, risk_level,
               gps_lat, gps_lon
        FROM cold_chain_fleet_status
        ORDER BY truck_id;
    """)
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            'truck_id','food_name','temperature','humidity','methane','co2',
            'storage_days','temp_deviation','gas_spoilage_score','risk_score',
            'spoiled_class','lat','lon','last_update'
        ])
    df = df.rename(columns={
        "storage_risk_score": "risk_score",
        "gps_lat": "lat",
        "gps_lon": "lon",
    })
    df["spoiled_class"] = df["risk_level"].map(RISK_LEVEL_TO_CLASS).fillna(0).astype(int)
    df["last_update"] = pd.to_datetime(df["event_time"]).dt.strftime("%H:%M:%S")
    return df


@st.cache_data(ttl=30)
def get_sensor_history(truck_id, hours=24):
    """
    Real per-truck reading history from cold_chain_sensor_history
    (populated from HDFS /processed, same table the fleet snapshot
    is derived from). Empty if this truck has no readings yet in
    the requested window — not backfilled with generated data.
    """
    df = _run_query("""
        SELECT event_time AS timestamp, temperature, humidity, methane, co2
        FROM cold_chain_sensor_history
        WHERE truck_id = %(truck_id)s
          AND event_time >= NOW() - INTERVAL '%(hours)s hours'
        ORDER BY event_time ASC;
    """, params={"truck_id": truck_id, "hours": hours})
    return df if df is not None else pd.DataFrame(columns=["timestamp","temperature","humidity","methane","co2"])


@st.cache_data(ttl=15)
def get_alert_history(n=40):
    """
    Real alerts fired by the pipeline, from cold_chain_alerts —
    populated by alert_to_postgres.py from HDFS /alerts (the
    CRITICAL / temp-breach stream spark_streaming.py produces).
    """
    df = _run_query("""
        SELECT created_at, truck_id, food_name, temperature, methane,
               risk_level, spoilage_prob
        FROM cold_chain_alerts
        WHERE truck_id != 'TRK-000'
        ORDER BY created_at DESC
        LIMIT %(n)s;
    """, params={"n": n})
    if df is None or df.empty:
        return pd.DataFrame(columns=['time','truck','food','temp','methane','risk','probability','action'])
    out = pd.DataFrame({
        'time':        pd.to_datetime(df['created_at']).dt.strftime("%Y-%m-%d %H:%M"),
        'truck':       df['truck_id'],
        'food':        df['food_name'],
        'temp':        df['temperature'],
        'methane':     df['methane'],
        'risk':        df['risk_level'].apply(lambda r: '🔴 CRITICAL' if r == 'CRITICAL' else '🟠 WARNING'),
        'probability': df['spoilage_prob'].apply(lambda p: f"{p:.1%}" if pd.notna(p) else "—"),
        'action':      df['risk_level'].apply(lambda r: 'Reroute dispatched' if r == 'CRITICAL' else 'Alert sent'),
    })
    return out.sort_values('time', ascending=False).reset_index(drop=True)


@st.cache_data(ttl=20)
def get_model_metrics():
    """Latest logged metrics per model from model_metrics (written by train_random_forest.py / train_lstm.py)."""
    df = _run_query("""
        SELECT DISTINCT ON (model_name) model_name, f1_score, accuracy,
               precision_val, recall_val, trained_at
        FROM model_metrics
        ORDER BY model_name, trained_at DESC;
    """)
    return df if df is not None else pd.DataFrame()


@st.cache_data(ttl=15)
def get_pipeline_health():
    """Real freshness/reachability checks — no hardcoded green checkmarks."""
    health = {"db_ok": False, "fleet_table_ready": False, "fleet_last_update": None,
              "fleet_rows": 0, "alerts_last_update": None, "model_api": None}
    ping_df = _run_query("SELECT 1 AS ok;")
    health["db_ok"] = ping_df is not None
    fleet_df = _run_query("SELECT MAX(updated_at) AS ts, COUNT(*) AS n FROM cold_chain_fleet_status;")
    if fleet_df is not None and not fleet_df.empty:
        health["fleet_table_ready"] = True
        health["fleet_last_update"] = fleet_df["ts"].iloc[0]
        health["fleet_rows"] = int(fleet_df["n"].iloc[0])
    alerts_df = _run_query("SELECT MAX(created_at) AS ts FROM cold_chain_alerts WHERE truck_id != 'TRK-000';")
    if alerts_df is not None and not alerts_df.empty:
        health["alerts_last_update"] = alerts_df["ts"].iloc[0]
    try:
        resp = requests.get(f"{MODEL_API_URL}/health", timeout=3)
        resp.raise_for_status()
        health["model_api"] = resp.json()
    except Exception:
        health["model_api"] = None
    return health

# model-api's internal Docker network address (override via env var if needed)
MODEL_API_URL = os.environ.get("MODEL_API_URL", "http://model-api:8000")


def _rule_based_predict_spoilage(food, temp, humidity, methane, co2, storage_days):
    """Fallback heuristic — used only if model-api is unreachable."""
    safe_max = SAFE_TEMP.get(food, 6)
    temp_dev = max(0, temp - safe_max)
    gas_score = min(1.0, (methane/175 + co2/2400) / 2)
    day_score = min(1.0, storage_days / 15)
    raw = 0.38*min(1.0, temp_dev/22) + 0.35*gas_score + 0.27*day_score
    if temp <= safe_max and methane < 10 and co2 < 500 and storage_days < 3:
        raw = min(raw, 0.18)
    if temp > safe_max + 8 or methane > 80 or co2 > 1400:
        raw = max(raw, 0.72)
    p0 = max(0, 1 - raw*2.2)
    p2 = max(0, raw*1.8 - 0.3) if raw > 0.5 else 0
    p1 = max(0, 1 - p0 - p2)
    total = p0+p1+p2 or 1
    p0,p1,p2 = p0/total, p1/total, p2/total
    cls = 0 if p0 >= p1 and p0 >= p2 else (1 if p1 >= p2 else 2)
    return cls, round(p0,3), round(p1,3), round(p2,3), "rule-based fallback"


def predict_spoilage(food, temp, humidity, methane, co2, storage_days):
    """
    Calls the real trained Random Forest model via model-api.
    Computes the same 5 engineered features spark_streaming.py
    uses, so inputs match exactly what the model was trained on.
    Falls back to a rule-based heuristic if model-api is
    unreachable or hasn't loaded a Production model yet, so the
    dashboard never hard-crashes.
    """
    safe_max = SAFE_TEMP.get(food, 6)
    temp_humidity_index = round(temp * humidity / 100.0, 4)
    gas_spoilage_score = round(min(1.0, (methane/175.0 + co2/2400.0) / 2.0), 4)
    methane_co2_ratio = round(methane / co2, 6) if co2 > 0 else 0.0
    temp_deviation = round(max(0.0, temp - safe_max), 2)
    storage_risk_score = round(
        0.40 * min(1.0, temp_deviation / 25.0) +
        0.35 * gas_spoilage_score +
        0.25 * min(1.0, storage_days / 15.0),
        4
    )

    payload = {
        "food_name": food,
        "temperature": temp,
        "humidity": humidity,
        "methane": methane,
        "co2": co2,
        "storage_days": storage_days,
        "temp_humidity_index": temp_humidity_index,
        "gas_spoilage_score": gas_spoilage_score,
        "methane_co2_ratio": methane_co2_ratio,
        "temp_deviation": temp_deviation,
        "storage_risk_score": storage_risk_score,
    }

    try:
        resp = requests.post(f"{MODEL_API_URL}/predict/spoilage", json=payload, timeout=5)
        resp.raise_for_status()
        result = resp.json()
        probs = result["probabilities"]
        p0 = probs.get("Fresh", 0.0)
        p1 = probs.get("At Risk", 0.0)
        p2 = probs.get("Spoiled", 0.0)
        return result["predicted_class"], round(p0, 3), round(p1, 3), round(p2, 3), "ML model (Random Forest)"
    except Exception as e:
        st.warning(f"⚠️ Could not reach the live model ({e}) — showing a rule-based estimate instead.")
        return _rule_based_predict_spoilage(food, temp, humidity, methane, co2, storage_days)


try:
    from food_shelf_life import compute_remaining_shelf_life_hours
except ImportError:
    compute_remaining_shelf_life_hours = None  # fallback disabled if module missing


def predict_shelf_life(truck_id, truck_info, hist):
    """
    Calls model-api's LSTM endpoint using the truck's last 10
    sensor readings. Falls back to the same heuristic formula
    used at training time (food_shelf_life.py) if model-api is
    unreachable, so this never hard-crashes the dashboard.
    """
    food = truck_info['food_name']
    storage_days = truck_info['storage_days']
    safe_max = SAFE_TEMP.get(food, 6)

    if hist.empty or len(hist) < 10:
        return None, "insufficient history"

    last10 = hist.tail(10)
    readings = []
    for _, row in last10.iterrows():
        temp, humidity, methane, co2 = row['temperature'], row['humidity'], row['methane'], row['co2']
        readings.append({
            "temperature": temp,
            "humidity": humidity,
            "methane": methane,
            "co2": co2,
            "temp_humidity_index": round(temp * humidity / 100.0, 4),
            "gas_spoilage_score": round(min(1.0, (methane/175.0 + co2/2400.0) / 2.0), 4),
            "methane_co2_ratio": round(methane / co2, 6) if co2 > 0 else 0.0,
            "temp_deviation": round(max(0.0, temp - safe_max), 2),
            "storage_risk_score": round(
                0.40 * min(1.0, max(0.0, temp - safe_max) / 25.0) +
                0.35 * min(1.0, (methane/175.0 + co2/2400.0) / 2.0) +
                0.25 * min(1.0, storage_days / 15.0),
                4
            ),
        })

    try:
        resp = requests.post(f"{MODEL_API_URL}/predict/shelf_life", json={"readings": readings}, timeout=5)
        resp.raise_for_status()
        return resp.json()["predicted_remaining_hours"], "ML model (LSTM)"
    except Exception as e:
        if compute_remaining_shelf_life_hours is not None:
            fallback_hours = compute_remaining_shelf_life_hours(food, storage_days, readings[-1]["storage_risk_score"])
            return round(max(0, fallback_hours), 1), f"heuristic fallback ({e})"
        return None, f"unavailable ({e})"


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚚 Cold Chain Monitor")
    st.caption("Agricultural Food Spoilage Prediction System")
    st.divider()

    auto_refresh = st.toggle("Auto-refresh (10s)", value=False)
    if auto_refresh:
        time.sleep(10)
        st.rerun()

    st.markdown("**Pipeline status**")
    health = get_pipeline_health()

    if not health["db_ok"]:
        st.error("Postgres unreachable — check PG_HOST/PG_PORT env vars and that the postgres container is up.")
    elif not health["fleet_table_ready"]:
        st.warning("Connected to Postgres, but cold_chain_fleet_status doesn't exist yet — alert_dag hasn't run yet.")
    else:
        fresh_mins = None
        if health["fleet_last_update"] is not None:
            fresh_mins = (datetime.now() - health["fleet_last_update"].to_pydatetime().replace(tzinfo=None)).total_seconds() / 60
        if fresh_mins is None:
            st.warning("Streaming — no fleet data synced yet")
        elif fresh_mins <= 20:
            st.success(f"Streaming ✓ ({health['fleet_rows']} trucks, updated {fresh_mins:.0f}m ago)")
        else:
            st.warning(f"Streaming — stale ({fresh_mins:.0f}m since last sync)")

        if health["alerts_last_update"] is not None:
            st.caption(f"Last alert: {health['alerts_last_update'].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.caption("No alerts fired yet")

    api = health["model_api"]
    if api is None:
        st.error("Model API unreachable")
    elif api.get("rf_model_loaded") and api.get("lstm_model_loaded"):
        st.success("Model API ✓ (RF + LSTM loaded)")
    else:
        missing = [m for m, loaded in [("RF", api.get("rf_model_loaded")), ("LSTM", api.get("lstm_model_loaded"))] if not loaded]
        st.warning(f"Model API up, but no Production model yet: {', '.join(missing)}")

    st.divider()
    st.markdown("**Filter fleet**")
    risk_filter = st.multiselect(
        "Risk level", ["Fresh","At Risk","Spoiled"],
        default=["Fresh","At Risk","Spoiled"]
    )
    food_filter = st.multiselect("Food type", FOOD_TYPES, default=FOOD_TYPES)
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh now", width='stretch'):
        st.cache_data.clear()
        st.rerun()

# ═══════════════════════════════════════════════════════════════
#  HEADER + SUMMARY METRICS
# ═══════════════════════════════════════════════════════════════
st.markdown("## 🌡️ Cold Chain Spoilage Detection Dashboard")
st.caption("Real-time monitoring of refrigerated trucks | Kafka → Spark → Hadoop → ML → Alerts")

fleet = get_fleet_data()

if fleet.empty:
    st.warning(
        "No fleet data yet. This dashboard now reads live from Postgres "
        "(`cold_chain_fleet_status`), which `fleet_status_to_postgres.py` "
        "populates from HDFS `/processed` every 15 minutes via `alert_dag`. "
        "Make sure `kafka_producer.py` is sending events, `spark_streaming.py` "
        "is running, and `alert_dag` has completed at least one run — then "
        "refresh this page."
    )
    st.stop()

RISK_MAP = {0:"Fresh", 1:"At Risk", 2:"Spoiled"}
fleet['risk_label'] = fleet['spoiled_class'].map(RISK_MAP)

# Apply sidebar filters
mask = (fleet['risk_label'].isin(risk_filter)) & (fleet['food_name'].isin(food_filter))
fleet_f = fleet[mask]

total    = len(fleet)
n_fresh  = (fleet['spoiled_class']==0).sum()
n_atrisk = (fleet['spoiled_class']==1).sum()
n_spoil  = (fleet['spoiled_class']==2).sum()
avg_risk = fleet['risk_score'].mean()

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Total trucks",   total)
c2.metric("🟢 Fresh",       n_fresh,  delta=f"{n_fresh/total:.0%}")
c3.metric("🟠 At risk",     n_atrisk, delta=f"{n_atrisk/total:.0%}", delta_color="inverse")
c4.metric("🔴 Spoiled",     n_spoil,  delta=f"{n_spoil/total:.0%}",  delta_color="inverse")
c5.metric("Avg risk score", f"{avg_risk:.2f}", delta="0–1 scale", delta_color="off")

st.divider()

# ═══════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚛 Fleet monitor", "📈 Sensor charts",
    "🔮 Predict spoilage", "🚨 Alert history", "📊 Analytics"
])

# ══════════════════════════════════════
#  TAB 1 — FLEET MONITOR
# ══════════════════════════════════════
with tab1:
    col_map, col_table = st.columns([1.1, 0.9])

    with col_map:
        st.markdown("#### Live truck locations")
        color_map = {0:"#28a745", 1:"#fd7e14", 2:"#dc3545"}
        fleet_f['color'] = fleet_f['spoiled_class'].map(color_map)
        fleet_f['size']  = fleet_f['risk_score'] * 20 + 8
        fleet_f['label'] = fleet_f.apply(
            lambda r: f"{r['truck_id']} | {r['food_name']} | {r['risk_label']}<br>"
                      f"Temp: {r['temperature']}°C | Methane: {r['methane']} ppm<br>"
                      f"Risk: {r['risk_score']:.2f}", axis=1)
        fig_map = px.scatter_map(
            fleet_f, lat="lat", lon="lon",
            color="risk_label",
            color_discrete_map={"Fresh":"#28a745","At Risk":"#fd7e14","Spoiled":"#dc3545"},
            size="size", size_max=18,
            hover_name="truck_id",
            hover_data={"food_name":True,"temperature":True,"methane":True,"risk_score":":.2f","lat":False,"lon":False,"size":False},
            zoom=4, center={"lat":20.5,"lon":78.9},
            map_style="open-street-map",
            height=420,
        )
        fig_map.update_layout(margin=dict(l=0,r=0,t=0,b=0), legend_title="Risk level")
        st.plotly_chart(fig_map, width='stretch')

    with col_table:
        st.markdown("#### Truck status table")
        risk_order = {"Spoiled":0,"At Risk":1,"Fresh":2}
        display = fleet_f.sort_values('spoiled_class', ascending=False)[
            ['truck_id','food_name','temperature','methane','co2',
             'storage_days','risk_score','risk_label','last_update']
        ].copy()
        display.columns = ['Truck','Food','Temp°C','CH₄ ppm',
                           'CO₂ ppm','Days','Risk','Status','Updated']

        def style_risk(val):
            if val=='Spoiled': return 'background-color:#f8d7da;color:#721c24;font-weight:600'
            if val=='At Risk': return 'background-color:#fff3cd;color:#856404;font-weight:600'
            return 'background-color:#d4edda;color:#155724;font-weight:600'

        def style_temp(val):
            if val > 15: return 'color:#dc3545;font-weight:600'
            if val > 8:  return 'color:#fd7e14;font-weight:600'
            return 'color:#28a745'

        styled = display.style\
            .map(style_risk, subset=['Status'])\
            .map(style_temp, subset=['Temp°C'])\
            .format({'Temp°C':'{:.1f}','CH₄ ppm':'{:.1f}','CO₂ ppm':'{:.0f}','Days':'{:.1f}','Risk':'{:.2f}'})
        st.dataframe(styled, height=400, width='stretch')

# ══════════════════════════════════════
#  TAB 2 — SENSOR CHARTS
# ══════════════════════════════════════
with tab2:
    st.markdown("#### 24-hour sensor history")
    selected_truck = st.selectbox("Select truck", fleet['truck_id'].sort_values().tolist(), key="chart_truck")
    truck_info = fleet[fleet['truck_id']==selected_truck].iloc[0]

    ci1,ci2,ci3,ci4 = st.columns(4)
    ci1.metric("Food type",   truck_info['food_name'])
    ci2.metric("Current temp", f"{truck_info['temperature']}°C",
               delta=f"Safe max: {SAFE_TEMP.get(truck_info['food_name'],6)}°C", delta_color="off")
    ci3.metric("Methane",      f"{truck_info['methane']} ppm")
    ci4.metric("Risk score",   f"{truck_info['risk_score']:.2f}",
               delta=truck_info['risk_label'],
               delta_color="normal" if truck_info['spoiled_class']==0 else "inverse")

    hist = get_sensor_history(selected_truck)
    if not hist.empty:
        fig = make_subplots(rows=2, cols=2,
            subplot_titles=("Temperature (°C)","Humidity (%)","Methane (ppm)","CO₂ (ppm)"),
            vertical_spacing=0.14, horizontal_spacing=0.1)
        safe_t = SAFE_TEMP.get(truck_info['food_name'], 6)
        color_t = "#dc3545" if truck_info['temperature'] > safe_t+5 else "#fd7e14" if truck_info['temperature'] > safe_t else "#28a745"

        fig.add_trace(go.Scatter(x=hist['timestamp'],y=hist['temperature'],
            mode='lines',name='Temperature',line=dict(color=color_t,width=2),fill='tozeroy',
            fillcolor=hex_to_rgba(color_t)), row=1,col=1)
        fig.add_hline(y=safe_t, line_dash="dash", line_color="#6c757d",
            annotation_text=f"Safe max {safe_t}°C", row=1, col=1)

        fig.add_trace(go.Scatter(x=hist['timestamp'],y=hist['humidity'],
            mode='lines',name='Humidity',line=dict(color='#4a90d9',width=2),fill='tozeroy',
            fillcolor='rgba(74,144,217,0.08)'), row=1,col=2)

        meth_color = "#dc3545" if truck_info['methane']>80 else "#fd7e14" if truck_info['methane']>20 else "#28a745"
        fig.add_trace(go.Scatter(x=hist['timestamp'],y=hist['methane'],
            mode='lines',name='Methane',line=dict(color=meth_color,width=2),fill='tozeroy',
            fillcolor=hex_to_rgba(meth_color)), row=2,col=1)

        co2_color = "#dc3545" if truck_info['co2']>1400 else "#fd7e14" if truck_info['co2']>700 else "#28a745"
        fig.add_trace(go.Scatter(x=hist['timestamp'],y=hist['co2'],
            mode='lines',name='CO₂',line=dict(color=co2_color,width=2),fill='tozeroy',
            fillcolor=hex_to_rgba(co2_color)), row=2,col=2)

        fig.update_layout(height=480, showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
        st.plotly_chart(fig, width='stretch')
    else:
        st.info(f"No sensor history synced for {selected_truck} yet in the last {24}h window — check back after the pipeline has run a few cycles.")

    # Risk gauge
    st.markdown("#### Composite risk score")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(truck_info['risk_score'],3),
        domain={'x':[0,1],'y':[0,1]},
        title={'text': f"{selected_truck} — {truck_info['food_name']}", 'font':{'size':16}},
        delta={'reference':0.5,'increasing':{'color':'#dc3545'},'decreasing':{'color':'#28a745'}},
        gauge={
            'axis':{'range':[0,1],'tickwidth':1,'tickcolor':'#6c757d'},
            'bar':{'color': "#dc3545" if truck_info['risk_score']>0.65 else "#fd7e14" if truck_info['risk_score']>0.35 else "#28a745"},
            'bgcolor':'white',
            'steps':[
                {'range':[0,0.35],'color':'#d4edda'},
                {'range':[0.35,0.65],'color':'#fff3cd'},
                {'range':[0.65,1],'color':'#f8d7da'},
            ],
            'threshold':{'line':{'color':'#343a40','width':3},'thickness':0.75,'value':0.7}
        }
    ))
    fig_gauge.update_layout(height=260, margin=dict(t=40,b=10,l=40,r=40))
    st.plotly_chart(fig_gauge, width='stretch')

    # Shelf-life prediction (LSTM)
    st.markdown("#### Estimated shelf life remaining")
    hours_left, sl_source = predict_shelf_life(selected_truck, truck_info, hist)
    if hours_left is None:
        st.info(f"Shelf-life estimate unavailable ({sl_source}).")
    else:
        days_left = hours_left / 24.0
        sl_col1, sl_col2 = st.columns([1, 2])
        sl_col1.metric("Hours remaining", f"{hours_left:.0f}h", delta=f"≈ {days_left:.1f} days")
        sl_col2.caption(f"Source: **{sl_source}** — based on {truck_info['food_name']}'s typical shelf life, "
                         f"this truck's last 10 sensor readings, and {truck_info['storage_days']:.1f} days in storage so far.")
        if hours_left < 12:
            st.error("⏰ Critical — very little shelf life remaining. Prioritize this shipment.")
        elif hours_left < 48:
            st.warning("⏰ Getting low — plan for near-term delivery or rerouting.")

# ══════════════════════════════════════
#  TAB 3 — PREDICTION FORM
# ══════════════════════════════════════
with tab3:
    st.markdown("#### Spoilage prediction — enter sensor readings")
    st.caption("Enter current sensor values to get an instant ML prediction from the trained Random Forest model.")

    with st.form("predict_form"):
        fc1, fc2 = st.columns(2)
        with fc1:
            p_food    = st.selectbox("Food type", FOOD_TYPES)
            p_temp    = st.slider("Temperature (°C)", -5.0, 35.0, 10.0, 0.1)
            p_humid   = st.slider("Humidity (%)",      30.0, 100.0, 63.0, 0.5)
        with fc2:
            p_methane = st.slider("Methane (ppm)",      0.0, 180.0, 5.0,  0.5)
            p_co2     = st.slider("CO₂ (ppm)",        300.0,2400.0,410.0, 10.0)
            p_days    = st.slider("Storage days",       0.0,  15.0,  2.0,  0.1)

        safe_t_preview = SAFE_TEMP.get(p_food, 6)
        st.info(f"Safe temperature max for **{p_food}**: {safe_t_preview}°C  |  "
                f"Current: {p_temp}°C  |  "
                f"Deviation: **{max(0, round(p_temp - safe_t_preview, 1))}°C above safe**")
        submitted = st.form_submit_button("🔮 Predict spoilage", width='stretch', type="primary")

    if submitted:
        cls, p0, p1, p2, source = predict_spoilage(p_food, p_temp, p_humid, p_methane, p_co2, p_days)
        labels = {0:"✅ FRESH — Safe to sell",1:"⚠️ AT RISK — Needs attention",2:"🚨 SPOILED — Do not sell"}
        colors = {0:"success", 1:"warning", 2:"error"}
        bg     = {0:"#d4edda",1:"#fff3cd",2:"#f8d7da"}
        tc     = {0:"#155724",1:"#856404",2:"#721c24"}

        st.caption(f"Prediction source: **{source}**")
        st.markdown(f"""
        <div style="background:{bg[cls]};color:{tc[cls]};border-radius:10px;
                    padding:16px 20px;text-align:center;margin:10px 0;
                    border: 2px solid {tc[cls]}30;">
          <div style="font-size:22px;font-weight:700;margin-bottom:4px">{labels[cls]}</div>
          <div style="font-size:13px;opacity:0.85">{p_food} | {p_temp}°C | Methane {p_methane} ppm | CO₂ {p_co2:.0f} ppm | {p_days} days</div>
        </div>""", unsafe_allow_html=True)

        pr1, pr2, pr3 = st.columns(3)
        pr1.metric("🟢 P(Fresh)",   f"{p0:.1%}", delta="class 0")
        pr2.metric("🟠 P(At Risk)", f"{p1:.1%}", delta="class 1")
        pr3.metric("🔴 P(Spoiled)", f"{p2:.1%}", delta="class 2")

        fig_prob = go.Figure(go.Bar(
            x=["Fresh (0)","At Risk (1)","Spoiled (2)"],
            y=[p0, p1, p2],
            marker_color=["#28a745","#fd7e14","#dc3545"],
            text=[f"{p0:.1%}",f"{p1:.1%}",f"{p2:.1%}"],
            textposition='outside',
        ))
        fig_prob.update_layout(
            title="Class probability distribution",
            yaxis=dict(range=[0,1.15], tickformat='.0%', showgrid=True, gridcolor='#f0f0f0'),
            height=300, margin=dict(t=40,b=20,l=10,r=10),
            plot_bgcolor='white', paper_bgcolor='white',
        )
        st.plotly_chart(fig_prob, width='stretch')

        st.markdown("**SHAP feature contribution (approximate)**")
        safe_t = SAFE_TEMP.get(p_food, 6)
        shap_vals = {
            'Temperature deviation': round((max(0,p_temp-safe_t)/22)*0.38, 4),
            'Gas spoilage score':    round(min(1,(p_methane/175+p_co2/2400)/2)*0.35, 4),
            'Storage days':          round((p_days/15)*0.27, 4),
        }
        fig_shap = go.Figure(go.Bar(
            x=list(shap_vals.values()),
            y=list(shap_vals.keys()),
            orientation='h',
            marker_color=["#dc3545" if v==max(shap_vals.values()) else "#4a90d9" for v in shap_vals.values()],
            text=[f"{v:.4f}" for v in shap_vals.values()],
            textposition='outside',
        ))
        fig_shap.update_layout(
            height=220, margin=dict(t=10,b=20,l=10,r=60),
            xaxis=dict(showgrid=True,gridcolor='#f0f0f0'),
            plot_bgcolor='white', paper_bgcolor='white',
        )
        st.plotly_chart(fig_shap, width='stretch')

        if cls == 2:
            st.error("🚨 CRITICAL ALERT — Rerouting recommendation triggered. Nearest market: dispatch logistics team immediately.")
        elif cls == 1:
            st.warning("⚠️ WARNING — Monitor closely. Consider expedited delivery or immediate sale.")
        else:
            st.success("✅ All sensors within safe range. No action required.")

# ══════════════════════════════════════
#  TAB 4 — ALERT HISTORY
# ══════════════════════════════════════
with tab4:
    st.markdown("#### Recent spoilage alerts")

    ah1, ah2, ah3 = st.columns(3)
    alerts = get_alert_history(40)
    n_crit = (alerts['risk'].str.contains('CRITICAL')).sum()
    n_warn = (alerts['risk'].str.contains('WARNING')).sum()
    ah1.metric("Total alerts (last 8h)", len(alerts))
    ah2.metric("🔴 Critical",  n_crit)
    ah3.metric("🟠 Warning",   n_warn)

    risk_f = st.multiselect("Filter by risk", ["🔴 CRITICAL","🟠 WARNING"],
                             default=["🔴 CRITICAL","🟠 WARNING"], key="alert_filter")
    filtered_alerts = alerts[alerts['risk'].isin(risk_f)] if risk_f else alerts

    def style_alerts(val):
        if 'CRITICAL' in str(val): return 'background:#f8d7da;color:#721c24;font-weight:600'
        if 'WARNING'  in str(val): return 'background:#fff3cd;color:#856404;font-weight:600'
        return ''

    st.dataframe(
        filtered_alerts.style.map(style_alerts, subset=['risk']),
        height=420, width='stretch'
    )

    st.markdown("#### Alerts over time (last 8h)")
    alerts['hour'] = pd.to_datetime(alerts['time']).dt.floor('30min')
    alert_ts = alerts.groupby(['hour','risk']).size().reset_index(name='count')
    fig_ts = px.bar(alert_ts, x='hour', y='count', color='risk',
        color_discrete_map={'🔴 CRITICAL':'#dc3545','🟠 WARNING':'#fd7e14'},
        labels={'hour':'Time','count':'Alerts','risk':'Risk level'},
        height=280)
    fig_ts.update_layout(margin=dict(t=10,b=10,l=10,r=10),
                          plot_bgcolor='white', paper_bgcolor='white',
                          legend_title='Risk level')
    st.plotly_chart(fig_ts, width='stretch')

# ══════════════════════════════════════
#  TAB 5 — ANALYTICS
# ══════════════════════════════════════
with tab5:
    st.markdown("#### Fleet analytics")

    an1, an2 = st.columns(2)
    with an1:
        fig_pie = px.pie(
            fleet, names='risk_label',
            color='risk_label',
            color_discrete_map={'Fresh':'#28a745','At Risk':'#fd7e14','Spoiled':'#dc3545'},
            title="Fleet risk distribution",
            hole=0.4,
        )
        fig_pie.update_layout(height=320, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig_pie, width='stretch')

    with an2:
        food_risk = fleet.groupby('food_name')['risk_score'].mean().sort_values(ascending=True).reset_index()
        fig_bar = px.bar(food_risk, x='risk_score', y='food_name',
            orientation='h',
            color='risk_score',
            color_continuous_scale=['#28a745','#fd7e14','#dc3545'],
            title="Average risk score by food type",
            labels={'risk_score':'Avg risk score','food_name':'Food'},
        )
        fig_bar.update_layout(height=320, margin=dict(t=40,b=10,l=10,r=10),
                               plot_bgcolor='white', paper_bgcolor='white',
                               coloraxis_showscale=False)
        st.plotly_chart(fig_bar, width='stretch')

    an3, an4 = st.columns(2)
    with an3:
        fig_temp = px.histogram(fleet, x='temperature', color='risk_label',
            color_discrete_map={'Fresh':'#28a745','At Risk':'#fd7e14','Spoiled':'#dc3545'},
            nbins=20, title="Temperature distribution across fleet",
            labels={'temperature':'Temperature (°C)','count':'Trucks'},
            barmode='overlay', opacity=0.75,
        )
        fig_temp.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=10),
                                plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig_temp, width='stretch')

    with an4:
        fig_scatter = px.scatter(fleet, x='temperature', y='methane',
            color='risk_label', size='risk_score',
            color_discrete_map={'Fresh':'#28a745','At Risk':'#fd7e14','Spoiled':'#dc3545'},
            hover_name='truck_id',
            hover_data={'food_name':True,'co2':True,'storage_days':True},
            title="Temperature vs methane (bubble = risk score)",
            labels={'temperature':'Temperature (°C)','methane':'Methane (ppm)'},
            size_max=18,
        )
        fig_scatter.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=10),
                                   plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig_scatter, width='stretch')

    st.markdown("#### ML model performance (from Postgres `model_metrics`, logged by train_random_forest.py / train_lstm.py)")
    metrics_df = get_model_metrics()
    if metrics_df.empty:
        st.info(
            "No rows in `model_metrics` yet — this table is written by the nightly "
            "`train_model_dag`. Run it at least once (or check MLflow directly at "
            "the MLflow UI) to see real numbers here."
        )
    else:
        for _, row in metrics_df.iterrows():
            mc1,mc2,mc3,mc4,mc5 = st.columns(5)
            mc1.metric("Model",      row["model_name"])
            mc2.metric("F1 score",   f"{row['f1_score']:.3f}" if pd.notna(row['f1_score']) else "—")
            mc3.metric("Accuracy",   f"{row['accuracy']:.1%}" if pd.notna(row['accuracy']) else "—")
            mc4.metric("Precision",  f"{row['precision_val']:.3f}" if pd.notna(row['precision_val']) else "—")
            mc5.metric("Recall",     f"{row['recall_val']:.3f}" if pd.notna(row['recall_val']) else "—")
            st.caption(f"Last trained: {pd.to_datetime(row['trained_at']).strftime('%Y-%m-%d %H:%M')}")
        st.caption(
            "Note: a per-class confusion matrix isn't currently persisted anywhere "
            "in the pipeline, so it isn't shown here rather than fabricated. To add "
            "one, log it as an MLflow artifact or an extra `model_metrics` column in "
            "train_random_forest.py."
        )
    