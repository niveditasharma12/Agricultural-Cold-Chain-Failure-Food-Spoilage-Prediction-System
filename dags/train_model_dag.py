"""
=============================================================
  Cold Chain Project — ML Training DAG
=============================================================
  Runs nightly at 2 AM (after feature_dag's 1 AM feature build):
    1. Train the Random Forest spoilage classifier
    2. Train the LSTM shelf-life regressor
       (both log to MLflow and register a new model version)
    3. Compare each new version against the current Production
       version and auto-promote if it's actually better

  IMPORTANT — dependency isolation:
  All pip installs below use `--target=/tmp/ml_deps` and set
  PYTHONPATH explicitly, INSTEAD of a plain `pip install`. This
  is deliberate: BashOperator tasks run as the same `airflow`
  user Airflow itself runs as, so a plain `pip install` writes
  into Airflow's own site-packages directory
  (/home/airflow/.local/...). Earlier versions of this DAG did
  exactly that, and mlflow's dependency resolution downgraded
  `typing_extensions` to satisfy itself — which broke Airflow's
  own `pydantic` dependency and crashed the scheduler entirely
  (ImportError: cannot import name 'TypeAliasType'). Installing
  to an isolated target directory and only extending PYTHONPATH
  for these specific subprocess calls prevents this from ever
  happening again. /tmp is wiped on container restart, so this
  also means a full clean reinstall each run — slower, but safe.
=============================================================
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "cold-chain-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

ML_DEPS_DIR = "/tmp/ml_deps"

PIP_INSTALL = (
    f"pip install --quiet --target={ML_DEPS_DIR} "
    "scikit-learn==1.3.2 mlflow==2.9.2 tensorflow==2.13.0 "
    "pandas numpy shap matplotlib psycopg2-binary joblib"
)

RUN_WITH_ISOLATED_DEPS = f"PYTHONPATH={ML_DEPS_DIR} python"

with DAG(
    dag_id="train_model_dag",
    description="Nightly training of the RF classifier and LSTM shelf-life model, with auto-deploy of the best version",
    default_args=default_args,
    schedule_interval="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["cold-chain", "ml", "mlflow", "nightly"],
) as dag:

    train_random_forest = BashOperator(
        task_id="train_random_forest",
        bash_command=(
            "cd /opt/airflow/data && "
            f"{PIP_INSTALL} && "
            f"{RUN_WITH_ISOLATED_DEPS} /opt/airflow/scripts/train_random_forest.py"
        ),
    )

    train_lstm = BashOperator(
        task_id="train_lstm",
        bash_command=(
            "cd /opt/airflow/data && "
            f"{PIP_INSTALL} && "
            f"{RUN_WITH_ISOLATED_DEPS} /opt/airflow/scripts/train_lstm.py"
        ),
    )

    promote_best_model = BashOperator(
        task_id="promote_best_model",
        bash_command=(
            f"pip install --quiet --target={ML_DEPS_DIR} mlflow==2.9.2 && "
            f"{RUN_WITH_ISOLATED_DEPS} /opt/airflow/scripts/promote_best_model.py"
        ),
    )

    train_random_forest >> train_lstm >> promote_best_model