"""
=============================================================
  Cold Chain Project — Nightly Best-Model Auto-Deploy
=============================================================
  For each registered model (the RF classifier and the LSTM
  regressor), compares the metric of the most recently trained
  version against whatever is currently in the "Production"
  stage, and promotes the new one ONLY if it's actually better.
  The previous Production version is archived, not deleted.

    - RF classifier:  higher f1_macro wins
    - LSTM regressor: lower test_mae_hours wins

  If a model has never been promoted before (no Production
  version exists yet), the latest version is promoted
  automatically as the first deployment.

  Run nightly, after both training scripts:
    python promote_best_model.py
=============================================================
"""

import os
import logging

from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("PromoteBestModel")

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# model_name -> (metric_key, "higher_is_better" | "lower_is_better")
MODELS_TO_CHECK = {
    "cold-chain-rf-classifier":     ("f1_macro", "higher_is_better"),
    "cold-chain-lstm-shelf-life":   ("test_mae_hours", "lower_is_better"),
}


def get_metric(client, run_id, metric_key):
    run = client.get_run(run_id)
    return run.data.metrics.get(metric_key)


def promote_if_better(client, model_name, metric_key, direction):
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        log.warning(f"No versions found for '{model_name}' — skipping.")
        return

    # Latest version = highest version number
    latest = max(versions, key=lambda v: int(v.version))
    latest_metric = get_metric(client, latest.run_id, metric_key)

    if latest_metric is None:
        log.warning(f"Latest version of '{model_name}' has no '{metric_key}' metric — skipping.")
        return

    current_prod = [v for v in versions if v.current_stage == "Production"]

    if not current_prod:
        log.info(f"No Production version yet for '{model_name}' — promoting v{latest.version} "
                  f"({metric_key}={latest_metric:.4f}) as the first deployment.")
        client.transition_model_version_stage(
            name=model_name, version=latest.version, stage="Production", archive_existing_versions=True
        )
        return

    prod_version = current_prod[0]
    prod_metric = get_metric(client, prod_version.run_id, metric_key)

    if prod_metric is None:
        log.warning(f"Current Production version of '{model_name}' has no '{metric_key}' metric "
                    f"— promoting latest as a precaution.")
        client.transition_model_version_stage(
            name=model_name, version=latest.version, stage="Production", archive_existing_versions=True
        )
        return

    is_better = (
        latest_metric >= prod_metric if direction == "higher_is_better"
        else latest_metric <= prod_metric
    )

    log.info(
        f"'{model_name}': Production v{prod_version.version} ({metric_key}={prod_metric:.4f}) "
        f"vs latest v{latest.version} ({metric_key}={latest_metric:.4f})"
    )

    if latest.version == prod_version.version:
        log.info(f"Latest version IS already Production for '{model_name}' — nothing to do.")
        return

    if is_better:
        log.info(f"✅ Promoting v{latest.version} → Production for '{model_name}'.")
        client.transition_model_version_stage(
            name=model_name, version=latest.version, stage="Production", archive_existing_versions=True
        )
    else:
        log.info(f"Keeping current Production version for '{model_name}' — no improvement found.")


def main():
    client = MlflowClient(tracking_uri=MLFLOW_URI)

    for model_name, (metric_key, direction) in MODELS_TO_CHECK.items():
        try:
            promote_if_better(client, model_name, metric_key, direction)
        except Exception as e:
            log.error(f"Failed to evaluate/promote '{model_name}': {e}")

    log.info("✅ Nightly model promotion check complete.")


if __name__ == "__main__":
    main()