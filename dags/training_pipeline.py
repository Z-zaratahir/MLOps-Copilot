"""Parcel-delay training DAG.

validate -> preprocess -> train -> evaluate_and_log -> quality_gate -> promote_candidate

Heavy ML work runs in an isolated virtualenv (/opt/mlenv) via training/stages.py, so
Airflow's own dependency set never has to match the model's. Deterministic failures
(bad data, failed gate) exit with code 2 and are not retried; anything else retries.
"""
import json
import logging
import os
import subprocess
from datetime import datetime, timedelta

from airflow.sdk import DAG, Param, get_current_context, task
from airflow.exceptions import AirflowFailException

log = logging.getLogger(__name__)
PROJECT = os.environ.get("PROJECT_DIR", ".")
PY = os.environ.get("ML_PYTHON", "/opt/mlenv/bin/python")


def run_stage(*args: str) -> dict:
    """Run one stage, log its output, return its JSON result."""
    proc = subprocess.run([PY, "-m", "training.stages", *args], cwd=PROJECT,
                          capture_output=True, text=True, timeout=900)
    for line in proc.stderr.splitlines()[-25:]:
        log.info("[%s] %s", args[0], line)
    last = (proc.stdout.strip().splitlines() or ["{}"])[-1]
    try:
        result = json.loads(last)
    except json.JSONDecodeError:
        result = {"ok": False, "error": f"unparseable stage output: {last[:200]}"}
    log.info("stage=%s exit=%s result=%s", args[0], proc.returncode, result)
    if proc.returncode == 2:
        raise AirflowFailException(f"{args[0]} rejected the pipeline: {result}")
    if proc.returncode != 0:
        raise RuntimeError(f"{args[0]} failed (exit {proc.returncode}): {result}")
    return result


with DAG(
    dag_id="parcel_delay_training",
    description="Validated, gated training of the parcel-delay classifier",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "parcel-delay"],
    default_args={"retries": 2, "retry_delay": timedelta(seconds=15)},
    params={
        "model": Param("rf", enum=["rf", "gbm", "logreg"]),
        "max_depth": Param(8, type="integer"),
        "min_f1": Param(0.60, type="number"),
        "min_recall": Param(0.55, type="number"),
        "min_auc": Param(0.70, type="number"),
    },
) as dag:

    @task
    def validate_data():
        return run_stage("validate")

    @task
    def preprocess():
        return run_stage("preprocess", "--work", _work())

    @task
    def train_model():
        p = get_current_context()["params"]
        return run_stage("train", "--work", _work(), "--model", p["model"],
                         "--max-depth", str(p["max_depth"]))

    @task
    def evaluate_and_log():
        return run_stage("evaluate", "--work", _work(),
                         "--airflow-run-id", get_current_context()["run_id"])

    @task
    def quality_gate(evaluation: dict):
        p = get_current_context()["params"]
        return run_stage("gate", "--run-id", evaluation["run_id"], "--min-f1", str(p["min_f1"]),
                         "--min-recall", str(p["min_recall"]), "--min-auc", str(p["min_auc"]))

    @task
    def promote_candidate(evaluation: dict, gate_result: dict):
        return run_stage("promote", "--run-id", evaluation["run_id"])

    def _work() -> str:
        return f"{PROJECT}/.work/{get_current_context()['run_id'].replace(':', '_')}"

    v = validate_data()
    pre = preprocess()
    tr = train_model()
    ev = evaluate_and_log()
    v >> pre >> tr >> ev
    gate = quality_gate(ev)
    promote_candidate(ev, gate)
