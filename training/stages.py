"""Individual pipeline stages, run by the Airflow DAG in an isolated environment.

Each stage prints one JSON object as its last stdout line. Exit codes:
  0 success | 2 deterministic failure (bad data / failed gate, retrying is pointless)
  1 anything else (infrastructure trouble, safe to retry)
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from training.train import CAT, DATA, NUM, build, dataset_provenance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s stage=%(name)s %(message)s")
EXPERIMENT = "parcel-delay"
REGISTERED = "parcel-delay-classifier"
EXPECTED_COLUMNS = set(NUM + CAT + ["delayed"])
MIN_ROWS = 1000


def emit(payload: dict, code: int = 0):
    print(json.dumps(payload))
    sys.exit(code)


def client() -> MlflowClient:
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(uri)
    return MlflowClient(uri)


def validate(a):
    log = logging.getLogger("validate")
    problems = []
    if not DATA.exists():
        emit({"stage": "validate", "ok": False, "problems": [f"missing file {DATA.name}"]}, 2)
    prov = dataset_provenance()
    df = pd.read_csv(DATA)
    if set(df.columns) != EXPECTED_COLUMNS:
        problems.append(f"schema mismatch: got {sorted(df.columns)}")
    else:
        if df.isnull().any().any():
            problems.append(f"null values in {df.columns[df.isnull().any()].tolist()}")
        if not 0.05 <= df["delayed"].mean() <= 0.95:
            problems.append(f"label rate {df['delayed'].mean():.3f} outside [0.05, 0.95]")
    if len(df) < MIN_ROWS:
        problems.append(f"only {len(df)} rows, need >= {MIN_ROWS}")
    if prov["dataset_file_matches_pointer"] != "true":
        problems.append("data file hash differs from its DVC pointer (modified or not checked out)")
    log.info("rows=%d problems=%s", len(df), problems)
    emit({"stage": "validate", "ok": not problems, "problems": problems, "rows": len(df),
          **prov}, 2 if problems else 0)


def preprocess(a):
    df = pd.read_csv(DATA)
    tr, te = train_test_split(df, test_size=0.2, stratify=df["delayed"], random_state=7)
    w = Path(a.work)
    w.mkdir(parents=True, exist_ok=True)
    tr.to_csv(w / "train.csv", index=False)
    te.to_csv(w / "test.csv", index=False)
    emit({"stage": "preprocess", "train_rows": len(tr), "test_rows": len(te),
          "train_delay_rate": round(float(tr["delayed"].mean()), 4)})


def train(a):
    w = Path(a.work)
    tr = pd.read_csv(w / "train.csv")
    params = dict(n_estimators=a.n_estimators, max_depth=a.max_depth, learning_rate=a.learning_rate)
    t0 = time.time()
    pipe = build(a.model, params).fit(tr[NUM + CAT], tr["delayed"])
    joblib.dump(pipe, w / "pipeline.joblib")
    (w / "params.json").write_text(json.dumps({"model_type": a.model, **params}))
    emit({"stage": "train", "model": a.model, "seconds": round(time.time() - t0, 2)})


def evaluate(a):
    w = Path(a.work)
    pipe = joblib.load(w / "pipeline.joblib")
    params = json.loads((w / "params.json").read_text())
    te = pd.read_csv(w / "test.csv")
    X, y = te[NUM + CAT], te["delayed"]
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = dict(f1=f1_score(y, pred), precision=precision_score(y, pred),
             recall=recall_score(y, pred), accuracy=accuracy_score(y, pred),
             roc_auc=roc_auc_score(y, proba))
    prov = dataset_provenance()
    client()
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"airflow-{a.airflow_run_id}"[:60]) as run:
        mlflow.log_params({**params, "test_rows": len(te)})
        mlflow.log_metrics({k: round(float(v), 4) for k, v in m.items()})
        mlflow.set_tags({**prov, "orchestrator": "airflow", "airflow_run_id": a.airflow_run_id,
                         "quality_gate": "pending"})
        info = mlflow.sklearn.log_model(pipe, name="model", signature=infer_signature(X, pred),
                                        serialization_format="cloudpickle",
                                        registered_model_name=REGISTERED)
        mlflow.set_tag("model_version", info.registered_model_version)
    emit({"stage": "evaluate", "run_id": run.info.run_id,
          "model_version": info.registered_model_version,
          "metrics": {k: round(float(v), 4) for k, v in m.items()}})


def gate(a):
    c = client()
    r = c.get_run(a.run_id)
    m = r.data.metrics
    checks = {"f1": (m["f1"], a.min_f1), "recall": (m["recall"], a.min_recall),
              "roc_auc": (m["roc_auc"], a.min_auc)}
    failed = {k: f"{v:.4f} < {t}" for k, (v, t) in checks.items() if v < t}
    verdict = "failed" if failed else "passed"
    c.set_tag(a.run_id, "quality_gate", verdict)
    c.set_tag(a.run_id, "quality_gate_detail", json.dumps(failed) if failed else "all thresholds met")
    emit({"stage": "gate", "ok": not failed, "failed_checks": failed,
          "thresholds": {k: t for k, (_, t) in checks.items()}}, 2 if failed else 0)


def promote(a):
    c = client()
    r = c.get_run(a.run_id)
    if r.data.tags.get("quality_gate") != "passed":
        emit({"stage": "promote", "ok": False, "reason": "quality gate not passed"}, 2)
    ver = r.data.tags["model_version"]
    c.set_registered_model_alias(REGISTERED, "candidate", ver)  # human approval still required
    emit({"stage": "promote", "ok": True, "alias": "candidate", "model_version": ver})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("validate").set_defaults(fn=validate)
    p = sp.add_parser("preprocess"); p.add_argument("--work", required=True); p.set_defaults(fn=preprocess)
    p = sp.add_parser("train"); p.add_argument("--work", required=True)
    p.add_argument("--model", default="rf"); p.add_argument("--n-estimators", type=int, default=150)
    p.add_argument("--max-depth", type=int, default=8); p.add_argument("--learning-rate", type=float, default=0.1)
    p.set_defaults(fn=train)
    p = sp.add_parser("evaluate"); p.add_argument("--work", required=True)
    p.add_argument("--airflow-run-id", default="manual"); p.set_defaults(fn=evaluate)
    p = sp.add_parser("gate"); p.add_argument("--run-id", required=True)
    p.add_argument("--min-f1", type=float, default=0.60); p.add_argument("--min-recall", type=float, default=0.55)
    p.add_argument("--min-auc", type=float, default=0.70); p.set_defaults(fn=gate)
    p = sp.add_parser("promote"); p.add_argument("--run-id", required=True); p.set_defaults(fn=promote)
    args = ap.parse_args()
    try:
        args.fn(args)
    except SystemExit:
        raise
    except Exception as exc:
        logging.getLogger("stages").exception("stage failed")
        emit({"stage": args.cmd, "ok": False, "error": f"{type(exc).__name__}: {exc}"}, 1)
