"""Train the parcel-delay classifier and log full provenance to MLflow.

Every run records the DVC data hash, the git commit that last changed the
dataset pointer, and the nearest dataset tag, so "which data trained this
model?" is answerable later from MLflow alone.
"""
import argparse
import os
import re
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "parcels.csv"
POINTER = ROOT / "data" / "parcels.csv.dvc"
NUM = ["distance_km", "hour", "rider_load", "weight_kg"]
CAT = ["weather", "zone"]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def dataset_provenance() -> dict:
    md5 = yaml.safe_load(POINTER.read_text())["outs"][0]["md5"]
    commit = git("log", "-1", "--format=%H", "--", "data/parcels.csv.dvc")
    tags = [t for t in git("tag", "--points-at", commit).split() if t.startswith("dataset-")]
    version = tags[0].removeprefix("dataset-") if tags else "untagged"
    dirty = bool(git("status", "--porcelain", "--", "data/parcels.csv.dvc"))
    return dict(dataset_version=version, dataset_md5=md5, dataset_git_commit=commit,
                dataset_pointer_dirty=str(dirty).lower())


def build(model: str, p: dict) -> Pipeline:
    clf = {
        "gbm": lambda: GradientBoostingClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=p["learning_rate"], random_state=7),
        "rf": lambda: RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"], random_state=7, n_jobs=2),
        "logreg": lambda: LogisticRegression(max_iter=500),
    }[model]()
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gbm", "rf", "logreg"], default="gbm")
    ap.add_argument("--n-estimators", type=int, default=150)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--experiment", default="parcel-delay")
    ap.add_argument("--register", default="parcel-delay-classifier")
    ap.add_argument("--run-name", default=None)
    a = ap.parse_args()
    params = dict(n_estimators=a.n_estimators, max_depth=a.max_depth,
                  learning_rate=a.learning_rate)

    df = pd.read_csv(DATA)
    X, y = df[NUM + CAT], df["delayed"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)
    prov = dataset_provenance()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(a.experiment)
    with mlflow.start_run(run_name=a.run_name) as run:
        pipe = build(a.model, params).fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = dict(f1=f1_score(yte, pred), precision=precision_score(yte, pred),
                       recall=recall_score(yte, pred), accuracy=accuracy_score(yte, pred),
                       roc_auc=roc_auc_score(yte, proba))
        mlflow.log_params({"model_type": a.model, **(params if a.model != "logreg" else {}),
                           "train_rows": len(Xtr), "test_rows": len(Xte)})
        mlflow.log_metrics({k: round(float(v), 4) for k, v in metrics.items()})
        mlflow.set_tags(prov)
        mlflow.log_artifact(str(POINTER), artifact_path="provenance")
        mlflow.sklearn.log_model(pipe, name="model", signature=infer_signature(Xte, pred),
                                 registered_model_name=a.register)
        print(re.sub(r"\s+", " ", f"run={run.info.run_id} {prov['dataset_version']} "
              + " ".join(f"{k}={v:.3f}" for k, v in metrics.items())))


if __name__ == "__main__":
    main()
