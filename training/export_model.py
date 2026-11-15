"""Export the best MLflow run (by metric, restricted to a dataset version) for serving."""
import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
from mlflow.tracking import MlflowClient

ap = argparse.ArgumentParser()
ap.add_argument("--dataset-version", default="v3")
ap.add_argument("--metric", default="f1")
ap.add_argument("--run-id", default=None)
ap.add_argument("--allow-ungated", action="store_true", help="serve a run that has no passed quality gate")
ap.add_argument("--out", default="models/served")
a = ap.parse_args()

uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(uri)
c = MlflowClient(uri)
exp = c.get_experiment_by_name("parcel-delay")
if a.run_id:
    run = c.get_run(a.run_id)
else:
    flt = f"tags.dataset_version = '{a.dataset_version}'"
    if not a.allow_ungated:  # only models that passed the Airflow quality gate may be served
        flt += " and tags.quality_gate = 'passed'"
    found = c.search_runs([exp.experiment_id], flt, order_by=[f"metrics.{a.metric} DESC"], max_results=1)
    if not found:
        raise SystemExit(f"no eligible run for {a.dataset_version} (quality gate required unless --allow-ungated)")
    run = found[0]
model = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
joblib.dump(model, out / "model.joblib")
meta = dict(run_id=run.info.run_id, run_name=run.info.run_name,
            dataset_version=run.data.tags["dataset_version"],
            dataset_md5=run.data.tags["dataset_md5"],
            validation_f1=run.data.metrics["f1"], quality_gate=run.data.tags.get("quality_gate"),
            exported_by_metric=a.metric)
(out / "meta.json").write_text(json.dumps(meta, indent=2))
print(meta)
