"""Static checks that the deployment and DAG keep satisfying the assignment's requirements."""
import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return yaml.safe_load((ROOT / "kubernetes" / name).read_text())


def test_deployment_meets_requirements():
    d = load("deployment.yaml")
    assert d["spec"]["replicas"] >= 2
    c = d["spec"]["template"]["spec"]["containers"][0]
    assert c["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert c["livenessProbe"]["httpGet"]["path"] == "/health"
    for kind in ("requests", "limits"):
        assert {"cpu", "memory"} <= set(c["resources"][kind])
    assert c["envFrom"][0]["configMapRef"]["name"] == load("configmap.yaml")["metadata"]["name"]


def test_service_is_clusterip_and_selects_the_deployment():
    s, d = load("service.yaml"), load("deployment.yaml")
    assert s["spec"]["type"] == "ClusterIP"
    assert s["spec"]["selector"].items() <= d["spec"]["template"]["metadata"]["labels"].items()


def test_configmap_holds_no_secrets():
    data = load("configmap.yaml")["data"]
    assert not any(w in k.lower() for k in data for w in ("key", "token", "password", "secret"))


def test_manifests_stay_in_the_approved_namespace():
    for f in ("deployment.yaml", "service.yaml", "configmap.yaml"):
        assert load(f)["metadata"]["namespace"] == "parcel-ml"


def test_training_dag_has_required_stages_retries_and_gate():
    src = (ROOT / "dags" / "training_pipeline.py").read_text()
    tree = ast.parse(src)
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert {"validate_data", "preprocess", "train_model", "evaluate_and_log", "quality_gate",
            "promote_candidate"} <= funcs
    assert '"retries": 2' in src and "AirflowFailException" in src
    assert "v >> pre >> tr >> ev" in src and "promote_candidate(ev, gate)" in src
