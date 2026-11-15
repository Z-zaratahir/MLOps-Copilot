"""Exercise every model-visible tool once against the REAL running systems (not part of CI).

Usage: python scripts/live_smoke.py          (requires MLflow, Airflow and the kind cluster to be up)
Mutating tools are only *requested* here; nothing is executed or triggered.
"""
import sys

from agent import registry
from agent.orchestrator import Orchestrator  # noqa: F401  (registers every tool)
from agent.registry import RequestContext, execute

ctx = RequestContext("live-smoke", "show me the current state")
pod = execute("k8s_list_pods", {}, ctx)
pod_name = pod.data["pods"][0]["name"] if pod.ok and pod.data["pods"] else "unknown"

ARGS = {
    "list_runs": {"limit": 3}, "get_run_details": {"run_ref": "25"}, "find_best_run": {},
    "compare_runs": {"last_n": 3}, "compare_experiments": {}, "get_model_registry": {},
    "dvc_compare_versions": {"version_a": "v1", "version_b": "v3"},
    "dvc_reproduce_instructions": {"version": "v2"}, "get_dataset_for_run": {"run_ref": "25"},
    "airflow_get_task_diagnostics": {"task_id": "validate_data"},
    "k8s_get_pod_status": {"pod": pod_name}, "k8s_request_restart": {"deployment": "parcel-delay-api"},
    "k8s_request_scale": {"deployment": "parcel-delay-api", "replicas": 2},
    "airflow_trigger_training": {"wait_seconds": 0},  # rejected: the request text never asks to train
    "assess_retraining_need": {},
}
EXPECTED = {"k8s_request_restart": "pending_confirmation", "k8s_request_scale": "pending_confirmation",
            "airflow_trigger_training": "rejected"}

bad = 0
for name in sorted(t.name for t in registry.TOOLS.values() if not t.hidden):
    r = execute(name, ARGS.get(name, {}), ctx)
    want = EXPECTED.get(name, "ok")
    flag = "PASS" if r.status == want else "FAIL"
    bad += flag == "FAIL"
    print(f"{flag}  {name:<32} {r.status:<22} {r.duration_ms:>6} ms  {(r.error or '')[:90]}")
print(f"\n{len(registry.TOOLS) - 1 - bad}/{len(registry.TOOLS) - 1} tools behaved as expected")
sys.exit(1 if bad else 0)
