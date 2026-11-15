"""Cross-system retraining assessment: a deterministic, evidence-backed policy check.

It only RECOMMENDS. It never starts training. Every trigger it reports is backed by data
fetched from the four systems; anything it cannot fetch is listed as an uncertainty.
"""
import re
from datetime import UTC, datetime

from pydantic import Field

from agent.config import get_settings
from agent.registry import ToolInput, current_request, tool
from agent.tools import airflow_tools, dvc_tools, kubernetes_tools, mlflow_tools

MODULE = "assessment"
DRIFT_DELAY_RATE_THRESHOLD = 0.05
DEGRADATION_WORDS = re.compile(r"degrad|dropp?ed|declin|decreas|worse|fell|deteriorat", re.I)


class AssessIn(ToolInput):
    run_ref: str | None = Field(None, description="Model run to assess; defaults to the best recent run")
    observed_degradation: float | None = Field(
        None, ge=0, le=1, description="Measured drop in live metric, only if the user supplied it")


def _try(label: str, fn, uncertainties: list):
    try:
        return fn()
    except Exception as exc:
        uncertainties.append(f"{label} unavailable ({type(exc).__name__}: {str(exc)[:120]})")
        return None


@tool(MODULE, "Assess whether retraining is justified by explicit evidence (freshness policy, newer/drifted data, "
              "failed quality gate, model mismatch, supplied degradation). Recommends only; never triggers training.",
      timeout=40)
def assess_retraining_need(a: AssessIn):
    s = get_settings()
    unc: list[str] = []
    triggers: list[dict] = []
    facts: dict = {}

    summary = _try("MLflow", lambda: (
        mlflow_tools.summarize(mlflow_tools.resolve_run(a.run_ref)) if a.run_ref else _best_summary()), unc)
    if summary is None:
        return {"decision": "insufficient_evidence", "triggers": [], "uncertainties": unc, "facts": facts,
                "note": "cannot assess without model information from MLflow"}
    facts["assessed_model"] = {k: summary[k] for k in ("run_name", "run_id", "dataset_version", "metrics", "quality_gate")}

    # 1. freshness policy
    ended = datetime.fromisoformat(summary["ended"]) if summary.get("ended") else None
    if ended:
        age = (datetime.now(UTC) - ended).days
        facts["model_age_days"] = age
        facts["freshness_policy_days"] = s.model_freshness_days
        if age > s.model_freshness_days:
            triggers.append({"type": "freshness_policy", "detail": f"model is {age} days old; policy is {s.model_freshness_days}"})

    # 2. newer / drifted data
    cur = _try("DVC", dvc_tools.current, unc)
    if cur:
        facts["current_dataset_version"] = cur["version"]
        if cur["pointer_md5"] != summary["dataset_md5"]:
            cmp_ = _try("DVC compare", lambda: dvc_tools.dvc_compare_versions(
                dvc_tools.CompareIn(version_a=summary["dataset_version"], version_b=cur["version"])), unc)
            delta = (cmp_ or {}).get("delta_stats") or {}
            facts["data_delta_vs_training_data"] = delta
            drift = abs(delta.get("delay_rate", 0)) >= DRIFT_DELAY_RATE_THRESHOLD
            triggers.append({"type": "newer_dataset" + ("_with_drift" if drift else ""),
                             "detail": f"model trained on {summary['dataset_version']}, current data is {cur['version']}"
                                       f"; delay-rate change {delta.get('delay_rate', 'n/a')}"})

    # 3. Airflow quality gate / latest training outcome
    runs = _try("Airflow", lambda: airflow_tools._get(f"/dags/{s.airflow_approved_dag}/dagRuns",
                                                      order_by="-start_date", limit=3)["dag_runs"], unc)
    if runs:
        facts["recent_airflow_runs"] = [{"id": r["dag_run_id"], "state": r["state"]} for r in runs]
        if runs[0]["state"] == "failed":
            failed = _try("Airflow tasks", lambda: airflow_tools._tasks(s.airflow_approved_dag, runs[0]["dag_run_id"]), unc)
            names = [t["task_id"] for t in failed or [] if t["state"] == "failed"]
            facts["latest_run_failed_tasks"] = names
            if "quality_gate" in names:
                triggers.append({"type": "failed_quality_gate", "detail": "most recent training run was rejected by the quality gate"})

    # 4. deployed model vs assessed model
    dep = _try("Kubernetes", lambda: kubernetes_tools.k8s_get_deployed_version(None), unc)
    if dep:
        facts["deployed"] = dep
        live = dep.get("running_service_reports")
        if isinstance(live, dict) and live.get("run_id") and live["run_id"] != summary["run_id"]:
            facts["deployed_model_differs_from_assessed"] = True
        elif not isinstance(live, dict):
            unc.append("running service did not report which model it serves")

    # 5. user-supplied degradation
    ctx = current_request.get()
    said = ctx.user_text if ctx else ""
    if a.observed_degradation is not None and not DEGRADATION_WORDS.search(said):
        unc.append(f"ignored degradation value {a.observed_degradation}: the user did not report any degradation")
        a.observed_degradation = None
    if a.observed_degradation is not None and a.observed_degradation > 0.05:
        triggers.append({"type": "supplied_degradation", "detail": f"user reported metric drop of {a.observed_degradation}"})
    elif a.observed_degradation is None:
        unc.append("no live-performance/degradation results were supplied, so real-world model quality is unverified")

    return {"decision": "retrain_recommended" if triggers else "retraining_not_indicated",
            "triggers": triggers, "facts": facts, "uncertainties": unc,
            "note": "recommendation only; no training run was started"}


def _best_summary() -> dict:
    runs = [mlflow_tools.summarize(r) for r in mlflow_tools.all_runs()]
    newest = max((r["dataset_version"] for r in runs if r["dataset_version"]), default=None)
    pool = [r for r in runs if r["dataset_version"] == newest and "f1" in r["metrics"]]
    return max(pool, key=lambda r: r["metrics"]["f1"])
