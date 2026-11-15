"""MLflow experiment-intelligence tools (MlflowClient only; no UI scraping)."""
import os
from datetime import UTC, datetime

# fail fast when the server is down instead of retrying for minutes
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")

from mlflow.tracking import MlflowClient  # noqa: E402
from pydantic import Field  # noqa: E402

from agent.config import get_settings  # noqa: E402
from agent.registry import NoArgs, ToolInput, tool  # noqa: E402

MODULE = "mlflow"
METRICS_OF_INTEREST = ["f1", "precision", "recall", "accuracy", "roc_auc"]
MIN_RECALL_FOR_CONFIDENCE = 0.5


def client() -> MlflowClient:
    return MlflowClient(get_settings().mlflow_tracking_uri)


def _iso(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="seconds") if ms else None


def _experiment_id(name: str | None = None) -> str:
    exp = client().get_experiment_by_name(name or get_settings().mlflow_experiment)
    if exp is None:
        raise LookupError(f"experiment '{name or get_settings().mlflow_experiment}' not found")
    return exp.experiment_id


def summarize(run) -> dict:
    t = run.data.tags
    return {"run_id": run.info.run_id, "run_name": run.info.run_name, "status": run.info.status,
            "started": _iso(run.info.start_time), "ended": _iso(run.info.end_time),
            "metrics": {k: run.data.metrics[k] for k in METRICS_OF_INTEREST if k in run.data.metrics},
            "model_type": run.data.params.get("model_type"),
            "dataset_version": t.get("dataset_version"), "dataset_md5": t.get("dataset_md5"),
            "dataset_git_commit": t.get("dataset_git_commit"),
            "quality_gate": t.get("quality_gate"), "model_version": t.get("model_version"),
            "orchestrator": t.get("orchestrator", "manual")}


def all_runs(experiment: str | None = None, only_finished: bool = True) -> list:
    """Runs newest-first."""
    flt = "attributes.status = 'FINISHED'" if only_finished else ""
    return client().search_runs([_experiment_id(experiment)], flt,
                                order_by=["attributes.start_time DESC"], max_results=500)


def resolve_run(ref: str):
    """Accepts a run id, a run name, or an ordinal ('25' = 25th oldest run)."""
    runs = all_runs(only_finished=False)
    chron = list(reversed(runs))
    if ref.isdigit() and len(ref) < 6:
        n = int(ref)
        if not 1 <= n <= len(chron):
            raise LookupError(f"experiment/run #{n} does not exist; {len(chron)} runs are recorded")
        return chron[n - 1]
    for r in runs:
        if ref in (r.info.run_id, r.info.run_name):
            return r
    raise LookupError(f"no run matches '{ref}'")


class ListRunsIn(ToolInput):
    experiment: str | None = Field(None, description="Experiment name; defaults to the project experiment")
    limit: int = Field(10, ge=1, le=50)
    dataset_version: str | None = Field(None, description="Only runs trained on this dataset version, e.g. v3")


class RunRefIn(ToolInput):
    run_ref: str = Field(description="Run id, run name, or an ordinal number such as '25' (nth oldest run)")


class BestRunIn(ToolInput):
    metric: str = Field("f1", description="Metric to rank by (f1, recall, precision, accuracy, roc_auc)")
    recent_runs: int | None = Field(None, ge=1, le=100, description="Only consider the N most recent runs")
    dataset_version: str | None = None


class CompareRunsIn(ToolInput):
    run_refs: list[str] | None = Field(None, description="Specific runs to compare")
    last_n: int = Field(5, ge=2, le=20, description="Used when run_refs is omitted")
    metric: str = "f1"


class CompareExperimentsIn(ToolInput):
    names: list[str] | None = Field(None, description="Experiment names; defaults to all")
    metric: str = "f1"


class RegistryIn(ToolInput):
    model_name: str | None = None


@tool(MODULE, "List MLflow experiments with their run counts.")
def list_experiments(_: NoArgs):
    c = client()
    out = []
    for e in c.search_experiments():
        n = len(c.search_runs([e.experiment_id], max_results=1000))
        out.append({"experiment_id": e.experiment_id, "name": e.name, "run_count": n})
    return out


@tool(MODULE, "List recent training runs (newest first) with metrics and the dataset version each used.")
def list_runs(a: ListRunsIn):
    runs = [summarize(r) for r in all_runs(a.experiment)]
    if a.dataset_version:
        runs = [r for r in runs if r["dataset_version"] == a.dataset_version]
    return {"count": len(runs), "runs": runs[: a.limit]}


@tool(MODULE, "Get one run's metrics, parameters, tags and artifacts. Accepts run id, name or ordinal number.")
def get_run_details(a: RunRefIn):
    r = resolve_run(a.run_ref)
    arts = [x.path for x in client().list_artifacts(r.info.run_id)]
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    return {**summarize(r), "params": r.data.params, "all_metrics": r.data.metrics,
            "tags": tags, "artifacts": arts}


def _assess(run: dict, newest_version: str | None) -> list[str]:
    """Reasons a run is NOT automatically production-ready."""
    issues = []
    m = run["metrics"]
    if run["quality_gate"] != "passed":
        issues.append("no passed quality gate recorded" if not run["quality_gate"]
                      else f"quality gate = {run['quality_gate']}")
    if run["dataset_version"] != newest_version:
        issues.append(f"trained on {run['dataset_version']}, newest data is {newest_version}")
    if m.get("recall", 1) < MIN_RECALL_FOR_CONFIDENCE:
        issues.append(f"low recall {m.get('recall')} (misses many delayed parcels)")
    return issues


@tool(MODULE, "Find the best run by a metric and assess whether it is actually safe to promote. "
              "Never treat the top metric alone as production-ready.")
def find_best_run(a: BestRunIn):
    runs = [summarize(r) for r in all_runs()]
    versions = sorted({r["dataset_version"] for r in runs if r["dataset_version"]})
    newest = versions[-1] if versions else None
    pool = runs[: a.recent_runs] if a.recent_runs else runs
    if a.dataset_version:
        pool = [r for r in pool if r["dataset_version"] == a.dataset_version]
    pool = [r for r in pool if a.metric in r["metrics"]]
    if not pool:
        raise LookupError("no runs with that metric in the selected scope")
    ranked = sorted(pool, key=lambda r: r["metrics"][a.metric], reverse=True)
    best = ranked[0]
    mixed = len({r["dataset_version"] for r in pool}) > 1
    return {"metric": a.metric, "considered_runs": len(pool), "newest_dataset_version": newest,
            "best": best, "readiness_concerns": _assess(best, newest),
            "comparability_warning": ("candidates were trained/tested on different dataset versions; "
                                      "metric values are not directly comparable") if mixed else None,
            "top_candidates": ranked[:5]}


@tool(MODULE, "Compare several runs side by side (explicit runs or the last N) and report per-metric winners.")
def compare_runs(a: CompareRunsIn):
    if a.run_refs:
        runs = [summarize(resolve_run(x)) for x in a.run_refs]
    else:
        runs = [summarize(r) for r in all_runs()[: a.last_n]]
    if len(runs) < 2:
        raise ValueError("need at least two runs to compare")
    winners = {m: max(runs, key=lambda r: r["metrics"].get(m, -1))["run_name"] for m in METRICS_OF_INTEREST}
    return {"runs": runs, "metric_winners": winners, "ranking_metric": a.metric,
            "ranked": [r["run_name"] for r in sorted(runs, key=lambda r: r["metrics"].get(a.metric, -1), reverse=True)],
            "mixed_dataset_versions": len({r["dataset_version"] for r in runs}) > 1}


@tool(MODULE, "Compare MLflow experiments: run counts, best metric and latest run for each.")
def compare_experiments(a: CompareExperimentsIn):
    c = client()
    out = []
    for e in c.search_experiments():
        if a.names and e.name not in a.names:
            continue
        runs = c.search_runs([e.experiment_id], "attributes.status = 'FINISHED'",
                             order_by=["attributes.start_time DESC"], max_results=500)
        scored = [r for r in runs if a.metric in r.data.metrics]
        best = max(scored, key=lambda r: r.data.metrics[a.metric]) if scored else None
        out.append({"experiment": e.name, "finished_runs": len(runs),
                    "best_run": best.info.run_name if best else None,
                    f"best_{a.metric}": best.data.metrics[a.metric] if best else None,
                    "latest_run": runs[0].info.run_name if runs else None})
    return out


@tool(MODULE, "Get model-registry information: versions, source runs, aliases (e.g. candidate/champion).")
def get_model_registry(a: RegistryIn):
    c = client()
    name = a.model_name or get_settings().mlflow_registered_model
    model = c.get_registered_model(name)
    aliases = dict(model.aliases)
    versions = []
    for v in c.search_model_versions(f"name='{name}'"):
        run = c.get_run(v.run_id) if v.run_id else None
        versions.append({"version": v.version, "run_id": v.run_id, "status": v.status,
                         "created": _iso(v.creation_timestamp),
                         "aliases": [k for k, ver in aliases.items() if ver == v.version],
                         "source_run_name": run.info.run_name if run else None,
                         "dataset_version": run.data.tags.get("dataset_version") if run else None,
                         "quality_gate": run.data.tags.get("quality_gate") if run else None})
    return {"model": name, "aliases": aliases,
            "versions": sorted(versions, key=lambda x: int(x["version"]), reverse=True)}
