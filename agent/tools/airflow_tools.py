"""Airflow 3 workflow tools (REST API v2, JWT-authenticated)."""
import time

import httpx
from pydantic import Field

from agent.config import get_settings
from agent.policy import PolicyViolation, trigger_allowed
from agent.registry import NoArgs, ToolInput, current_request, tool

MODULE = "airflow"
_token: dict = {"value": None, "exp": 0.0, "base": None}


def _http() -> httpx.Client:
    s = get_settings()
    if _token["value"] is None or time.time() > _token["exp"] or _token["base"] != s.airflow_base_url:
        r = httpx.post(f"{s.airflow_base_url}/auth/token", timeout=8,
                       json={"username": s.airflow_username, "password": s.airflow_password})
        r.raise_for_status()
        _token.update(value=r.json()["access_token"], exp=time.time() + 600, base=s.airflow_base_url)
    return httpx.Client(base_url=f"{s.airflow_base_url}/api/v2", timeout=20,
                        headers={"Authorization": f"Bearer {_token['value']}"})


def _get(path: str, **params):
    with _http() as c:
        r = c.get(path, params=params)
        r.raise_for_status()
        return r.json()


def _approved(dag_id: str | None) -> str:
    approved = get_settings().airflow_approved_dag
    if dag_id and dag_id != approved:
        raise PolicyViolation(f"DAG '{dag_id}' is not approved for operations (allowed: '{approved}')")
    return approved


def _run_brief(r: dict) -> dict:
    return {"dag_run_id": r["dag_run_id"], "state": r["state"], "run_type": r.get("run_type"),
            "start": r.get("start_date"), "end": r.get("end_date"), "conf": r.get("conf")}


def _tasks(dag_id: str, run_id: str) -> list[dict]:
    items = _get(f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances")["task_instances"]
    return [{"task_id": t["task_id"], "state": t["state"], "try_number": t["try_number"],
             "max_tries": t.get("max_tries"), "duration_s": round(t["duration"], 1) if t.get("duration") else None}
            for t in items]


class RunsIn(ToolInput):
    dag_id: str | None = None
    limit: int = Field(5, ge=1, le=20)


class RunIdIn(ToolInput):
    dag_run_id: str | None = Field(None, description="Run id; defaults to the latest run")
    dag_id: str | None = None


class TriggerIn(ToolInput):
    model: str = Field("rf", pattern="^(rf|gbm|logreg)$")
    max_depth: int = Field(8, ge=1, le=20)
    min_f1: float = Field(0.60, ge=0, le=1)
    wait_seconds: int = Field(0, ge=0, le=100, description="Seconds to wait for the run to finish")


class DiagIn(ToolInput):
    task_id: str
    dag_run_id: str | None = None
    try_number: int = Field(1, ge=1, le=5)


def _latest_run_id(dag_id: str) -> str:
    runs = _get(f"/dags/{dag_id}/dagRuns", order_by="-start_date", limit=1)["dag_runs"]
    if not runs:
        raise LookupError("no DAG runs recorded yet")
    return runs[0]["dag_run_id"]


@tool(MODULE, "List Airflow DAGs and whether they are paused.")
def airflow_list_dags(_: NoArgs):
    dags = _get("/dags")["dags"]
    return [{"dag_id": d["dag_id"], "paused": d["is_paused"], "tags": [t["name"] for t in d.get("tags", [])],
             "approved_for_operations": d["dag_id"] == get_settings().airflow_approved_dag} for d in dags]


@tool(MODULE, "Get recent DAG runs with their state (success/failed/running) for the training DAG.")
def airflow_get_dag_runs(a: RunsIn):
    dag = _approved(a.dag_id)
    data = _get(f"/dags/{dag}/dagRuns", order_by="-start_date", limit=a.limit)
    return {"dag_id": dag, "total": data["total_entries"], "runs": [_run_brief(r) for r in data["dag_runs"]]}


@tool(MODULE, "Get one DAG run's state and the state of every task in it (defaults to the latest run).")
def airflow_get_run_status(a: RunIdIn):
    dag = _approved(a.dag_id)
    rid = a.dag_run_id or _latest_run_id(dag)
    run = _get(f"/dags/{dag}/dagRuns/{rid}")
    return {**_run_brief(run), "tasks": _tasks(dag, rid)}


@tool(MODULE, "Identify failed tasks in a DAG run (defaults to the latest run) and which tasks were blocked downstream.")
def airflow_identify_failed_tasks(a: RunIdIn):
    dag = _approved(a.dag_id)
    rid = a.dag_run_id or _latest_run_id(dag)
    tasks = _tasks(dag, rid)
    return {"dag_run_id": rid, "run_state": _get(f"/dags/{dag}/dagRuns/{rid}")["state"],
            "failed": [t for t in tasks if t["state"] == "failed"],
            "blocked_downstream": [t["task_id"] for t in tasks if t["state"] == "upstream_failed"],
            "all_tasks": tasks}


@tool(MODULE, "Get execution details and failure diagnostics (log excerpt) for one task of a DAG run.")
def airflow_get_task_diagnostics(a: DiagIn):
    dag = _approved(None)
    rid = a.dag_run_id or _latest_run_id(dag)
    ti = _get(f"/dags/{dag}/dagRuns/{rid}/taskInstances/{a.task_id}")
    with _http() as c:
        r = c.get(f"/dags/{dag}/dagRuns/{rid}/taskInstances/{a.task_id}/logs/{a.try_number}",
                  headers={"Accept": "application/json"})
        r.raise_for_status()
        lines = r.json().get("content", [])
    keep = []
    for e in lines:
        ev = e.get("event", "") if isinstance(e, dict) else str(e)
        lvl = e.get("level", "") if isinstance(e, dict) else ""
        if lvl in ("error", "critical", "warning") or "stage=" in ev or "rejected" in ev:
            keep.append(f"[{lvl}] {ev}"[:350])
    return {"task_id": a.task_id, "dag_run_id": rid, "state": ti["state"], "try_number": ti["try_number"],
            "start": ti.get("start_date"), "end": ti.get("end_date"),
            "duration_s": ti.get("duration"), "log_excerpt": keep[-12:]}


@tool(MODULE, "Trigger the approved training DAG. Only allowed when the user explicitly asks to train/run the pipeline.",
      mutating=True, timeout=110)
def airflow_trigger_training(a: TriggerIn):
    ctx = current_request.get()
    if not trigger_allowed(ctx.user_text if ctx else ""):
        return {"_status": "rejected",
                "message": "training was not explicitly requested by the user; refusing to start a run"}
    dag = _approved(None)
    conf = {"model": a.model, "max_depth": a.max_depth, "min_f1": a.min_f1}
    with _http() as c:
        r = c.post(f"/dags/{dag}/dagRuns", json={"logical_date": None, "conf": conf})
        r.raise_for_status()
        rid = r.json()["dag_run_id"]
    # The run now exists. From here on, never lose its id: the API can be slow while a task executes.
    state, deadline, poll_errors = "queued", time.time() + a.wait_seconds, []
    while a.wait_seconds and time.time() < deadline:
        try:
            state = _get(f"/dags/{dag}/dagRuns/{rid}")["state"]
            if state in ("success", "failed"):
                break
        except httpx.HTTPError as exc:
            poll_errors.append(type(exc).__name__)
        time.sleep(4)
    try:
        tasks = _tasks(dag, rid)
    except httpx.HTTPError as exc:
        tasks, poll_errors = None, poll_errors + [type(exc).__name__]
    out = {"dag_id": dag, "dag_run_id": rid, "triggered_with": conf,
           "state_when_last_checked": state, "tasks": tasks}
    if poll_errors:
        out["status_check_problems"] = f"{len(poll_errors)} status check(s) failed ({sorted(set(poll_errors))}); " \
                                       "the run itself was created. Query its status again for the outcome."
    return out
