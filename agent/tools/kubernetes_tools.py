"""Kubernetes operations tools. Reads are scoped to the approved namespace; writes need
explicit human confirmation (see agent/policy.py)."""
import json
from datetime import UTC, datetime

from kubernetes import client, config
from pydantic import Field

from agent import policy
from agent.config import get_settings
from agent.registry import NoArgs, ToolInput, tool

MODULE = "kubernetes"
TIMEOUT = 8
_loaded = {"ctx": None}


def _api():
    s = get_settings()
    if _loaded["ctx"] != s.k8s_context:
        config.load_kube_config(context=s.k8s_context)
        _loaded["ctx"] = s.k8s_context
    return client.CoreV1Api(), client.AppsV1Api()


def _ns() -> str:
    return get_settings().k8s_approved_namespace


def _age(ts) -> str:
    secs = int((datetime.now(UTC) - ts).total_seconds()) if ts else 0
    return f"{secs // 3600}h{secs % 3600 // 60}m" if secs >= 3600 else f"{secs // 60}m{secs % 60}s"


def _container_states(pod) -> list[dict]:
    out = []
    for cs in pod.status.container_statuses or []:
        state = cs.state
        if state.running:
            cur = {"state": "running", "since": str(state.running.started_at)}
        elif state.waiting:
            cur = {"state": "waiting", "reason": state.waiting.reason, "message": state.waiting.message}
        else:
            cur = {"state": "terminated", "reason": state.terminated.reason, "exit_code": state.terminated.exit_code}
        last = cs.last_state.terminated if cs.last_state and cs.last_state.terminated else None
        out.append({"name": cs.name, "ready": cs.ready, "restart_count": cs.restart_count, **cur,
                    "last_termination": {"reason": last.reason, "exit_code": last.exit_code} if last else None,
                    "image": cs.image})
    return out


def _pod_brief(pod) -> dict:
    cs = pod.status.container_statuses or []
    return {"name": pod.metadata.name, "phase": pod.status.phase,
            "ready": f"{sum(1 for c in cs if c.ready)}/{len(pod.spec.containers)}",
            "restarts": sum(c.restart_count for c in cs), "age": _age(pod.metadata.creation_timestamp),
            "node": pod.spec.node_name}


def _selector() -> str:
    return get_settings().k8s_app_label


class PodIn(ToolInput):
    pod: str


class LogsIn(ToolInput):
    pod: str | None = Field(None, description="Pod name; defaults to the first app pod")
    tail_lines: int = Field(40, ge=1, le=200)
    previous: bool = Field(False, description="Logs of the previous (crashed) container instance")


class RestartIn(ToolInput):
    deployment: str
    namespace: str | None = Field(None, description="Defaults to the approved namespace")


class ScaleIn(ToolInput):
    deployment: str
    replicas: int = Field(ge=0, le=20)
    namespace: str | None = None


class ApplyIn(ToolInput):
    token: str


@tool(MODULE, "List deployments in the application namespace with desired/ready replicas and image.")
def k8s_list_deployments(_: NoArgs):
    _, apps = _api()
    return [{"name": d.metadata.name, "desired": d.spec.replicas, "ready": d.status.ready_replicas or 0,
             "available": d.status.available_replicas or 0, "updated": d.status.updated_replicas or 0,
             "image": d.spec.template.spec.containers[0].image}
            for d in apps.list_namespaced_deployment(_ns(), _request_timeout=TIMEOUT).items]


@tool(MODULE, "List application pods with phase, readiness, restart count and age.")
def k8s_list_pods(_: NoArgs):
    core, _a = _api()
    pods = core.list_namespaced_pod(_ns(), label_selector=_selector(), _request_timeout=TIMEOUT).items
    return {"namespace": _ns(), "count": len(pods), "pods": [_pod_brief(p) for p in pods]}


@tool(MODULE, "Get detailed status of one pod: container states, restart reasons, conditions.")
def k8s_get_pod_status(a: PodIn):
    core, _a = _api()
    p = core.read_namespaced_pod(a.pod, _ns(), _request_timeout=TIMEOUT)
    return {**_pod_brief(p), "conditions": {c.type: c.status for c in p.status.conditions or []},
            "containers": _container_states(p), "labels": p.metadata.labels}


@tool(MODULE, "Retrieve recent log lines of an application pod.")
def k8s_get_pod_logs(a: LogsIn):
    core, _a = _api()
    pod = a.pod or core.list_namespaced_pod(_ns(), label_selector=_selector(),
                                            _request_timeout=TIMEOUT).items[0].metadata.name
    text = core.read_namespaced_pod_log(pod, _ns(), tail_lines=a.tail_lines, previous=a.previous,
                                        _request_timeout=TIMEOUT)
    return {"pod": pod, "previous": a.previous, "lines": text.strip().splitlines()[-a.tail_lines:]}


@tool(MODULE, "Inspect Services: type, cluster IP, ports, selector and number of ready endpoints.")
def k8s_list_services(_: NoArgs):
    core, _a = _api()
    out = []
    for s in core.list_namespaced_service(_ns(), _request_timeout=TIMEOUT).items:
        ep = core.read_namespaced_endpoints(s.metadata.name, _ns(), _request_timeout=TIMEOUT)
        ready = sum(len(sub.addresses or []) for sub in ep.subsets or [])
        out.append({"name": s.metadata.name, "type": s.spec.type, "cluster_ip": s.spec.cluster_ip,
                    "ports": [f"{p.port}->{p.target_port}" for p in s.spec.ports],
                    "selector": s.spec.selector, "ready_endpoints": ready})
    return out


@tool(MODULE, "Find unhealthy pods (not ready, restarting, crash/pull errors, OOM) and the recent warning events.")
def k8s_find_unhealthy_pods(_: NoArgs):
    core, apps = _api()
    pods = core.list_namespaced_pod(_ns(), label_selector=_selector(), _request_timeout=TIMEOUT).items
    problems = []
    for p in pods:
        reasons = []
        if p.status.phase != "Running":
            reasons.append(f"phase={p.status.phase}")
        for c in _container_states(p):
            if not c["ready"]:
                reasons.append(f"container {c['name']} not ready ({c.get('reason') or c['state']})")
            if c["restart_count"]:
                reasons.append(f"{c['restart_count']} restart(s), last: "
                               f"{(c['last_termination'] or {}).get('reason', 'unknown')}")
        if reasons:
            problems.append({**_pod_brief(p), "reasons": reasons})
    events = core.list_namespaced_event(_ns(), _request_timeout=TIMEOUT).items
    warns = sorted((e for e in events if e.type == "Warning"),
                   key=lambda e: e.last_timestamp or e.event_time or e.metadata.creation_timestamp)[-8:]
    deps = [{"name": d.metadata.name, "desired": d.spec.replicas, "ready": d.status.ready_replicas or 0}
            for d in apps.list_namespaced_deployment(_ns(), _request_timeout=TIMEOUT).items]
    return {"namespace": _ns(), "pods_checked": len(pods), "unhealthy": problems,
            "healthy": not problems and all(d["ready"] == d["desired"] for d in deps), "deployments": deps,
            "warning_events": [{"object": e.involved_object.name, "reason": e.reason,
                                "message": (e.message or "")[:220], "count": e.count} for e in warns]}


@tool(MODULE, "Report the currently deployed application version: image tag, config version, and what the running service reports.")
def k8s_get_deployed_version(_: NoArgs):
    core, apps = _api()
    s = get_settings()
    d = apps.read_namespaced_deployment(s.k8s_deployment, _ns(), _request_timeout=TIMEOUT)
    img = d.spec.template.spec.containers[0].image
    cm = core.read_namespaced_config_map("parcel-delay-config", _ns(), _request_timeout=TIMEOUT)
    live: dict | str
    try:
        raw = core.connect_get_namespaced_service_proxy_with_path(
            f"{s.k8s_service}:80", _ns(), path="version", _preload_content=False, _request_timeout=TIMEOUT)
        live = json.loads(raw.data)
    except Exception as exc:
        live = f"unavailable: {type(exc).__name__}"
    return {"deployment": d.metadata.name, "image": img, "image_tag": img.rsplit(":", 1)[-1],
            "configmap_app_version": cm.data.get("APP_VERSION"),
            "ready_replicas": f"{d.status.ready_replicas or 0}/{d.spec.replicas}",
            "generation": d.metadata.generation, "observed_generation": d.status.observed_generation,
            "running_service_reports": live}


@tool(MODULE, "REQUEST a rollout restart of a deployment. Does not execute: returns a confirmation token the human must approve.",
      mutating=True)
def k8s_request_restart(a: RestartIn):
    ns = a.namespace or _ns()
    try:
        pa = policy.request_action("restart_deployment", {"deployment": a.deployment}, ns,
                                   f"rollout restart deployment/{a.deployment} in namespace {ns}")
    except policy.PolicyViolation as exc:
        return {"_status": "rejected", "message": str(exc)}
    return {"_status": "pending_confirmation", "token": pa.token, "action": pa.description,
            "expires_in_s": policy.CONFIRM_TTL_SECONDS,
            "message": f"Not executed. The user must type: confirm {pa.token}"}


@tool(MODULE, "REQUEST scaling a deployment. Does not execute: returns a confirmation token the human must approve.",
      mutating=True)
def k8s_request_scale(a: ScaleIn):
    ns = a.namespace or _ns()
    try:
        pa = policy.request_action("scale_deployment", {"deployment": a.deployment, "replicas": a.replicas}, ns,
                                   f"scale deployment/{a.deployment} to {a.replicas} replicas in {ns}")
    except policy.PolicyViolation as exc:
        return {"_status": "rejected", "message": str(exc)}
    return {"_status": "pending_confirmation", "token": pa.token, "action": pa.description,
            "expires_in_s": policy.CONFIRM_TTL_SECONDS,
            "message": f"Not executed. The user must type: confirm {pa.token}"}


@tool(MODULE, "Internal: apply a human-confirmed cluster action.", mutating=True, hidden=True)
def k8s_apply_confirmed(a: ApplyIn):
    try:
        pa = policy.consume(a.token)
    except policy.PolicyViolation as exc:
        return {"_status": "rejected", "message": str(exc)}
    _core, apps = _api()
    dep = pa.params["deployment"]
    if pa.action == "restart_deployment":
        body = {"spec": {"template": {"metadata": {"annotations": {
            "kubectl.kubernetes.io/restartedAt": datetime.now(UTC).isoformat(timespec="seconds")}}}}}
        apps.patch_namespaced_deployment(dep, pa.namespace, body, _request_timeout=TIMEOUT)
    else:
        apps.patch_namespaced_deployment_scale(dep, pa.namespace, {"spec": {"replicas": pa.params["replicas"]}},
                                               _request_timeout=TIMEOUT)
    return {"executed": pa.description, "namespace": pa.namespace}
