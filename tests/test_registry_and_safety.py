import json
import logging
import time

import pytest

from agent import policy, registry
from agent.logging_utils import REDACTED, redact
from agent.registry import NoArgs, RequestContext, ToolInput, execute, tool
from agent.tools import airflow_tools, dvc_tools, kubernetes_tools  # noqa: F401


def ctx(text="hello"):
    return RequestContext(request_id="req-test", user_text=text)


def test_invalid_arguments_are_rejected_with_reason():
    r = execute("list_runs", {"limit": 999}, ctx())
    assert r.status == "invalid_arguments" and "limit" in r.error
    r = execute("list_runs", {"surprise": 1}, ctx())
    assert r.status == "invalid_arguments"


def test_unknown_tool_is_reported_not_raised():
    assert execute("drop_database", {}, ctx()).status == "unknown_tool"


def test_slow_tool_times_out(monkeypatch):
    class Args(ToolInput):
        pass

    @tool("test", "sleeps", timeout=0.2)
    def slow_tool(a: Args):
        time.sleep(1.5)

    try:
        r = execute("slow_tool", {}, ctx())
    finally:
        registry.TOOLS.pop("slow_tool")
    assert r.status == "timeout" and "exceeded" in r.error


def test_service_failure_becomes_structured_error(monkeypatch):
    monkeypatch.setattr(registry.TOOLS["list_experiments"], "func",
                        lambda a: (_ for _ in ()).throw(ConnectionError("mlflow down")))
    r = execute("list_experiments", {}, ctx())
    assert r.status == "error" and "ConnectionError" in r.error and r.data is None


def test_restart_in_unapproved_namespace_is_rejected():
    r = execute("k8s_request_restart", {"deployment": "parcel-delay-api", "namespace": "kube-system"}, ctx())
    assert r.status == "rejected" and "not approved" in r.error
    assert policy._pending == {}


def test_scale_beyond_limit_is_rejected():
    r = execute("k8s_request_scale", {"deployment": "parcel-delay-api", "replicas": 15}, ctx())
    assert r.status == "rejected"


def test_cluster_change_needs_human_confirmation(monkeypatch):
    applied = []

    class FakeApps:
        def patch_namespaced_deployment(self, name, ns, body, **kw):
            applied.append((name, ns))

    monkeypatch.setattr(kubernetes_tools, "_api", lambda: (None, FakeApps()))
    pending = execute("k8s_request_restart", {"deployment": "parcel-delay-api"}, ctx())
    assert pending.status == "pending_confirmation" and applied == []
    token = pending.data["token"]
    # the model cannot call the hidden executor even though it can see the token
    assert execute("k8s_apply_confirmed", {"token": token}, ctx()).status == "unknown_tool"
    assert applied == []
    # the human path executes exactly once
    assert execute("k8s_apply_confirmed", {"token": token}, ctx(), allow_hidden=True).status == "ok"
    assert applied == [("parcel-delay-api", "parcel-ml")]
    assert execute("k8s_apply_confirmed", {"token": token}, ctx(), allow_hidden=True).status == "rejected"


def test_confirmation_token_expires(monkeypatch):
    pending = execute("k8s_request_restart", {"deployment": "parcel-delay-api"}, ctx())
    policy._pending[pending.data["token"]].expires_at = time.time() - 1
    r = execute("k8s_apply_confirmed", {"token": pending.data["token"]}, ctx(), allow_hidden=True)
    assert r.status == "rejected" and "expired" in r.error


def test_training_trigger_requires_explicit_user_intent():
    r = execute("airflow_trigger_training", {}, ctx("find the best model and tell me if we need retraining"))
    assert r.status == "rejected" and "not explicitly requested" in r.error
    assert policy.trigger_allowed("Trigger the model training pipeline and report the result")
    assert not policy.trigger_allowed("Explain whether retraining is necessary")


def test_dvc_commands_are_allow_listed():
    with pytest.raises(policy.PolicyViolation):
        dvc_tools.dvc("push")
    with pytest.raises(policy.PolicyViolation):
        dvc_tools.dvc("gc", "-f")


def test_hidden_tools_are_not_offered_to_the_model():
    names = {t["function"]["name"] for t in registry.llm_schemas()}
    assert "k8s_apply_confirmed" not in names and "k8s_request_restart" in names
    assert len(names) >= 25


def test_logs_are_structured_and_hide_secrets(tmp_path):
    secret = "gsk_" + "A1b2C3d4E5f6G7h8I9j0"
    execute("list_runs", {"limit": 999, "api_key": secret}, ctx())
    assert redact({"password": "x", "nested": {"token": "y"}, "msg": f"key {secret}"}) == {
        "password": REDACTED, "nested": {"token": REDACTED}, "msg": f"key {REDACTED}"}
    for h in logging.getLogger("agent").handlers:
        h.flush()
    line = json.loads((tmp_path / "agent.jsonl").read_text().splitlines()[-1])
    assert {"tool", "status", "duration_ms", "request_id"} <= set(line)
    assert line["request_id"] == "req-test" and secret not in json.dumps(line)


def test_tools_cover_all_four_systems_with_three_or_more_each():
    by_module = {}
    for t in registry.TOOLS.values():
        by_module.setdefault(t.module, []).append(t.name)
    for m in ("dvc", "mlflow", "airflow", "kubernetes"):
        assert len(by_module[m]) >= 3, m


def test_noargs_tool_accepts_empty_input():
    assert NoArgs.model_validate({}) is not None
