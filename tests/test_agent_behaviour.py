import httpx
import pytest

from agent import grounding, registry
from agent.llm import ChatClient, FallbackLLM, LLMError, RateLimited
from agent.orchestrator import Orchestrator, missing_areas
from agent.schemas import ToolResult
from agent.tools import assessment_tools, mlflow_tools
from tests.conftest import FakeLLM, call, fake_run, say


def patch_tool(monkeypatch, name, fn):
    monkeypatch.setattr(registry.TOOLS[name], "func", fn)


def test_agent_selects_and_runs_the_right_tool(monkeypatch):
    patch_tool(monkeypatch, "get_dataset_for_run",
               lambda a: {"run": "r25", "dataset_version": "v2.1", "dataset_git_commit": "a4c92e1"})
    llm = FakeLLM(call("get_dataset_for_run", run_ref="25"),
                  say("Experiment 25 used dataset version v2.1 at commit a4c92e1 [E1]."))
    ans = Orchestrator(llm).run("Which dataset version was used for experiment 25?")
    assert [e.tool for e in ans.evidence] == ["get_dataset_for_run"]
    assert ans.evidence[0].ok and ans.grounded and "v2.1" in ans.answer


def test_agent_does_not_invent_results_when_tools_are_down(monkeypatch):
    def down(a):
        raise ConnectionError("connection refused")
    patch_tool(monkeypatch, "find_best_run", down)
    llm = FakeLLM(call("find_best_run"), say("The best model is run 3df3 with F1 0.9123 on v3 [E1]."))
    ans = Orchestrator(llm).run("What is the best model?")
    assert "0.9123" not in ans.answer and "draft_discarded_no_evidence" in ans.notes
    assert "find_best_run" in ans.answer and "connection refused" in ans.answer


def test_failed_tool_is_disclosed_even_when_other_tools_worked(monkeypatch):
    patch_tool(monkeypatch, "dvc_current_version", lambda a: {"version": "v3"})
    patch_tool(monkeypatch, "k8s_list_pods", lambda a: (_ for _ in ()).throw(TimeoutError("api timeout")))
    llm = FakeLLM(call("dvc_current_version"), call("k8s_list_pods"), say("Data is on v3 [E1]."))
    ans = Orchestrator(llm).run("data version and pods?")
    assert "Tool failures" in ans.answer and "k8s_list_pods" in ans.answer


def test_llm_outage_yields_safe_message_without_guessing():
    class Dead:
        def complete(self, m, t):
            raise LLMError("no provider")
    ans = Orchestrator(Dead()).run("Is the model deployed?")
    assert "could not reach any language model" in ans.answer and ans.evidence == []


def test_unsupported_numbers_and_citations_are_flagged(monkeypatch):
    patch_tool(monkeypatch, "find_best_run", lambda a: {"best": {"f1": 0.7123}})
    llm = FakeLLM(call("find_best_run"), say("F1 is 0.7123 [E1] but recall is 0.8899 [E7]."))
    ans = Orchestrator(llm).run("best model?")
    assert not ans.grounded and "0.8899" in ans.answer and "[E7]" in ans.answer.split("Grounding warning")[1]


def test_grounding_accepts_values_and_differences_present_in_evidence():
    res = [ToolResult(tool="t", status="ok", data={"a": 0.7123, "b": 0.7003}, evidence_id="E1")]
    assert grounding.check("a=0.7123, gap 0.012 [E1]", res) == {"bad_citations": [], "unsupported_values": []}


def test_multi_system_question_is_pushed_to_query_every_system(monkeypatch):
    patch_tool(monkeypatch, "find_best_run", lambda a: {"best": "r1"})
    patch_tool(monkeypatch, "get_dataset_for_run", lambda a: {"dataset_version": "v3"})
    patch_tool(monkeypatch, "k8s_get_deployed_version", lambda a: {"image": "x:1"})
    patch_tool(monkeypatch, "airflow_get_dag_runs", lambda a: {"runs": []})
    llm = FakeLLM(call("find_best_run"), say("done early"),
                  call("get_dataset_for_run", run_ref="1"), call("k8s_get_deployed_version"),
                  call("airflow_get_dag_runs"), say("Complete answer."))
    q = "Find the best model, its dataset version, the deployment status and whether retraining is needed"
    ans = Orchestrator(llm).run(q)
    assert {e.tool for e in ans.evidence} >= {"find_best_run", "get_dataset_for_run", "k8s_get_deployed_version"}
    assert missing_areas(q, ans.evidence) == []
    assert any("not queried" in str(m) or "have not queried" in str(m) for call_ in llm.calls for m in call_)


def test_best_run_is_not_called_production_ready_just_for_top_f1(monkeypatch):
    runs = [fake_run("old-high", "r1", 0.80, recall=0.30, version="v1"), fake_run("new", "r2", 0.70, gate="passed")]
    monkeypatch.setattr(mlflow_tools, "all_runs", lambda *a, **k: runs)
    out = mlflow_tools.find_best_run(mlflow_tools.BestRunIn())
    assert out["best"]["run_name"] == "old-high"
    joined = " ".join(out["readiness_concerns"])
    assert "no passed quality gate" in joined and "newest data is v3" in joined and "low recall" in joined
    assert out["comparability_warning"]


def test_retraining_assessment_ignores_degradation_the_user_never_reported(monkeypatch):
    monkeypatch.setattr(assessment_tools, "_best_summary", lambda: {
        "run_name": "r", "run_id": "r1", "dataset_version": "v3", "dataset_md5": "m", "metrics": {}, "quality_gate": "passed",
        "ended": None})
    monkeypatch.setattr(assessment_tools.dvc_tools, "current", lambda: {"version": "v3", "pointer_md5": "m"})
    monkeypatch.setattr(assessment_tools.airflow_tools, "_get", lambda *a, **k: {"dag_runs": [{"dag_run_id": "d", "state": "success"}]})
    monkeypatch.setattr(assessment_tools.kubernetes_tools, "k8s_get_deployed_version", lambda a: {"running_service_reports": "unavailable"})
    r = registry.execute("assess_retraining_need", {"observed_degradation": 0.4}, registry.RequestContext("r", "is retraining needed?"))
    assert r.ok and r.data["decision"] == "retraining_not_indicated"
    assert any("ignored degradation" in u for u in r.data["uncertainties"])


def test_failed_quality_gate_and_stale_data_trigger_a_recommendation(monkeypatch):
    monkeypatch.setattr(assessment_tools, "_best_summary", lambda: {
        "run_name": "r", "run_id": "r1", "dataset_version": "v1", "dataset_md5": "old", "metrics": {}, "quality_gate": "passed",
        "ended": None})
    monkeypatch.setattr(assessment_tools.dvc_tools, "current", lambda: {"version": "v3", "pointer_md5": "new"})
    monkeypatch.setattr(assessment_tools.dvc_tools, "dvc_compare_versions", lambda a: {"delta_stats": {"delay_rate": 0.24}})
    monkeypatch.setattr(assessment_tools.airflow_tools, "_get", lambda *a, **k: {"dag_runs": [{"dag_run_id": "d", "state": "failed"}]})
    monkeypatch.setattr(assessment_tools.airflow_tools, "_tasks", lambda *a: [{"task_id": "quality_gate", "state": "failed"}])
    monkeypatch.setattr(assessment_tools.kubernetes_tools, "k8s_get_deployed_version", lambda a: {})
    r = registry.execute("assess_retraining_need", {}, registry.RequestContext("r", "retrain?"))
    kinds = {t["type"] for t in r.data["triggers"]}
    assert r.data["decision"] == "retrain_recommended" and {"newer_dataset_with_drift", "failed_quality_gate"} <= kinds
    assert "no training run was started" in r.data["note"]


def test_llm_chain_switches_model_when_one_is_rate_limited(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    from agent.config import get_settings
    get_settings.cache_clear()
    seen = []

    def fake(self, messages, tools):
        seen.append(self.model)
        if len(seen) == 1:
            raise RateLimited("limited", 30)
        return {"content": "ok", "tool_calls": None}

    monkeypatch.setattr(ChatClient, "complete", fake)
    llm = FallbackLLM()
    assert llm.complete([], [])["content"] == "ok" and len(set(seen)) == 2 and llm.last_used.startswith("groq/")


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_model_reply_is_not_passed_off_as_an_answer(bad):
    ans = Orchestrator(FakeLLM(say(bad))).run("hello")
    assert "step_budget_exhausted" in ans.notes


def test_httpx_errors_are_handled_as_llm_failures(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    from agent.config import get_settings
    get_settings.cache_clear()

    def boom(self, m, t):
        raise httpx.ConnectError("network down")
    monkeypatch.setattr(ChatClient, "complete", boom)
    with pytest.raises(LLMError):
        FallbackLLM().complete([], [])
