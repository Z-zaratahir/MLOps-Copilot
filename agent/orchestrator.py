"""Agent loop: LLM chooses tools, the registry runs them, grounding checks guard the answer."""
import json
import re
import uuid
from datetime import date

from agent import grounding, registry
from agent.config import get_settings
from agent.llm import LLMError
from agent.logging_utils import log_event
from agent.schemas import AgentAnswer, ToolResult
from agent.tools import (  # noqa: F401  (imports register the tools)
    airflow_tools,
    assessment_tools,
    dvc_tools,
    kubernetes_tools,
    mlflow_tools,
)

MAX_STEPS = 8
MAX_CALLS_PER_STEP = 6
MAX_TOOL_JSON = 6000
CONFIRM = re.compile(r"^\s*confirm\s+([0-9a-f]{6})\s*$", re.I)

SYSTEM_PROMPT = """You are the MLOps Copilot for a parcel-delay prediction model. You operate real systems through tools: \
DVC (dataset versions), MLflow (experiments, registry), Airflow 3 (training DAG '{dag}'), Kubernetes (namespace '{ns}').
Today is {today}.

Rules:
1. Every operational claim must come from a tool result in THIS conversation. Cite it as [E<number>] using the evidence_id. \
Never answer operational questions from memory. If a tool failed, timed out or was rejected, say so plainly and do not guess \
what it would have returned.
2. Do not call tools you do not need, but do call every tool a question requires. For a combined question use: MLflow \
(find_best_run) -> DVC (get_dataset_for_run) -> Kubernetes (k8s_get_deployed_version, k8s_find_unhealthy_pods) -> Airflow \
(airflow_get_dag_runs) -> assess_retraining_need, then correlate.
3. The highest metric is not automatically production-ready: use readiness_concerns and comparability warnings from the tools.
4. Never start training unless the user explicitly asks you to run/trigger it. Recommend retraining only when \
assess_retraining_need reports explicit triggers (freshness policy, failed quality gate, newer/drifted data, supplied degradation).
5. Cluster changes: you can only REQUEST them (k8s_request_restart / k8s_request_scale). Tell the user to type the exact \
`confirm <token>` themselves. Never claim a change happened before confirmation.
6. For combined or diagnostic questions answer with three headed sections: **Known facts** (cited), **Uncertainties**, \
**Recommended actions**. For simple lookups, answer briefly with citations.
7. Quote numbers and ids exactly as returned by tools."""


class Orchestrator:
    def __init__(self, llm):
        self.llm = llm
        self.history: list[dict] = []

    # ---------- public ----------
    def run(self, question: str) -> AgentAnswer:
        m = CONFIRM.match(question)
        if m:
            return self._confirm(m.group(1).lower())
        ctx = registry.RequestContext(request_id=uuid.uuid4().hex[:10], user_text=question)
        log_event("request_start", request_id=ctx.request_id, question_chars=len(question))
        s = get_settings()
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(
            dag=s.airflow_approved_dag, ns=s.k8s_approved_namespace, today=date.today().isoformat())},
            *self.history[-6:], {"role": "user", "content": question}]
        schemas, cache = registry.llm_schemas(), {}
        draft = None
        try:
            for _ in range(MAX_STEPS):
                msg = self.llm.complete(messages, schemas)
                calls = (msg.get("tool_calls") or [])[:MAX_CALLS_PER_STEP]
                if not calls:
                    draft = msg.get("content") or ""
                    break
                messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
                for c in calls:
                    messages.append(self._run_call(c, ctx, cache))
            else:
                draft = ""  # step budget exhausted without a final answer
        except LLMError as exc:
            log_event("llm_unavailable", request_id=ctx.request_id, error=str(exc)[:200])
            ans = self._finalize(ctx, "", llm_error=str(exc))
            return ans
        ans = self._finalize(ctx, draft)
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": ans.answer}]
        log_event("request_end", request_id=ctx.request_id, tools=len(ctx.results), grounded=ans.grounded)
        return ans

    # ---------- internals ----------
    def _run_call(self, call: dict, ctx, cache: dict) -> dict:
        fn = call.get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments") or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
        except (json.JSONDecodeError, ValueError) as exc:
            args = None
            res = registry.execute("__invalid_json__", {}, ctx)
            res.error = f"unparseable tool arguments for {name}: {exc}"
        key = (name, json.dumps(args, sort_keys=True))
        if args is not None:
            if key in cache:
                res = cache[key]
            else:
                res = cache[key] = registry.execute(name, args, ctx)
        payload = res.model_dump(include={"tool", "status", "data", "error", "evidence_id"})
        text = json.dumps(payload, default=str)
        if len(text) > MAX_TOOL_JSON:
            text = text[:MAX_TOOL_JSON] + '..."[truncated]"'
        return {"role": "tool", "tool_call_id": call.get("id", ""), "content": text}

    def _confirm(self, token: str) -> AgentAnswer:
        ctx = registry.RequestContext(request_id=uuid.uuid4().hex[:10], user_text=f"confirm {token}")
        res = registry.execute("k8s_apply_confirmed", {"token": token}, ctx, allow_hidden=True)
        if res.ok:
            text = f"Confirmed and executed: {res.data['executed']} [{res.evidence_id}]."
        else:
            text = f"Not executed: {res.error} [{res.evidence_id}]"
        log_event("confirmation", request_id=ctx.request_id, status=res.status)
        return AgentAnswer(request_id=ctx.request_id, answer=text, evidence=ctx.results)

    def _finalize(self, ctx, draft: str, llm_error: str | None = None) -> AgentAnswer:
        results: list[ToolResult] = ctx.results
        ok = [r for r in results if r.ok]
        failed = [r for r in results if not r.ok and r.status != "pending_confirmation"]
        notes: list[str] = []
        grounded = True
        if llm_error:
            answer = ("I could not reach any language model, so I cannot interpret results. "
                      f"Reason: {llm_error}")
            notes.append("llm_unavailable")
        elif not draft.strip():
            answer = "I could not complete this request within the allowed number of steps."
            notes.append("step_budget_exhausted")
        elif not ok and grounding.looks_operational(draft):
            # nothing verified, yet the draft makes operational claims: refuse to pass it on
            answer = ("I could not verify any operational facts for this request because the required tools did "
                      "not return data, so I will not guess.")
            notes.append("draft_discarded_no_evidence")
        else:
            answer = draft.strip()
            problems = grounding.check(answer, results)
            if problems["bad_citations"] or problems["unsupported_values"]:
                grounded = False
                notes.append(f"grounding_issues={problems}")
                answer += ("\n\n> Grounding warning: these cited references/values were not found in any tool "
                           f"output and should not be trusted: "
                           f"{', '.join(problems['bad_citations'] + problems['unsupported_values'])}")
        if failed:
            lines = [f"- {r.tool} [{r.evidence_id}]: {r.status} — {r.error}" for r in failed]
            answer += "\n\n**Tool failures (no data was obtained from these):**\n" + "\n".join(lines)
        pending = [r for r in results if r.status == "pending_confirmation"]
        for r in pending:
            answer += f"\n\n**Awaiting your confirmation** [{r.evidence_id}]: type `confirm {r.data['token']}` to run it."
        return AgentAnswer(request_id=ctx.request_id, answer=answer, evidence=results, grounded=grounded, notes=notes)
