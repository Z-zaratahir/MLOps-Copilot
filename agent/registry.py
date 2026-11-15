"""Tool registry: schemas for the LLM, validated + time-boxed + logged execution."""
import time
import typing
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, ConfigDict, ValidationError

from agent.config import get_settings
from agent.logging_utils import log_event
from agent.schemas import ToolResult


class ToolInput(BaseModel):
    """Base for tool arguments; unknown fields are rejected, not ignored."""
    model_config = ConfigDict(extra="forbid")


class NoArgs(ToolInput):
    pass


@dataclass
class RequestContext:
    request_id: str
    user_text: str = ""
    counter: int = 0
    results: list[ToolResult] = field(default_factory=list)

    def next_evidence_id(self) -> str:
        self.counter += 1
        return f"E{self.counter}"


current_request: ContextVar[RequestContext | None] = ContextVar("current_request", default=None)


@dataclass
class ToolSpec:
    name: str
    module: str
    description: str
    input_model: type[ToolInput]
    func: Callable
    mutating: bool = False
    timeout: float | None = None
    hidden: bool = False  # never offered to, nor callable by, the LLM


TOOLS: dict[str, ToolSpec] = {}
_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="tool")


def tool(module: str, description: str, mutating: bool = False, timeout: float | None = None,
         hidden: bool = False):
    def deco(fn):
        hints = typing.get_type_hints(fn)
        model = next((v for k, v in hints.items() if k != "return"), NoArgs)
        TOOLS[fn.__name__] = ToolSpec(fn.__name__, module, description, model, fn, mutating, timeout,
                                      hidden)
        return fn
    return deco


def llm_schemas(only: set[str] | None = None) -> list[dict]:
    out = []
    for t in TOOLS.values():
        if t.hidden or (only and t.name not in only):
            continue
        schema = t.input_model.model_json_schema()
        schema.pop("title", None)
        out.append({"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": schema}})
    return out


def execute(name: str, raw_args: dict | None, ctx: RequestContext,
            allow_hidden: bool = False) -> ToolResult:
    """Run one tool call. Never raises: every outcome is a structured ToolResult.

    allow_hidden is True only on the human-confirmation path, never for LLM requests."""
    started = time.perf_counter()
    eid = ctx.next_evidence_id()

    def finish(status: str, data=None, error: str | None = None) -> ToolResult:
        res = ToolResult(tool=name, status=status, data=data, error=error, evidence_id=eid,
                         duration_ms=int((time.perf_counter() - started) * 1000),
                         request_id=ctx.request_id)
        ctx.results.append(res)
        log_event("tool_call", request_id=ctx.request_id, tool=name, status=res.status,
                  duration_ms=res.duration_ms, evidence_id=eid,
                  error=(error or "")[:200] or None)
        return res

    spec = TOOLS.get(name)
    if spec is None or (spec.hidden and not allow_hidden):
        return finish("unknown_tool", error=f"no such tool: {name}")
    try:
        args = spec.input_model.model_validate(raw_args or {})
    except ValidationError as exc:
        msgs = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return finish("invalid_arguments", error=msgs)

    token = current_request.set(ctx)
    limit = spec.timeout or get_settings().agent_tool_timeout_seconds
    try:
        fut = _pool.submit(_run_in_context, spec.func, args, ctx)
        data = fut.result(timeout=limit)
        if isinstance(data, dict) and data.get("_status") in ("rejected", "pending_confirmation"):
            status = data.pop("_status")
            return finish(status, data=data, error=data.get("message") if status == "rejected" else None)
        return finish("ok", data=data)
    except FutureTimeout:
        return finish("timeout", error=f"{name} exceeded {limit:.0f}s")
    except Exception as exc:  # service down, bad credentials, etc.
        return finish("error", error=f"{type(exc).__name__}: {exc}")
    finally:
        current_request.reset(token)


def _run_in_context(fn, args, ctx):
    token = current_request.set(ctx)
    try:
        return fn(args)
    finally:
        current_request.reset(token)
