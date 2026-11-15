"""Structured inputs/outputs shared by tools, orchestrator and CLI."""
from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["ok", "error", "timeout", "invalid_arguments", "rejected",
                 "pending_confirmation", "unknown_tool"]


class ToolResult(BaseModel):
    """What every tool call returns, whether it worked or not."""
    tool: str
    status: Status
    data: Any = None
    error: str | None = None
    duration_ms: int = 0
    evidence_id: str = ""
    request_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class PendingAction(BaseModel):
    token: str
    action: str
    params: dict
    namespace: str
    expires_at: float
    description: str


class AgentAnswer(BaseModel):
    request_id: str
    answer: str
    evidence: list[ToolResult] = Field(default_factory=list)
    grounded: bool = True
    notes: list[str] = Field(default_factory=list)
