"""Tool-calling LLM clients (Groq primary, Gemini fallback) over OpenAI-compatible endpoints.

Free-tier limits are per model (e.g. 8k tokens/minute on Groq), so on a rate limit the chain
switches to the next model first and only sleeps when every model is limited.
"""
import re
import time

import httpx

from agent.config import get_settings
from agent.logging_utils import log_event

ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}
MAX_WAIT_SECONDS = 65
ROUNDS = 3


class LLMError(Exception):
    """No configured provider could produce a response."""


class RateLimited(LLMError):
    def __init__(self, msg: str, delay: float):
        super().__init__(msg)
        self.delay = delay


def _retry_delay(resp: httpx.Response) -> float:
    hint = resp.headers.get("retry-after")
    if hint:
        try:
            return float(hint)
        except ValueError:
            pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", resp.text)
    return int(m.group(1) or 0) * 60 + float(m.group(2)) if m else 5.0


class ChatClient:
    def __init__(self, provider: str, api_key: str, model: str):
        self.provider, self.api_key, self.model = provider, api_key, model

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": 0.1}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        r = httpx.post(ENDPOINTS[self.provider], json=body, timeout=60,
                       headers={"Authorization": f"Bearer {self.api_key}"})
        if r.status_code == 429:
            raise RateLimited(f"{self.label} rate limited", _retry_delay(r))
        if r.status_code == 413:
            raise LLMError(f"{self.label} request too large for its token limit")
        if r.status_code >= 400:
            raise LLMError(f"{self.label} HTTP {r.status_code}: {r.text[:160]}")
        return r.json()["choices"][0]["message"]


class FallbackLLM:
    """Preferred provider's models first, then the other provider if configured."""

    def __init__(self):
        s = get_settings()
        groq_models = [s.groq_model] + [m.strip() for m in s.groq_fallback_models.split(",") if m.strip()]
        chains = {"groq": [ChatClient("groq", s.groq_api_key, m) for m in groq_models],
                  "gemini": [ChatClient("gemini", s.gemini_api_key, s.gemini_model)]}
        order = [s.llm_provider] + [p for p in chains if p != s.llm_provider]
        self.clients = [c for p in order for c in chains[p] if c.api_key]
        self.last_used: str | None = None

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        errors: list[str] = []
        for round_no in range(ROUNDS):
            delays = []
            for c in self.clients:
                try:
                    msg = c.complete(messages, tools)
                    self.last_used = c.label
                    return msg
                except RateLimited as exc:
                    delays.append(exc.delay)
                    log_event("llm_rate_limited", model=c.label, retry_in_s=round(exc.delay, 1))
                except (LLMError, httpx.HTTPError) as exc:
                    errors.append(f"{type(exc).__name__}: {str(exc)[:140]}")
            if delays and round_no < ROUNDS - 1 and min(delays) <= MAX_WAIT_SECONDS:
                time.sleep(min(delays) + 0.5)  # every model is limited: wait for the soonest reset
                continue
            if delays:
                errors.append(f"all models rate limited (soonest reset {min(delays):.0f}s)")
            break
        raise LLMError("; ".join(errors) or "no LLM provider configured (missing API keys)")
