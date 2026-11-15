"""Tool-calling LLM clients (Groq primary, Gemini fallback) over OpenAI-compatible endpoints.

Every model has its own rate-limit bucket, so the chain tries several Groq models before
falling back to Gemini.
"""
import re
import time

import httpx

from agent.config import get_settings

ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}
MAX_WAIT_SECONDS = 20


class LLMError(Exception):
    """No configured provider could produce a response."""


def _retry_delay(resp: httpx.Response) -> float | None:
    """Seconds the server asks us to wait, if short enough to be worth waiting."""
    hint = resp.headers.get("retry-after")
    if not hint:
        m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", resp.text)
        hint = (int(m.group(1) or 0) * 60 + float(m.group(2))) if m else None
    try:
        delay = float(hint) if hint is not None else 3.0
    except ValueError:
        delay = 3.0
    return delay + 0.5 if delay <= MAX_WAIT_SECONDS else None


class ChatClient:
    def __init__(self, provider: str, api_key: str, model: str):
        self.provider, self.api_key, self.model = provider, api_key, model

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": 0.1}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        for attempt in range(3):
            r = httpx.post(ENDPOINTS[self.provider], json=body, timeout=60,
                           headers={"Authorization": f"Bearer {self.api_key}"})
            if r.status_code == 429:
                delay = _retry_delay(r)
                if delay is not None and attempt < 2:
                    time.sleep(delay)
                    continue
                raise LLMError(f"{self.provider}/{self.model} rate limited")
            if r.status_code >= 400:
                raise LLMError(f"{self.provider}/{self.model} HTTP {r.status_code}: {r.text[:160]}")
            return r.json()["choices"][0]["message"]
        raise LLMError(f"{self.provider}/{self.model} rate limited")


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
        errors = []
        for c in self.clients:
            try:
                msg = c.complete(messages, tools)
                self.last_used = f"{c.provider}/{c.model}"
                return msg
            except (LLMError, httpx.HTTPError) as exc:
                errors.append(f"{type(exc).__name__}: {str(exc)[:140]}")
        raise LLMError("; ".join(errors) or "no LLM provider configured (missing API keys)")
