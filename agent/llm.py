"""Tool-calling LLM clients (Groq primary, Gemini fallback) over OpenAI-compatible endpoints."""
import time

import httpx

from agent.config import get_settings

ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}


class LLMError(Exception):
    """No configured provider could produce a response."""


class ChatClient:
    def __init__(self, provider: str, api_key: str, model: str):
        self.provider, self.api_key, self.model = provider, api_key, model

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": 0.1}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        for attempt in range(2):
            r = httpx.post(ENDPOINTS[self.provider], json=body, timeout=45,
                           headers={"Authorization": f"Bearer {self.api_key}"})
            if r.status_code == 429 and attempt == 0:
                time.sleep(min(float(r.headers.get("retry-after", 3)), 8))
                continue
            if r.status_code >= 400:
                raise LLMError(f"{self.provider} HTTP {r.status_code}: {r.text[:200]}")
            return r.json()["choices"][0]["message"]
        raise LLMError(f"{self.provider} rate limited")


class FallbackLLM:
    """Tries the preferred provider first, then the other one if configured."""

    def __init__(self):
        s = get_settings()
        clients = {"groq": ChatClient("groq", s.groq_api_key, s.groq_model),
                   "gemini": ChatClient("gemini", s.gemini_api_key, s.gemini_model)}
        order = [s.llm_provider] + [p for p in clients if p != s.llm_provider]
        self.clients = [clients[p] for p in order if clients[p].api_key]
        self.last_provider: str | None = None

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        errors = []
        for c in self.clients:
            try:
                msg = c.complete(messages, tools)
                self.last_provider = c.provider
                return msg
            except (LLMError, httpx.HTTPError) as exc:
                errors.append(f"{c.provider}: {type(exc).__name__}: {str(exc)[:160]}")
        raise LLMError("; ".join(errors) or "no LLM provider configured (missing API keys)")
