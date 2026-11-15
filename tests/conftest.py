import json
import logging
from types import SimpleNamespace

import pytest

from agent import policy
from agent.config import get_settings


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Fresh settings, log file and confirmation store for every test."""
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "agent.jsonl"))
    monkeypatch.setenv("AIRFLOW_PASSWORD", "unit-test-password")
    get_settings.cache_clear()
    lg = logging.getLogger("agent")
    for h in list(lg.handlers):
        lg.removeHandler(h)
    policy.clear_pending()
    yield
    get_settings.cache_clear()


class FakeLLM:
    """Scripted stand-in for the model: returns queued assistant messages in order."""

    def __init__(self, *messages):
        self.queue = list(messages)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append([m.get("content") for m in messages])
        return self.queue.pop(0)


def call(name, **args):
    return {"content": None, "tool_calls": [{"id": f"c-{name}", "type": "function",
                                             "function": {"name": name, "arguments": json.dumps(args)}}]}


def say(text):
    return {"content": text, "tool_calls": None}


def fake_run(name, rid, f1, recall=0.7, version="v3", gate=None, ended_ms=1_760_000_000_000):
    tags = {"dataset_version": version, "dataset_md5": f"md5-{version}", "dataset_git_commit": "abc"}
    if gate:
        tags["quality_gate"] = gate
    return SimpleNamespace(
        info=SimpleNamespace(run_id=rid, run_name=name, status="FINISHED", start_time=ended_ms, end_time=ended_ms),
        data=SimpleNamespace(tags=tags, params={"model_type": "rf"},
                             metrics={"f1": f1, "recall": recall, "precision": 0.6, "accuracy": 0.7, "roc_auc": 0.7}))
