"""Safety policy: approved scope, two-step confirmation, intent checks.

The LLM can only *request* a cluster change. The change runs only when the human
types the matching `confirm <token>` themselves; that path never goes through the LLM.
"""
import re
import secrets
import time

from agent.config import get_settings
from agent.schemas import PendingAction

CONFIRM_TTL_SECONDS = 120
ALLOWED_K8S_ACTIONS = {"restart_deployment", "scale_deployment"}
MAX_REPLICAS = 5
TRIGGER_INTENT = re.compile(r"\b(trigger|run|start|launch|kick ?off|execute)\b.*\b(train|training|pipeline|dag)\b"
                            r"|\b(train|retrain)\b.*\b(now|again|model)\b", re.I)
ALLOWED_DVC_COMMANDS = {("status",), ("diff",), ("remote", "list"), ("list",)}


class PolicyViolation(Exception):
    pass


_pending: dict[str, PendingAction] = {}


def check_namespace(namespace: str) -> None:
    approved = get_settings().k8s_approved_namespace
    if namespace != approved:
        raise PolicyViolation(f"namespace '{namespace}' is not approved (allowed: '{approved}')")


def request_action(action: str, params: dict, namespace: str, description: str) -> PendingAction:
    if action not in ALLOWED_K8S_ACTIONS:
        raise PolicyViolation(f"action '{action}' is not an approved cluster operation")
    check_namespace(namespace)
    if action == "scale_deployment" and not 0 <= params.get("replicas", -1) <= MAX_REPLICAS:
        raise PolicyViolation(f"replicas must be between 0 and {MAX_REPLICAS}")
    pa = PendingAction(token=secrets.token_hex(3), action=action, params=params, namespace=namespace,
                       expires_at=time.time() + CONFIRM_TTL_SECONDS, description=description)
    _pending[pa.token] = pa
    return pa


def consume(token: str) -> PendingAction:
    """Single-use. Re-validates scope at execution time, not just request time."""
    pa = _pending.pop(token, None)
    if pa is None:
        raise PolicyViolation("unknown or already-used confirmation token")
    if time.time() > pa.expires_at:
        raise PolicyViolation("confirmation token expired; request the action again")
    check_namespace(pa.namespace)
    return pa


def trigger_allowed(user_text: str) -> bool:
    """A training run may only start when the user's own message asks for one."""
    return bool(TRIGGER_INTENT.search(user_text or ""))


def dvc_command_allowed(args: list[str]) -> bool:
    return any(tuple(args[: len(a)]) == a for a in ALLOWED_DVC_COMMANDS)


def clear_pending() -> None:
    _pending.clear()
