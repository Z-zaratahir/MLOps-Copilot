"""Structured JSON logging with secret redaction."""
import json
import logging
import re
import sys
import time
from pathlib import Path

SENSITIVE_KEY = re.compile(r"(key|token|password|secret|authorization|credential)", re.I)
SECRET_VALUE = re.compile(r"(gsk_[A-Za-z0-9]{10,}|AIza[0-9A-Za-z_\-]{20,}|eyJ[A-Za-z0-9_\-]{20,}\.[\w\-.]+"
                          r"|Bearer\s+[A-Za-z0-9_\-.]{12,})")
REDACTED = "***redacted***"


def redact(value):
    """Recursively mask secret-looking keys and values."""
    if isinstance(value, dict):
        return {k: (REDACTED if SENSITIVE_KEY.search(str(k)) else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return SECRET_VALUE.sub(REDACTED, value)
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
                   "level": record.levelname, "event": record.getMessage()}
        payload.update(getattr(record, "fields", {}))
        return json.dumps(redact(payload), default=str)


def get_logger(log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("agent")
    want = str(Path(log_file).resolve()) if log_file else None
    for h in list(logger.handlers):  # re-attach when the target file changes
        if getattr(h, "baseFilename", None) != want:
            logger.removeHandler(h)
            h.close()
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = JsonFormatter()
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    else:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def log_event(event: str, **fields) -> None:
    from agent.config import get_settings
    get_logger(get_settings().log_file).info(event, extra={"fields": fields})
