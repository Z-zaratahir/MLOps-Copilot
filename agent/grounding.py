"""Post-checks that keep answers tied to tool evidence."""
import json
import re

from agent.schemas import ToolResult

CITE = re.compile(r"\[E(\d+)\]")
NUMBER = re.compile(r"(?<![\w.])\d+\.\d{2,}(?![\w.])")
HEX_ID = re.compile(r"\b[0-9a-f]{32}\b")
OPERATIONAL = re.compile(r"\b[0-9a-f]{32}\b|\bv\d\b|\b\d+/\d+\b|\b\d+(\.\d+)?\s?%|\b\d+\.\d{2,}\b", re.I)


def _floats(obj, out: set):
    if isinstance(obj, bool):
        return
    if isinstance(obj, float):
        out.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _floats(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _floats(v, out)


def evidence_text(results: list[ToolResult]) -> str:
    return " ".join(json.dumps(r.data, default=str) for r in results if r.ok)


def check(answer: str, results: list[ToolResult]) -> dict:
    """Return problems found in a draft answer: bad citations, numbers/ids not in evidence."""
    ok = [r for r in results if r.ok]
    known = {int(r.evidence_id[1:]) for r in results}
    bad_cites = sorted({f"[E{n}]" for n in map(int, CITE.findall(answer)) if n not in known})
    blob = evidence_text(ok)
    floats: set = set()
    for r in ok:
        _floats(r.data, floats)
    unsupported = []
    for tok in NUMBER.findall(answer):
        v = float(tok)
        decimals = len(tok.split(".")[1])
        if tok in blob or any(round(f, decimals) == v for f in floats):
            continue
        unsupported.append(tok)
    unsupported += [h for h in HEX_ID.findall(answer) if h not in blob]
    return {"bad_citations": bad_cites, "unsupported_values": sorted(set(unsupported))}


def looks_operational(text: str) -> bool:
    return bool(OPERATIONAL.search(text))
