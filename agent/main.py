"""Command-line interface: `python -m agent.main` (interactive) or `-q "question"`."""
import argparse
import json
import time
from pathlib import Path

from agent.config import ROOT
from agent.llm import FallbackLLM
from agent.orchestrator import Orchestrator
from agent.schemas import AgentAnswer

TRANSCRIPTS = ROOT / "logs" / "transcripts"


def render(ans: AgentAnswer) -> str:
    rows = [f"  {r.evidence_id:<4} {r.tool:<32} {r.status:<20} {r.duration_ms:>6} ms" for r in ans.evidence]
    ev = "\n".join(rows) if rows else "  (no tools called)"
    return f"\n{ans.answer}\n\n--- evidence (request {ans.request_id}) ---\n{ev}\n"


def save_transcript(question: str, ans: AgentAnswer, name: str | None) -> Path:
    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPTS / f"{name or time.strftime('%Y%m%d-%H%M%S')}.md"
    ev = "\n".join(f"### {r.evidence_id} `{r.tool}` -> {r.status} ({r.duration_ms} ms)\n```json\n"
                   f"{json.dumps(r.data if r.ok else {'error': r.error}, indent=1, default=str)[:3500]}\n```"
                   for r in ans.evidence)
    path.write_text(f"# Question\n{question}\n\n# Answer\n{ans.answer}\n\n# Tool evidence\n{ev}\n"
                    f"\ngrounded: {ans.grounded}  notes: {ans.notes}\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="MLOps Copilot")
    ap.add_argument("-q", "--question", help="ask one question and exit")
    ap.add_argument("--save", metavar="NAME", help="save a transcript under logs/transcripts/NAME.md")
    a = ap.parse_args()
    agent = Orchestrator(FallbackLLM())
    if a.question:
        ans = agent.run(a.question)
        print(render(ans))
        if a.save:
            print("transcript:", save_transcript(a.question, ans, a.save))
        return
    print("MLOps Copilot. Ask about datasets, experiments, pipelines or the deployment. "
          "Type 'confirm <token>' to approve a requested cluster change, 'exit' to quit.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            ans = agent.run(q)
            print(render(ans))
            if a.save:
                save_transcript(q, ans, f"{a.save}-{ans.request_id}")


if __name__ == "__main__":
    main()
