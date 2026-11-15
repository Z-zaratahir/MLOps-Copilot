"""DVC data-management tools. Only allow-listed read-only DVC/git commands are ever run."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from pydantic import Field

from agent.config import ROOT
from agent.policy import PolicyViolation, dvc_command_allowed
from agent.registry import NoArgs, ToolInput, tool
from agent.tools import mlflow_tools

MODULE = "dvc"
POINTER = "data/parcels.csv.dvc"
DATA_FILE = "data/parcels.csv"


def _run(cmd: list[str], timeout: int = 20) -> str:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} failed: {(p.stderr or p.stdout).strip()[:300]}")
    return p.stdout.strip()


def git(*args: str) -> str:
    return _run(["git", *args])


def dvc(*args: str) -> str:
    if not dvc_command_allowed(list(args)):
        raise PolicyViolation(f"dvc {' '.join(args)} is not an approved command")
    return _run([sys.executable, "-m", "dvc", *args], timeout=40)


def _norm(v: str) -> str:
    return v if v.startswith("dataset-") else f"dataset-{v}"


def _pointer_at(ref: str | None) -> dict:
    text = git("show", f"{ref}:{POINTER}") if ref else (ROOT / POINTER).read_text()
    out = yaml.safe_load(text)["outs"][0]
    return {"md5": out["md5"], "size_bytes": out["size"], "path": out["path"]}


def versions() -> list[dict]:
    res = []
    for tag in git("tag", "--list", "dataset-*").split():
        commit = git("rev-list", "-n", "1", tag)
        res.append({"version": tag.removeprefix("dataset-"), "tag": tag, "git_commit": commit,
                    "committed": git("show", "-s", "--format=%cI", commit), **_pointer_at(tag)})
    return sorted(res, key=lambda v: v["committed"])


def _remote_dir() -> Path | None:
    line = dvc("remote", "list").splitlines()
    if not line:
        return None
    path = Path(line[0].split()[1])
    return path if path.is_absolute() else (ROOT / ".dvc" / path).resolve()


def blob_path(md5: str) -> tuple[Path | None, str]:
    rel = Path("files") / "md5" / md5[:2] / md5[2:]
    for label, base in (("local DVC cache", ROOT / ".dvc" / "cache"), ("DVC remote", _remote_dir())):
        if base and (base / rel).exists():
            return base / rel, label
    return None, "not found"


def stats(md5: str) -> dict | None:
    path, _ = blob_path(md5)
    if not path:
        return None
    df = pd.read_csv(path)
    return {"rows": len(df), "delay_rate": round(float(df["delayed"].mean()), 4),
            "mean_distance_km": round(float(df["distance_km"].mean()), 3),
            "mean_rider_load": round(float(df["rider_load"].mean()), 3),
            "storm_or_rain_share": round(float(df["weather"].isin(["rain", "storm"]).mean()), 4)}


def current() -> dict:
    ptr = _pointer_at(None)
    vs = {v["md5"]: v for v in versions()}
    match = vs.get(ptr["md5"])
    actual = hashlib.md5((ROOT / DATA_FILE).read_bytes()).hexdigest() if (ROOT / DATA_FILE).exists() else None
    return {"dataset": DATA_FILE, "pointer_md5": ptr["md5"],
            "version": match["version"] if match else "untagged",
            "git_commit": match["git_commit"] if match else git("rev-parse", "HEAD"),
            "workspace_file_md5": actual, "workspace_matches_pointer": actual == ptr["md5"],
            "head_commit": git("rev-parse", "--short", "HEAD")}


class CompareIn(ToolInput):
    version_a: str | None = Field(None, description="e.g. 'v1'; defaults to the version before version_b")
    version_b: str | None = Field(None, description="e.g. 'v3'; defaults to the current version")


class VersionIn(ToolInput):
    version: str = Field(description="Dataset version, e.g. 'v2'")


class RunRefIn(ToolInput):
    run_ref: str = Field(description="Run id, run name, or ordinal number like '25'")


@tool(MODULE, "List datasets tracked by DVC and every recorded dataset version (tag, hash, git commit).")
def dvc_list_datasets(_: NoArgs):
    tracked = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.dvc")
               if ".venv" not in p.parts and ".dvc" not in p.parts]
    return {"tracked_pointer_files": tracked, "versions": versions()}


@tool(MODULE, "Get the dataset version currently checked out, its git commit and whether the file matches its DVC pointer.")
def dvc_current_version(_: NoArgs):
    return current()


@tool(MODULE, "Detect whether the tracked data changed (runs `dvc status`, compares hashes and git state).")
def dvc_detect_changes(_: NoArgs):
    cur = current()
    status = dvc("status", "--json")
    dirty = git("status", "--porcelain", "--", "data")
    return {"changed": bool(json.loads(status or "{}")) or not cur["workspace_matches_pointer"] or bool(dirty),
            "dvc_status": json.loads(status or "{}") or "up to date",
            "workspace_matches_pointer": cur["workspace_matches_pointer"],
            "uncommitted_git_changes": dirty or None, "current_version": cur["version"]}


@tool(MODULE, "Compare two dataset versions: hash/size differences plus distribution statistics (rows, delay rate, weather, load).")
def dvc_compare_versions(a: CompareIn):
    vs = versions()
    names = [v["version"] for v in vs]
    b = a.version_b or current()["version"]
    if b not in names:
        raise LookupError(f"unknown version '{b}'; known: {names}")
    prev = names[names.index(b) - 1] if names.index(b) > 0 else None
    av = a.version_a or prev
    if av not in names:
        raise LookupError(f"unknown version '{av}'; known: {names}")
    pa, pb = (next(v for v in vs if v["version"] == x) for x in (av, b))
    sa, sb = stats(pa["md5"]), stats(pb["md5"])
    diff = {k: round(sb[k] - sa[k], 4) for k in sa} if sa and sb else None
    return {"from": {"version": av, **pa, "stats": sa}, "to": {"version": b, **pb, "stats": sb},
            "identical": pa["md5"] == pb["md5"], "delta_stats": diff,
            "note": None if sa and sb else "data blob unavailable in cache/remote; only hashes compared"}


@tool(MODULE, "Explain, with verified commands, how to reproduce a given dataset version.")
def dvc_reproduce_instructions(a: VersionIn):
    match = next((v for v in versions() if v["version"] == a.version), None)
    if not match:
        raise LookupError(f"unknown version '{a.version}'")
    path, where = blob_path(match["md5"])
    return {"version": a.version, "git_commit": match["git_commit"], "md5": match["md5"],
            "data_available_in": where,
            "steps": [f"git checkout {match['tag']} -- {POINTER}   # restore the version's pointer file",
                      f"dvc checkout {POINTER}                      # restore data from the DVC cache",
                      f"# if the cache is empty: dvc pull {POINTER}  # fetch from the configured remote",
                      f"md5sum {DATA_FILE}   # must equal {match['md5']}",
                      f"git checkout HEAD -- {POINTER} && dvc checkout   # return to current version"],
            "verified_retrievable": path is not None}


@tool(MODULE, "Find which dataset version (hash and git commit) was used to train a given run/experiment, from recorded MLflow provenance.")
def get_dataset_for_run(a: RunRefIn):
    run = mlflow_tools.resolve_run(a.run_ref)
    t = run.data.tags
    if "dataset_md5" not in t:
        return {"run": run.info.run_name, "provenance_recorded": False,
                "message": "this run has no recorded dataset provenance"}
    path, where = blob_path(t["dataset_md5"])
    return {"run": run.info.run_name, "run_id": run.info.run_id, "provenance_recorded": True,
            "dataset_version": t["dataset_version"], "dataset_md5": t["dataset_md5"],
            "dataset_git_commit": t["dataset_git_commit"],
            "file_matched_pointer_at_training": t.get("dataset_file_matches_pointer"),
            "retrievable_from": where,
            "reproduce_with": f"git checkout dataset-{t['dataset_version']} -- {POINTER} && dvc checkout {POINTER}"}
