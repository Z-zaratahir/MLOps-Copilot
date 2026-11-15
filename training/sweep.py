"""Reproducible hyper-parameter sweep across dataset versions.

Older pointers are restored with `git show` (worktree only), never `git checkout <tag>`,
so nothing is staged and the repository's recorded dataset pointer is untouched.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POINTER = "data/parcels.csv.dvc"

SWEEP = {
    "v2": [("gbm", dict(max_depth=d, learning_rate=lr, n_estimators=150))
           for d in (2, 3, 4) for lr in (0.05, 0.1)],
    "v3": [("gbm", dict(max_depth=d, learning_rate=0.1, n_estimators=n)) for d in (2, 3, 4, 5) for n in (100, 200)]
          + [("rf", dict(max_depth=d, n_estimators=150, learning_rate=0.1)) for d in (4, 6, 10, 12)]
          + [("logreg", dict(max_depth=3, learning_rate=0.1, n_estimators=100))],
}


def sh(*cmd: str, **kw) -> str:
    return subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True, **kw).stdout


def use_version(version: str) -> None:
    text = sh("git", "show", f"dataset-{version}:{POINTER}")
    (ROOT / POINTER).write_text(text)
    sh(sys.executable, "-m", "dvc", "checkout", POINTER)


BASELINES = {v: [("logreg", dict(max_depth=3, learning_rate=0.1, n_estimators=100)),
                 ("rf", dict(max_depth=6, learning_rate=0.1, n_estimators=100))] for v in ("v1", "v2", "v3")}


def main() -> None:
    baselines = "--baselines" in sys.argv  # second experiment, used to compare experiments
    plan = BASELINES if baselines else SWEEP
    extra = ["--experiment", "parcel-delay-baselines", "--register", "parcel-delay-baseline"] if baselines else []
    try:
        for version, configs in plan.items():
            use_version(version)
            for i, (model, p) in enumerate(configs, 1):
                name = f"{'baseline' if baselines else 'sweep'}-{version}-{model}-{i}"
                out = sh(sys.executable, "training/train.py", "--model", model, "--run-name", name, *extra,
                         "--max-depth", str(p["max_depth"]), "--learning-rate", str(p["learning_rate"]),
                         "--n-estimators", str(p["n_estimators"]))
                print(name, out.strip().splitlines()[-1][:120])
    finally:
        sh("git", "checkout", "HEAD", "--", POINTER)  # index already equals HEAD, so nothing is staged
        sh(sys.executable, "-m", "dvc", "checkout", POINTER)


if __name__ == "__main__":
    main()
