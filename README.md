# MLOps Copilot — Parcel Delay Operations Agent

A conversational agent that answers operational questions about a **campus parcel-delay prediction model** by
calling real tools against **DVC**, **MLflow**, **Apache Airflow 3** and **Kubernetes**. Every operational claim in
an answer must be backed by a tool result from the same conversation; the agent cites evidence ids (`[E1]`),
discloses failed tools, and refuses to guess when a system is unavailable.

> Example: *"Find the best recent model, identify the dataset version used to train it, check the deployment status,
> and explain whether retraining is necessary."* — the agent queries MLflow, DVC, Kubernetes and Airflow, correlates
> the evidence, and answers with **Known facts / Uncertainties / Recommended actions**. It never starts training
> or changes the cluster on its own.

See [`architecture.md`](architecture.md) for the design, tool list, permissions and safety model.

## What is in the box

| Area | What it does |
|---|---|
| `data/`, `training/make_data.py` | Seeded generator for three dataset versions (v1 baseline, v2 exam-week surge, v3 monsoon) tracked with **DVC** (local remote, git tags `dataset-v1..v3`) |
| `training/train.py`, `stages.py`, `sweep.py` | Training that records dataset provenance (hash, git commit, version) as **MLflow** tags; stage runner used by Airflow; reproducible hyper-parameter sweep |
| `dags/training_pipeline.py` | **Airflow 3** DAG: validate → preprocess → train → evaluate/log → quality gate → promote; retries, logging, gated promotion |
| `api/main.py`, `Dockerfile`, `kubernetes/` | FastAPI inference service; Deployment (2 replicas, startup/readiness/liveness probes, requests/limits), ClusterIP Service, ConfigMap |
| `agent/` | Tool registry, 30 model-visible tools, orchestrator, grounding checks, safety policy, structured logging, CLI |
| `tests/` | 38 pytest tests; all external systems mocked |

## Prerequisites

Linux / WSL2 with Docker, Python 3.12, [`uv`](https://docs.astral.sh/uv/), `kubectl`, and [`kind`](https://kind.sigs.k8s.io/).
At least 8 GB RAM is recommended (Airflow + MLflow + a kind node run side by side).

## 1. Install and configure

```bash
git clone https://github.com/Z-zaratahir/MLOps-Copilot.git && cd MLOps-Copilot
uv sync                          # reproducible environment from uv.lock
cp .env.example .env             # then edit .env
```

Edit `.env` (it is git-ignored — never commit it):

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY`, `GEMINI_API_KEY` | LLM providers (Groq primary, Gemini optional fallback) |
| `GROQ_MODEL`, `GROQ_FALLBACK_MODELS` | Tool-calling models; fallbacks are tried on rate limits |
| `AIRFLOW_PASSWORD` | Password for the Airflow `admin` user (choose any value) |
| `PROJECT_DIR`, `A3_DIR` | Absolute paths used by `docker compose` (repo dir, and its parent) |
| `K8S_APPROVED_NAMESPACE`, `AIRFLOW_APPROVED_DAG` | The only namespace/DAG the agent may operate on |
| `MODEL_FRESHNESS_DAYS` | Freshness policy used by the retraining assessment |

## 2. Data (DVC)

```bash
. .venv/bin/activate
dvc init                                              # already done in this repo
dvc remote add -d localstore ../dvc-store             # local remote outside the repo
for v in v1 v2 v3; do                                 # how the three versions were recorded
  python training/make_data.py --version $v
  dvc add data/parcels.csv && git add data/parcels.csv.dvc data/.gitignore
  git commit -m "Recorded parcel dataset $v under DVC" && git tag dataset-$v && dvc push
done
```

Reproduce any version (worktree only, nothing staged):

```bash
git show dataset-v1:data/parcels.csv.dvc > data/parcels.csv.dvc
dvc checkout data/parcels.csv.dvc
md5sum data/parcels.csv        # must equal the md5 in the pointer file
git checkout HEAD -- data/parcels.csv.dvc && dvc checkout data/parcels.csv.dvc   # back to current
```

## 3. MLflow

```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --allowed-hosts "localhost:5000,127.0.0.1:5000,host.docker.internal:5000" \
  --backend-store-uri sqlite:///../mlflow-store/mlflow.db --default-artifact-root ../mlflow-store/artifacts
python training/train.py --model gbm --run-name first-run     # one tracked run
python training/sweep.py                                       # 19 more runs across v2/v3
```

## 4. Airflow 3 (Docker)

```bash
printf '{"admin": "%s"}\n' "$AIRFLOW_PASSWORD" > airflow/passwords.json   # git-ignored
cd airflow && docker compose --env-file ../.env up -d --build
```

UI: <http://localhost:8080> (user `admin`). The DAG `parcel_delay_training` starts paused: unpause it in the UI
or via the API. The training stages run in an isolated virtualenv inside the image (`/opt/mlenv`).

## 5. Kubernetes (kind)

```bash
python training/export_model.py          # best *quality-gated* run for the newest data -> models/served/
docker build -t parcel-delay-api:1.0.1 .
kind create cluster --name parcel-lab
kind load docker-image parcel-delay-api:1.0.1 --name parcel-lab
kubectl apply -f kubernetes/namespace.yaml -f kubernetes/configmap.yaml \
              -f kubernetes/deployment.yaml -f kubernetes/service.yaml
kubectl -n parcel-ml rollout status deploy/parcel-delay-api
```

## 6. Talk to the agent

```bash
python -m agent.main                                   # interactive
python -m agent.main -q "Which dataset version was used for experiment 25?" --save demo
```

Things to try:

* *Which dataset version was used for experiment 25, and how do I reproduce it?*
* *Compare the last five training runs and recommend the best candidate based on F1 score.*
* *Trigger the model training pipeline and report whether execution succeeded.*
* *Check whether the prediction API is healthy and show me the most recent logs.*
* *Find the best recent model, identify its dataset version, check the deployment and explain whether retraining is necessary.*
* *Why is my prediction API unavailable?* (after breaking the ConfigMap, see below)

Cluster changes are two-step. The agent can only **request** a restart/scale; it returns a token, and **you** type
`confirm <token>` to execute it. Requests outside the approved namespace are rejected.

## 7. Tests and CI

```bash
uv run ruff check .
uv run pytest                 # 38 tests, no external services needed
```

`.github/workflows/ci.yml` runs lint and tests on pull requests and pushes to `main`. Kubernetes, Airflow, MLflow and
the LLM are mocked in CI.

## 8. Reproducing the injected failures

| Failure | How to inject | What the agent finds | Resolution |
|---|---|---|---|
| Bad configuration | `kubectl -n parcel-ml patch configmap parcel-delay-config -p '{"data":{"MODEL_DIR":"models/missing"}}'` then `kubectl -n parcel-ml delete pod -l app=parcel-delay-api` | Pods `0/1` Ready, readiness 503, `FileNotFoundError` in logs, 0 service endpoints, wrong `MODEL_DIR` | `kubectl apply -f kubernetes/configmap.yaml`, then ask the agent to restart and `confirm` |
| Corrupted data | blank some `distance_km` values in `data/parcels.csv`, then ask the agent to run the pipeline | `validate_data` fails (nulls + hash mismatch vs DVC pointer), downstream tasks `upstream_failed` | `dvc checkout --force data/parcels.csv.dvc` |
| Unavailable service | `MLFLOW_TRACKING_URI=http://127.0.0.1:59999 python -m agent.main -q "..."` | MLflow tools fail; agent reports the errors and does not invent results | restore the URI |

## Secrets

No API keys, kubeconfig or tokens are committed. Secrets come from `.env` / environment variables; logs redact
anything that looks like a key, token or password.

## Known limitations

* The model is baked into the container image; shipping a new model means rebuilding the image.
* Free-tier LLM rate limits (8k tokens/minute per Groq model) make multi-tool answers slow; the agent switches models
  and waits when limited.
* Airflow runs in `standalone` mode (SQLite, single container) — suitable for development, not production.
* The agent has no persistent memory beyond the current CLI session.
