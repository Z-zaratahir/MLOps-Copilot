"""Parcel-delay inference service (the workload the agent monitors on Kubernetes)."""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/served"))
APP_VERSION = os.getenv("APP_VERSION", "dev")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
state: dict = {"model": None, "meta": {}, "started": time.time(), "error": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        state["model"] = joblib.load(MODEL_DIR / "model.joblib")
        state["meta"] = json.loads((MODEL_DIR / "meta.json").read_text())
    except Exception as exc:  # surfaced via /ready and logs so the agent can diagnose it
        state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"MODEL_LOAD_FAILED {state['error']}", flush=True)
    yield


app = FastAPI(title="Parcel Delay API", lifespan=lifespan)


class Parcel(BaseModel):
    distance_km: float = Field(gt=0, le=15)
    hour: int = Field(ge=0, le=23)
    rider_load: float = Field(ge=0, le=12)
    weather: str = Field(pattern="^(clear|rain|storm)$")
    zone: str = Field(pattern="^(hostel|library|admin|sports|gate)$")
    weight_kg: float = Field(gt=0, le=40)


@app.get("/health")
def health():
    """Liveness: process is up and serving."""
    return {"status": "alive", "uptime_s": round(time.time() - state["started"], 1)}


@app.get("/ready")
def ready():
    """Readiness: a model is loaded and can answer."""
    if state["model"] is None:
        raise HTTPException(503, detail=state["error"] or "model not loaded")
    return {"status": "ready"}


@app.get("/version")
def version():
    return {"app_version": APP_VERSION, **state["meta"]}


@app.post("/predict")
def predict(p: Parcel):
    if state["model"] is None:
        raise HTTPException(503, detail="model not loaded")
    proba = float(state["model"].predict_proba(pd.DataFrame([p.model_dump()]))[0, 1])
    return {"delay_probability": round(proba, 4), "delayed": proba >= DECISION_THRESHOLD,
            "model_run_id": state["meta"].get("run_id")}
