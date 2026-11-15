"""Central settings, read from environment / .env (never hard-coded)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment: str = "parcel-delay"
    mlflow_registered_model: str = "parcel-delay-classifier"

    airflow_base_url: str = "http://127.0.0.1:8080"
    airflow_username: str = "admin"
    airflow_password: str = ""
    airflow_approved_dag: str = "parcel_delay_training"

    k8s_approved_namespace: str = "parcel-ml"
    k8s_app_label: str = "app=parcel-delay-api"
    k8s_deployment: str = "parcel-delay-api"
    k8s_service: str = "parcel-delay-api"
    k8s_context: str = "kind-parcel-lab"

    agent_tool_timeout_seconds: float = 15.0
    model_freshness_days: int = 14
    log_file: str = str(ROOT / "logs" / "agent.jsonl")


@lru_cache
def get_settings() -> Settings:
    return Settings()
