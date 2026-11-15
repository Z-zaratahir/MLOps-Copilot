FROM apache/airflow:3.0.6

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /opt/mlenv && chown airflow:0 /opt/mlenv && chmod g+w /opt/mlenv

USER airflow
# Isolated environment for the training stages; pins match uv.lock
RUN python -m venv /opt/mlenv && /opt/mlenv/bin/pip install --no-cache-dir \
    scikit-learn==1.9.1 pandas==3.1.1 numpy==2.5.4 joblib==1.6.0 cloudpickle==3.1.2 \
    mlflow-skinny==3.18.0 pyyaml==6.0.3
