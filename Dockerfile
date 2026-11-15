FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY api/requirements.txt api/requirements.txt
RUN pip install -r api/requirements.txt

COPY api/ api/
# exported by training/export_model.py (model.joblib + meta.json)
COPY models/served/ models/served/

RUN useradd --uid 10001 --no-create-home appuser
USER 10001

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
