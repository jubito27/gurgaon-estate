# syntax=docker/dockerfile:1.7
# Keep the runtime aligned with the Python 3.14 environment used by CI/model artifacts.
FROM python:3.14.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# XGBoost's runtime requires libgomp. Install no build toolchain or optional OS packages.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY flask-app/requirements.txt /tmp/requirements.txt

# Wheels only: avoids compiling scientific packages and keeps the final image lean.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-compile --only-binary=:all: -r /tmp/requirements.txt

COPY flask-app/ /app/app/
COPY data/processed/feature_selection/gurgaon_properties_post_feature_selection_v2.csv /app/data/processed/feature_selection/gurgaon_properties_post_feature_selection_v2.csv
COPY model/pipeline.pkl /app/model/pipeline.pkl
COPY model/recommendation/recommendation_model.joblib /app/model/recommendation/recommendation_model.joblib
COPY model/recommendation/property_catalog.csv /app/model/recommendation/property_catalog.csv

RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app app \
    && mkdir -p /tmp/matplotlib \
    && chown -R app:app /app /tmp/matplotlib

USER app
WORKDIR /app/app

EXPOSE 5000

# One process keeps the loaded data/model in memory once; threads handle concurrent requests.
CMD ["gunicorn", "--bind=0.0.0.0:5000", "--workers=1", "--threads=4", "--timeout=120", "--access-logfile=-", "--error-logfile=-", "app:app"]
