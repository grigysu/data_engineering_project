# Streamlit dashboard image. Phase 2c.
# Slim Python base + our runtime deps (predictor needs torch, pyarrow, etc.).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU torch from the dedicated index so we don't pull the multi-GB CUDA wheel.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY docker/dashboard/requirements-dashboard.txt /tmp/requirements-dashboard.txt
RUN pip install -r /tmp/requirements-dashboard.txt

# Source is bind-mounted at runtime (see docker-compose). Default to running
# Streamlit; compose-level command can override.
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
