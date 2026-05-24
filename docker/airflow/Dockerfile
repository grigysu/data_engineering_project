# Custom Airflow image: adds our runtime deps + docker CLI so BashOperator
# can `docker exec` into the spark-master container for spark-submit jobs.

FROM apache/airflow:2.10.5-python3.11

USER root
# docker CLI only — the daemon socket is mounted in at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends docker.io \
 && rm -rf /var/lib/apt/lists/*

# Give the airflow user (UID 50000) access to the host's docker.sock by
# adding it to the host docker group (GID set at runtime via group_add in compose).
# Nothing to do here at build time — see compose for group_add.

USER airflow

# CUDA 12.8 torch wheel — matches the host's GPU (RTX 50-series + NVIDIA
# Container Toolkit). The scheduler service in docker-compose.yml requests
# GPU access via deploy.resources so this wheel actually has a device to
# run on; without that passthrough torch.cuda.is_available() returns False
# and training silently falls back to CPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt
