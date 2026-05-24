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

# Install CPU torch via the dedicated index so it doesn't grab the (huge) CUDA wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt
