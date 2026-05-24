-- Postgres init: backing DB for Airflow's metadata store.
-- Runs on first postgres init only; for existing volumes, create manually.

CREATE USER airflow WITH PASSWORD 'airflow';
CREATE DATABASE airflow_db OWNER airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow;
