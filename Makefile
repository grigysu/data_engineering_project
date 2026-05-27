# Weather pipeline — make targets.
#
# Run from the repo root. Most targets either drive Docker (the stack) or
# the local venv (tests, seed scripts). On Windows, run inside Git Bash or
# WSL2 — recipes use POSIX-ish shell (cat, &&, etc.). PowerShell users can
# still invoke targets via `make.exe` from Git for Windows.
#
# Override PYTHON if your venv lives elsewhere:
#     make PYTHON=.venv/bin/python test

ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
# Force Git Bash as the recipe shell so `cat`, single-quoted `echo`, etc.
# behave like POSIX. GNU Make on Windows defaults to cmd.exe otherwise.
# Override if Git is installed elsewhere:
#     make SHELL='C:/Program Files (x86)/Git/usr/bin/bash.exe' help
SHELL := C:/Program Files/Git/usr/bin/bash.exe
.SHELLFLAGS := -c
# Recipes inherit Make's env; PowerShell's PATH doesn't include Git Bash
# coreutils, so `head`, `cat`, etc. would be missing. Prepend the Git Bash
# bin dir so recipes find them. Harmless if Git is elsewhere (override SHELL
# above + adjust this line in tandem).
export PATH := C:/Program Files/Git/usr/bin:$(PATH)
else
PYTHON ?= .venv/bin/python
endif

COMPOSE             := docker compose
# Airflow scheduler — container_name (for `docker exec`)
SCHEDULER           := weather_airflow_scheduler
# Airflow scheduler — compose service name (for `docker compose build`)
SCHEDULER_SERVICE   := airflow_scheduler
POSTGRES            := weather_postgres
MINIO_NETWORK       := data_engine_default
DAG                 := weather_pipeline

DDL := warehouse/ddl/001_dim_location.sql \
       warehouse/ddl/002_dim_time.sql \
       warehouse/ddl/003_fact_weather_observations.sql \
       warehouse/ddl/004_predictions.sql \
       warehouse/ddl/005_backtest_groups.sql

# Source .env into Make's own env so every recipe (Python *and* shell)
# sees the same variables. The wildcard guard makes this a no-op when
# .env doesn't exist yet (e.g. first clone).
ifneq (,$(wildcard .env))
include .env
export
endif

.DEFAULT_GOAL := help

# ---------- Help ----------
help:
	@echo "Common targets:"
	@echo ""
	@echo "  Stack:"
	@echo "    up / down / restart / logs    docker compose lifecycle (logs tails the scheduler)"
	@echo ""
	@echo "  DAG:"
	@echo "    unpause / trigger             unpause + manually trigger weather_pipeline"
	@echo "    runs                          list recent DAG runs"
	@echo "    train                         clear + rerun train_model -> walk_forward -> backfill"
	@echo "    walk-forward                  clear + rerun walk_forward -> backfill (reuses checkpoint)"
	@echo "    show-config                   print the resolved config/train.yaml as JSON"
	@echo "    ingest                        trigger full DAG over a date range"
	@echo "                                  (vars: START=YYYY-MM-DD END=YYYY-MM-DD; default = last year)"
	@echo ""
	@echo "  Clear (removal only — chain with walk-forward / trigger to rebuild):"
	@echo "    clear-predictions             TRUNCATE predictions + backtest_groups"
	@echo "    clear-checkpoints             rm -rf checkpoints/*"
	@echo "    clear-ml                      clear-predictions + clear-checkpoints + clear Airflow ML tasks"
	@echo ""
	@echo "  Note: Open-Meteo archive lags ~2 days; data_range_end is normally (today - 2d)."
	@echo ""
	@echo "  Services:"
	@echo "    psql / adminer / dashboard / minio / airflow"
	@echo "                                  open a shell or print the URL"
	@echo ""
	@echo "  Dev:"
	@echo "    test / lint / format / check  quality gates (uses local venv)"
	@echo "    smoke                         run every read-only target end-to-end (Makefile self-test)"
	@echo "    status                        what's in MinIO + warehouse + checkpoint right now"
	@echo "    seed                          regenerate ingestion/locations.json from Open-Meteo geocoding"
	@echo ""
	@echo "  Data layer:"
	@echo "    init-db                       apply warehouse DDL (idempotent on a clean DB)"
	@echo "    clear-db                      DROP TABLE … CASCADE for all warehouse tables"
	@echo "    clear-lake                    wipe MinIO bronze/silver/gold/manifest"
	@echo "    clear-all                     clear-lake + clear-db (full data nuke; checkpoints untouched)"
	@echo ""
	@echo "  GPU:"
	@echo "    gpu-build                     rebuild airflow image with CUDA torch + recreate scheduler"
	@echo "    verify-gpu                    check torch.cuda.is_available() inside scheduler"

# ---------- Stack lifecycle ----------
up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

restart: down up

logs:
	$(COMPOSE) logs -f $(SCHEDULER)

# ---------- DAG / scheduler ----------
unpause:
	docker exec $(SCHEDULER) airflow dags unpause $(DAG)

trigger: unpause
	docker exec $(SCHEDULER) airflow dags trigger $(DAG)

runs:
	docker exec $(SCHEDULER) airflow dags list-runs --dag-id $(DAG)

# Trigger the full DAG over a custom date range. Override START / END:
#     make ingest START=2025-05-26 END=2026-05-24
# Open-Meteo archive lags ~2 days; END should normally be (today - 2d).
# Ingest is additive (manifest-aware), so re-running the same range is cheap.
START ?= $(shell date -u -d '1 year ago'  +%Y-%m-%d 2>/dev/null || powershell -c "(Get-Date).AddYears(-1).ToString('yyyy-MM-dd')")
END   ?= $(shell date -u -d '2 days ago'  +%Y-%m-%d 2>/dev/null || powershell -c "(Get-Date).AddDays(-2).ToString('yyyy-MM-dd')")
ingest:
	docker exec $(SCHEDULER) airflow dags trigger $(DAG) \
		--conf '{"start_date": "$(START)", "end_date": "$(END)"}'

# Clear + rerun training. Downstream (walk_forward, backfill_actuals) get
# cleared too so the new checkpoint propagates into predictions.
# Use when the data hasn't changed but you want a fresh model from existing gold.
train:
	docker exec $(SCHEDULER) airflow tasks clear $(DAG) \
		-t "train_model|walk_forward|backfill_actuals" --yes

# Clear + rerun walk-forward only (reuses the existing checkpoint).
# Use when the checkpoint is fine but predictions are stale or missing.
walk-forward:
	docker exec $(SCHEDULER) airflow tasks clear $(DAG) \
		-t "walk_forward|backfill_actuals" --yes

# Print the resolved config (YAML + dataclass defaults) as JSON. Useful
# when wondering 'what hyperparams will the next DAG run use?'.
show-config:
	$(PYTHON) -c "import json; from ml.config import load_config; \
		print(json.dumps(load_config().to_dict(), indent=2))"

# ---------- Clear (removal only — chain with walk-forward / trigger to rebuild) ----------

# Wipe every prediction snapshot + the derived BT-groups rows.
# Safe — neither table has inbound FKs, so TRUNCATE doesn't cascade.
# To repopulate: `make walk-forward` (reuses checkpoint) or `make trigger`.
clear-predictions:
	docker exec $(POSTGRES) psql -U weather -d weather_dw -c \
		"TRUNCATE predictions, backtest_groups;"

# Wipe every training artifact under checkpoints/. Next `make train` or
# the DAG's train_model task starts from a clean slate.
clear-checkpoints:
	rm -rf checkpoints/*

# Wipe ML state end-to-end: checkpoints + predictions + BT groups +
# stale Airflow task states for the training/prediction chain. Doesn't
# touch gold/silver/bronze.
clear-ml: clear-predictions clear-checkpoints
	docker exec $(SCHEDULER) airflow tasks clear $(DAG) \
		-t "train_model|walk_forward|backfill_actuals" --yes

# ---------- Quality gates ----------
test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

# Mirrors the pre-commit gate (lint + format-check + tests).
check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m pytest -q

# ---------- Data layer ----------
seed:
	$(PYTHON) -m ingestion.seed_locations

init-db:
	cat $(DDL) | docker exec -i $(POSTGRES) psql -U weather -d weather_dw -v ON_ERROR_STOP=1

clear-db:
	docker exec -i $(POSTGRES) psql -U weather -d weather_dw -c \
		"DROP TABLE IF EXISTS backtest_groups, predictions, fact_weather_observations, dim_time, dim_location CASCADE;"

clear-lake:
	docker run --rm --network $(MINIO_NETWORK) --entrypoint sh minio/mc:latest -c \
		"mc alias set local http://minio:9000 minioadmin minioadmin >/dev/null && \
		 mc rm --recursive --force local/weather-lake/bronze/   >/dev/null 2>&1 || true && \
		 mc rm --recursive --force local/weather-lake/silver/   >/dev/null 2>&1 || true && \
		 mc rm --recursive --force local/weather-lake/gold/     >/dev/null 2>&1 || true && \
		 mc rm --recursive --force local/weather-lake/manifest/ >/dev/null 2>&1 || true && \
		 echo 'lake wiped (bronze + silver + gold + manifest)'"

# Lake + DB only. Checkpoints stay (use `clear-checkpoints` for those).
# To repopulate: `make init-db && make trigger`.
clear-all: clear-lake clear-db
	@echo ""
	@echo "Lake + DB wiped. Run 'make init-db && make trigger' to repopulate."
	@echo "(checkpoints/ untouched — use 'make clear-checkpoints' for those.)"

# ---------- Service shortcuts ----------
psql:
	docker exec -it $(POSTGRES) psql -U weather -d weather_dw

dashboard:
	@echo "Open http://localhost:8501"

minio:
	@echo "Open http://localhost:9001  (minioadmin / minioadmin)"

adminer:
	@echo "Open http://localhost:8082"
	@echo "  System: PostgreSQL  Server: postgres  User: weather  Password: weather  DB: weather_dw"

airflow:
	@echo "Open http://localhost:8081  (admin / admin)"

# ---------- Status (what data do I have right now?) ----------
status:
	@echo "=== MinIO lake (object counts per layer) ==="
	@docker run --rm --network $(MINIO_NETWORK) --entrypoint sh minio/mc:latest -c \
		"mc alias set local http://minio:9000 minioadmin minioadmin >/dev/null && \
		 for layer in bronze silver gold; do \
		   n=\$$(mc ls --recursive local/weather-lake/\$$layer/ 2>/dev/null | wc -l); \
		   echo \"  \$$layer: \$$n objects\"; \
		 done" 2>/dev/null || echo "  (minio unreachable)"
	@echo ""
	@echo "=== Warehouse: dim_location ==="
	@docker exec $(POSTGRES) psql -U weather -d weather_dw -c \
		"SELECT region, lat, lon FROM dim_location ORDER BY region;" 2>/dev/null \
		|| echo "  (postgres unreachable or empty)"
	@echo "=== Warehouse: fact_weather_observations ==="
	@docker exec $(POSTGRES) psql -U weather -d weather_dw -c \
		"SELECT count(*) AS rows, min(t.observed_at) AS min_obs, max(t.observed_at) AS max_obs \
		 FROM fact_weather_observations f JOIN dim_time t ON t.time_id = f.time_id;" 2>/dev/null \
		|| true
	@echo "=== Warehouse: predictions ==="
	@docker exec $(POSTGRES) psql -U weather -d weather_dw -c \
		"SELECT count(*) AS rows, count(DISTINCT prediction_made_at) AS snapshots, \
		        count(*) FILTER (WHERE actual_value IS NOT NULL) AS with_actuals, \
		        max(prediction_made_at) AS latest_made_at FROM predictions;" 2>/dev/null \
		|| true
	@echo "=== Checkpoint (model on disk) ==="
	@if [ -f checkpoints/best.pt ]; then \
		echo "  checkpoints/best.pt $$(stat -c '%y  %s bytes' checkpoints/best.pt 2>/dev/null || stat -f '%Sm  %z bytes' checkpoints/best.pt 2>/dev/null)"; \
		$(PYTHON) -c "import torch; c=torch.load('checkpoints/best.pt',map_location='cpu',weights_only=False); \
			print(f\"  model_version={c.get('model_version')}\"); \
			print(f\"  trained_at={c.get('trained_at')}\"); \
			print(f\"  data_range={c.get('data_range_start')} -> {c.get('data_range_end')}\"); \
			print(f\"  best_val_mse={c.get('best_val_mse')}\"); \
			print(f\"  seq_in={c['spec']['seq_in']}  seq_out={c['spec']['seq_out']}\")" 2>/dev/null || true; \
	else \
		echo "  (no checkpoint yet)"; \
	fi

# ---------- Smoke ----------
# Runs every read-only target in sequence — use this to verify the Makefile
# itself + the .env load + the stack is reachable. Does NOT touch data
# (no trigger / reset-* / gpu-build / down).
smoke:
	@echo "=== env check ==="
	@echo "POSTGRES_USER=$$POSTGRES_USER  MINIO_BUCKET=$$MINIO_BUCKET"
	@echo "=== help (first 4 lines) ==="
	@"$(MAKE)" -s help | sed -n '1,4p'
	@echo "=== lint ==="
	@"$(MAKE)" -s lint
	@echo "=== format-check ==="
	@$(PYTHON) -m ruff format --check .
	@echo "=== test ==="
	@"$(MAKE)" -s test
	@echo "=== runs (first 5 lines) ==="
	@"$(MAKE)" -s runs 2>&1 | sed -n '1,5p' || true
	@echo "=== verify-gpu ==="
	@"$(MAKE)" -s verify-gpu 2>&1 || true
	@echo "=== dashboard URLs ==="
	@"$(MAKE)" -s dashboard
	@"$(MAKE)" -s adminer
	@"$(MAKE)" -s minio
	@"$(MAKE)" -s airflow
	@echo "smoke OK"

# ---------- GPU build ----------
gpu-build:
	$(COMPOSE) build $(SCHEDULER_SERVICE)
	$(COMPOSE) up -d --force-recreate $(SCHEDULER_SERVICE)

verify-gpu:
	docker exec $(SCHEDULER) python -c \
		"import torch; print('cuda available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

.PHONY: help \
        up down restart logs \
        unpause trigger runs train walk-forward show-config ingest \
        clear-predictions clear-checkpoints clear-ml \
        test lint format check smoke status \
        seed init-db clear-db clear-lake clear-all \
        psql dashboard minio adminer airflow \
        gpu-build verify-gpu
