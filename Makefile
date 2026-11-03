.PHONY: up down doctor run test

up:
	docker compose up -d

down:
	docker compose down

doctor:
	python scripts/check_env.py

run:
	.venv/bin/uvicorn backend.app.main:app --reload --port 8000

test:
	.venv/bin/pytest -q
