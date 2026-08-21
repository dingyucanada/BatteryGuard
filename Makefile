.PHONY: sync test coverage lint typecheck verify demo api ui docker-build

sync:
	uv sync --extra dev

test:
	uv run pytest

coverage:
	uv run pytest --cov=batteryguard --cov-report=term --cov-fail-under=82

lint:
	uv run ruff check src tests apps

typecheck:
	uv run mypy src/batteryguard

verify: lint typecheck coverage

demo:
	uv run batteryguard demo --cell random --seed 42 --offline --no-reveal

api:
	uv run uvicorn batteryguard.api.app:app --host 127.0.0.1 --port 8000

ui:
	uv run streamlit run apps/streamlit_app.py

docker-build:
	docker build -f docker/Dockerfile -t batteryguard:local .
