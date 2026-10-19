.PHONY: setup install run test test-slow lint format voice preflight smoke clean

PYTHON ?= python
VENV ?= .venv
BIN := $(VENV)/bin

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

install:
	$(BIN)/pip install -e ".[dev]"

run:
	$(BIN)/python -m cosmion dev

test:
	$(BIN)/pytest -m "not slow"

test-slow:
	$(BIN)/pytest -m slow

lint:
	$(BIN)/ruff check .

format:
	$(BIN)/ruff format .

voice:
	$(BIN)/python scripts/download_voice.py

preflight:
	$(BIN)/python scripts/preflight.py

smoke:
	$(BIN)/python scripts/smoke_tts.py

clean:
	rm -rf .pytest_cache .ruff_cache artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
