.PHONY: help install test lint format typecheck

help:
	@echo "Available targets:"
	@echo "  install    Install all packages in editable mode"
	@echo "  test       Run tests with pytest"
	@echo "  lint       Run ruff linter with auto-fix"
	@echo "  format     Run ruff formatter"
	@echo "  typecheck  Run mypy type checker"

# On Lightning AI Studio, install into the conda base environment.
# Locally, activate a venv or conda env first — this guard prevents polluting system Python.
install:
	@if [ -z "$$VIRTUAL_ENV" ] && [ -z "$$CONDA_DEFAULT_ENV" ]; then \
		echo "ERROR: No active venv or conda environment detected. Activate one before running make install."; \
		exit 1; \
	fi
	pip install --upgrade pip setuptools wheel
	pip install -e .
	pip install -e projects/affinitydiff_rl

test:
	python -m pytest

lint:
	ruff check --fix .

format:
	ruff format .

typecheck:
	mypy ai4science/
