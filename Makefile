.PHONY: install test lint format typecheck

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip setuptools wheel
	.venv/bin/pip install -e .
	.venv/bin/pip install -e projects/affinitydiff_rl

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/ruff check --fix .

format:
	.venv/bin/ruff format .

typecheck:
	.venv/bin/mypy ai4science/
