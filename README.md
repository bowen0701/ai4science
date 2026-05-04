# AI4Science

`ai4science` is an AI for Science research and engineering repository organized around a reusable library package and experiment-first workflows.

## Repository Layout

```text
ai4science/
├── ai4science/
│   ├── data/
│   ├── learners/
│   ├── models/
│   └── utils/
├── experiments/
│   └── <project>/
│       ├── config.yaml
│       ├── train.py
│       ├── eval.py
│       └── notes.md
├── scripts/
├── tests/
├── notebooks/
├── pyproject.toml
└── README.md
```

## Working Model

- `ai4science/` contains reusable code for data pipelines, models, learners, and utilities.
- `experiments/` contains experiment-local configs and entrypoints.
- `scripts/` contains thin orchestration commands for repeatable runs and ablations.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install pytest
```

Train the scaffolded experiment:

```bash
bash scripts/run_train.sh
```

Run the sampling pass:

```bash
python3 -m experiments.<project>.eval
```

Run tests:

```bash
python -m pytest
```
