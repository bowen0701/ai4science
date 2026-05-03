# DL-Eng: Deep Learning Research & Engineering

`dl-eng` is a modular framework for deep learning research prototyping with an engineering-oriented structure. The goal is to keep model code, learner logic, data pipelines, and interfaces cleanly separated so experiments stay easy to extend and reason about.

---

## Project Architecture
The repository is organized around reusable ML building blocks where models stay focused on forward logic, learners own optimization, and data modules handle preparation and sharding:

```text
dl-eng/
├── dl_eng/                     # core Python package
│   ├── data/                   # datasets, sharding, and data modules
│   │   └── datasets/           # dataset-specific preparation logic
│   ├── learners/               # optimization loops by backend
│   ├── models/                 # linear / logistic / softmax models
│   ├── infra/                  # logging and support infrastructure
│   ├── interfaces/             # contracts for models and learners
│   └── utils/                  # shared helpers
├── examples/                   # runnable training examples
├── scripts/                    # thin orchestration entrypoints
├── notebooks/                  # exploratory analysis and demos
├── tests/                      # integration coverage
├── pyproject.toml
└── README.md
```

### Mental Model
```text
                ┌──────────────┐
                │   scripts    │
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │    data      │
                └──────┬───────┘
                       ↓
                ┌──────────────┐
                │   learners   │  optimization + training loop
                └──────┬───────┘
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       models      interfaces      infra
```

## Quick Start

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install pytest
```

### 1. Run Example Training
NumPy linear regression:
```bash
python examples/linear_regression.py
```

NumPy logistic regression:
```bash
python examples/logistic_regression.py
```

PyTorch softmax regression:
```bash
python examples/softmax_regression.py
```

### 2. Run the Test Suite
```bash
python -m pytest
```

### 3. Prepare Dataset Pipelines
Prepare AG News shards:
```bash
python scripts/manage_data.py prepare --dataset ag_news --shards 4 --format csv
```

Prepare ProteinNet data:
```bash
python scripts/manage_data.py prepare --dataset proteinnet --shards 4
```

Simulate distributed dataset setup:
```bash
python scripts/manage_data.py test_setup --dataset ag_news --rank 0 --world_size 2
```

## Notebooks
The `notebooks/` directory contains exploratory walkthroughs for:

- linear regression
- PCA / SVD
- correspondence analysis

## Engineering Standards
*   **Architecture**: Keep interfaces explicit and backend-specific learner logic isolated.
*   **Examples**: Prefer thin runnable scripts under `examples/` for quick validation.
*   **Data**: Centralize dataset preparation and sharding in `dl_eng/data/`.
*   **Linting/Formatting**: Managed via `ruff`.
*   **Typing**: `mypy` is configured with pragmatic defaults for iterative prototyping.

## Roadmap
- expand learner coverage beyond the current NumPy / PyTorch examples
- add richer experiment configuration and artifact management
- grow dataset and notebook coverage for more end-to-end workflows
