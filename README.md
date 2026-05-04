# AI4Science: Deep Learning Research & Engineering

`ai4science` is a modular framework for deep learning research prototyping with an engineering-oriented structure. It is designed to keep model code, optimization logic, data pipelines, and interfaces cleanly separated so experiments remain easy to extend, test, and adapt across different deep learning workflows and backends.

---

## 🏗 Project Architecture
The repository is organized around reusable ML building blocks where models stay focused on forward logic, learners own optimization, and data modules handle preparation and sharding:

```text
ai4science/
├── ai4science/                 # core Python package
│   ├── data/                   # datasets, sharding, and data modules
│   │   └── datasets/           # dataset-specific preparation logic
│   ├── learners/               # optimization loops by backend
│   ├── models/                 # linear / logistic / softmax models
│   ├── infra/                  # logging and support infrastructure
│   ├── interfaces/             # contracts for models and learners
│   └── utils/                  # shared helpers
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

## 🚀 Quick Start

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install pytest
```

### 1. Run the Test Suite
```bash
python -m pytest
```

### 2. Explore in Notebooks
The `notebooks/` directory is the main interactive surface for experimentation and walkthroughs.

### 3. Prepare Dataset Pipelines
Example: Prepare AG News shards:
```bash
python scripts/manage_data.py prepare_data --dataset ag_news --shards 4 --format csv
```

Force a clean rebuild of dataset artifacts:
```bash
python scripts/manage_data.py prepare_data --dataset ag_news --shards 4 --format csv --force
```

Simulate distributed dataset setup:
```bash
python scripts/manage_data.py setup --dataset ag_news --stage train --rank 0 --world_size 2
```

## 📓 Notebooks
The `notebooks/` directory contains exploratory analysis and modeling walkthroughs.

## 🛠 Engineering Standards
*   **Architecture**: Keep interfaces explicit and backend-specific learner logic isolated.
*   **Workflow**: Use tests for validation and notebooks for interactive experimentation.
*   **Data**: Centralize dataset preparation and sharding in `ai4science/data/`.
*   **Linting/Formatting**: Managed via `ruff`.
*   **Typing**: `mypy` is configured with pragmatic defaults for iterative prototyping.

## 🗺 Roadmap
- add more deep learning components, including areas such as transformers, diffusion models, and AI for science
- add richer experiment configuration and artifact management
- grow dataset and notebook coverage for more end-to-end workflows
