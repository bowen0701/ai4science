# AI4Science

`ai4science` is an AI for Science research and engineering monorepo organized around a shared library and independent research projects.

## Repository Layout

```text
ai4science/
├── ai4science/               # Shared library package (pip install -e .)
│   ├── data/                 # BaseDataModule ABC + DataSharder for DDP/FSDP
│   ├── learners/             # TorchLearner, JaxLearner, NumPyLearner, DiffusionLearner
│   ├── models/               # Reference model implementations (PyTorch + NumPy)
│   ├── utils/                # LearnerProtocol, ModelProtocol, logging
│   ├── geometry/             # 3D geometry: SE(3), equivariance, rotation representations
│   ├── bio_utils/            # Biological structure utils: PDB/SDF/SMILES parsing
│   └── plotting/             # Molecular visualization
├── projects/                 # One subdirectory per research project
│   └── <project>/            # e.g., affinitydiff_rl
│       ├── <project>/        # Project package (pip install -e projects/<project>)
│       │   ├── data.py
│       │   ├── model.py
│       │   └── ...
│       ├── configs/
│       │   └── config.yaml   # Hyperparameters and paths
│       ├── runs/             # Git-ignored; run outputs land here
│       │   └── <run_id>/     # e.g., {name}_{yyyymmdd}_{timestamp}_s{seed}_g{git_hash}
│       │       ├── config.yml
│       │       ├── train_metrics.csv
│       │       ├── eval_metrics.csv
│       │       ├── train_curve.png
│       │       ├── eval_curve.png
│       │       └── checkpoints/
│       ├── train.py          # Training entrypoint
│       ├── eval.py           # Evaluation / sampling entrypoint
│       ├── pyproject.toml
│       └── README.md
├── exports/                  # Frozen, versioned model releases
│   └── <project_v0.x>/
│       ├── config.yaml
│       ├── export_metadata.yaml
│       └── checkpoints/      # Git-ignored
├── scripts/                  # Thin shell wrappers for repeatable runs and ablations
├── tests/                    # Integration and unit tests
├── notebooks/                # Exploratory notebooks
├── pyproject.toml
└── README.md
```

## Working Model

- `ai4science/` is the shared library: framework (learners, data, models, utils) and domain utilities (geometry, bio_utils, plotting).
- `projects/<project>/` is a self-contained research project with its own installable package, configs, and entrypoints.
- `scripts/` contains thin orchestration commands for repeatable runs and ablations.

## Quick Start

**Lightning AI Studio** (conda base env is active by default):

```bash
git clone https://github.com/bowen0701/ai4science.git && cd ai4science && make install
```

**Local development** (activate a venv or conda env first):

```bash
git clone https://github.com/bowen0701/ai4science.git && cd ai4science
python3 -m venv .venv && source .venv/bin/activate
make install
```

`make install` installs the shared library and all project packages into the active environment.

## Makefile Targets

| Target          | Description                          |
|-----------------|--------------------------------------|
| `make install`  | Create `.venv` and install all packages |
| `make test`     | Run test suite                       |
| `make lint`     | Lint and auto-fix with ruff          |
| `make format`   | Format with ruff                     |
| `make typecheck`| Type-check `ai4science/` with mypy   |

## Running a Project

```bash
python projects/affinitydiff_rl/train.py
python projects/affinitydiff_rl/eval.py
```
