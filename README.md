# EPFL CS-433 Machine Learning — Project 1

This repository is the team workspace for EPFL CS-433 Machine Learning Project 1. It contains the project structure, experiments, and the final submission entry point.

## Setup

1. Clone this repository.
2. Create and activate a Python environment compatible with the course requirements.
3. Install the packages required by the course (at minimum, NumPy).
4. Place the AIcrowd data files as described below.

## Data

Download the project data from AIcrowd and place the CSV files directly in the `data/` directory:

```text
data/
├── x_train.csv
├── y_train.csv
└── x_test.csv
```

The `data/` directory and all CSV files are intentionally excluded from Git. Do not commit the competition data.

## Required files

- `implementations.py` — required functions specified by the project statement.
- `run.py` — entry point for the final prediction pipeline.
- `README.md` — project instructions and setup notes.
- `.gitignore` — prevents local data, generated files, and environment artefacts from being committed.
- `experiments/` — optional notebooks or scripts for reproducible experiments.

## Team workflow

Create a feature branch for each contribution, open a pull request, and merge reviewed changes into `main`.

