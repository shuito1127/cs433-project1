# EPFL CS-433 Machine Learning — Project 1

Team implementation, experiments and submission pipeline for Project 1.

## Setup and data

Use a Python environment compatible with the [official course grading environment](https://github.com/epfml/ML_course/tree/main/projects/project1/grading_tests).
Anthony's experiments were verified with Python 3.9.23 and NumPy 1.23.1. The notebook also needs Jupyter/ipykernel; the standalone comparison needs only NumPy.

Download the course data from AIcrowd and place it here:

```text
data/
├── x_train.csv
├── y_train.csv
└── x_test.csv
```

Data, local course documents, IDE settings and the local grading-test copy are excluded from Git.

## Anthony's contribution

- **Required methods:** `mean_squared_error_gd` and `mean_squared_error_sgd` in `implementations.py`.
- **Experiment guide:** [settings, results and reproduction](experiments/anthony_README.md).
- **Baseline notebook:** [GD and SGD on the smaller feature set](experiments/anthony.ipynb).
- **Feature comparison:** [standalone GD + one-hot experiment](experiments/gd_onehot_comparison/README.md).

Run the feature comparison from the repository root:

```sh
python experiments/gd_onehot_comparison/run.py
```

On the same validation split and at threshold -0.55, the smaller GD baseline achieved F1 0.3442 after longer training; adding one-hot categorical features achieved 0.4300. These are local validation scores, not AIcrowd scores. The experiment guide separates the effects of extra features, more training and threshold tuning.

## Submission integration

On this branch, the GD/SGD functions are implemented and their three selected public tests pass. The other four required functions and the root `run.py` remain placeholders awaiting the team's integration. The experimental runner under `experiments/` is separate from that final submission entry point.

Once the team's implementations and final runner are combined, check all six methods, reproduce the selected submission CSV and finish the report. Develop on individual branches and merge reviewed contributions into `main`.
