# EPFL CS-433 Project 1 — Yusen's contribution

Least Squares and Ridge are evaluated with Shu's final preprocessing at
`f6baef4`. The six course methods are integrated locally for testing. This branch
provides linear-model candidate submissions; the team must still select its final
model, review the integration and finish the two-page report.

## Setup and data

Runtime: Python 3.9+ and NumPy. Jupyter is optional for notebook viewing. No external
ML/data libraries are used by the pipeline. From the repository root:

```sh
python -m pip install numpy
```

Put the course files in `dataset/x_train.csv`, `dataset/y_train.csv` and
`dataset/x_test.csv`. Data, training caches and prediction CSVs are ignored by
Git. Small fitted parameters are included in `models/` for quick reproduction.
The CDC documents explain fields; they are not replacement training data.

## Run

```sh
# Validation and model comparison; includes all labeled observations.
python -m experiments.evaluate_linear

# Generate both candidates, using included fitted parameters when compatible.
python run.py --model both

# Repeat prediction from matching saved weights and preprocessing.
python run.py --model both --predict-only
```

`python run.py` generates the Ridge candidate by default. It uses the settings in
`experiments/yusen_final_results.json`, trains if matching parameters are absent,
and writes `artifacts/submissions/yusen_ridge.csv`. The other candidate is
`yusen_least_squares.csv`. Loading checks sample IDs and file hashes. Checkpoints
avoid repeating matching L1 selection; remove the corresponding local
`artifacts/validation/` or `artifacts/full_train/` cache if a fresh selection run is
needed. `--retrain` refits the final weights but may reuse the selection cache.

L1 selection uses 2,000 full-batch iterations and can take tens of minutes on a
laptop. Compact NumPy matrix products reduce memory use. BLAS thread settings and
hardware affect runtime. The CLIs default to one MKL/OpenMP thread unless the
caller has configured them. Multi-threaded MKL solves were abnormally slow on the
local Windows ARM/x64 setup. In PowerShell, set these before starting Python or a
fresh Jupyter kernel if necessary:

```powershell
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
```

## Where to look

| File | Purpose |
| --- | --- |
| `implementations.py` | All six required methods; shared by experiments |
| [Fitted parameters](models/README.md) | Reproduce candidates without retraining |
| [Latest experiment](experiments/Yusen_final_preprocessing.ipynb) | Review results or rerun the experiment |
| [Results for the report](docs/yusen_final_results.md) | Metrics, settings, comparisons and limitations |
| [Experiment guide](experiments/README.md) | New experiment versus historical baseline |
| [Integration handoff](docs/integration.md) | Ownership, exact source commits, files to merge |
| [Preprocessing guide](preprocessing/README.md) | Shared fit/transform and feature selection |
| [Algorithm notes](docs/least_squares_ridge.md) | Formulas and returned-loss conventions |
| [Verification record](docs/verification.md) | Actual checks and remaining final-team checks |
| [Report guide](report/README.md) | LaTeX draft and current results snippet |
| [Codebook guide](codebook/README.md) | BRFSS field meanings and special codes |

## Tests

Install pytest for the additional adapter tests: `python -m pip install pytest`.

```sh
python -m pytest tests -q
conda env create -f grading_tests/environment.yml -n project1-grading
conda activate project1-grading
python -m pytest grading_tests --github_link . -k "not test_github_link_format"
```

The excluded URL-format check expects an immutable GitHub commit URL, whereas the
command above tests uncommitted local files. Black formatting is advisory in the
course tests. See the verification record for both checks. Git must be on PATH
for GitPython; with GitHub Desktop only, set `GIT_PYTHON_GIT_EXECUTABLE` to its
`resources/app/git/cmd/git.exe`. On Windows ARM, the course's old dependencies may
require a Windows x64 Conda environment (`CONDA_SUBDIR=win-64`).

The official tests are at
https://github.com/epfml/ML_course/tree/main/projects/project1/grading_tests.
The supplied local snapshot was checked against the official source. Passing
public tests does not replace team review or hidden grading tests.

## Final team integration

Keep validation outside `run.py`, reconcile `implementations.py` by function,
and preserve whichever pipeline generated the team's selected AIcrowd submission.
This branch's `run.py` is a **linear candidate**, not Shu's scored logistic pipeline.
The old baseline and original course material remain available for provenance.
The final report and immutable submission URL still require team approval.
