# EPFL CS-433 Machine Learning — Project 1

This repository is the team's working directory. It contains Shu's experiments, course starter materials, and the interfaces for the final submission. Open `cs433-project1/` as your IDE workspace; daily work does not require the sibling course starter folder.

## Repository layout

```text
cs433-project1/
├── dataset/                     # Local competition CSVs (ignored by Git)
│   ├── x_train.csv
│   ├── y_train.csv
│   ├── x_test.csv
│   └── sample_submission.csv
├── codebook/                    # Three BRFSS PDFs and reading guide
├── docs/project1_description.pdf
├── experiments/
│   ├── Shu_notebook.ipynb        # Exploratory analysis and baseline experiments
│   ├── Shu.run.ipynb             # Train/validation preprocessing and ridge baseline
│   ├── Yusen_least_squares_ridge.ipynb # Reproduce Shu and compare shared models
│   ├── yusen_baseline_results.json    # Saved validation metrics and provenance
│   └── README.md                # Experiment instructions
├── preprocessing/
│   ├── preprocess.py            # Split and standardization helpers
│   └── shu_baseline_config.py   # Snapshot of Shu's feature groups and code rules
├── grading_tests/               # Unmodified snapshot of course public tests
├── report/latex-example-paper/  # Complete course LaTeX example and supporting files
├── helpers.py                   # Course CSV loading and submission helpers
├── implementations.py           # Least squares and ridge implemented; four stubs remain
└── run.py                       # Final pipeline entry point (currently a stub)
```

## Setup and data

1. Create a Python environment compatible with the [project requirements](docs/project1_description.pdf). The existing notebooks use NumPy and Matplotlib; install Jupyter support to run them.
2. Place the course/AIcrowd CSV files in `dataset/`, directly under the repository root. This local workspace already contains copies from the course starter dataset. Teammates cloning the repository must supply their own data.
3. Read [codebook/README.md](codebook/README.md) for field meanings, special value codes, and derived-variable documentation.

`dataset/`, the legacy `data/` path, and all CSV files are ignored by Git. Do not commit competition data or generated submission CSVs. The CDC documentation describes the source survey; use the course-provided CSVs for model development.

## Running Shu's experiments

From the repository root, launch Jupyter with:

```sh
cd experiments
jupyter lab
```

Open `Shu.run.ipynb` for the focused baseline or `Shu_notebook.ipynb` for the exploratory work. With a kernel working directory of `experiments/`, the existing `../dataset` paths and parent-directory helper imports resolve within this repository. In an IDE, check `Path.cwd()` in the notebook and select `experiments/` as the kernel working directory if necessary.

The notebook code and saved outputs have been preserved. Saved metrics are historical outputs, not a fresh validation of this integration. `Shu.run.ipynb` splits training and validation data before fitting imputation and scaling parameters; parts of `Shu_notebook.ipynb` impute before splitting and should be treated as exploratory work.

`implementations.py` now implements `least_squares` and `ridge_regression` using the lab ex03 normal equations. Both return `(w, loss)`, with `loss = mean((y - tx @ w)**2) / 2`; Ridge's returned loss excludes the regularization penalty. See [Least Squares and Ridge notes](docs/least_squares_ridge.md) for formulas, assumptions, and a usage example.

The other four algorithms and `run.py` still contain `pass` placeholders. Running `python run.py` does not yet train a model or generate a submission. Shu's notebooks retain their original local function definitions; importing these implementations and unpacking `(w, loss)` is necessary to use the shared versions.

## Yusen's baseline comparison

For a comparison on the actual dataset, see [Yusen's baseline notebook](experiments/Yusen_least_squares_ridge.ipynb) and the [experiment guide](experiments/README.md). It reproduces Shu's fixed split, preprocessing, Ridge regularization, and threshold. The original matrix has two exact redundant columns, so after reproducing Ridge on the original inputs, it compares shared Least Squares and Ridge on the same reduced, full-rank inputs. Validation metrics and file hashes are saved in [the results summary](experiments/yusen_baseline_results.json).

## Course public tests

The original instructions and environment specification are in [grading_tests/INSTRUCTIONS.md](grading_tests/INSTRUCTIONS.md) and [grading_tests/environment.yml](grading_tests/environment.yml). With Conda and Git installed, create the supplied grading environment from the repository root:

```sh
conda env create --file grading_tests/environment.yml --name project1-grading
conda activate project1-grading
python -m pytest grading_tests --github_link . -k "not test_github_link_format"
```

The local command excludes only the GitHub URL-format check. Tests for the four unimplemented algorithms are expected to fail. To run just the three numerical public tests for Least Squares and Ridge:

```sh
python -m pytest grading_tests --github_link . -k "test_least_squares or test_ridge_regression"
```

The grading environment includes testing dependencies; it is not a list of libraries permitted in the submitted implementation. Follow the project description for those restrictions.

The course recommends running tests from its own repository to receive updates. A snapshot is included here to support working entirely inside this folder; check the course repository for updates before final validation. Passing public tests does not guarantee full grading credit.

## Report

The complete starter template is in [report/latex-example-paper/](report/latex-example-paper/), including the example PDF, TeX source, bibliography, class/style files, and figures. Work from that directory when compiling the template so its relative references resolve.

## Integration notes

Course starter materials were copied into this workspace as follows:

| Original path in the sibling `project1/` folder | Destination here |
| --- | --- |
| `data/dataset_to_release/dataset/*.csv` | `dataset/` |
| `file/project1_description.pdf` | `docs/project1_description.pdf` |
| `grading_tests/` | `grading_tests/` |
| `latex-example-paper/` | `report/latex-example-paper/` |
| `helpers.py` | Existing `helpers.py` retained; SHA-256 matched the starter file |

The original starter folder remains available as a reference. Shu's notebooks, preprocessing code, algorithm interfaces, and entry point were preserved. The separate `reference_4_proj1/` lab examples were not part of this starter-folder integration.

## Team workflow

Work on your own feature branch, review changes through a pull request, and merge agreed changes into `main`. Share code and documentation through Git; each teammate maintains their local `dataset/` directory.
