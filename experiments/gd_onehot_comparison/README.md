# One experiment: does one-hot encoding help Anthony's GD?

## Run it

From the **repository root**, using Python 3.9+ and NumPy (verified: Python 3.9.23 / NumPy 1.23.1):

```sh
python experiments/gd_onehot_comparison/run.py
```

Or open this folder's `run.py` in PyCharm, select a compatible interpreter (locally **project1-work**), and run it.
The working directory does not matter: data paths are resolved relative to this file.

It prints progress and the final comparison. Read **results/summary.md** afterwards. The verified full run took about 33 seconds.

[recorded_results.json](recorded_results.json) is the fixed, versioned snapshot used in the report. Subsequent runs update only the ignored `results/` folder; they do not overwrite this snapshot.
It uses only NumPy and Python's standard library; no installation is needed in your existing environments.
Results are overwritten on another run. This folder's `.gitignore` excludes generated results.

If you move this folder elsewhere, it remains runnable with explicit data location:

```sh
python run.py --data-dir /path/to/project/data
```

Required inputs are `x_train.csv` and `y_train.csv`. The script reads them; it does not copy or alter them.
The competition test CSV is not used. No other project code, notebooks, grading folder, or GitHub access is needed.

## What you are comparing

1. **The existing smaller baseline:** 5 survey-presence fields + 111 ordered fields + intercept.
2. **That same baseline, plus Shu's 157 categorical fields:** one-hot encoded using categories found in training rows.

The seed-42, 75%/25% split is exactly the procedure in `../anthony.ipynb`. Labels remain -1/+1.
The existing mean filling, scaling, special-code rules and intercept are preserved.
The five corrected missing codes are also preserved: MAXVO2_/FC60_=999; PAFREQ1_/PAFREQ2_/STRFREQ_=99.

For the categorical fields, this follows Shu's encoding: each distinct raw response code is a category;
missing values get their own category when present in training. A category seen only in validation
gets all zeros for that field. These categorical fields are not given the ordered-field missing-code rules.
No category is dropped. Redundant columns are acceptable for GD; no matrix inverse is used.
No decoded extra fields, L1 selection, L2 regularization or logistic regression are added.

This is a comparison of **including these extra fields with one-hot encoding**, not proof that one-hot
is better than every alternative encoding of those same fields. The smaller model reproduces the original notebook's 300-step result.

## How to read the results

**First table: fixed threshold -0.55.** Both feature sets are shown after 300 updates and at the end of longer GD.
Compare the baseline's 300-step and final rows to see the effect of additional optimization.
Compare both final rows for the feature extension, checking their convergence status.
The step size is calculated separately for each training matrix using the same `1 / largest eigenvalue` rule.
Equal update counts do not imply equal convergence when features and step sizes differ.

**Second table: threshold tuning.** Each final weight vector is kept fixed. The code searches all distinct validation
scores for the threshold giving highest validation F1. This separates changing the decision rule from retraining.
The selected threshold uses validation labels, so its F1 is a tuning result with selection optimism.
Do not describe it as independent test performance or compare it directly with an AIcrowd score.
A fresh holdout would be needed for a separate estimate after choosing features/settings/thresholds.

**Convergence:** start at zero weights, run at least 300 updates, and check the gradient every 100 updates.
Stop when its norm is at most `max(1e-10, 0.0001 * initial_gradient_norm)`, or at 100,000 updates.
This is a numerical stopping rule, not a guarantee of best validation F1 or exact optimal weights.
The script explicitly reports `iteration_cap_reached` if the tolerance was not reached.
The training curve records loss and gradient norm, and an unexpected loss increase/non-finite value raises an error.

If a run hits the cap and you want to investigate further:

```sh
python experiments/gd_onehot_comparison/run.py --max-iters 300000 --output-dir experiments/gd_onehot_comparison/results/longer_run
```

## Why this runs without a huge one-hot array

For squared-loss GD, `(X.T @ (X @ w - y)) / n` equals `(X.T @ X / n) @ w - X.T @ y / n`.
The script calculates the two constant parts once, in chunks, then performs the same full-batch GD update repeatedly.
It never switches to a closed-form solver. `reference_gd.py` is a snapshot of Anthony's implementation;
`verify.py` checks the cached calculation against it, including a redundant one-hot design.
Floating-point rounding can differ slightly because the sums are grouped differently.
Raw data and the small baseline matrix are held in memory; the full expanded matrix is only built one chunk at a time.
Thread defaults are limited to four where supported, unless you already set them in your environment.

## Files and provenance

| File | Purpose / origin |
|---|---|
| `run.py` | New standalone comparison, cached full-batch GD, validation, threshold search and reporting. |
| `baseline.py` | Behavior-preserving snapshot of the three preprocessing functions from Anthony's `preprocessing/anthony_baseline.py`, 2026-10-26. |
| `feature_schema.py` | Anthony's corrected `baseline_schema.py`, plus the categorical name list from Shu's notebook. |
| `reference_gd.py` | Anthony's original `mean_squared_error_gd`, copied unchanged from `implementations.py`, commit f97a921. Used for verification. |
| `verify.py` | Tests for GD equivalence, split consistency, preprocessing, missing/unseen categories, ID alignment and threshold selection. |
| `recorded_results.json` | Verified metrics/settings snapshot retained in Git for the report. |
| `results/summary.md` | Human-readable result tables and convergence checks. |
| `results/metrics.json` | Exact metrics, settings, source hashes and training/validation ID-order hashes. |
| `results/training_curve.csv` | Training loss and gradient norm every 100 updates. |
| `results/feature_names.json` | Ordered names of all expanded model columns. |
| `results/model_state.npz` | Final and 300-step weights, training-fitted preprocessing and category vocabularies; numeric arrays only. |

Shu source: [categorical_baseline.ipynb at df0feda](https://github.com/shuito1127/cs433-project1/blob/df0feda4891b10c69e2be8e4f8e715e1f166637f/experiments/categorical_baseline.ipynb).
This is a fixed snapshot: later edits elsewhere in the project will not silently change this experiment.
Inherited ordered-field choices (for example ASTHMAGE 97 -> 10) remain baseline assumptions, not newly validated rules.

Optional small checks, without loading the dataset:

```sh
python experiments/gd_onehot_comparison/verify.py
```
