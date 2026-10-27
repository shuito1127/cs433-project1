# Experiments

## Current: Shu's final preprocessing + Yusen's linear models

Run `python -m experiments.evaluate_linear` from the repository root. The review
notebook is [Yusen_final_preprocessing.ipynb](Yusen_final_preprocessing.ipynb).
It displays saved results by default; set `RERUN = True` to train/evaluate again.
The experiment imports the shared algorithms from `implementations.py`.

- Source: Shu `f6baef4`; 5 one-value, 111 ordered, 14 decoded, 157 categorical inputs.
- Split: NumPy `default_rng(42)`, shuffled 75%/25%, 246,102 / 82,033 observations,
  not stratified. This matches the earlier baseline's sample split.
- Fit means/scales/categories and supervised L1 selection on training rows only.
- L1: lambda 0.02, gamma 0.2, 2,000 iterations, cutoff 0.001; keep the intercept.
- Diagnose rank after selection; fit Least Squares and Ridge on the same
  independent columns. Also record Ridge before rank reduction.
- Fixed comparison: threshold -0.55, Ridge lambda 0.001, as in the earlier baseline.
- Search: Ridge lambdas `1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1`; thresholds -0.90 to 0.00
  inclusive in steps of 0.01. Maximize validation F1, then accuracy; threshold ties
  prefer the higher threshold. Lambda ties retain the earlier ascending-grid entry.
- Labels are -1/+1 for linear models. Scores are not probabilities. Returned loss
  is half-MSE and excludes the Ridge penalty, including when scoring training data.

The search reuses validation labels, so selected scores are model-selection
estimates, not independent test performance. No cross-validation or AIcrowd
evaluation is claimed. `python run.py --model both` reproduces candidate CSVs
from the included full-data fitted state. Use `--retrain` to refit weights.

Read [report-ready results](../docs/yusen_final_results.md). Exact values, confusion
counts, all search settings and source/data hashes are in
[yusen_final_results.json](yusen_final_results.json). Candidate CSV hashes and
full-data feature counts are in [yusen_submissions.json](yusen_submissions.json).

## Historical partial-feature baseline

[Yusen_least_squares_ridge.ipynb](Yusen_least_squares_ridge.ipynb) and
[yusen_baseline_results.json](yusen_baseline_results.json) preserve the original
experiment: 111 ordered variables plus 6 one-value inputs, 118 columns including
bias. `CTELNUM1` and `CELLFON2` were exactly `-CTELENUM`; dropping them gave 116
independent columns. Ridge lambda 0.001 and threshold -0.55 gave F1 0.343401;
Least Squares gave 0.344603. Both used the same seed/split as above.

The original `Shu.run.ipynb` and `Shu_notebook.ipynb` remain unchanged. Their saved
outputs are historical. The new experiment does not call their local model
implementations or reuse preprocessing fitted on all labeled data.

See [integration notes](../docs/integration.md) for rank-handling and source details.
