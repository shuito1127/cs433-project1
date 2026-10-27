# Yusen: final-preprocessing linear-model results

Measured on all 328,135 labeled observations using Shu's preprocessing at
`f6baef4dd878ac79d030aa3c65cdda6e905a903b`. Source: [exact results](../experiments/yusen_final_results.json).
These are local validation results, not AIcrowd scores.

## Report table

| Model | Input/settings | Threshold | Validation accuracy | Validation F1 |
| --- | --- | ---: | ---: | ---: |
| Earlier Least Squares | Partial features | -0.55 | 0.877123 | 0.344603 |
| Earlier Ridge | Partial features | -0.55 | 0.878098 | 0.343401 |
| least_squares | Final preprocessing, fixed settings | -0.55 | 0.875782 | 0.428940 |
| ridge | Final preprocessing, fixed settings | -0.55 | 0.875184 | 0.429614 |
| least_squares | Final preprocessing, validation-selected settings | -0.57 | 0.869260 | 0.429127 |
| ridge | Final preprocessing, validation-selected settings | -0.56 | 0.872783 | 0.429789 |

Earlier Ridge used lambda 0.001; the new fixed Ridge comparison uses the same
lambda. Both old and new experiments use the same 75%/25% split and seed 42.
The new pipeline adds categorical/decoded inputs and supervised feature selection
together, so this is a comparison of pipeline versions, not an isolated decoding
ablation. The fixed-setting rows isolate the benefit of that combined pipeline
change from the later parameter search.

## Selected settings and diagnostics

| Model | Lambda | Threshold | Precision | Recall | Train half-MSE | Validation half-MSE | TP / FP / FN / TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| least_squares | None | -0.57 | 0.349035 | 0.556922 | 0.129122 | 0.130538 | 4031 / 7518 / 3207 / 67277 |
| ridge | 0.0001 | -0.56 | 0.355477 | 0.543382 | 0.129242 | 0.130442 | 3933 / 7131 / 3305 / 67664 |

- Training / validation: 246,102 / 82,033;
  positives 21,737 / 7,238.
- NumPy `default_rng(42)` shuffle, not stratified; one holdout, no cross-validation.
- Inputs: 5 one-value, 111 ordered, 14 decoded and 157 categorical raw columns.
  Expanded columns including intercept: 1100.
- L1 selector: lambda 0.02, step 0.2, 2,000 iterations, absolute-weight cutoff 0.001.
  Training-only selection retained 844 columns.
- Rank handling retained 734 columns; dropped 110.
  Squared relative residual tolerance: 1e-10; largest discarded
  relative residual: 8.22e-07; normalized
  condition number of retained design: 8.22e+03.
  Exact column names and diagnostics are in the JSON.
- All-missing numeric columns: 0; constant numeric
  columns: 0. Unseen validation categories are recorded
  per raw column in the JSON and encoded as all-zero.
- Original selected-input Ridge (before rank handling), lambda 0.001 / threshold
  -0.55: F1 0.429598, accuracy
  0.875209. Removing columns
  changes Ridge's L2 penalty representation; its effect must not be described as
  merely fixing the solver.
- Always-negative baseline: accuracy 0.911767, F1 0.
- Runtime environment: Python 3.14.6, NumPy 2.4.6.

## Interpretation and limits

Lambda was selected from 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1; thresholds from -0.90
through 0.00 in increments of 0.01. Selection maximized validation F1. Therefore
the selected-setting scores use the same data for tuning and evaluation and can
be optimistic; do not describe them as an untouched test estimate. The fixed
comparison is provided separately. Regression scores are not probabilities.

The task is imbalanced, so accuracy alone can favor an always-negative classifier.
Use F1 with precision/recall and the confusion counts above. A small difference
between two methods on one split is not evidence of general superiority.

After model selection, preprocessing, L1 selection and both linear models were
refitted on all labeled rows. `python run.py --model both` reproduces predictions
from the included fitted state; use `--retrain` to refit weights. Full-fit counts and
verified CSV hashes are in [yusen_submissions.json](../experiments/yusen_submissions.json).
No hidden test labels are available locally, and neither generated candidate has
been uploaded by this workflow. Shu's reported AIcrowd F1 0.444 is a separate
team-reported competition result, not a directly comparable local validation row.

## Reproduce and integrate

```sh
python -m experiments.evaluate_linear
python -m experiments.export_linear_results
python run.py --model both
```

See [integration notes](integration.md) and [verification](verification.md).
The Overleaf-ready excerpt is [report/yusen_results.tex](../report/yusen_results.tex).
