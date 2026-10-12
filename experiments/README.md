# Baseline experiments

## Yusen: reproduce Shu and compare shared models

Open [Yusen_least_squares_ridge.ipynb](Yusen_least_squares_ridge.ipynb) and run all cells from a fresh Python kernel. Start Jupyter from the repository root or `experiments/`; the notebook resolves paths from either location. Install NumPy and your preferred Jupyter frontend. Keep `x_train.csv` and `y_train.csv` in `dataset/`. Several GB of available RAM are needed for the full dataset and model matrices.

The notebook:

1. Verifies that the extracted feature configuration matches the literal definitions in `Shu.run.ipynb`.
2. Loads the labeled training data in chunks, checks feature/label sample IDs, and reproduces the 75%/25% split with seed 42.
3. Applies Shu's one-observed-value handling and 111 ordered/numeric feature rules. Imputation and standardization parameters come only from the training split.
4. Reproduces Shu's Ridge baseline with `lambda_=0.001` and threshold `-0.55`, checking his saved confusion counts.
5. Calls shared Ridge from `implementations.py` and verifies it matches the original reference. Diagnoses two exact redundant columns, removes them using training-only decisions, and then fits both shared Least Squares and shared Ridge on the same reduced matrices and threshold.
6. Reports half-MSE, accuracy, precision, recall, F1, and confusion counts. An always-negative baseline puts accuracy into context.
7. Saves settings, feature names, file hashes, environment versions, and metrics to [yusen_baseline_results.json](yusen_baseline_results.json). Running the final cell replaces this experiment's previous summary.

The configuration in [preprocessing/shu_baseline_config.py](../preprocessing/shu_baseline_config.py) is an explicit snapshot of Shu's feature groups and code mappings, not a new audit of their correctness. The notebook checks for all-missing and zero-variance ordered features, nonfinite model inputs, and any remaining rank deficiency after exact redundant-column removal; it stops rather than silently changing the experiment further.

## Why there are original and reduced comparisons

Shu's original matrix has 118 columns including the intercept, but rank 116. After his missing-value replacement, `CTELNUM1` and `CELLFON2` are both exactly `-CTELENUM`. Ridge with a positive lambda can solve this system; unregularized Least Squares does not have unique weights.

The notebook first reproduces Ridge on all 118 columns. It then detects identical/sign-reversed columns in the one-observed-value group using training values only, keeps the first representative, and removes the two redundant columns from both training and validation matrices. Both shared models are compared on the resulting 116-column, full-rank matrices. The shared Least Squares function is used directly, without a pseudoinverse or an added penalty.

Removing redundant columns preserves the set of possible linear predictions but changes the coefficient representation and therefore Ridge's L2 penalty. The original Ridge result and reduced Ridge result are explicitly labeled separately; differences between them should not be attributed solely to a model implementation change.

## Scope and interpretation

### Recorded baseline run

The notebook was executed end to end on the full labeled dataset: 246,102 training rows and 82,033 validation rows, with Python 3.14.6 and NumPy 2.4.6. All nine code cells completed, including configuration equality, sample-ID alignment, historical Ridge confusion counts, shared/reference Ridge equality, and full-rank comparison checks. This used the local environment rather than the course's pinned grading environment.

| Model | Columns including intercept | Validation accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shu Ridge reference | 118 | 0.878098 | 0.327202 | 0.361288 | 0.343401 |
| Shared Ridge, original inputs | 118 | 0.878098 | 0.327202 | 0.361288 | 0.343401 |
| Shared Ridge, reduced inputs | 116 | 0.878098 | 0.327202 | 0.361288 | 0.343401 |
| Shared Least Squares, reduced inputs | 116 | 0.877123 | 0.325473 | 0.366123 | 0.344603 |
| Always negative | 0 | 0.911767 | 0 | 0 | 0 |

On the matched 116-column inputs, Least Squares has slightly higher F1 and recall, while Ridge has slightly higher precision and accuracy. This small difference on one split with a fixed threshold is not evidence of a general winner. The high accuracy of the always-negative baseline illustrates why accuracy alone is insufficient here. Exact metrics and input hashes are in the JSON summary; the notebook preserves actual execution outputs.

### Limits

- The original `Shu.run.ipynb` and `Shu_notebook.ipynb` are preserved.
- No additional feature groups, hyperparameter search, or threshold tuning are introduced.
- Classification uses `score >= -0.55` for the positive class (+1). This inherited threshold is held fixed, not optimized independently for each model.
- Shared algorithms return `(w, loss)`, where loss is training half-MSE. The Ridge loss excludes the regularization penalty.
- Scores are continuous regression predictions, not calibrated probabilities.
- These are local validation results, not competition test scores. The unlabeled `x_test.csv` is not loaded and no submission is generated.
- Numerical equality can vary slightly across NumPy/BLAS environments. A failed reproduction assertion should be investigated before interpreting model differences.

For the formulas and interfaces, see [Least Squares and Ridge notes](../docs/least_squares_ridge.md). Keep this baseline stable and evaluate subsequent preprocessing changes in a separate experiment using the same split.
