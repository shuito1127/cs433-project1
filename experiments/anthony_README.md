# Anthony's GD/SGD contribution

Two reproducible experiments support the required GD/SGD implementations: the original smaller-feature notebook and a controlled extension adding Shu's categorical features to GD. Both use only training CSVs and the same seed-42 validation split.

## Start here

**For the latest feature comparison**, run from the repository root:

```sh
python experiments/gd_onehot_comparison/run.py
```

It takes roughly 30–40 seconds on the verification machine. Read `gd_onehot_comparison/results/summary.md` afterwards. The [standalone README](gd_onehot_comparison/README.md) explains its settings, files and checks; [recorded_results.json](gd_onehot_comparison/recorded_results.json) preserves the verified results in Git. New runs write to the ignored `results/` folder.

**For the original GD/SGD comparison**, open [anthony.ipynb](anthony.ipynb), select a Jupyter-capable environment and use **Restart Kernel and Run All**. Launch from the repository root or `experiments/`. The final cell writes [anthony_baseline_results.json](anthony_baseline_results.json).

Both were verified with Python 3.9.23 and NumPy 1.23.1. Locally, `project1-work` supports the notebook; either `project1-work` or `project1-grading` supports the script. Other team members may use equivalent environments. Place `x_train.csv` and `y_train.csv` in `data/`; loaders verify feature/label ID alignment. The competition test CSV is not used.

## Verified results

Fixed score threshold **-0.55**; positive class **+1**:

| Model / experiment | Model columns | Updates | Validation half-MSE | Validation F1 |
|---|---:|---:|---:|---:|
| Original GD | 117 | 300 | 0.142665 | 0.3433 |
| Original SGD | 117 | 738,306 | 0.143658 | 0.3277 |
| GD, longer training | 117 | 3,600 | 0.142470 | 0.3442 |
| GD + one-hot, same 300-step checkpoint | 1,086 | 300 | 0.136063 | 0.4126 |
| GD + one-hot, longer training | 1,086 | 63,100 | 0.130466 | 0.4300 |

The two longer GD runs met the same relative-gradient stopping rule. The step size is calculated separately for each feature matrix, so 300 updates do not imply equal convergence. Longer training barely changed the smaller baseline's F1; including the extra categorical fields produced the larger improvement. This comparison tests those added fields with one-hot encoding; it does not isolate one-hot encoding against another encoding of the same fields.

A separate validation-threshold sweep, with final weights held fixed, gave F1 **0.3662** for the smaller GD model (threshold -0.633755) and **0.4311** for GD + one-hot (threshold -0.552754). These are scores on the validation set used for threshold selection, with selection optimism. None of these results is an independent post-selection test or an AIcrowd score.

The original SGD settings were not tuned and its budget differs from GD's. The scores do not establish that SGD is inherently worse. The five missing-code corrections are shared across the experiments; their individual effect has not been measured.

## Shared data preparation

| Setting | Value |
|---|---|
| Split | Random, not stratified; validation fraction 0.25; seed 42 |
| Rows | 246,102 training; 82,033 validation |
| Labels / initial weights | -1/+1; all weights start at zero |
| Smaller baseline | 111 ordered/numeric fields + 5 survey-presence fields + intercept |
| Missing ordered values | Fill with training-column means |
| Scaling | Standardize ordered fields using training statistics; preserve the presence-field encoding |
| Empty/constant fields | Entirely missing training fields contribute zero; zero standard deviations use scale 1 |
| Expanded model | Add 157 categorical source fields, yielding 969 one-hot columns |
| Category vocabulary | Training rows only; missing is a category; unseen held-out categories contribute all zeros for that field |

The categorical extension adds no decoded extra features, regularization or feature selection. All ordered-field cleaning remains identical to the smaller baseline.

## Optimizer settings

- **GD:** step size `1 / largest eigenvalue of X.T @ X / n`: 0.125769918 for the smaller model and 0.00848478904 for the expanded model. The original notebook uses 300 updates.
- **Longer GD:** check the training-gradient norm every 100 updates, after at least 300. Stop at `max(1e-10, 1e-4 * initial_gradient_norm)` or 100,000 updates; explicitly report if the cap is reached. This is a numerical stopping criterion, not a guarantee of best validation performance.
- **SGD:** 738,306 single-row updates, sampled uniformly with replacement; seed 42; step size 1.96500766593e-06, the reciprocal of the largest squared training-row length. This is three dataset-sized amounts of work, not three guaranteed epochs.
- **Cached GD:** the standalone runner precomputes `X.T @ X / n` and `X.T @ y / n` in chunks, then performs the same iterative full-batch updates. It does not substitute a closed-form least-squares solver. Numerical equivalence is tested against the original GD function.

## Verification

With the official grading folder copied locally, run from the project root:

```sh
python -m pytest grading_tests --github_link . -k test_mean_squared_error
python experiments/gd_onehot_comparison/verify.py
```

The selected public tests cover Anthony's two required methods: **3 passed**. The standalone comparison has **7 passing checks** for numerical equivalence, data alignment, split consistency, training-only preparation, category handling, stopping criteria and threshold selection. This does not claim that the other four methods or final submission pipeline pass.

The two JSON result snapshots contain aggregate results/settings only; no data rows or prediction vectors.
