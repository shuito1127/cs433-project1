# Least Squares and Ridge: lab-to-project notes

Implemented in [implementations.py](../implementations.py), following the solution cells in the course lab `ex03.ipynb` (provided locally under `reference_4_proj1/solution/`). The implementation only depends on NumPy; the lab folder is not needed at runtime.

## Inputs and returned loss

- `y` has shape `(N,)`; `tx` has shape `(N, D)`.
- Both functions return `(w, loss)`, where `w` has shape `(D,)`.
- The course defines the data loss as `||y - tx @ w||² / (2N)`, or `0.5 * np.mean((y - tx @ w)**2)`.
- Ridge returns this same data loss, **without** adding the regularization penalty. The lab's Ridge function only returned `w`; the project public tests require both values.

## Least Squares

Minimize `L(w) = ||y - Xw||² / (2N)`, where `X = tx`.

Setting its gradient to zero gives:

```text
X.T @ (X @ w - y) / N = 0
(X.T @ X) @ w = X.T @ y
```

As in the lab, form `a = X.T @ X` and `b = X.T @ y`, then use `np.linalg.solve(a, b)`. Do not explicitly calculate the inverse. The columns of `X` must be linearly independent; singular systems can raise `np.linalg.LinAlgError`, and nearly dependent columns can cause numerical instability.

## Ridge Regression

Minimize `L(w) = ||y - Xw||² / (2N) + lambda_ * ||w||²`.

The penalty gradient is `2 * lambda_ * w`, so the normal equations become:

```text
(X.T @ X + 2 * N * lambda_ * I) @ w = X.T @ y
```

The factor `2 * N` comes from the course's loss normalization. `I` is a `D x D` identity matrix. Following the lab, all weights are penalized, including an intercept weight if present. With `lambda_ = 0`, Ridge reduces to Least Squares when the system is nonsingular. A positive `lambda_` makes the system invertible in exact arithmetic even with dependent features.

## Using the functions after preprocessing

```python
import numpy as np
from implementations import least_squares, ridge_regression

# X_train and X_val must already be cleaned and transformed consistently.
tx_train = np.c_[np.ones(X_train.shape[0]), X_train]
tx_val = np.c_[np.ones(X_val.shape[0]), X_val]

w_ls, loss_ls = least_squares(y_train, tx_train)
w_ridge, loss_ridge = ridge_regression(y_train, tx_train, lambda_=0.001)
val_scores = tx_val @ w_ridge
```

Neither function adds an intercept, imputes missing values, standardizes features, nor chooses a classification threshold. Supply finite numeric inputs and a nonnegative `lambda_`. Remove redundant columns for Least Squares. Learn preprocessing parameters on the training split and reuse them on validation data.

For this binary classification project, the output scores are continuous regression values, not probabilities. Select a threshold using validation data to obtain `-1`/`1` predictions and compute F1, precision, recall, and accuracy separately. The returned training loss is not a classification score.

## Verification

From the repository root, with the supplied grading dependencies installed:

```sh
python -m pytest grading_tests --github_link . -k "test_least_squares or test_ridge_regression"
```

These three public tests check Least Squares and Ridge at `lambda_ = 0` and `lambda_ = 1`, including weights, loss, and output shapes. Full-dataset model evaluation is a separate step.
