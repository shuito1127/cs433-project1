"""Anthony's original GD, copied for the numerical equivalence check.

The experiment uses the same GD updates after precomputing X.T @ X / n
and X.T @ y / n once. See run.py and verify.py.
"""
import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.

    Returns:
        - w;        Final model weights.
        - loss;     (0.5*) MSE, after full-batch GD.
    """
    w = initial_w.copy()
    n = len(y)

    for _ in range(max_iters):
        errors = tx @ w - y
        gradient = (tx.T @ errors) / n
        w = w - (gamma * gradient)

    errors = tx @ w - y
    loss = 0.5 * np.mean(errors**2)
    return w, loss
