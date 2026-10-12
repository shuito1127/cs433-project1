"""Required function interfaces for EPFL CS-433 Project 1.

Implement the algorithms collaboratively after deriving and testing them.
"""

import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Compute linear regression with gradient descent.

    Parameters and return values must follow the Project 1 specification.
    """
    pass


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Compute linear regression with stochastic gradient descent.

    Parameters and return values must follow the Project 1 specification.
    """
    pass


def least_squares(y, tx):
    """Solve least squares using the normal equations, as in lab ex03.

    Args:
        y: Target array of shape (N,).
        tx: Design matrix of shape (N, D), with full column rank.

    Returns:
        w: Weight array of shape (D,).
        loss: Scalar mean squared error with the course's factor of 1/2.
    """
    # Solve (X.T X) w = X.T y without explicitly computing an inverse.
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)

    e = y - tx.dot(w)
    loss = 0.5 * np.mean(e**2)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Solve ridge regression using the normal equations, as in lab ex03.

    Args:
        y: Target array of shape (N,).
        tx: Design matrix of shape (N, D).
        lambda_: Nonnegative L2 regularization strength.

    Returns:
        w: Weight array of shape (D,).
        loss: Scalar half-MSE, excluding the regularization penalty.
    """
    # The objective is ||y - Xw||^2 / (2N) + lambda_ * ||w||^2.
    # As in the lab, regularize every weight, including any bias column.
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)

    # Project interface: return the data loss, not the penalized objective.
    e = y - tx.dot(w)
    loss = 0.5 * np.mean(e**2)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Compute logistic regression.

    Parameters and return values must follow the Project 1 specification.
    """
    pass


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Compute regularized logistic regression.

    Parameters and return values must follow the Project 1 specification.
    """
    pass

