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
    """Compute least-squares regression using the course-specified approach.

    Parameters and return values must follow the Project 1 specification.
    """
    pass


def ridge_regression(y, tx, lambda_):
    """Compute ridge regression with regularization strength ``lambda_``.

    Parameters and return values must follow the Project 1 specification.
    """
    pass

def sigmoid(t):
    t = np.clip(t, -500, 500)
    return 1 / (1 + np.exp(-t))


def calculate_logistic_loss(y, tx, w):
    """
    y:  (N,) with values {0, 1}
    tx: (N, D)
    w:  (D,)
    """
    z = tx @ w

    loss = np.mean(np.logaddexp(0, z) - y * z)

    return float(loss)


def calculate_logistic_gradient(y, tx, w):
    """
    Gradient of logistic loss.
    """
    p = sigmoid(tx @ w)

    grad = tx.T @ (p - y) / len(y)

    return grad


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Compute logistic regression.

    Parameters and return values must follow the Project 1 specification.
    """
    w = initial_w.copy()

    for _ in range(max_iters):
        grad = calculate_logistic_gradient(y, tx, w)
        w = w - gamma * grad

    loss = calculate_logistic_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Compute regularized logistic regression.

    Parameters and return values must follow the Project 1 specification.
    """
    w = initial_w.copy()

    for _ in range(max_iters):
        grad = calculate_logistic_gradient(y, tx, w) + 2*lambda_*w
        w = w - gamma * grad

    loss = calculate_logistic_loss(y, tx, w)

    return w, loss

def l1reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w.copy()

    for _ in range(max_iters):
        grad = calculate_logistic_gradient(y, tx, w) + lambda_*np.sign(w)
        w = w - gamma * grad

    loss = calculate_logistic_loss(y, tx, w)

    return w, loss

