import numpy as np

"""Required function interfaces for EPFL CS-433 Project 1.

Implement the algorithms collaboratively after deriving and testing them.
"""


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


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent.

    Returns:
        - w;        Final model weights.
        - loss;     full-data, (0.5*) MSE, After single-example SGD.
    """
    w = initial_w.copy()
    n = len(y)

    for _ in range(max_iters):
        i = np.random.randint(n)    # Pick random row (w/ replacement)
        error = tx[i] @ w - y[i]
        gradient = tx[i] * error    # No division by n. Only uses 1 example.
        w = w - (gamma * gradient)

    errors = tx @ w - y
    loss = 0.5 * np.mean(errors**2)
    return w, loss

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

