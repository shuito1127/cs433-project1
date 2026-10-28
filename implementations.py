"""Six course methods: GD/SGD (Anthony), least squares/ridge (Yusen), logistic methods (Shu)."""


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
        i = np.random.randint(n)  # Pick random row (w/ replacement)
        error = tx[i] @ w - y[i]
        gradient = tx[i] * error  # No division by n. Only uses 1 example.
        w = w - (gamma * gradient)

    errors = tx @ w - y
    loss = 0.5 * np.mean(errors**2)
    return w, loss


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

    return np.float64(loss)


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
        grad = calculate_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad

    loss = calculate_logistic_loss(y, tx, w)

    return w, loss


def l1reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Shu's L1 subgradient selector; penalizes every weight, including bias."""
    w = initial_w.copy()

    for _ in range(max_iters):
        grad = calculate_logistic_gradient(y, tx, w) + lambda_ * np.sign(w)
        w = w - gamma * grad

    loss = calculate_logistic_loss(y, tx, w)

    return w, loss
