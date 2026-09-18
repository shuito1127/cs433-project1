import numpy as np
def train_val_split(X, y, val_ratio, seed=42):
    assert len(X) == len(y)
    assert 0 < val_ratio < 1
    N = len(y)

    rng = np.random.default_rng(seed)

    indices = np.arange(N)

    rng.shuffle(indices)

    n_val = int(N*val_ratio)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_train = X[train_indices]
    X_val = X[val_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]

    return X_train, X_val, y_train, y_val