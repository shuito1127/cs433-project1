"""Reusable, train-fitted preparation for Anthony's first GD/SGD comparison."""
import csv
from itertools import islice
from pathlib import Path

import numpy as np

from .baseline_schema import ONE_VALUE_NAMES, ORDERED_NAMES, ORDERED_RULES

FEATURE_NAMES = ONE_VALUE_NAMES + ORDERED_NAMES


def load_training_data(data_dir):
    """Read only selected training columns in chunks; verify label ID alignment.

    The competition test CSV is deliberately unnecessary for validation work.
    """
    data_dir = Path(data_dir)
    labels = np.loadtxt(data_dir / 'y_train.csv', delimiter=',', skiprows=1)
    chunks = []
    with (data_dir / 'x_train.csv').open() as handle:
        header = next(csv.reader([next(handle)]))
        columns = [0] + [header.index(name) for name in FEATURE_NAMES]
        while True:
            lines = list(islice(handle, 5000))
            if not lines:
                break
            chunks.append(np.genfromtxt(lines, delimiter=',', usecols=columns, ndmin=2))
    data = np.concatenate(chunks)
    if data.shape[0] != labels.shape[0] or not np.array_equal(data[:, 0], labels[:, 0]):
        raise ValueError('Training feature IDs and label IDs do not match in order.')
    y = labels[:, 1].astype(int)
    if not np.all(np.isin(labels[:, 1], [-1, 1])):
        raise ValueError('Expected labels -1 and +1.')
    return data[:, 1:].copy(), y


def clean_features(x):
    """Apply fixed survey-code rules; no averages are learned here."""
    x = np.asarray(x, dtype=float).copy()
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError('Input must contain FEATURE_NAMES in their listed order.')
    if np.isinf(x).any():
        raise ValueError('Unexpected infinite raw feature value.')
    n_flags = len(ONE_VALUE_NAMES)
    # Preserve Shu's observed code versus missing (-1) distinction.
    x[:, :n_flags] = np.where(np.isnan(x[:, :n_flags]), -1, x[:, :n_flags])
    for j, name in enumerate(ORDERED_NAMES, start=n_flags):
        rule = ORDERED_RULES.get(name, {})
        original = x[:, j].copy()
        for old, new in rule.get('remap_codes', {}).items():
            x[original == old, j] = new
        for code in rule.get('zero_codes', []):
            x[original == code, j] = 0
        for code in rule.get('nan_codes', []):
            x[original == code, j] = np.nan
    return x


def fit_preprocessor(x_train):
    """Learn imputation and scaling exclusively from the training split.

    Entirely missing training columns use zero. Constant columns use scale one.
    The first five fields retain Shu's scaling; ordered fields are standardized.
    """
    x = clean_features(x_train)
    count = np.sum(~np.isnan(x), axis=0)
    means = np.divide(np.nansum(x, axis=0), count,
                      out=np.zeros(x.shape[1]), where=count > 0)
    filled = np.where(np.isnan(x), means, x)
    center = np.mean(filled, axis=0)
    scale = np.std(filled, axis=0)
    n_flags = len(ONE_VALUE_NAMES)
    center[:n_flags] = 0
    scale[:n_flags] = 1
    scale[scale == 0] = 1
    state = {'impute': means, 'center': center, 'scale': scale,
             'all_missing': count == 0}
    return transform_features(x_train, state), state


def transform_features(x, state):
    """Reuse training statistics and append an explicit intercept column."""
    x = clean_features(x)
    x = np.where(np.isnan(x), state['impute'], x)
    # No training information exists for these columns; ignore held-out values.
    x[:, state['all_missing']] = 0
    tx = np.column_stack((np.ones(len(x)), (x - state['center']) / state['scale']))
    if not np.isfinite(tx).all():
        raise ValueError('Prepared matrix contains non-finite values.')
    return tx


def classification_metrics(y, predictions):
    """Positive class is +1. Return F1 and accuracy without sklearn."""
    tp = np.sum((y == 1) & (predictions == 1))
    fp = np.sum((y == -1) & (predictions == 1))
    fn = np.sum((y == 1) & (predictions == -1))
    denominator = 2 * tp + fp + fn
    return {'f1': float(2 * tp / denominator) if denominator else 0.0,
            'accuracy': float(np.mean(y == predictions))}
