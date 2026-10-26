"""Snapshot of Anthony's existing train-fitted baseline preprocessing.

Copied unchanged in behavior on 2026-10-26; standalone local imports only.
"""
import numpy as np
from feature_schema import ONE_VALUE_NAMES, ORDERED_NAMES, ORDERED_RULES

FEATURE_NAMES = ONE_VALUE_NAMES + ORDERED_NAMES


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
