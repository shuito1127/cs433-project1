"""Reusable, training-only adapter for Shu's final preprocessing (NumPy only)."""

import numpy as np

from implementations import l1reg_logistic_regression
from preprocessing import shu_reference as shu


class EncodedDesign:
    """Exact numeric + one-hot matrix products without allocating dense one-hot.

    This supports Shu's unchanged L1 optimizer via ``X @ w`` and ``X.T @ r``.
    Missing categories have their own level; unseen categories produce zeros.
    """

    def __init__(self, numeric, codes, widths, names):
        self.numeric = np.ascontiguousarray(numeric)
        self.codes = np.asfortranarray(codes, dtype=np.int16)
        self.widths = widths
        self.names = names
        self.shape = (len(numeric), numeric.shape[1] + sum(widths))

    def __matmul__(self, w):
        result = self.numeric @ w[: self.numeric.shape[1]]
        offset = self.numeric.shape[1]
        for j, width in enumerate(self.widths):
            # The last, zero coefficient represents an unseen level (code -1).
            lookup = np.r_[w[offset : offset + width], 0.0]
            result += lookup[self.codes[:, j]]
            offset += width
        return result

    @property
    def T(self):
        return _Transpose(self)

    def dense(self, columns):
        """Materialize only the selected columns needed by normal equations."""
        columns = np.asarray(columns, dtype=int)
        result = np.empty((self.shape[0], len(columns)), dtype=float)
        offsets = np.cumsum([self.numeric.shape[1]] + list(self.widths))
        for k, index in enumerate(columns):
            if index < self.numeric.shape[1]:
                result[:, k] = self.numeric[:, index]
            else:
                j = np.searchsorted(offsets, index, side="right") - 1
                result[:, k] = self.codes[:, j] == index - offsets[j]
        return result


class _Transpose:
    def __init__(self, design):
        self.design = design

    def __matmul__(self, residual):
        x = self.design
        result = np.empty(x.shape[1])
        result[: x.numeric.shape[1]] = x.numeric.T @ residual
        offset = x.numeric.shape[1]
        for j, width in enumerate(x.widths):
            # Bin 0 is the unseen category; discard its gradient.
            result[offset : offset + width] = np.bincount(
                x.codes[:, j].astype(np.int32) + 1,
                weights=residual,
                minlength=width + 1,
            )[1:]
            offset += width
        return result


class FinalPreprocessor:
    """Fit mappings, means, scales and category vocabulary on training rows only."""

    def fit_transform(self, raw, feature_names, rows=None):
        self.feature_names = list(feature_names)
        self.mapping = {name: i for i, name in enumerate(feature_names)}
        self.categories = None
        self.numeric_mean = None
        self.numeric_scale = None
        return self._transform(raw, rows, fit=True)

    def transform(self, raw, rows=None):
        return self._transform(raw, rows, fit=False)

    def _read(self, raw, names, rows):
        idx = [self.mapping[name] for name in names]
        if rows is None:
            return np.asarray(raw[:, idx], dtype=float)
        return np.asarray(raw[np.ix_(rows, idx)], dtype=float)

    def _transform(self, raw, rows, fit):
        one = self._read(raw, shu.one_value_names, rows)
        one[np.isnan(one)] = -1
        ordered = shu.apply_ordered_rules(
            self._read(raw, shu.ordered_names, rows),
            shu.ordered_names,
            shu.ordered_rules,
        )
        decoded = shu.apply_decode_rules(
            self._read(raw, shu.decode_basic_names, rows),
            shu.decode_basic_names,
            shu.decode_rules,
        )
        continuous = np.column_stack([ordered, decoded])
        if fit:
            counts = np.sum(~np.isnan(continuous), axis=0)
            self.all_missing = np.flatnonzero(counts == 0).tolist()
            self.numeric_mean = np.nansum(continuous, axis=0) / np.maximum(counts, 1)
        missing = np.where(np.isnan(continuous))
        continuous[missing] = self.numeric_mean[missing[1]]
        if fit:
            self.numeric_center = continuous.mean(axis=0)
            self.numeric_scale = continuous.std(axis=0)
            self.constant = np.flatnonzero(self.numeric_scale == 0).tolist()
            self.numeric_scale[self.numeric_scale == 0] = 1.0
        continuous -= self.numeric_center
        continuous /= self.numeric_scale
        numeric = np.column_stack([np.ones(len(one)), one, continuous])
        if not np.isfinite(numeric).all():
            raise ValueError("Nonfinite preprocessed numeric values")
        names = (
            ["intercept"]
            + shu.one_value_names
            + shu.ordered_names
            + shu.decode_basic_names
        )
        raw_cats = self._read(raw, shu.categorical_names, rows)
        if fit:
            self.categories = shu.fit_onehot(raw_cats)
        codes = np.full(raw_cats.shape, -1, dtype=np.int16, order="F")
        self.unseen_counts = {}
        widths = []
        for j, (name, levels) in enumerate(zip(shu.categorical_names, self.categories)):
            for k, value in enumerate(levels):
                mask = (
                    np.isnan(raw_cats[:, j])
                    if np.isnan(value)
                    else raw_cats[:, j] == value
                )
                codes[mask, j] = k
                names.append(f"{name}={value:g}")
            self.unseen_counts[name] = int(np.sum(codes[:, j] < 0))
            widths.append(len(levels))
        return EncodedDesign(numeric, codes, widths, names)

    def state(self):
        """JSON-compatible fitted state; NaN categorical levels use null."""
        return {
            "feature_names": self.feature_names,
            "numeric_mean": self.numeric_mean.tolist(),
            "numeric_center": self.numeric_center.tolist(),
            "numeric_scale": self.numeric_scale.tolist(),
            "all_missing": self.all_missing,
            "constant": self.constant,
            "categories": [
                [None if np.isnan(v) else float(v) for v in c] for c in self.categories
            ],
        }

    @classmethod
    def from_state(cls, state):
        obj = cls()
        obj.feature_names = state["feature_names"]
        obj.mapping = {n: i for i, n in enumerate(obj.feature_names)}
        for name in ("numeric_mean", "numeric_center", "numeric_scale"):
            setattr(obj, name, np.asarray(state[name]))
        obj.all_missing = state["all_missing"]
        obj.constant = state["constant"]
        obj.categories = [
            np.array([np.nan if v is None else v for v in c])
            for c in state["categories"]
        ]
        return obj


def fit_selection(design, y, max_iters=2000):
    """Use Shu's exact L1 settings and label conversion, then keep the bias."""
    weights, loss = l1reg_logistic_regression(
        (y == 1).astype(float), design, 0.02, np.zeros(design.shape[1]), max_iters, 0.2
    )
    selected = np.flatnonzero(np.r_[True, np.abs(weights[1:]) > 0.001])
    return selected, weights, loss


def independent_columns(x, tolerance=1e-10):
    """Select a numerical full-rank column basis using pivoted Gram elimination.

    Normalize column norms before pivoting; keep the intercept first. Decisions
    depend on training X only. No pseudoinverse or least-squares solver is used.
    The tolerance applies to squared relative residual norms, so near-dependencies
    may also be removed. Record the discarded span residual instead of assuming
    every discarded column is an exact duplicate.
    """
    gram = x.T @ x
    norms = np.sqrt(np.maximum(np.diag(gram), 0))
    safe = np.where(norms > 0, norms, 1)
    corr = gram / safe[:, None] / safe[None, :]
    residual = np.maximum(np.diag(corr).copy(), 0)
    factors = np.zeros_like(corr)
    keep = []
    for step in range(len(norms)):
        pivot = 0 if step == 0 and residual[0] > tolerance else int(np.argmax(residual))
        if residual[pivot] <= tolerance:
            break
        col = corr[:, pivot] - factors[:, :step] @ factors[pivot, :step]
        factors[:, step] = col / np.sqrt(residual[pivot])
        keep.append(pivot)
        residual = np.maximum(residual - factors[:, step] ** 2, 0)
        residual[keep] = 0
    keep = np.asarray(sorted(keep), dtype=int)
    dropped = np.setdiff1d(np.arange(x.shape[1]), keep)
    normalized = corr[np.ix_(keep, keep)]
    eigenvalues = np.linalg.eigvalsh(normalized)
    if eigenvalues[0] <= 0:
        raise ValueError("Reduced columns are not numerically independent")
    return keep, {
        "tolerance": tolerance,
        "input_columns": x.shape[1],
        "retained_columns": len(keep),
        "dropped_indices": dropped.tolist(),
        "max_discarded_relative_residual": float(
            np.sqrt(max(residual[dropped], default=0))
        ),
        "normalized_condition_number": float(np.sqrt(eigenvalues[-1] / eigenvalues[0])),
    }
