"""Chunked CSV loading and small metric helpers (standard library + NumPy)."""

import hashlib
from itertools import islice
import json
from pathlib import Path

import numpy as np


class NormalEquationDesign:
    """Cache X.T @ X for repeated calls to the unchanged shared algorithms.

    The experiment does not mutate the underlying matrix. This small array-like
    wrapper supplies exactly the dot products used by Least Squares and Ridge;
    it avoids recomputing the expensive Gram matrix for each lambda.
    """

    def __init__(self, matrix, gram=None):
        self.matrix = matrix
        self.shape = matrix.shape
        self._gram = gram

    @property
    def gram(self):
        if self._gram is None:
            self._gram = self.matrix.T @ self.matrix
        return self._gram

    @property
    def T(self):
        return _NormalTranspose(self)

    def dot(self, value):
        return self.matrix.dot(value)

    __matmul__ = dot


class _NormalTranspose:
    def __init__(self, parent):
        self.parent = parent

    def dot(self, value):
        if value is self.parent:
            return self.parent.gram
        return self.parent.matrix.T.dot(value)

    __matmul__ = dot


def encoded_gram(design, columns):
    """Compute the same Gram matrix using category counts for one-hot blocks.

    Numeric/numeric uses BLAS; numeric/category uses weighted bin counts; pairs
    of categories use contingency counts. This avoids multiplying millions of
    explicit zeros on a laptop. Unseen levels contribute zeros, just as in X.
    """
    numeric = np.asfortranarray(design.numeric)
    codes = np.asfortranarray(design.codes, dtype=np.int64) + 1
    width_numeric = numeric.shape[1]
    gram = np.zeros((design.shape[1], design.shape[1]))
    gram[:width_numeric, :width_numeric] = numeric.T @ numeric
    offsets = np.cumsum([width_numeric] + list(design.widths))
    for j, width in enumerate(design.widths):
        left, right = offsets[j], offsets[j + 1]
        for k in range(width_numeric):
            sums = np.bincount(codes[:, j], weights=numeric[:, k], minlength=width + 1)[
                1:
            ]
            gram[k, left:right] = sums
            gram[left:right, k] = sums
        counts = np.bincount(codes[:, j], minlength=width + 1)[1:]
        gram[left:right, left:right] = np.diag(counts)
        for i in range(j):
            joint = codes[:, i] * (width + 1) + codes[:, j]
            counts = np.bincount(joint, minlength=(design.widths[i] + 1) * (width + 1))
            block = counts.reshape(design.widths[i] + 1, width + 1)[1:, 1:]
            a, b = offsets[i], offsets[i + 1]
            gram[a:b, left:right] = block
            gram[left:right, a:b] = block.T
    return gram[np.ix_(columns, columns)]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_features(path, cache_dir):
    """Cache numeric CSV data as a read-only mmap, keyed by its SHA-256."""
    path, cache_dir = Path(path), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256(path)
    cache = cache_dir / f"{path.stem}_{digest[:16]}.npy"
    with path.open() as stream:
        header = stream.readline().strip().split(",")
        if not cache.exists():
            count = sum(1 for _ in stream)
    if not cache.exists():
        temporary = cache.with_suffix(".partial.npy")
        data = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=float, shape=(count, len(header))
        )
        with path.open() as stream:
            next(stream)
            start = 0
            while lines := list(islice(stream, 2000)):
                block = np.genfromtxt(lines, delimiter=",", ndmin=2)
                data[start : start + len(block)] = block
                start += len(block)
        assert start == count
        data.flush()
        del data
        temporary.replace(cache)
    data = np.load(cache, mmap_mode="r")
    return data[:, 1:], data[:, 0].astype(np.int64), header[1:], digest


def load_labels(path, ids):
    labels = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.int64)
    np.testing.assert_array_equal(labels[:, 0], ids)
    if not np.isin(labels[:, 1], [-1, 1]).all():
        raise ValueError("Labels must be -1/+1")
    return labels[:, 1]


def metrics(y, scores, threshold):
    positive = scores >= threshold
    truth = y == 1
    tp = int(np.sum(positive & truth))
    fp = int(np.sum(positive & ~truth))
    fn = int(np.sum(~positive & truth))
    tn = int(np.sum(~positive & ~truth))
    return {
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "half_mse": float(0.5 * np.mean((y - scores) ** 2)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
