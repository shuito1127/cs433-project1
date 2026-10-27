"""Regression checks for the adapter, selection math and course interfaces."""

import numpy as np

from implementations import l1reg_logistic_regression, least_squares, ridge_regression
from preprocessing.final_pipeline import (
    EncodedDesign,
    FinalPreprocessor,
    independent_columns,
)
from preprocessing import shu_reference as shu
from experiments.linear_io import NormalEquationDesign, encoded_gram


def test_cached_normal_equations_match_shared_dense_models():
    rng = np.random.default_rng(13)
    matrix = rng.normal(size=(50, 8))
    y = rng.normal(size=50)
    cached = NormalEquationDesign(matrix)
    for method, args in [
        (least_squares, ()),
        (ridge_regression, (0.0,)),
        (ridge_regression, (0.1,)),
    ]:
        actual_w, actual_loss = method(y, cached, *args)
        dense_w, dense_loss = method(y, matrix, *args)
        np.testing.assert_allclose(actual_w, dense_w, atol=1e-12)
        np.testing.assert_allclose(actual_loss, dense_loss, atol=1e-12)
    keep1, _ = independent_columns(cached)
    keep2, _ = independent_columns(matrix)
    np.testing.assert_array_equal(keep1, keep2)


def test_compressed_design_matches_dense_selection():
    rng = np.random.default_rng(7)
    numeric = np.column_stack([np.ones(30), rng.normal(size=(30, 3))])
    codes = rng.integers(-1, 3, size=(30, 2))
    design = EncodedDesign(numeric, codes, [3, 3], list(range(10)))
    dense = design.dense(np.arange(10))
    np.testing.assert_allclose(
        encoded_gram(design, np.arange(10)), dense.T @ dense, atol=1e-12
    )
    w, residual = rng.normal(size=10), rng.normal(size=30)
    np.testing.assert_allclose(design @ w, dense @ w, atol=1e-12)
    np.testing.assert_allclose(design.T @ residual, dense.T @ residual, atol=1e-12)
    y = rng.integers(0, 2, size=30)
    actual = l1reg_logistic_regression(y, design, 0.02, np.zeros(10), 2000, 0.2)
    expected = l1reg_logistic_regression(y, dense, 0.02, np.zeros(10), 2000, 0.2)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-12)


def test_adapter_matches_shu_and_does_not_refit_validation():
    names = (
        shu.one_value_names
        + shu.ordered_names
        + shu.decode_basic_names
        + shu.categorical_names
    )
    rng = np.random.default_rng(7)
    raw = rng.integers(1, 5, size=(30, len(names))).astype(float)
    mapping = {n: i for i, n in enumerate(names)}
    for name in shu.decode_basic_names:
        raw[:, mapping[name]] = rng.choice([101, 102, 201, 202, np.nan], 30)
    prep = FinalPreprocessor()
    design = prep.fit_transform(raw, names)
    ordered = shu.apply_ordered_rules(
        raw[:, [mapping[n] for n in shu.ordered_names]],
        shu.ordered_names,
        shu.ordered_rules,
    )
    decoded = shu.apply_decode_rules(
        raw[:, [mapping[n] for n in shu.decode_basic_names]],
        shu.decode_basic_names,
        shu.decode_rules,
    )
    for values in [ordered, decoded]:
        values[:], _ = shu.fit_mean_imputation(values)
        values[:], _, _ = shu.fit_standardize(values)
    cats = raw[:, [mapping[n] for n in shu.categorical_names]]
    dense = np.column_stack(
        [
            np.ones(30),
            raw[:, : len(shu.one_value_names)],
            ordered,
            decoded,
            shu.transform_onehot(cats, shu.fit_onehot(cats)),
        ]
    )
    np.testing.assert_allclose(
        design.dense(np.arange(design.shape[1])), dense, atol=1e-12
    )
    import json

    before = json.dumps(prep.state(), sort_keys=True)
    changed = raw.copy()
    changed[:, mapping[shu.categorical_names[0]]] = 12345
    transformed = prep.transform(changed)
    assert json.dumps(prep.state(), sort_keys=True) == before
    assert (transformed.codes[:, 0] == -1).all()
    loaded = FinalPreprocessor.from_state(prep.state()).transform(changed)
    np.testing.assert_array_equal(loaded.numeric, transformed.numeric)
    np.testing.assert_array_equal(loaded.codes, transformed.codes)


def test_rank_reduction_preserves_predictions():
    x = np.column_stack([np.ones(20), np.linspace(-1, 1, 20)])
    redundant = np.column_stack([x, 2 * x[:, 1], np.zeros(20)])
    keep, info = independent_columns(redundant)
    assert len(keep) == 2
    assert info["max_discarded_relative_residual"] < 1e-6
    y = 1 + 3 * x[:, 1]
    w, loss = least_squares(y, redundant[:, keep])
    np.testing.assert_allclose(redundant[:, keep] @ w, y, atol=1e-12)
    assert loss < 1e-24
    w0, loss0 = ridge_regression(y, redundant[:, keep], 0)
    np.testing.assert_allclose(w0, w)
    np.testing.assert_allclose(loss0, loss)
