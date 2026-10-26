"""Small numerical and data-handling checks; no project data required."""
import csv
from pathlib import Path
import tempfile
import unittest

import run
import numpy as np
from baseline import FEATURE_NAMES, fit_preprocessor, transform_features
from feature_schema import CATEGORICAL_NAMES
from reference_gd import mean_squared_error_gd


class ExperimentChecks(unittest.TestCase):
    def test_split_matches_existing_notebook(self):
        train, val = run.split_indices(20)
        expected = np.arange(20)
        np.random.default_rng(42).shuffle(expected)
        np.testing.assert_array_equal(val, expected[:5])
        np.testing.assert_array_equal(train, expected[5:])
        self.assertFalse(set(train) & set(val))
        self.assertEqual(len(set(train) | set(val)), 20)

    def test_training_only_vocabulary_missing_and_unseen(self):
        raw = np.array([[1., 10.], [2., 20.], [np.nan, 10.], [3., 30.]])
        categories = run.fit_categories(raw, np.array([0, 1, 2]))
        codes = run.category_codes(raw, categories)
        np.testing.assert_array_equal(codes, [[0, 3], [1, 4], [2, 3], [-1, -1]])
        base = np.ones((4, 1))
        matrix = run.design_block(base, codes, 5)
        np.testing.assert_array_equal(matrix[:, 1:].sum(axis=1), [2, 2, 2, 0])
        weights = np.array([.5, 1., 2., 3., 4., 5.])
        np.testing.assert_allclose(run.predict_scores(base, codes, weights, 2), matrix @ weights)
        np.testing.assert_allclose(run.predict_scores(base, codes, weights[:1], 2), [.5] * 4)

    def test_training_only_preprocessing_and_five_corrected_codes(self):
        train = np.ones((4, len(FEATURE_NAMES)))
        age = FEATURE_NAMES.index('_AGE80')
        train[:, age] = [20., 30., 40., 50.]
        entirely_missing = FEATURE_NAMES.index('NUMMEN')
        train[:, entirely_missing] = np.nan
        _, state = fit_preprocessor(train)
        before = {key: value.copy() for key, value in state.items()}
        heldout = np.ones((1, len(FEATURE_NAMES)))
        heldout[0, age] = 10000
        heldout[0, entirely_missing] = 999
        corrections = {'MAXVO2_': 999, 'FC60_': 999,
                       'PAFREQ1_': 99, 'PAFREQ2_': 99, 'STRFREQ_': 99}
        for name, code in corrections.items():
            heldout[0, FEATURE_NAMES.index(name)] = code
        result = transform_features(heldout, state)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result[0, entirely_missing + 1], 0)
        self.assertAlmostEqual(result[0, age + 1], (10000 - 35) / np.std([20, 30, 40, 50]))
        for name in corrections:
            self.assertEqual(result[0, FEATURE_NAMES.index(name) + 1], 0)
        for key in state:
            np.testing.assert_array_equal(state[key], before[key])

    def test_cached_full_batch_gd_matches_original_including_singular_design(self):
        rng = np.random.default_rng(7)
        base = np.column_stack((np.ones(41), rng.normal(size=(41, 3))))
        raw = rng.integers(0, 3, size=(41, 2)).astype(float)
        categories = run.fit_categories(raw, np.arange(len(raw)))
        codes = run.category_codes(raw, categories)
        matrix = run.design_block(base, codes, sum(map(len, categories)))
        y = rng.choice([-1, 1], size=len(base))
        gram, rhs, mean_y_squared = run.sufficient_statistics(base, codes, y, 6, 7)
        np.testing.assert_allclose(gram, matrix.T @ matrix / len(y), atol=1e-14)
        np.testing.assert_allclose(rhs, matrix.T @ y / len(y), atol=1e-14)
        w, at_300, details, curve = run.gd_from_statistics(gram, rhs, mean_y_squared, 300, 1e-12)
        reference, loss = mean_squared_error_gd(y, matrix, np.zeros(matrix.shape[1]), 300, details['gamma'])
        np.testing.assert_allclose(w, reference, atol=1e-12, rtol=1e-10)
        np.testing.assert_array_equal(w, at_300)
        self.assertAlmostEqual(details['training_half_mse'], loss, places=12)
        self.assertTrue(all(b['training_half_mse'] <= a['training_half_mse'] + 1e-12
                            for a, b in zip(curve, curve[1:])))
        self.assertEqual(details['updates'], 300)

    def test_convergence_and_cap_are_distinguished(self):
        _, _, details, _ = run.gd_from_statistics(np.eye(2), np.array([1., 2.]), 5., 400, 1e-4)
        self.assertTrue(details['gradient_tolerance_reached'])
        self.assertEqual(details['updates'], 300)
        _, _, details, _ = run.gd_from_statistics(np.diag([1., 1e-5]), np.array([1., 1.]), 1e6, 300, 1e-6)
        self.assertEqual(details['stopping_reason'], 'iteration_cap_reached')

    def test_threshold_sweep_handles_ties_and_matches_bruteforce(self):
        for seed in range(8):
            rng = np.random.default_rng(seed)
            y = rng.choice([-1, 1], size=100)
            scores = rng.integers(-4, 5, size=100) / 10.
            actual = run.best_validation_threshold(y, scores)
            candidates = [run.score_metrics(y, scores, t) for t in np.unique(scores)[::-1]]
            expected = max(candidates, key=lambda item: item['f1'])
            self.assertEqual(actual, expected)
        self.assertEqual(run.score_metrics(np.array([-1, -1]), np.zeros(2), 1)['f1'], 0)

    def test_csv_loader_reorders_features_and_rejects_misaligned_ids(self):
        names = FEATURE_NAMES + CATEGORICAL_NAMES
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with (path / 'x_train.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['Id'] + names[::-1])
                writer.writerow([10] + list(range(len(names)))[::-1])
                writer.writerow([20] + [1] * len(names))
            (path / 'y_train.csv').write_text('Id,Prediction\n10,1\n20,-1\n')
            x, y, ids = run.read_data(path, 1)
            np.testing.assert_array_equal(x[0], np.arange(len(names)))
            np.testing.assert_array_equal(y, [1, -1])
            np.testing.assert_array_equal(ids, [10, 20])
            (path / 'y_train.csv').write_text('Id,Prediction\n20,1\n10,-1\n')
            with self.assertRaisesRegex(ValueError, 'IDs'):
                run.read_data(path, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
