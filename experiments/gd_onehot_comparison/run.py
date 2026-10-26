"""One controlled experiment: Anthony's GD baseline versus added one-hot fields.

Run this file directly. Only Python 3.9+ and NumPy are required.
"""
import os

# Keep laptop resource use bounded; respect existing thread settings.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "4")

import argparse
import csv
import hashlib
from itertools import islice
import json
from pathlib import Path
import platform
import time

import numpy as np

from baseline import FEATURE_NAMES, fit_preprocessor, transform_features
from feature_schema import CATEGORICAL_NAMES

HERE = Path(__file__).resolve().parent
FIXED_THRESHOLD = -0.55


def say(message):
    print(message, flush=True)


def split_indices(n, seed=42, validation_fraction=0.25):
    """Same row order and split procedure as the existing notebook."""
    indices = np.arange(n)
    np.random.default_rng(seed).shuffle(indices)
    n_val = int(n * validation_fraction)
    return indices[n_val:], indices[:n_val]


def read_data(data_dir, chunk_size):
    """Read selected raw columns, checking row IDs against the label file."""
    labels = np.loadtxt(data_dir / "y_train.csv", delimiter=",", skiprows=1, ndmin=2)
    if labels.shape[1] != 2 or not np.isin(labels[:, 1], [-1, 1]).all():
        raise ValueError("y_train.csv must have ID and -1/+1 label columns.")
    if not np.isfinite(labels).all() or len(np.unique(labels[:, 0])) != len(labels):
        raise ValueError("Label IDs must be finite and unique.")
    names = FEATURE_NAMES + CATEGORICAL_NAMES
    if len(set(names)) != len(names):
        raise ValueError("The feature groups overlap.")
    raw = np.empty((len(labels), len(names)), dtype=np.float64)
    start = 0
    with (data_dir / "x_train.csv").open() as handle:
        header = next(csv.reader([next(handle)]))
        missing = [name for name in names if name not in header]
        if missing:
            raise ValueError("Missing CSV fields: " + ", ".join(missing))
        if len(set(header)) != len(header):
            raise ValueError("Duplicate CSV column names.")
        columns = [0] + [header.index(name) for name in names]
        while True:
            lines = list(islice(handle, chunk_size))
            if not lines:
                break
            block = np.genfromtxt(lines, delimiter=",", usecols=columns, ndmin=2)
            end = start + len(block)
            if end > len(labels) or not np.array_equal(block[:, 0], labels[start:end, 0]):
                raise ValueError("Training feature IDs and label IDs do not match in order.")
            if np.isinf(block[:, 1:]).any():
                raise ValueError("Infinite raw feature value.")
            raw[start:end] = block[:, 1:]
            start = end
            if start % (chunk_size * 20) == 0:
                say(f"  Read {start:,}/{len(labels):,} rows")
    if start != len(labels):
        raise ValueError("Different numbers of feature and label rows.")
    return raw, labels[:, 1].astype(int), labels[:, 0]


def fit_categories(raw_categories, training_indices):
    """Missing is a category; vocabulary is learned from training rows only."""
    return [np.unique(raw_categories[training_indices, j])
            for j in range(raw_categories.shape[1])]


def category_codes(raw_categories, categories):
    """Compact one-hot column indices. -1 means unseen, encoded as all zeros."""
    codes = np.empty(raw_categories.shape, dtype=np.int32)
    offset = 0
    for j, values in enumerate(categories):
        column = raw_categories[:, j]
        positions = np.searchsorted(values, column)
        safe = np.minimum(positions, len(values) - 1)
        matches = (values[safe] == column) | (np.isnan(values[safe]) & np.isnan(column))
        codes[:, j] = np.where((positions < len(values)) & matches, safe + offset, -1)
        offset += len(values)
    return codes


def design_block(base, codes, one_hot_width):
    matrix = np.zeros((len(base), base.shape[1] + one_hot_width))
    matrix[:, :base.shape[1]] = base
    rows = np.repeat(np.arange(len(base)), codes.shape[1])
    columns = codes.ravel()
    valid = columns >= 0
    matrix[rows[valid], base.shape[1] + columns[valid]] = 1.0
    return matrix


def sufficient_statistics(base, codes, y, one_hot_width, chunk_size):
    """Compute full-data GD statistics without keeping the big one-hot matrix."""
    d = base.shape[1] + one_hot_width
    gram = np.zeros((d, d))
    rhs = np.zeros(d)
    for start in range(0, len(y), chunk_size):
        end = min(start + chunk_size, len(y))
        matrix = design_block(base[start:end], codes[start:end], one_hot_width)
        gram += matrix.T @ matrix
        rhs += matrix.T @ y[start:end]
        if start // chunk_size % 12 == 0 or end == len(y):
            say(f"  Prepared training statistics: {end:,}/{len(y):,} rows")
    gram /= len(y)
    rhs /= len(y)
    return gram, rhs, float(np.mean(y.astype(float) ** 2))


def gd_from_statistics(gram, rhs, mean_y_squared, max_iters, gradient_rtol, label="GD"):
    """Same full-batch GD as reference_gd.py, with X.T@X and X.T@y cached.

    This is still iterative GD, with no direct least-squares solve or penalty.
    Stop using training gradients only, after at least the 300-step checkpoint.
    """
    largest = float(np.linalg.eigvalsh(gram)[-1])
    if not np.isfinite(largest) or largest <= 0:
        raise ValueError("Expected a finite positive largest Gram eigenvalue.")
    gamma = 1.0 / largest
    w = np.zeros(len(rhs))
    initial_norm = float(np.linalg.norm(rhs))
    tolerance = max(1e-10, gradient_rtol * initial_norm)
    curve = []
    checkpoint = None
    previous_loss = float("inf")
    start_time = time.perf_counter()
    for step in range(max_iters + 1):
        gradient = gram @ w - rhs
        if step % 100 == 0 or step == max_iters:
            loss = float(0.5 * (w @ (gradient - rhs) + mean_y_squared))
            norm = float(np.linalg.norm(gradient))
            if not np.isfinite(w).all() or not np.isfinite(loss):
                raise FloatingPointError("GD produced non-finite weights or loss.")
            if loss > previous_loss + 1e-10:
                raise FloatingPointError("Training loss increased despite the conservative step.")
            curve.append({"step": step, "training_half_mse": loss, "gradient_norm": norm})
            previous_loss = loss
            if step == 300:
                checkpoint = w.copy()
            if step >= 300 and norm <= tolerance:
                reason = "gradient_tolerance_reached"
                break
            if step == max_iters:
                reason = "iteration_cap_reached"
                break
            if step and step % 10000 == 0:
                say(f"  {label}: {step:,} steps; training half-MSE {loss:.6f}")
        w -= gamma * gradient
    details = {
        "updates": step, "gamma": gamma, "training_half_mse": curve[-1]["training_half_mse"],
        "gradient_norm": curve[-1]["gradient_norm"], "initial_gradient_norm": initial_norm,
        "gradient_tolerance": tolerance, "stopping_reason": reason,
        "gradient_tolerance_reached": reason == "gradient_tolerance_reached",
        "seconds_in_gd_loop": time.perf_counter() - start_time,
    }
    return w, checkpoint, details, curve


def predict_scores(base, codes, w, chunk_size):
    scores = base @ w[:base.shape[1]]
    if len(w) == base.shape[1]:
        return scores
    categorical_w = w[base.shape[1]:]
    for start in range(0, len(base), chunk_size):
        selected = codes[start:start + chunk_size]
        terms = categorical_w[np.maximum(selected, 0)]
        terms[selected < 0] = 0
        scores[start:start + len(selected)] += terms.sum(axis=1)
    return scores


def score_metrics(y, scores, threshold):
    predictions = scores >= threshold
    positive = y == 1
    tp = int(np.sum(predictions & positive))
    fp = int(np.sum(predictions & ~positive))
    fn = int(np.sum(~predictions & positive))
    return {
        "threshold": float(threshold), "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        "precision": tp / (tp + fp) if tp + fp else 0.,
        "recall": tp / (tp + fn) if tp + fn else 0.,
        "accuracy": float(np.mean(predictions == positive)), "tp": tp, "fp": fp, "fn": fn,
    }


def best_validation_threshold(y, scores):
    """Exact F1 sweep over distinct scores, treating tied scores as one group.

    This selects on validation labels. Its F1 is a tuning result, not a fresh
    test-set estimate. In a tie, choose the highest threshold.
    """
    if len(y) != len(scores) or not len(y) or not np.isfinite(scores).all():
        raise ValueError("Threshold tuning requires aligned, nonempty finite scores.")
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    group_ends = np.flatnonzero(np.r_[ordered[1:] != ordered[:-1], True])
    tp = np.cumsum(y[order] == 1)[group_ends]
    predicted = group_ends + 1
    f1 = 2 * tp / (int(np.sum(y == 1)) + predicted)
    threshold = ordered[group_ends[int(np.argmax(f1))]]
    return score_metrics(y, scores, threshold)


def array_hash(array):
    return hashlib.sha256(np.asarray(array, dtype="<i8").tobytes()).hexdigest()


def describe_features(categories):
    return ["intercept"] + FEATURE_NAMES + [
        name + "=" + ("missing" if np.isnan(value) else str(float(value)))
        for name, values in zip(CATEGORICAL_NAMES, categories) for value in values
    ]


def markdown_summary(result):
    lines = ["# GD: smaller baseline versus added one-hot features", "",
             "Local validation only. The split, labels, existing preprocessing, and fixed threshold are shared.",
             "", "## Fixed threshold (-0.55): feature and training comparison", "",
             "| Features | Updates | Validation F1 | Validation half-MSE | Stop |",
             "|---|---:|---:|---:|---|"]
    for name, model in result["models"].items():
        for stage, row in model["evaluations"].items():
            stop = "300-step reference" if stage == "at_300_steps" else model["training"]["stopping_reason"]
            lines.append(f"| {name} | {row['updates']:,} | {row['fixed_threshold']['f1']:.6f} | "
                         f"{row['validation_half_mse']:.6f} | {stop} |")
    lines += ["", "## Threshold tuning: same final weights, different decision threshold", "",
              "| Features | F1 at -0.55 | Selected threshold | Tuned validation F1 |",
              "|---|---:|---:|---:|"]
    for name, model in result["models"].items():
        final = model["evaluations"]["final"]
        tuned = model["threshold_tuning"]
        lines.append(f"| {name} | {final['fixed_threshold']['f1']:.6f} | {tuned['threshold']:.6f} | {tuned['f1']:.6f} |")
    lines += ["", "The tuned thresholds use this validation set's labels. Their scores are optimistic after selection; "
              "they are not independent test results or AIcrowd scores.", "",
              "## Training checks", ""]
    for name, model in result["models"].items():
        train = model["training"]
        lines.append(f"- {name}: {model['model_columns']} model columns; gamma {train['gamma']:.9g}; "
                     f"final gradient norm {train['gradient_norm']:.3g}, target {train['gradient_tolerance']:.3g}. "
                     f"{train['stopping_reason']}.")
    lines += ["", "An iteration cap means the requested gradient tolerance was NOT reached. "
              "Inspect training_curve.csv; the script does not silently label capped runs converged.", "",
              f"Train/validation rows: {result['split']['training_rows']:,}/{result['split']['validation_rows']:,}; "
              f"seed {result['split']['seed']}; total runtime {result['runtime_seconds']:.1f} seconds.",
              f"Validation rows with an unseen category: {result['one_hot']['validation_rows_with_unseen_category']}. "
              "An unseen field contributes all zeros to its one-hot block.", "",
              "See metrics.json for full settings, metrics, source fingerprints and split fingerprints.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE.parent.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--max-iters", type=int, default=100000)
    parser.add_argument("--gradient-rtol", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()
    if args.max_iters < 300 or args.chunk_size < 1 or not 0 < args.gradient_rtol < 1:
        parser.error("Use max-iters >= 300, chunk-size >= 1 and 0 < gradient-rtol < 1.")
    data_dir = args.data_dir.expanduser().resolve()
    for name in ("x_train.csv", "y_train.csv"):
        if not (data_dir / name).is_file():
            parser.error(f"Cannot find {data_dir / name}. Pass --data-dir /path/to/data.")
    start_time = time.perf_counter()
    say("1/5 Reading local training data (the competition test data is not used).")
    raw, y, ids = read_data(data_dir, args.chunk_size)
    train_idx, val_idx = split_indices(len(y))
    if not len(train_idx) or not len(val_idx):
        raise ValueError("Need enough rows for a nonempty training/validation split.")
    y_train, y_val = y[train_idx], y[val_idx]
    say("2/5 Fitting preprocessing on training rows only.")
    base_width = len(FEATURE_NAMES)
    base_train, state = fit_preprocessor(raw[train_idx, :base_width])
    base_val = transform_features(raw[val_idx, :base_width], state)
    categories = fit_categories(raw[:, base_width:], train_idx)
    codes = category_codes(raw[:, base_width:], categories)
    train_codes, val_codes = codes[train_idx], codes[val_idx]
    if (train_codes < 0).any():
        raise AssertionError("Training category not in training vocabulary.")
    del raw, codes
    width = sum(map(len, categories))
    say(f"  {base_train.shape[1]} baseline columns; {width} added one-hot columns; "
        f"{base_train.shape[1] + width} expanded columns.")
    say("3/5 Computing full-training-data GD statistics in chunks.")
    gram, rhs, mean_y_squared = sufficient_statistics(
        base_train, train_codes, y_train, width, args.chunk_size)
    del base_train, train_codes
    result = {
        "experiment": "GD: corrected smaller baseline vs added Shu one-hot fields",
        "python": platform.python_version(), "numpy": np.__version__,
        "split": {"seed": 42, "validation_fraction": .25, "stratified": False,
                  "training_rows": len(y_train), "validation_rows": len(y_val),
                  "training_positive_rows": int(np.sum(y_train == 1)),
                  "validation_positive_rows": int(np.sum(y_val == 1)),
                  "training_id_order_sha256": array_hash(ids[train_idx]),
                  "validation_id_order_sha256": array_hash(ids[val_idx])},
        "settings": {"initial_weights": "zeros", "labels": [-1, 1],
                     "max_iters": args.max_iters, "gradient_rtol": args.gradient_rtol,
                     "minimum_iters": 300, "gradient_check_every": 100,
                     "gamma_rule": "1 / largest eigenvalue of training X.T @ X / n",
                     "fixed_threshold": FIXED_THRESHOLD, "positive_rule": "score >= threshold",
                     "regularization": None, "decoded_extra_features": False,
                     "chunk_size": args.chunk_size},
        "one_hot": {"raw_fields": len(categories), "encoded_columns": width,
                    "training_only_vocabulary": True, "missing_is_own_category": True,
                    "validation_rows_with_unseen_category": int(np.sum(np.any(val_codes < 0, axis=1))),
                    "unseen_by_field": {name: int(np.sum(val_codes[:, j] < 0))
                                        for j, name in enumerate(CATEGORICAL_NAMES) if np.any(val_codes[:, j] < 0)}},
        "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                          for name in ("run.py", "baseline.py", "feature_schema.py", "reference_gd.py")},
        "models": {},
    }
    say("4/5 Training both models; then evaluating the same held-out rows.")
    curves = []
    saved_weights = {}
    for label, d in (("smaller_baseline", base_width + 1), ("added_one_hot", len(rhs))):
        # Contiguous baseline submatrix avoids slow strided BLAS calls.
        model_gram = np.ascontiguousarray(gram[:d, :d])
        w, at_300, training, curve = gd_from_statistics(
            model_gram, rhs[:d], mean_y_squared, args.max_iters, args.gradient_rtol, label)
        model = {"model_columns": d, "training": training, "evaluations": {}}
        for stage, weights, steps in (("at_300_steps", at_300, 300), ("final", w, training["updates"])):
            scores = predict_scores(base_val, val_codes, weights, args.chunk_size)
            model["evaluations"][stage] = {
                "updates": steps, "validation_half_mse": float(.5 * np.mean((scores - y_val) ** 2)),
                "fixed_threshold": score_metrics(y_val, scores, FIXED_THRESHOLD),
                "zero_threshold": score_metrics(y_val, scores, 0.),
            }
        model["threshold_tuning"] = best_validation_threshold(y_val, scores)
        result["models"][label] = model
        saved_weights[label] = w
        saved_weights[label + "_at_300"] = at_300
        curves.extend({"model": label, **row} for row in curve)
        say(f"  {label}: fixed-threshold F1 {model['evaluations']['final']['fixed_threshold']['f1']:.4f}; "
            f"tuned validation F1 {model['threshold_tuning']['f1']:.4f}; {training['stopping_reason']}")
    result["runtime_seconds"] = time.perf_counter() - start_time
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    say("5/5 Saving results.")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output_dir / "summary.md").write_text(markdown_summary(result))
    (output_dir / "feature_names.json").write_text(json.dumps(describe_features(categories), indent=2) + "\n")
    # Include preprocessing state and vocabulary so weight order remains interpretable.
    np.savez_compressed(output_dir / "model_state.npz", **saved_weights,
                        **{"baseline_" + key: value for key, value in state.items()},
                        **{"category_" + str(j): values for j, values in enumerate(categories)})
    with (output_dir / "training_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "step", "training_half_mse", "gradient_norm"])
        writer.writeheader()
        writer.writerows(curves)
    say("\n" + markdown_summary(result))
    say(f"Results: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
