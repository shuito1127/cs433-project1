"""Evaluate shared Least Squares/Ridge with Shu's final preprocessing.

Run from the repository root: python -m experiments.evaluate_linear
Validation is also used for the explicitly labeled hyperparameter search;
selected scores are model-selection estimates, not untouched test estimates.
"""

import gc
import json
import os
from pathlib import Path
import platform
import time

# Avoid severe multi-threaded MKL solve slowdowns observed on Windows ARM/x64.
# Respect explicit caller settings; notebooks should set these before NumPy loads.
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from experiments.linear_io import (
    NormalEquationDesign,
    encoded_gram,
    load_features,
    load_labels,
    metrics,
    save_json,
    sha256,
)
from implementations import least_squares, ridge_regression
from preprocessing.final_pipeline import (
    FinalPreprocessor,
    fit_selection,
    independent_columns,
)
from preprocessing import shu_reference as shu

ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
THRESHOLDS = np.round(np.linspace(-0.9, 0.0, 91), 2)


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def prepare_selected(raw, names, y, rows, cache_dir, signature):
    """Fit once per exact dataset/code/split; checkpoint only after completion."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "selection.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["signature"] == signature:
            log("Reusing matching fitted preprocessing and L1 selection checkpoint")
            preprocessor = FinalPreprocessor.from_state(metadata["preprocessor"])
            design = preprocessor.transform(raw, rows)
            columns = np.array(metadata["selected_indices"])
            metadata["checkpoint_reused"] = True
            return finish_selection(preprocessor, design, columns, metadata, cache_dir)
    log("Fitting numeric preprocessing and categorical vocabulary")
    preprocessor = FinalPreprocessor()
    design = preprocessor.fit_transform(raw, names, rows)
    log(f"L1 selection: {design.shape}, 2000 iterations, lambda=0.02, gamma=0.2")
    started = time.perf_counter()
    columns, weights, loss = fit_selection(design, y)
    log(
        f"L1 finished in {time.perf_counter() - started:.1f}s; selected {len(columns)} columns"
    )
    metadata = {
        "signature": signature,
        "preprocessor": preprocessor.state(),
        "expanded_columns": design.shape[1],
        "selected_indices": columns.tolist(),
        "selected_names": [design.names[i] for i in columns],
        "selector_weights": weights.tolist(),
        "selector_loss": float(loss),
        "checkpoint_reused": False,
        "selector_seconds": time.perf_counter() - started,
        "selector_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
    }
    save_json(metadata_path, metadata)
    return finish_selection(preprocessor, design, columns, metadata, cache_dir)


def finish_selection(preprocessor, design, columns, metadata, cache_dir):
    gram_path = cache_dir / "gram.npy"
    manifest_path = cache_dir / "gram_manifest.json"
    manifest = {
        "signature": metadata["signature"],
        "columns": columns.tolist(),
        "algorithm": "encoded_counts_v1",
    }
    if not (
        gram_path.exists()
        and manifest_path.exists()
        and json.loads(manifest_path.read_text()) == manifest
    ):
        log("Computing normal-equation statistics using category counts")
        np.save(gram_path, encoded_gram(design, columns))
        save_json(manifest_path, manifest)
    metadata["gram_path"] = str(gram_path)
    return preprocessor, design.dense(columns), metadata


def code_signature():
    return {
        name: sha256(ROOT / name)
        for name in [
            "implementations.py",
            "preprocessing/shu_reference.py",
            "preprocessing/final_pipeline.py",
        ]
    }


def evaluate():
    start = time.perf_counter()
    log("Loading all labeled observations (chunked CSV / cached mmap)")
    raw, ids, names, digest = load_features(
        ROOT / "dataset/x_train.csv", ROOT / "artifacts/cache"
    )
    y = load_labels(ROOT / "dataset/y_train.csv", ids)
    rng = np.random.default_rng(42)
    permutation = rng.permutation(len(y))
    n_val = int(len(y) * 0.25)
    val_rows, train_rows = permutation[:n_val], permutation[n_val:]
    signature = {
        "data": digest,
        "labels": sha256(ROOT / "dataset/y_train.csv"),
        "seed": 42,
        "validation_ratio": 0.25,
        "code": code_signature(),
    }
    preprocessor, train, selection = prepare_selected(
        raw, names, y[train_rows], train_rows, ROOT / "artifacts/validation", signature
    )
    validation_design = preprocessor.transform(raw, val_rows)
    validation = validation_design.dense(selection["selected_indices"])
    unseen = preprocessor.unseen_counts.copy()
    del validation_design, raw
    gc.collect()
    log("Checking rank and fitting the original selected-input Ridge reference")
    design = NormalEquationDesign(train, np.load(selection["gram_path"]))
    keep, rank = independent_columns(design)
    rank["dropped_names"] = [
        selection["selected_names"][i] for i in rank["dropped_indices"]
    ]
    rank["retained_names"] = [selection["selected_names"][i] for i in keep]
    w_original, loss_original = ridge_regression(y[train_rows], design, 0.001)
    original = {
        "name": "ridge_selected_original",
        "lambda": 0.001,
        "threshold": -0.55,
        "columns": train.shape[1],
        "training_loss": float(loss_original),
        "validation": metrics(y[val_rows], validation @ w_original, -0.55),
    }
    gram = design.gram[np.ix_(keep, keep)]
    del design
    train = np.take(train, keep, axis=1)
    validation = np.take(validation, keep, axis=1)
    design = NormalEquationDesign(train, gram)
    log(f"Matched full-rank inputs: {train.shape[1]} columns")
    y_train, y_val = y[train_rows], y[val_rows]
    records = []
    for method, lambda_ in [("least_squares", None)] + [
        ("ridge", value) for value in LAMBDAS
    ]:
        log(f"Fitting {method}, lambda={lambda_}")
        t0 = time.perf_counter()
        if method == "least_squares":
            w, loss = least_squares(y_train, design)
        else:
            w, loss = ridge_regression(y_train, design, lambda_)
        train_scores, val_scores = train @ w, validation @ w
        assert np.isfinite(w).all()
        np.testing.assert_allclose(
            loss, 0.5 * np.mean((y_train - train_scores) ** 2), rtol=1e-12
        )
        gradient = train.T @ (train_scores - y_train) / len(y_train)
        if lambda_ is not None:
            gradient += 2 * lambda_ * w
        assert np.max(np.abs(gradient)) < 1e-7, "Normal-equation residual too large"
        sweep = [
            {"threshold": float(t), **metrics(y_val, val_scores, t)} for t in THRESHOLDS
        ]
        # Deterministic tie rule: F1, then accuracy, then higher threshold.
        chosen = max(
            sweep, key=lambda row: (row["f1"], row["accuracy"], row["threshold"])
        )
        record = {
            "method": method,
            "lambda": lambda_,
            "columns": train.shape[1],
            "seconds": time.perf_counter() - t0,
            "training_loss": float(loss),
            "max_normal_equation_residual": float(np.max(np.abs(gradient))),
            "fixed_threshold": -0.55,
            "fixed_train": metrics(y_train, train_scores, -0.55),
            "fixed_validation": metrics(y_val, val_scores, -0.55),
            "selected_threshold": chosen["threshold"],
            "selected_validation": chosen,
            "selected_train": metrics(y_train, train_scores, chosen["threshold"]),
            "threshold_sweep": sweep,
        }
        records.append(record)
        log(
            f"Fixed F1={record['fixed_validation']['f1']:.6f}; tuned F1={chosen['f1']:.6f}"
        )
    best = {}
    for method in ("least_squares", "ridge"):
        options = [r for r in records if r["method"] == method]
        row = max(
            options,
            key=lambda r: (
                r["selected_validation"]["f1"],
                r["selected_validation"]["accuracy"],
            ),
        )
        best[method] = {
            "lambda": row["lambda"],
            "threshold": row["selected_threshold"],
            "validation": row["selected_validation"],
            "training": row["selected_train"],
        }
    result = {
        "source": json.loads((ROOT / "preprocessing/shu_source.json").read_text()),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "selector_environment": selection.get("selector_environment"),
        },
        "signature": signature,
        "evaluation_code_sha256": sha256(Path(__file__)),
        "linear_io_sha256": sha256(ROOT / "experiments/linear_io.py"),
        "split": {
            "seed": 42,
            "stratified": False,
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
            "train_positives": int(np.sum(y_train == 1)),
            "validation_positives": int(np.sum(y_val == 1)),
            "train_ids_sha256": __import__("hashlib")
            .sha256(ids[train_rows].tobytes())
            .hexdigest(),
            "validation_ids_sha256": __import__("hashlib")
            .sha256(ids[val_rows].tobytes())
            .hexdigest(),
        },
        "preprocessing": {
            "one_value": len(shu.one_value_names),
            "ordered": len(shu.ordered_names),
            "decoded": shu.decode_basic_names,
            "categorical": len(shu.categorical_names),
            "decode_not_used": [
                n for n in shu.decode_ordered_names if n not in shu.decode_basic_names
            ],
            "dropped_raw": shu.drop_candidate_names,
            "expanded_columns": selection["expanded_columns"],
            "l1_selected_columns": len(selection["selected_indices"]),
            "l1_settings": {
                "lambda": 0.02,
                "gamma": 0.2,
                "iterations": 2000,
                "cutoff": 0.001,
            },
            "all_missing_numeric": preprocessor.all_missing,
            "constant_numeric": preprocessor.constant,
            "unseen_validation_categories": unseen,
        },
        "rank": rank,
        "ridge_original_selected": original,
        "always_negative": metrics(y_val, -np.ones(len(y_val)), 0.0),
        "models": records,
        "selected_models": best,
        "limitations": [
            "One shuffled holdout, not cross-validation.",
            "Threshold and lambda selection use validation labels; selected metrics are not independent test scores.",
            "Shu's preprocessing code is reproduced, not a new medical codebook audit.",
            "Removing columns changes Ridge's penalty representation; original-input Ridge is reported separately.",
            "No AIcrowd score was obtained in this run.",
        ],
        "seconds": time.perf_counter() - start,
        "selector_training_seconds": selection["selector_seconds"],
        "selection_checkpoint_reused": selection["checkpoint_reused"],
    }
    save_json(ROOT / "experiments/yusen_final_results.json", result)
    log("Saved experiments/yusen_final_results.json")
    return result


if __name__ == "__main__":
    evaluate()
