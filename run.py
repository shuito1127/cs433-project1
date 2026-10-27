"""Reproduce Yusen's linear-model candidates with Shu's preprocessing.

This entry point does not reproduce Shu's separately scored logistic submission.
Validation and model comparison stay in experiments/.
"""

import argparse
import gc
import json
import os
from pathlib import Path

# Set before importing NumPy; an explicit caller setting takes precedence.
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from experiments.evaluate_linear import code_signature, log, prepare_selected
from experiments.linear_io import (
    NormalEquationDesign,
    load_features,
    load_labels,
    save_json,
    sha256,
)
from helpers import create_csv_submission
from implementations import least_squares, ridge_regression
from preprocessing.final_pipeline import FinalPreprocessor, independent_columns

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["least_squares", "ridge", "both"], default="ridge"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Refit weights; reuse matching selection cache.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Require a matching trained artifact.",
    )
    args = parser.parse_args()
    results_path = ROOT / "experiments/yusen_final_results.json"
    if not results_path.exists():
        parser.error("Run python -m experiments.evaluate_linear first.")
    selected = json.loads(results_path.read_text())["selected_models"]
    raw, ids, names, digest = load_features(
        ROOT / "dataset/x_train.csv", ROOT / "artifacts/cache"
    )
    y = load_labels(ROOT / "dataset/y_train.csv", ids)
    signature = {
        "data": digest,
        "labels": sha256(ROOT / "dataset/y_train.csv"),
        "code": code_signature(),
        "fit": "all_labeled_rows",
    }
    model_path = ROOT / "models/yusen_linear_models.json"
    artifact = json.loads(model_path.read_text()) if model_path.exists() else None
    matches = (
        artifact is not None
        and artifact["signature"] == signature
        and artifact["settings"] == selected
    )
    if args.predict_only and (not matches or args.retrain):
        parser.error("A matching trained artifact is required for --predict-only.")
    if args.retrain or not matches:
        log("Refitting preprocessing and L1 selection on all labeled observations")
        prep, matrix, selection = prepare_selected(
            raw, names, y, None, ROOT / "artifacts/full_train", signature
        )
        design = NormalEquationDesign(matrix, np.load(selection["gram_path"]))
        keep, rank = independent_columns(design)
        gram = design.gram[np.ix_(keep, keep)]
        del design
        matrix = np.take(matrix, keep, axis=1)
        design = NormalEquationDesign(matrix, gram)
        models = {}
        for method in ("least_squares", "ridge"):
            w, loss = (
                least_squares(y, design)
                if method == "least_squares"
                else ridge_regression(y, design, selected[method]["lambda"])
            )
            if not np.isfinite(w).all():
                raise ValueError("Nonfinite final weights")
            models[method] = {
                "weights": w.tolist(),
                "training_loss": float(loss),
                "threshold": selected[method]["threshold"],
                "lambda": selected[method]["lambda"],
            }
        artifact = {
            "signature": signature,
            "settings": selected,
            "preprocessor": prep.state(),
            "expanded_columns": selection["expanded_columns"],
            "selected_columns": len(selection["selected_indices"]),
            "model_columns": matrix.shape[1],
            "rank": rank,
            "columns": np.asarray(selection["selected_indices"])[keep].tolist(),
            "models": models,
        }
        save_json(model_path, artifact)
        del matrix, design
    del raw
    gc.collect()
    prep = FinalPreprocessor.from_state(artifact["preprocessor"])
    raw_test, test_ids, test_names, test_digest = load_features(
        ROOT / "dataset/x_test.csv", ROOT / "artifacts/cache"
    )
    if test_names != names:
        raise ValueError("Training/test feature headers differ")
    methods = ["least_squares", "ridge"] if args.model == "both" else [args.model]
    scores = {method: np.empty(len(test_ids)) for method in methods}
    log(f"Predicting {len(test_ids)} test rows")
    for start in range(0, len(test_ids), 8192):
        stop = min(start + 8192, len(test_ids))
        design = prep.transform(raw_test, np.arange(start, stop))
        matrix = design.dense(artifact["columns"])
        for method in methods:
            scores[method][start:stop] = matrix @ np.asarray(
                artifact["models"][method]["weights"]
            )
    manifest_path = ROOT / "experiments/yusen_submissions.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    outputs = (
        existing.get("outputs", {})
        if existing.get("training_signature") == signature
        else {}
    )
    for method in methods:
        if not np.isfinite(scores[method]).all():
            raise ValueError("Nonfinite predictions")
        labels = np.where(
            scores[method] >= artifact["models"][method]["threshold"], 1, -1
        )
        output = ROOT / f"artifacts/submissions/yusen_{method}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        create_csv_submission(test_ids, labels, str(output))
        # Reopen the actual CSV and verify its IDs, labels and row count.
        with output.open() as stream:
            assert stream.readline().strip() == "Id,Prediction"
        saved = np.loadtxt(output, delimiter=",", skiprows=1, dtype=np.int64)
        np.testing.assert_array_equal(saved[:, 0], test_ids)
        np.testing.assert_array_equal(saved[:, 1], labels)
        assert len(np.unique(test_ids)) == len(test_ids)
        outputs[method] = {
            "path": output.relative_to(ROOT).as_posix(),
            "sha256": sha256(output),
            "rows": len(labels),
            "positive_predictions": int(np.sum(labels == 1)),
            "lambda": artifact["models"][method]["lambda"],
            "threshold": artifact["models"][method]["threshold"],
            "training_loss": artifact["models"][method]["training_loss"],
        }
        log(f"Wrote and verified {output.name}")
    save_json(
        manifest_path,
        {
            "training_signature": signature,
            "test_data_sha256": test_digest,
            "model_artifact": {
                "path": model_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(model_path),
            },
            "expanded_columns": artifact["expanded_columns"],
            "l1_selected_columns": artifact["selected_columns"],
            "model_columns": artifact["model_columns"],
            "outputs": outputs,
            "aicrowd_uploaded": False,
        },
    )


if __name__ == "__main__":
    main()
