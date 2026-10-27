"""Export measured JSON results to concise Markdown and an Overleaf snippet."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "experiments/yusen_final_results.json").read_text())
    selected = data["selected_models"]
    fixed = [
        r
        for r in data["models"]
        if r["method"] == "least_squares" or r["lambda"] == 0.001
    ]
    rows = []
    for label, accuracy, f1 in [
        ("Earlier Least Squares", 0.8771226213840772, 0.34460338101430427),
        ("Earlier Ridge", 0.8780978386746797, 0.3434011818778726),
    ]:
        rows.append(
            f"| {label} | Partial features | -0.55 | {accuracy:.6f} | {f1:.6f} |"
        )
    for row in fixed:
        metrics = row["fixed_validation"]
        rows.append(
            f"| {row['method']} | Final preprocessing, fixed settings | -0.55 | {metrics['accuracy']:.6f} | {metrics['f1']:.6f} |"
        )
    for method, row in selected.items():
        metrics = row["validation"]
        rows.append(
            f"| {method} | Final preprocessing, validation-selected settings | {row['threshold']:.2f} | {metrics['accuracy']:.6f} | {metrics['f1']:.6f} |"
        )
    details = []
    for method, row in selected.items():
        v, t = row["validation"], row["training"]
        details.append(
            f"| {method} | {row['lambda']} | {row['threshold']:.2f} | {v['precision']:.6f} | {v['recall']:.6f} | {t['half_mse']:.6f} | {v['half_mse']:.6f} | {v['tp']} / {v['fp']} / {v['fn']} / {v['tn']} |"
        )
    p, rank = data["preprocessing"], data["rank"]
    source = data["source"]["shu_branch"]["commit"]
    md = f"""# Yusen: final-preprocessing linear-model results

Measured on all 328,135 labeled observations using Shu's preprocessing at
`{source}`. Source: [exact results](../experiments/yusen_final_results.json).
These are local validation results, not AIcrowd scores.

## Report table

| Model | Input/settings | Threshold | Validation accuracy | Validation F1 |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(rows)}

Earlier Ridge used lambda 0.001; the new fixed Ridge comparison uses the same
lambda. Both old and new experiments use the same 75%/25% split and seed 42.
The new pipeline adds categorical/decoded inputs and supervised feature selection
together, so this is a comparison of pipeline versions, not an isolated decoding
ablation. The fixed-setting rows isolate the benefit of that combined pipeline
change from the later parameter search.

## Selected settings and diagnostics

| Model | Lambda | Threshold | Precision | Recall | Train half-MSE | Validation half-MSE | TP / FP / FN / TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(details)}

- Training / validation: {data['split']['train_rows']:,} / {data['split']['validation_rows']:,};
  positives {data['split']['train_positives']:,} / {data['split']['validation_positives']:,}.
- NumPy `default_rng(42)` shuffle, not stratified; one holdout, no cross-validation.
- Inputs: 5 one-value, 111 ordered, 14 decoded and 157 categorical raw columns.
  Expanded columns including intercept: {p['expanded_columns']}.
- L1 selector: lambda 0.02, step 0.2, 2,000 iterations, absolute-weight cutoff 0.001.
  Training-only selection retained {p['l1_selected_columns']} columns.
- Rank handling retained {rank['retained_columns']} columns; dropped {len(rank['dropped_indices'])}.
  Squared relative residual tolerance: {rank['tolerance']:g}; largest discarded
  relative residual: {rank['max_discarded_relative_residual']:.3g}; normalized
  condition number of retained design: {rank['normalized_condition_number']:.3g}.
  Exact column names and diagnostics are in the JSON.
- All-missing numeric columns: {len(p['all_missing_numeric'])}; constant numeric
  columns: {len(p['constant_numeric'])}. Unseen validation categories are recorded
  per raw column in the JSON and encoded as all-zero.
- Original selected-input Ridge (before rank handling), lambda 0.001 / threshold
  -0.55: F1 {data['ridge_original_selected']['validation']['f1']:.6f}, accuracy
  {data['ridge_original_selected']['validation']['accuracy']:.6f}. Removing columns
  changes Ridge's L2 penalty representation; its effect must not be described as
  merely fixing the solver.
- Always-negative baseline: accuracy {data['always_negative']['accuracy']:.6f}, F1 0.
- Runtime environment: Python {data['environment']['python']}, NumPy {data['environment']['numpy']}.

## Interpretation and limits

Lambda was selected from 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1; thresholds from -0.90
through 0.00 in increments of 0.01. Selection maximized validation F1. Therefore
the selected-setting scores use the same data for tuning and evaluation and can
be optimistic; do not describe them as an untouched test estimate. The fixed
comparison is provided separately. Regression scores are not probabilities.

The task is imbalanced, so accuracy alone can favor an always-negative classifier.
Use F1 with precision/recall and the confusion counts above. A small difference
between two methods on one split is not evidence of general superiority.

After model selection, preprocessing, L1 selection and both linear models were
refitted on all labeled rows. `python run.py --model both` reproduces predictions
from the included fitted state; use `--retrain` to refit weights. Full-fit counts and
verified CSV hashes are in [yusen_submissions.json](../experiments/yusen_submissions.json).
No hidden test labels are available locally, and neither generated candidate has
been uploaded by this workflow. Shu's reported AIcrowd F1 0.444 is a separate
team-reported competition result, not a directly comparable local validation row.

## Reproduce and integrate

```sh
python -m experiments.evaluate_linear
python -m experiments.export_linear_results
python run.py --model both
```

See [integration notes](integration.md) and [verification](verification.md).
The Overleaf-ready excerpt is [report/yusen_results.tex](../report/yusen_results.tex).
"""
    (ROOT / "docs/yusen_final_results.md").write_text(md, encoding="utf-8")
    table = []
    for row in fixed:
        v = row["fixed_validation"]
        label = (
            "Least squares, fixed"
            if row["method"] == "least_squares"
            else "Ridge, fixed"
        )
        table.append(f"{label} & {v['accuracy']:.4f} & {v['f1']:.4f} \\\\")
    for method, row in selected.items():
        v = row["validation"]
        label = "Least squares, tuned" if method == "least_squares" else "Ridge, tuned"
        table.append(f"{label} & {v['accuracy']:.4f} & {v['f1']:.4f} \\\\")
    tex = (
        r"""% Measured results excerpt. Requires booktabs; adapt within the two-page report.
\paragraph{Linear-model comparison.}
We used a shuffled 75/25 training/validation split (seed 42;
246,102/82,033 observations). Imputation, scaling, category vocabularies and
supervised feature selection were fitted only on training observations.
Shu's preprocessing used 5 one-value, 111 ordered, 14 decoded and 157 categorical
variables. The L1 logistic selector used $\lambda=0.02$, step size $0.2$,
2,000 iterations and an absolute-weight cutoff of $0.001$, retaining the bias.
"""
        + f"The expanded design had {p['expanded_columns']} columns; L1 retained {p['l1_selected_columns']},\n"
        + f"and training-only rank reduction retained {rank['retained_columns']} independent columns\n"
        + r"""for both least squares and ridge. Both used normal equations and returned
$\|y-Xw\|^2/(2N)$, excluding the Ridge penalty. The fixed comparison used
threshold $-0.55$ and Ridge $\lambda=0.001$.

\begin{table}[t]
\centering
\caption{Local validation scores. Tuned rows reuse validation data for selection.}
\begin{tabular}{lrr}
\toprule
Model & Accuracy & F1 \\
\midrule
Earlier least squares & 0.8771 & 0.3446 \\
Earlier ridge & 0.8781 & 0.3434 \\
"""
        + "\n".join(table)
        + r"""
\bottomrule
\end{tabular}
\end{table}

"""
        + f"The selected Least Squares threshold was ${selected['least_squares']['threshold']:.2f}$;\n"
        + f"Ridge selected $\\lambda={selected['ridge']['lambda']:g}$ and threshold ${selected['ridge']['threshold']:.2f}$.\n"
        + r"""We searched six logarithmically spaced Ridge penalties from $10^{-5}$ to $1$
and thresholds from $-0.90$ to $0.00$ in steps of $0.01$, maximizing validation F1.
These tuned scores are model-selection estimates, not independent test results.
The earlier and current fixed-setting comparisons share the same split; the
pipeline change combines categorical encoding, decoding and feature selection,
so its effect cannot be attributed to decoding alone. Full-data refitting
generates separate candidate CSVs; their AIcrowd scores are not yet measured.
"""
    )
    (ROOT / "report/yusen_results.tex").write_text(tex, encoding="utf-8")
    print("Exported docs/yusen_final_results.md and report/yusen_results.tex")


if __name__ == "__main__":
    main()
