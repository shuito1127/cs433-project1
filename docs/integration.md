# Yusen's integration handoff

## Scope and ownership

This branch combines reviewed source definitions to evaluate Yusen's Least Squares
and Ridge with Shu's latest preprocessing. No remote branches were merged or
rewritten. This contribution is published on `yusen_branch`; it does not merge
into `main`, create a PR, or upload predictions to AIcrowd.

| Component | Source |
| --- | --- |
| Least Squares / Ridge | Yusen, `e8e3e5f349b881a61b5932769d1676982fd9c7fc` |
| GD / SGD | Anthony, `0af44c8a11ceaedbe786b699dc3f01a7c5a6d5c1` |
| Logistic / regularized logistic / L1 selector | Shu, `f6baef4dd878ac79d030aa3c65cdda6e905a903b` |
| Feature definitions and transformations | Shu's `run.py` at the same commit |

The six required methods remain in `implementations.py`. Yusen's normal-equation
implementations are unchanged. Anthony's algorithms are copied without algorithmic
changes. Shu's logistic loss now returns `np.float64(loss)` instead of a Python
`float`, because the public tests inspect `loss.ndim`. This does not change its
value. The extra L1 method is Shu's subgradient optimizer, not a proximal solver.

## Files to integrate

- `implementations.py`: reconcile by function with each author's branch; do not
  replace teammates' later changes with an older whole-file copy.
- `preprocessing/shu_reference.py`, `shu_source.json`: extracted source definitions
  and immutable provenance. The original executable `main()` and imports are omitted.
- `preprocessing/final_pipeline.py`: fit/transform adapter, compact one-hot matrix
  products, L1 selection, training-only rank reduction, serializable fitted state.
- `experiments/evaluate_linear.py`, `linear_io.py`: validation, parameter search,
  chunked data loading and metrics. Only NumPy and standard-library runtime imports.
- `experiments/Yusen_final_preprocessing.ipynb`: concise review notebook with saved
  results and an optional rerun switch.
- `experiments/yusen_final_results.json`, `yusen_submissions.json`: measured results,
  settings, source/data hashes and candidate CSV metadata.
- `tests/test_final_pipeline.py`: adapter, leakage and numerical regression checks.
- `models/yusen_linear_models.json`: small fitted state for both candidates,
  including preprocessing, column selection and weights; enables fast reproduction.
- `run.py`: reproducible **linear candidate** entry point. Keep Shu's logistic
  default in the final team repository if that is still the chosen submission;
  either expose a model option or retain this entry point under a separate name.

The historical notebooks, partial-feature baseline and course helper/test files
are retained. Data, caches and generated CSVs live locally under ignored
`dataset/` and `artifacts/`; cloning the branch does not download them. The small
`models/` parameter file is intended to be committed with the code.

## Preprocessing differences that must be reported

1. Shu's latest code uses 5 one-value, 111 ordered, 14 decoded and 157 categorical
   variables. The 29 explicitly dropped variables and 5 undecoded variables are
   not used. The latter are `HEIGHT3`, `EXERHMM1`, `EXERHMM2`, `FLSHTMY2`, `HIVTSTD3`.
2. All fitting, including the supervised L1 selector, happens after splitting.
   Test/validation rows only receive transformations. L1 uses labels 0/1;
   Least Squares and Ridge use labels -1/+1.
3. The adapter preserves Shu's mappings and one-hot levels, including a missing
   category. Unseen categories are all-zero, as in the source. It guards all-missing
   and constant numeric columns to avoid NaN/division-by-zero; actual counts are
   recorded in the results JSON.
4. Compact matrix multiplication avoids a multi-GB dense one-hot allocation. It is
   algebraically equivalent to the dense representation, tested through 2,000 L1
   iterations. Floating-point reduction order can differ across BLAS environments.
   Category/contingency counts compute the same normal-equation Gram matrix
   without multiplying explicit zeros. A tested dot-product wrapper caches these
   statistics while calling the unchanged shared Least Squares/Ridge functions.
5. L1 settings are unchanged: lambda 0.02, step 0.2, 2,000 iterations, retain
   `abs(weight) > 0.001`, always keep the intercept.
6. Least Squares requires independent columns. A training-only pivoted Gram
   procedure removes numerical dependencies from L1-selected inputs (squared
   relative residual tolerance `1e-10`). Both linear methods use the same retained
   columns. Original selected-input Ridge is also recorded because changing the
   representation changes its L2 penalty. This is not the old two-column rule.

## Before the final team submission

- Review this contribution, then merge contributions one at a time.
- Choose the actual submitted model and make the final `run.py` reproduce that CSV.
- Run the course tests on the immutable **final commit URL**, not just a local path.
- Update the two-page report with the results and limitations in
  `docs/yusen_final_results.md`; distinguish local scores from AIcrowd scores.
- Compile the team report in Overleaf and verify its page limit. The revised
  `report/main.tex` includes the latest linear results; other team contributions
  and the final submission choice still need to be added.
