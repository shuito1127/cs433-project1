# Verification record

Verified on 2026-10-27. This records local work, not a claim that the current
uncommitted changes have been tested from a new public GitHub submission URL.

## Course tests

The official test revision is
[`22f74faa9d408853c381640f240f741ea196841a`](https://github.com/epfml/ML_course/tree/22f74faa9d408853c381640f240f741ea196841a/projects/project1/grading_tests),
the latest commit affecting the test directory when checked. Its four files
match our snapshot after newline normalization. The course test files were not
edited to obtain passes.

| Check | Actual result |
| --- | --- |
| All numerical tests for all six methods | Passed |
| Function existence, signatures as exercised, docstrings, required files, remaining-work scan | Passed |
| Black 22.6.0 formatting check | Passed after formatting |
| Local course suite with `--github_link .` | 27 passed; 1 expected URL-format failure |
| URL-format test alone with the existing `e8e3e5f` GitHub commit URL | Passed; checks URL syntax only |
| Additional adapter/regression tests | 4 passed |

The local URL failure occurs because `.` is a local path. It is not a numerical
failure. After committing/pushing the reviewed changes, run the entire course
suite against the **new final commit URL**. The separate URL-syntax check above
does not establish that the new code is already published or tested remotely.

Environment: the supplied `grading_tests/environment.yml`, installed as Windows
x64 on the local ARM machine: Python 3.9.25, NumPy 1.23.1, pytest 7.1.2,
GitPython 3.1.18, Black 22.6.0. Core checks also passed in the working Python 3.14.6 /
NumPy 2.4.6 environment. The model experiment itself ran in that working environment;
it was not fully retrained in the old grading environment.

```sh
python -m pytest grading_tests --github_link . -k "not test_github_link_format"
python -m pytest tests -q
```

XML logs are local under `artifacts/checks/`. Formatting was checked to preserve
the executable AST (allowing docstring whitespace normalization); details are
in `formatting_verification.json`. Changes to the old preprocessing helpers and
baseline configuration are formatting only. The earlier notebook/JSON are intact.

## Pipeline checks

- Compare compact matrix/vector products and weighted category counts against
  explicit dense one-hot matrices, including unseen categories.
- Compare 2,000 iterations of the shared L1 selector on compact versus dense inputs.
- Compare preprocessing with Shu's original functions on synthetic data; verify
  validation transformations do not alter fitted means/scales/categories.
- Round-trip serialized preprocessing state and test unknown-category behavior.
- Check rank reduction on known dependent columns and compare cached normal
  equations against direct shared Least Squares/Ridge on dense arrays.
- On the real data: match feature/label IDs; split before any fitted preprocessing;
  verify finite features/weights, returned half-MSE and normal-equation residuals.
- Record exact split/ID hashes, feature counts, source commits, data hashes,
  train/validation metrics, confusion counts, parameter grids and limitations.

The adapter intentionally guards all-missing and constant numeric columns;
neither case occurred in the recorded validation fit. Shu's survey-code rules
are reproduced, not independently re-audited against every codebook definition.

## Candidate submissions and remaining team checks

Preprocessing/selection and both linear models were fitted on all labeled rows.
`run.py --model both` reuses the included `models/yusen_linear_models.json` when
compatible (or trains if absent). It reopens each CSV and checks header, row count, exact test-ID
order, unique IDs and labels restricted to -1/+1. The actual output hashes and
feature counts are recorded in `experiments/yusen_submissions.json`.

Prediction reproducibility is checked by rerunning `run.py --model both
--predict-only` and comparing both CSV hashes. This checks inference from the same
saved fitted state; it does not claim cross-platform bitwise equivalence or two
independent full retrainings.

Remaining team-level steps are external to these local checks: select and upload
the actual candidate to AIcrowd, record its returned score, choose the final
`run.py` pipeline, review/merge code, test the final GitHub commit URL and compile
the completed report within the page limit. This record describes the local
verification before publication; no AIcrowd upload, PR or merge is included.
Publishing `yusen_branch` is a separate step from final team review and submission.
