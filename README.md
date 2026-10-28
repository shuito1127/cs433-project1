# CS-433 Project 1

Anthony Glegola, Shu Ito and Yusen Huang.

## Run the final model

Use the course environment (verified with Python 3.9.23 and NumPy 1.23.1).
Put `x_train.csv`, `y_train.csv` and `x_test.csv` in `data/`, then run:

```sh
python run.py
```

This trains the final model and writes **`reg_logistic.csv`** in the project folder.
Upload that CSV to AIcrowd to obtain a score. The Python script runs locally.
A full cached run took about five minutes on the verification Mac; first-time CSV
loading adds time. Several GB of memory are needed.

The runner also accepts `dataset/` or `--data-dir /path/to/csvs`.
Data, predictions and generated caches are ignored by Git.

## Final method

Shu's codebook-based preprocessing, mean imputation, standardization and one-hot
encoding are fitted on training rows. L1 logistic selection uses lambda 0.02,
2,000 updates, step size 0.2 and coefficient cutoff 0.001 (intercept retained).
L2 logistic regression uses lambda 0.00001, 2,000 updates, step size 0.2 and
probability threshold 0.21. Both stages start from zero weights.
The full fit retained 838 of 1,104 expanded columns.

`implementations.py` contains all six required algorithms. `helpers.py` contains
the course CSV helper. Only NumPy and the Python standard library are used.

## Experiment sources

- Anthony's GD/SGD and categorical comparison: [experiment guide](experiments/anthony_README.md).
- Yusen's least-squares/ridge experiments and recorded results: [published revision](https://github.com/shuito1127/cs433-project1/tree/ed97461d1b417b2a395ce081962c72859cd156d2/experiments).
- Shu's preprocessing, logistic experiments and final settings: [published contribution](https://github.com/shuito1127/cs433-project1/blob/65c5f340bb381021360f6447208bce0600cdb4af/shu_contribution.md).

These source links preserve the experiments without adding duplicate runners or
copies of every teammate's working files to this checkout.

## Verification

The six algorithms pass the local numerical/interface checks. The final runner
was executed on all 328,135 training rows and produced 109,379 test predictions,
with IDs and -1/+1 labels checked. Its CSV matched Shu's published recipe exactly:

```text
SHA-256: 4cf36de9293440958724d4ce024424d4cf23556be623724de76dc21319337553
```

This identifies the reproduced recipe; the historical AIcrowd upload was not
available locally for a file comparison.

Copy the official course grading folder locally to check the code. Existing
experiment files retain their original formatting; the course's Black check is
advisory. Run the numerical/interface checks with:

```sh
python -m pytest grading_tests --github_link . -k 'not test_github_link_format and not test_black_format'
```

For the course submission, supply the report PDF and an immutable GitHub commit
URL ending in `/tree/FULL_COMMIT_SHA`.
