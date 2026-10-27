# Shu's Contribution

## 1. Main Contributions

My main contributions to Project 1 were:

- Implemented `logistic_regression`.
- Implemented `reg_logistic_regression`.
- Designed the codebook-based preprocessing pipeline.
- Classified the 321 raw BRFSS features into:
  - `one_value`
  - `ordered`
  - `decode_ordered`
  - `categorical`
  - `drop_candidate`
- Implemented variable-specific missing / special-value handling.
- Implemented mean imputation and standardization using NumPy.
- Implemented decoding rules for survey-specific numerical variables.
- Implemented one-hot encoding for categorical variables.
- Implemented L1-Regularized Logistic Regression for feature selection.
- Performed validation for regularization and classification threshold.
- Integrated the final preprocessing and prediction pipeline into `run.py`.

---

## 2. Preprocessing

The BRFSS dataset contains many numerical-looking variables that actually represent categories, missing codes, or encoded quantities.

The preprocessing pipeline therefore uses the official codebook to determine how each variable should be handled.

### Numerical / Ordered Features

The main preprocessing steps were:

1. Apply variable-specific rules for special values.
2. Convert missing values to `NaN`.
3. Impute missing values using the training mean.
4. Standardize using the training mean and standard deviation.
5. Apply the same fitted parameters to validation and test data.

```text
Training:
fit + transform

Validation / Test:
transform only
```

### Decoded Numerical Features

Several survey-specific variables were converted into more meaningful numerical representations.

Examples include:

| Variable                 | Processing                                                 |
| ------------------------ | ---------------------------------------------------------- |
| `WEIGHT2`                | Converted weight codes to kilograms                        |
| `HEIGHT3`                | Converted feet/inches or metric codes to centimeters       |
| `ALCDAY5`                | Converted weekly/monthly frequency to a common scale       |
| Food-frequency variables | Converted daily/weekly/monthly codes to common frequencies |
| `EXEROFT1`, `EXEROFT2`   | Converted exercise-frequency encodings                     |
| `EXERHMM1`, `EXERHMM2`   | Converted HHMM values to minutes                           |
| `BLDSUGAR`               | Converted different frequency units to a common scale      |
| `FLSHTMY2`, `HIVTSTD3`   | Converted MMYYYY values to a month index                   |

### Categorical Features

Categorical variables were encoded using one-hot encoding implemented with NumPy.

Categories were fitted using the training data and the same category mapping was reused for validation and test data.

In one version of the pipeline:

```text
157 categorical variables
→ approximately 969 one-hot features
```

### Missing-Code Corrections

Some values in the course CSV differed from the raw codebook representation.

The following values were verified directly from the training CSV:

| Variable   | Missing code |
| ---------- | -----------: |
| `MAXVO2_`  |        `999` |
| `FC60_`    |        `999` |
| `PAFREQ1_` |         `99` |
| `PAFREQ2_` |         `99` |
| `STRFREQ_` |         `99` |

---

## 3. Model Implementations

My assigned implementations were:

```python
logistic_regression(
    y,
    tx,
    initial_w,
    max_iters,
    gamma
)
```

and:

```python
reg_logistic_regression(
    y,
    tx,
    lambda_,
    initial_w,
    max_iters,
    gamma
)
```

The original dataset uses labels:

```text
{-1, +1}
```

so the labels are converted to:

```text
{0, 1}
```

before Logistic Regression training.

I also implemented an L1-Regularized Logistic Regression using a subgradient for feature-selection experiments.

---

## 4. Performance Progression

| Stage                      | Main change                                  | Validation F1 |  AIcrowd F1 |
| -------------------------- | -------------------------------------------- | ------------: | ----------: |
| Initial baseline           | One-value + ordered, Ridge Regression        |         0.343 |           — |
| Add categorical features   | One-hot categorical features                 |         0.426 |           — |
| Logistic Regression        | Logistic + classification-threshold tuning   |         0.432 |     \~0.445 |
| Early L1 feature selection | L1-selected feature set                      |       \~0.431 | 0.439–0.442 |
| Final pipeline             | L1 selection + L2 Reg-Log + threshold tuning |               |   **0.447** |

Using directly comparable validation results:

```text
0.343 → 0.432
```

which corresponds to an absolute improvement of approximately:

```text
+0.089
```

The largest improvement came from adding properly encoded categorical variables.

---

## 5. L1 Feature Selection

L1-Regularized Logistic Regression was used to select features based on the magnitude of the learned weights.

The workflow was:

```text
full feature matrix
→ L1 Logistic Regression
→ select features using |w|
→ retain the same features for train/test
→ retrain L2-Regularized Logistic Regression
```

Current feature-selection parameters:

| Parameter                   |   Value |
| --------------------------- | ------: |
| L1 lambda                   |  `0.02` |
| Feature-selection threshold | `0.001` |

Feature selection substantially reduced the number of features while maintaining similar performance.

After tuning the downstream model, the feature-selected version became the current best pipeline.

---

## 6. Final Best Pipeline

The current best pipeline is:

```text
raw data
→ codebook-based preprocessing
→ categorical one-hot encoding
→ feature concatenation
→ L1 feature selection
→ L2-Regularized Logistic Regression
→ classification threshold
→ submission
```

Final configuration:

| Parameter                   |   Value |
| --------------------------- | ------: |
| L1 lambda                   |  `0.02` |
| Feature-selection threshold | `0.001` |
| Final L2 lambda             |  `1e-5` |
| Classification threshold    |  `0.21` |
| Gamma                       |   `0.2` |
| Max iterations              |  `2000` |

Current best AIcrowd result:

| Metric       |     Score |
| ------------ | --------: |
| **F1 Score** | **0.447** |
| **Accuracy** | **0.882** |

---

## 7. `run.py`

I consolidated the final pipeline into `run.py`.

It performs:

```text
raw CSV loading
→ preprocessing
→ feature concatenation
→ feature selection
→ model training
→ test prediction
→ submission CSV generation
```

To execute the final pipeline:

```bash
python run.py
```

Validation and model-comparison experiments are kept in the experiment notebook.

The model section of `run.py` can also be replaced with another implementation from `implementations.py` to evaluate another model using the same preprocessing pipeline.
