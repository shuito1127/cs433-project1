# Shu's Contribution

## 1. Main Contribution

My main responsibilities were:

- Implementing Logistic Regression and L2-Regularized Logistic Regression.
- Designing the codebook-based preprocessing pipeline.
- Classifying the 321 raw BRFSS features into numerical, ordered, decoded numerical, categorical, and drop-candidate groups.
- Handling variable-specific missing and special-value codes.
- Implementing mean imputation, standardization, numerical decoding, and one-hot encoding with NumPy.
- Implementing and evaluating L1-based feature selection.
- Performing validation and classification-threshold tuning.
- Integrating the final preprocessing and prediction workflow into `run.py`.

The goal was to make the full pipeline reproducible from the raw CSV files without relying on notebook execution order.

---

## 2. Preprocessing

The BRFSS dataset contains many numerical-looking values that actually represent categories, missing codes, or encoded quantities. Therefore, I used the official codebook to determine how each feature should be handled.

The main preprocessing steps were:

- Variable-specific missing / special-code handling.
- Mean imputation using statistics fitted only on the training data.
- Standardization using training mean and standard deviation.
- Decoding survey-specific variables such as weight, height, frequency, time, and date encodings.
- One-hot encoding categorical variables using categories identified from the training data.

An important issue was that some values in the course CSV differed from the raw codebook representation. For example, `MAXVO2_` and `FC60_` use `999` as the missing code in the actual course CSV, while `PAFREQ1_`, `PAFREQ2_`, and `STRFREQ_` use `99`. These mappings were verified directly from the training data.

The general principle was:

```text
Training data: fit + transform
Validation/test data: transform only
```

---

## 3. Experiments and Performance

The pipeline was improved incrementally.

| Stage | Main change | Validation F1 | AIcrowd F1 |
| --- | --- | ---: | ---: |
| Initial baseline | One-value + ordered features, Ridge Regression | 0.343 | — |
| Add categorical features | One-hot categorical features | 0.426 | — |
| Logistic Regression | Logistic model + threshold tuning | 0.432 | ~0.445 |
| Early L1 feature selection | L1-based feature reduction | ~0.431 | 0.439–0.442 |
| Final pipeline | L1 selection + L2 Reg-Log + threshold tuning | best region around 0.20–0.21 | **0.447** |

Using directly comparable validation results, F1 improved from approximately **0.343 to 0.432**, an absolute improvement of about **0.089**.

The largest early improvement came from representing categorical survey information correctly rather than simply increasing model complexity.

---

## 4. Final Best Pipeline

The current best pipeline is:

```text
raw data
→ codebook-based preprocessing
→ one-hot encoding
→ L1 feature selection
→ L2-Regularized Logistic Regression
→ classification threshold
→ submission
```

Final configuration:

| Parameter | Value |
| --- | ---: |
| L1 lambda | `0.02` |
| Feature-selection threshold | `0.001` |
| Final L2 lambda | `1e-5` |
| Classification threshold | `0.21` |
| Gamma | `0.2` |
| Max iterations | `2000` |

Best AIcrowd result:

| Metric | Score |
| --- | ---: |
| **F1 Score** | **0.447** |
| **Accuracy** | **0.882** |

The L1 feature-selection step initially only reduced dimensionality while keeping similar performance. However, after tuning the downstream L2 regularization and classification threshold, the feature-selected pipeline became the best-performing version.

---

## 5. Debugging and Lessons

Several bugs occurred while building the pipeline.

One issue came from notebook execution order: training features were standardized while some test features were still using raw values. This caused severe train/test distribution mismatch.

Another issue was label representation. The original dataset uses `{-1, +1}`, while the logistic loss expects `{0, 1}`.

I also introduced a sigmoid sign error in the first `run.py` version:

```python
1 / (1 + np.exp(z))
```

instead of:

```python
1 / (1 + np.exp(-z))
```

This produced almost all-positive predictions and an AIcrowd result of approximately F1 `0.159` and accuracy `0.087`.

These problems were diagnosed using:

- feature ranges,
- NaN counts,
- prediction counts,
- positive prediction rate,
- probability distributions,
- validation / submission metrics.

This taught me to debug the whole pipeline systematically instead of assuming that a low score always means the model itself is bad.

---

## 6. Collaboration

I am currently studying abroad at EPFL and am still improving my confidence in technical discussions in English.

Rather than avoiding tasks requiring discussion, I intentionally worked on preprocessing, where many decisions were ambiguous and required communication with teammates.

My workflow was:

```text
form a hypothesis
→ check the codebook
→ inspect the actual data
→ implement
→ discuss with teammates
→ revise based on feedback
```

For example, when a teammate pointed out possible inconsistencies in several missing-value codes, I verified the actual values in the training CSV and corrected the preprocessing rules.

This project showed me that even when communication is not perfect, technical collaboration works well when decisions are supported by concrete evidence.

---

## 7. Final Integration

The final workflow was consolidated into `run.py`, which performs preprocessing, feature selection, model training, prediction, and submission generation in a single reproducible execution.

To run the final pipeline:

```bash
python run.py
```

Validation and model-comparison experiments remain in the experiment notebook.

The model section in `run.py` can also be replaced with other implementations from `implementations.py`, allowing the same preprocessing pipeline to be reused across different models.