# Preprocessing

- `shu_reference.py`: source definitions extracted from Shu's `run.py` at
  `f6baef4`; see `shu_source.json` for provenance.
- `final_pipeline.py`: reusable training-only fit/transform, memory-efficient
  one-hot products, Shu's L1 selection and linear-model rank handling.
- `preprocess.py`, `shu_baseline_config.py`: unchanged helpers/configuration for
  the earlier partial-feature baseline.

Use `FinalPreprocessor.fit_transform(train, names)` and then
`transform(validation_or_test)`. Fit the L1 selector on training labels only.
The experiment and `run.py` manage this automatically. Do not reuse a selector
fitted on all labeled rows when reporting validation performance.

See [integration notes](../docs/integration.md) for exact settings and differences
from Shu's standalone pipeline. No medical-code mappings have been redesigned.
