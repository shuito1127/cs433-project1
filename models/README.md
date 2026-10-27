# Fitted linear candidates

`yusen_linear_models.json` contains the full-data fitted preprocessing, selected
columns and both models' weights. It is about 90 KB and contains no training rows.
Commit it with the code to let teammates reproduce predictions without rerunning
L1 selection. Data/source hashes are checked before reusing it.

```sh
python run.py --model both --predict-only
```

Training used all 328,135 labeled observations: 1,104 expanded columns, 838 kept
by Shu's L1 selector, and 731 retained after numerical rank reduction. Validation
has its own training-only fit and different column counts; see the experiment guide.

To refit weights, run `python run.py --model both --retrain`. On a fresh clone,
this also fits preprocessing and L1 selection. A matching local selection cache
can be reused; to repeat selection from scratch, remove the local
`artifacts/full_train/` cache first. Training code is in `run.py`,
`preprocessing/final_pipeline.py` and `experiments/evaluate_linear.py`.

These are Yusen's candidate models, not the team's confirmed final AIcrowd model.
Keep only the chosen model's artifact in the final submission if desired.
