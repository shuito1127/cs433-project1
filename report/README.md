# Report sources

- `main.tex`: the revised English Overleaf draft, including the latest linear-model
  results and a marked pending-team-integration paragraph. It uses an inline bibliography.
- `references.bib`, `IEEEtran.cls`, `IEEEtran.bst`: supporting bibliography and course
  template files. The BibTeX file is retained as an optional alternative.
- `yusen_results.tex`: current measured linear-model results, ready to adapt into
  the shared Overleaf report. Values come from `experiments/yusen_final_results.json`.
- [Results and interpretation](../docs/yusen_final_results.md): exact settings,
  old/new comparison, model-selection caveats and source revision.
- `latex-example-paper/`: unmodified course example.

In Overleaf, keep the course `10pt,conference,compsocconf` template and use
pdfLaTeX. The latest baseline and extended-pipeline comparisons are already in
`main.tex`; avoid presenting validation-tuned scores as independent test scores.
The target is at most two pages of body text, plus an optional references-only
third page. The current sources have not been locally compiled or page-verified.

Authors in the draft: Anthony Glegola, Shu Ito, Yusen Haung, as supplied.

Remaining team content: Anthony's GD/SGD experiment details, Shu's confirmed
logistic results, the final selected submission and its exact reproduction
command, confirmed AIcrowd score, and final author/page-layout review. Shu's
reported AIcrowd F1 0.444 is a different evaluation source from our local scores;
do not compare it as if measured on the same validation split.

Use the maintained source files when uploading. Any older
`output/overleaf-report.zip` is a historical bundle and does not include the new
results snippet until regenerated.
