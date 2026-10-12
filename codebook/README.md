# BRFSS 2015 dataset documentation

Official index: https://www.cdc.gov/brfss/annual_data/annual_2015.html

## Field meanings and value codes (primary reference)

The existing [codebook15_llcp.pdf](codebook15_llcp.pdf) is the project's local codebook reference. The official **2015 BRFSS Codebook** describes variable names, survey questions, value labels, missing/special codes, and frequencies.

Official PDF: https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf

Use this document when interpreting CSV column names and writing per-feature preprocessing rules. For example, PHYSHLTH uses 88 for zero days, 77 for unknown, and 99 for refused. These rules are specific to each variable; do not apply them globally.

## Derived feature definitions (supplementary reference)

[Calculated Variables in the 2015 Data File of the BRFSS](2015_Calculated_Variables_Version4_08_10_17-508c.pdf) explains how calculated variables are constructed.

Official PDF: https://www.cdc.gov/brfss/annual_data/2015/pdf/2015_Calculated_Variables_Version4_08_10_17-508c.pdf

## Survey background (supplementary reference)

[2015 BRFSS Overview](overview_2015.pdf) describes survey design, data collection, and processing.

Official PDF: https://www.cdc.gov/brfss/annual_data/2015/pdf/overview_2015.pdf

## Project usage

- Match documentation to the actual competition CSV column names. The CDC source dataset and the course-provided dataset need not have identical rows or columns.
- The codebook's `Column` positions describe the original source layout, not NumPy column indices in the competition CSV.
- Use the course-provided training and test CSVs for the project; these documents are interpretation references.
- Imputation, scaling, encoding, and feature selection are modeling choices, rather than instructions mandated by the codebook.

All three PDFs are available locally in this directory. Official source links were checked on 2026-10-12; the supplementary PDFs were downloaded manually. Byte-for-byte identity with the current official downloads has not been verified.

The competition CSVs belong in `../dataset/`, separate from these documents. See the [project README](../README.md) for the workspace layout and experiment instructions, and the [course project description](../docs/project1_description.pdf) for assignment requirements.
