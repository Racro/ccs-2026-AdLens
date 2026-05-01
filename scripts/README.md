# Scripts

Utility and analysis scripts for the AdLens paper.

## f1_score_table2.py

Self-contained script that reproduces Table 2 — Precision, Recall, and F1 for every model across all three violation types. All data is defined inline as confusion-matrix counts; no external files are required.

```bash
python scripts/f1_score_table2.py
```

Outputs three sections:

| Section | Description |
|---|---|
| (a) Ensemble Classifiers | Golden set (1283 items), per-model results |
| (b) Judge Models | Golden set, disagreement cases only |
| (c) Golden Test Dataset | Held-out test set (seed-42, max 50/folder) |

> **Erratum — Table 2, section (c):** In the submission manuscript, the per-class scores for **Deceptive Claims** and **Scareware** in section (c) (Golden Test Dataset) were inadvertently transposed when copying results from the script output into the paper. The values in `f1_score_table2.py` are correct. This will be corrected in the final version. This error affects only the column-level P/R/F1 breakdown for those two classes in the temporal-robustness evaluation; the **Macro F1 scores are unaffected** because they are the unweighted average of all three F1 values, which remains the same regardless of column order. Importantly, the transposition does not alter the core finding: performance does not degrade on the held-out test set but in fact improves relative to the golden set evaluation, confirming the temporal robustness of the pipeline.

## plots/

Python scripts that generate the paper figures. See [`plots/README.md`](plots/README.md) for details.

## pdns/

Passive DNS (pDNS) lookup tooling for ad landing domains.

| File | Description |
|---|---|
| `query_pdns.sh` | Shell script that queries a pDNS API for each domain in the input list |
| `all_ad_link_domains.txt` | Deduplicated list of ad landing domains to look up |

### Usage

```bash
cd scripts/pdns
bash query_pdns.sh
```

Results are written to `progress/` and `screenshots/` subdirectories.
