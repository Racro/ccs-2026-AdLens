# Scripts

Utility and analysis scripts for the AdLens paper.

## eval_pipeline_results.py

Recomputes Table 2 from the output of `detection/adlens_pipeline.py` (classifier caches, ensemble labels, judge labels) against the ground-truth `label` in the metadata. Reports (a) each classifier and the majority-vote ensemble, (b) the judge on disagreement cases only, (c) the full pipeline. Standard library only.

```bash
python detection/adlens_pipeline.py --eval        # run the models on the labeled golden set
python scripts/eval_pipeline_results.py           # Table 2(a)/(b) (+ full-pipeline numbers)
```

Table 2(c) (test set): see [`../datasets/README.md`](../datasets/README.md#golden_test_data). Options: `--metadata`, `--results-dir`, `--classifier-models`, `--judge-name`, `--json-out`.

## f1_score_table2.py

Self-contained script that reproduces Table 2 — Precision, Recall, and F1 for every model across all three violation types, plus macro-averaged Precision, Recall, and F1. All data is defined inline as confusion-matrix counts; no external files are required.

```bash
python scripts/f1_score_table2.py
```

Outputs three sections:

| Section | Description |
|---|---|
| (a) Ensemble Classifiers | Golden set (1283 items), per-model results |
| (b) Judge Models | Golden set, disagreement cases only |
| (c) Golden Test Dataset | Held-out test set (seed-42, max 50/folder) |

*DC = Deceptive Claims · SW = Scareware · MD = Misleading Design*

#### (a) Ensemble Classifiers — Golden Set (1,283 items)

| Model | DC P | DC R | DC F1 | SW P | SW R | SW F1 | MD P | MD R | MD F1 | Macro F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5:9b | 0.973 | 0.787 | 0.870 | 0.989 | 0.949 | 0.969 | 0.859 | 0.970 | 0.911 | 0.917 |
| Gemma3:12b | 0.873 | 0.822 | 0.847 | 0.994 | 0.803 | 0.888 | 0.966 | 0.586 | 0.729 | 0.821 |
| GLM-4.6V-Flash:9b | 0.994 | 0.791 | 0.881 | 0.930 | 1.000 | 0.964 | 0.963 | 0.109 | 0.195 | 0.680 |

#### (b) Judge Models — Golden Set, Disagreement Cases Only

| Model | DC P | DC R | DC F1 | SW P | SW R | SW F1 | MD P | MD R | MD F1 | Macro F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma4:26b | 0.897 | 0.929 | 0.912 | 0.967 | 0.935 | 0.951 | 0.861 | 0.946 | 0.902 | 0.922 |
| Qwen3.5:27b | 0.628 | 0.964 | 0.761 | 0.939 | 1.000 | 0.969 | 0.861 | 0.946 | 0.902 | 0.877 |
| Mistral-small3.2:24b | 0.800 | 0.714 | 0.755 | 0.968 | 0.968 | 0.968 | 0.897 | 0.848 | 0.872 | 0.865 |

#### (c) Full Pipeline — Test Set (look-forward)

| Model | DC P | DC R | DC F1 | SW P | SW R | SW F1 | MD P | MD R | MD F1 | Macro P | Macro R | Macro F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5:9b | 0.978 | 0.880 | 0.926 | 1.000 | 0.958 | 0.979 | 1.000 | 1.000 | 1.000 | 0.993 | 0.946 | 0.968 |
| Gemma3:12b | 0.873 | 0.960 | 0.914 | 0.978 | 0.938 | 0.957 | 1.000 | 1.000 | 1.000 | 0.950 | 0.966 | 0.957 |
| Gemma4:26b (judge) | 1.000 | 1.000 | 1.000 | 0.980 | 1.000 | 0.990 | 1.000 | 1.000 | 1.000 | **0.993** | **1.000** | **0.997** |

Overall (macro-averaged over the three violation types) look-forward performance of the full pipeline: **precision 0.993, recall 1.000, F1 0.997**. Misleading Design has only 3 positive test ads, so its perfect scores should be read with that in mind.

## plots/

Python scripts that generate the paper figures. See [`plots/README.md`](plots/README.md) for details.

## pdns/

Protective DNS (PDNS) lookup tooling for ad landing domains.

| File | Description |
|---|---|
| `query_pdns.sh` | Shell script that queries PDNS providers (Cloudflare, Quad9, Cisco, CIRA) for each domain in the input list |
| `all_ad_link_domains.txt` | Deduplicated list of ad landing domains to look up |
| `reformat-links.ipynb` | Deduplicates ad landing domains across categories from `unique_domains_all.json` into `all_ad_link_domains.txt` |
| `analyze-pdns-results.ipynb` | Cross-references PDNS results from multiple resolvers to identify ad domains flagged as malicious |

### Usage

```bash
cd scripts/pdns
bash query_pdns.sh
```

Results are written as `pdns_<resolver>_ad_links.jsonl` files in the same directory.
