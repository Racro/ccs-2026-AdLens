# Datasets

Labeled ad datasets used for evaluation and as confirmed-violation examples in the paper.

## Structure

```
datasets/
├── golden_main_data/       # Main golden evaluation set (1,283 images)
│   ├── scareware/
│   │   ├── tp/             # True positives (198)
│   │   └── tn/             # True negatives (202)
│   ├── misleading/
│   │   ├── tp/             # True positives (225)
│   │   └── tn/             # True negatives (224)
│   ├── ad_design/
│   │   ├── tp/             # True positives (239)
│   │   └── tn/             # True negatives (195)
│   └── metadata.json       # Per-ad records with OCR text, crawl metadata, and ground-truth labels
├── golden_test_data/        # Held-out look-forward test set (301 images + metadata.json)
│   ├── scareware/
│   │   ├── tp/             # True positives (48)
│   │   └── tn/             # True negatives (67)
│   ├── misleading/
│   │   ├── tp/             # True positives (52)
│   │   └── tn/             # True negatives (64)
│   └── ad_design/
│       ├── tp/             # True positives (3)
│       └── tn/             # True negatives (67)
│   └── metadata.json       # Same schema as golden_main_data/metadata.json
└── violations/              # Sample confirmed-violation ad images used as paper examples (600 total)
    ├── deceptive_claims/    # 200 images
    ├── misleading_ad_design/ # 200 images
    └── scareware/           # 200 images
```

## golden_test_data

Held-out set for Table 2(c): 301 ad images (299 distinct creatives; two appear under two violation types) in `<violation>/<tp|tn>/<ad_id>.png`, with the same folder layout as `golden_main_data/`, and a `metadata.json` carrying `ad_id`, `violation_type`, `label`, `ocr_text` (PaddleOCR PP-OCRv5, regenerated with `ocr/process.py`) and `translated_ocr_text` (`translategemma:4b`), plus crawl fields where available.

Reproduce Table 2(c) (classifiers + judge on this set):

```bash
python detection/adlens_pipeline.py \
    --metadata datasets/golden_test_data/metadata.json \
    --images-dir datasets/golden_test_data \
    --out-dir detection/results/test_set_results
python scripts/eval_pipeline_results.py \
    --metadata datasets/golden_test_data/metadata.json \
    --results-dir detection/results/test_set_results
```

The paper's Table 2(c) evaluates the classifiers on a seed-42 sample (at most 50 ads per folder) and the judge on the full set, as described in `scripts/f1_score_table2.py`; the command above scores every ad.

## metadata.json Schema

`golden_main_data/metadata.json` contains one record per ad with the following fields:

| Field | Description |
|---|---|
| `ad_id` | Unique creative+version identifier (e.g. `CR00102789777257922561-v0`) |
| `crid` | Creative ID without version suffix |
| `violation_type` | `scareware` / `deceptive_claim` / `misleading_design` |
| `label` | Ground-truth label (`tp` = true positive, `tn` = true negative) |
| `category` | Ad software category (`mobile`, `computer`, `software`) |
| `advertiserID` | Google Ads advertiser ID |
| `advertiserName` | Advertiser display name |
| `creativeURL` | URL on `adstransparency.google.com` |
| `adFormat` | Ad format (e.g. `video`, `image`) |
| `targetUrl` | Full ad click URL (with tracking params) |
| `adUrl` | Final landing URL |
| `timestamp` | Crawl timestamp (ISO 8601) |
| `adText` | Ad text extracted by the crawler |
| `ocr_text` | Raw OCR output from PaddleOCR |
| `translated_ocr_text` | English translation of `ocr_text` (if source was non-English) |

## Violation Categories

| Category | Description |
|---|---|
| `scareware` | Assertive threat / panic-inducing claims |
| `deceptive_claim` | False device-state, fake recovery, financial bait |
| `misleading_design` | CTA-only ads with no identifiable advertiser |
| `misconfigured` | Blank, QR-code, or otherwise broken ad creatives — additional analysis only, not part of the main classification pipeline |

## Image Filename Format

Images follow the Google Ad Transparency Center creative ID scheme:

```
<creative_id>-v<version>.png
```

For example, `CR00173184909714653185-v1.png` is creative 1 of ad ID `CR00173184909714653185`.

## violations/

Each subfolder contains ~200 ad screenshots classified as violations by the AdLens ensemble pipeline and manually verified.

| Folder | Violation Type |
|---|---|
| `deceptive_claims/` | False device-state, fake recovery, or financial bait claims |
| `misleading_ad_design/` | CTA-only ads with no identifiable advertiser |
| `scareware/` | Assertive threat or panic-inducing claims |
