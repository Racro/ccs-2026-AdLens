# Datasets

Labeled ad datasets used for evaluation and as confirmed-violation examples in the paper.

## Structure

```
datasets/
├── golden_main_data/       # Main golden evaluation set (1,283 labeled ads)
│   ├── images/             # 1,271 PNGs, <ad_id>.png (12 ads carry two labels and share one image)
│   └── metadata.json       # Per-ad records with OCR text, crawl metadata, and ground-truth labels
├── golden_test_data/        # Held-out look-forward test set (253 labeled ads)
│   ├── images/             # 251 PNGs, <ad_id>.png (2 ads carry two labels and share one image)
│   └── metadata.json       # Same schema as golden_main_data/metadata.json
└── violations/              # Sample confirmed-violation ad images used as paper examples (600 total)
    ├── deceptive_claims/    # 200 images
    ├── misleading_ad_design/ # 200 images
    └── scareware/           # 200 images
```

## golden_test_data

Held-out set for Table 2(c): **253 ads**, a balanced sample of the look-forward crawl: up to 50 true negatives per violation type, 50 true positives for scareware and deceptive claims where available (48 scareware and 52 deceptive-claim positives exist, all kept), and all 3 misleading-design positives. Images are in `images/<ad_id>.png` and `metadata.json` carries `ad_id`, `violation_type`, `label`, `ocr_text` (PaddleOCR PP-OCRv5, regenerated with `ocr/process.py`) and `translated_ocr_text` (`translategemma:4b`), plus crawl fields where available. Different versions of the same creative (`-v0`, `-v1`, ...) are distinct images and count as separate ads.

Reproduce Table 2(c) (classifiers + judge on this set):

```bash
python detection/adlens_pipeline.py \
    --metadata datasets/golden_test_data/metadata.json \
    --images-dir datasets/golden_test_data/images \
    --out-dir detection/results/test_set_results
python scripts/eval_pipeline_results.py \
    --metadata datasets/golden_test_data/metadata.json \
    --results-dir detection/results/test_set_results
```

The shipped set is the 253-ad test set behind Table 2(c), approximately 50 ads per violation type and label (deceptive claims 52 / 50, scareware 48 / 50, misleading design 3 / 50 positives / negatives), so the command above runs and scores all 253 ads.

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
