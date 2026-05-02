# Sample Data

Ad dataset used by the detection pipeline and search platform.

## Structure

```
sample_data/
├── metadata.json               # Per-ad records (ad_id, ocr_text, translated_ocr_text, …)
├── cached_translations.json    # OCR translations (translategemma:4b) keyed by ad_id
└── images/                     # Flat directory of ad screenshots (<ad_id>.png)
```

Images are stored flat — all screenshots sit directly in `images/` with filenames matching `ad_id` (e.g. `CR00102789777257922561-v0.png`).

## metadata.json Schema

Each record in `metadata.json` has the following fields:

| Field | Description |
|---|---|
| `ad_id` | Unique creative+version identifier (e.g. `CR00102789777257922561-v0`) |
| `crid` | Creative ID without version suffix |
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

Images are resolved as `sample_data/images/<ad_id>.png`.
