# Sample Data

Labeled ad dataset used by the detection pipeline and search platform.

## Structure

```
sample_data/
├── metadata.json               # Per-ad records (ad_id, violation_type, label, ocr_text, …)
├── cached_translations.json    # NLLB-200 OCR translations keyed by ad_id
└── images/                     # Ad screenshots organised by violation and label
    ├── scareware/
    │   ├── positive/           # Ground-truth scareware ads
    │   └── negative/
    ├── deceptive_claim/
    │   ├── positive/
    │   └── negative/
    └── misleading_design/
        ├── positive/
        └── negative/
```

## metadata.json Schema

Each record in `metadata.json` has at minimum:

| Field | Description |
|---|---|
| `ad_id` | Unique creative identifier |
| `violation_type` | `scareware` / `deceptive_claim` / `misleading_design` |
| `label` | Ground-truth label (`positive` / `negative`) |
| `ocr_text` | Raw OCR output from PaddleOCR |
| `translated_ocr_text` | English translation (if source was non-English) |

Images are resolved as `sample_data/<violation_type>/<label>/<ad_id>.png`.
