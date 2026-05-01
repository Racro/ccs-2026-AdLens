# AdLens — CCS 2026 Cycle B

A measurement and detection system for malvertising and deceptive ad practices on the **Google Ad Transparency Center**. AdLens crawls ad creatives at scale, extracts text via OCR, and classifies ads into violation categories using an ensemble of vision-language models.

## Requirements

- **Python 3.12** (most stable tested version)
- **Node.js 18+** (crawler only)
- **Ollama** with models pulled locally (detection pipeline)
- **CUDA-capable GPU** (OCR and VLM inference)
- **Google Cloud credentials** (BigQuery queries)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install crawler dependencies:

```bash
cd crawler && npm install
```

## End-to-End Pipeline

### Step 1 — Crawl

Collect ad creatives and screenshots from `adstransparency.google.com` for a given advertiser:

```bash
cd crawler
node main.js AR16735076323512287233 200 --output ../sample_data/ads_raw.json
```

See [`crawler/README.md`](crawler/README.md) for authenticated crawling and bulk URL input.

### Step 2 — OCR

Extract text from the downloaded screenshots using PaddleOCR PP-OCRv5:

```bash
python ocr/process.py \
    --input-file sample_data/ads_raw.json \
    --output-file sample_data/metadata.json
```

See [`ocr/README.md`](ocr/README.md) for GPU/CPU options and batching flags.

### Step 3 — Translate

Translate non-English OCR text to English using `translategemma:4b` via Ollama:

```bash
ollama pull translategemma:4b

python detection/translate_ocr.py
# Reads  : sample_data/metadata.json
# Writes : sample_data/cached_translations.json
#          sample_data/metadata_with_translations.json
```

### Step 4 — Detect

Classify ads with a two-model ensemble (qwen3.5:9b + gemma3:12b) and resolve disagreements with a judge (gemma4:26b):

```bash
ollama pull qwen3.5:9b
ollama pull gemma3:12b
ollama pull gemma4:26b

python detection/adlens_pipeline.py
# Output: detection/results/pipeline_results/
#   scareware_results.json
#   misleading_results.json (deceptive_claim)
#   ad_design_results.json  (misleading_design)
#   classify_cache.json     (resumable)
#   latency_calls.json
```

Run specific violations or cap records for a quick test:

```bash
python detection/adlens_pipeline.py --violations scareware --limit 50
```

For the `misleading_design` violation the detection uses `detect_misconfigured.py` as a pre-filter and `misleading_design.py` for VLM confirmation before the ensemble pipeline. The `llm_judge.py` step runs automatically via `adlens_pipeline.py` for all disagreements; it can also be run standalone against any existing results directory.

## Repository Layout

```
.
├── crawler/          # Puppeteer crawler for Google Ad Transparency Center
├── detection/        # Ensemble VLM classification pipeline
│   ├── adlens_pipeline.py    # main entry point (steps 3–4)
│   ├── ensemble.py           # classifier prompts + standalone runner
│   ├── misleading_design.py  # CTA/undisclosed-advertiser detection
│   ├── detect_misconfigured.py
│   ├── llm_judge.py          # disagreement judge (standalone or via pipeline)
│   └── translate_ocr.py      # step 3 — OCR translation
├── ocr/              # PaddleOCR text extraction pipeline
├── plots/            # Paper figure scripts
├── sample_data/      # Labeled ad dataset (images + metadata)
├── scripts/          # Utility scripts (PDNS lookups)
├── search_platform/  # Gradio semantic search UI
├── sql_queries/      # BigQuery SQL queries and Python client
└── utils/            # Shared Python utilities
```

## Violation Categories

| Category | Description |
|---|---|
| `scareware` | Assertive threat / panic-inducing claims |
| `deceptive_claim` | False device-state, fake recovery, financial bait |
| `misleading_design` | CTA-only ads with no identifiable advertiser |
| `misconfigured` | Blank, QR-code, or otherwise broken ad creatives |
