# Detection

Ensemble VLM classification pipeline for three ad violation categories. Two classifier models independently label each ad; a larger judge model resolves disagreements via majority-vote ensemble.

## Scripts

| Script | Purpose |
|---|---|
| `adlens_pipeline.py` | Main entry point — runs classify + judge phases end-to-end |
| `ensemble.py` | Scareware & deceptive-claim prompts; standalone runner with embedding-based ranking — evaluates all three tested models, of which 2 are selected |
| `misleading_design.py` | Misleading design detection (CTA-only / undisclosed advertiser) |
| `detect_misconfigured.py` | Standalone utility for misconfigured ad detection (blank, QR-code, broken creatives) — not part of the pipeline |
| `llm_judge.py` | LLM judge for classifier disagreements — evaluates three candidate judge models |
| `translate_ocr.py` | OCR translation cache builder using `translategemma:4b` via Ollama |

## Requirements

- Python 3.12
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- Models pulled:

```bash
# Classifiers
ollama pull qwen3.5:9b
ollama pull gemma3:12b

# Judge models (pull whichever you use)
ollama pull gemma4:26b
ollama pull qwen3.5:27b
ollama pull mistral-small3.2:24b
```

- `sample_data/metadata.json` and images in `sample_data/images/<ad_id>.png`

## Usage

### adlens_pipeline.py

```bash
cd detection

# Full pipeline — all three violations
python adlens_pipeline.py

# Specific violations only
python adlens_pipeline.py --violations scareware deceptive_claim

# Skip classification, re-run judge only (needs existing cache)
python adlens_pipeline.py --skip-classify

# Cap to 50 total records (quick test)
python adlens_pipeline.py --limit 50

# Text-only mode (no images passed to models)
python adlens_pipeline.py --no-images

# Custom Ollama URL or judge model
python adlens_pipeline.py --ollama-url http://localhost:11434 --judge-model qwen3.5:27b
```

### misleading_design.py

```bash
# Full run via Ollama
python misleading_design.py --ollama --vlm-model gemma3:12b

# Quick test — cap candidates
python misleading_design.py --ollama --vlm-model gemma3:12b --limit 20

# Dry-run — OCR filter + keyword matching only, no VLM
python misleading_design.py --no-vlm
```

### ensemble.py

```bash
# Full run on GPU (embedding rank + 3-model LLM ensemble)
python ensemble.py --device cuda --skip-translate

# Single violation, top-50 candidates
python ensemble.py --violations scareware --top-n 50 --device cuda --skip-translate

# Ranking only, skip LLM
python ensemble.py --skip-llm --device cuda --skip-translate
```

### llm_judge.py

```bash
# Judge scareware disagreements with all three judge models
python llm_judge.py --violation scareware --judge-model qwen3.5:27b
python llm_judge.py --violation scareware --judge-model gemma4:26b
python llm_judge.py --violation scareware --judge-model mistral-small3.2:24b

# Judge misleading_design disagreements (reads from results/misleading_design/)
python llm_judge.py --violation misleading_design --judge-model gemma4:26b

# Judge a single ad by ID
python llm_judge.py --violation scareware --crid CR00102789777257922561-v0
```

### translate_ocr.py

```bash
# Translate all ads (appends to cached_translations.json)
python translate_ocr.py

# Quick test — first 10 ads only
python translate_ocr.py --limit 10
```

## Violation Types

| Violation | Positive label | Classifier models |
|---|---|---|
| `scareware` | `SCAREWARE` | qwen3.5:9b, gemma3:12b |
| `deceptive_claim` | `MISLEADING` | qwen3.5:9b, gemma3:12b |
| `misleading_design` | `Undisclosed Advertiser` | qwen3.5:9b, gemma3:12b |

## Output

All output is written to `results/pipeline_results/` (override with `--out-dir`):

| File | Contents |
|---|---|
| `classify_cache.json` | Per-(ad_id, violation, model) classification cache |
| `scareware_results.json` | Records with per-model, ensemble, and judge labels |
| `deceptive_claim_results.json` | Same for deceptive-claim violation |
| `misleading_design_results.json` | Same for misleading-design violation |
| `judge_cache_<violation>.json` | Judge responses keyed by ad_id |
| `latency_calls.json` | Per-call timing log (wall/load/eval ms, token counts) |

The pipeline is fully resumable — all intermediate results are cached to disk.

## Note on `ensemble.py`

`adlens_pipeline.py` imports only the prompt strings (`PROMPT_SCAREWARE`, `PROMPT_MISLEADING`) from `ensemble.py` and classifies all records directly. It does **not** use the embedding-based cosine-similarity ranking that `ensemble.py` implements as a standalone script. When run standalone, `ensemble.py` first ranks all ads by semantic similarity to reference statements, then runs the LLM classifiers on the top-N candidates — a two-stage approach used during the original large-scale measurement. `adlens_pipeline.py` skips the ranking step and classifies the full labeled dataset directly.
