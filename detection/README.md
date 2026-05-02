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
ollama pull qwen3.5:9b
ollama pull gemma3:12b
ollama pull gemma4:26b    # judge model
```

- `sample_data/metadata.json` and images in `sample_data/images/<ad_id>.png`

## Usage

```bash
cd detection

# Full pipeline — all three violations
python adlens_pipeline.py

# Specific violations only
python adlens_pipeline.py --violations scareware deceptive_claim

# Skip classification, re-run judge only (needs existing cache)
python adlens_pipeline.py --skip-classify

# Cap to 50 records per label per violation (quick test)
python adlens_pipeline.py --limit 50

# Text-only mode (no images passed to models)
python adlens_pipeline.py --no-images

# Custom Ollama URL or judge model
python adlens_pipeline.py --ollama-url http://localhost:11434 --judge-model qwen3.5:27b
```

## Violation Types

| Violation | Positive label | Classifier models |
|---|---|---|
| `scareware` | `SCAREWARE` | qwen3.5:9b, gemma3:12b |
| `deceptive_claim` | `DECEPTIVE_CLAIM` | qwen3.5:9b, gemma3:12b |
| `misleading_design` | `Undisclosed Advertiser` | qwen3.5:9b, gemma3:12b |

## Output

All output is written to `results/pipeline_results/` (override with `--out-dir`):

| File | Contents |
|---|---|
| `classify_cache.json` | Per-(ad_id, violation, model) classification cache |
| `scareware_results.json` | Records with per-model, ensemble, and judge labels |
| `misleading_results.json` | Same for deceptive-claim violation |
| `ad_design_results.json` | Same for misleading-design violation |
| `judge_cache_<violation>.json` | Judge responses keyed by ad_id |
| `latency_calls.json` | Per-call timing log (wall/load/eval ms, token counts) |

The pipeline is fully resumable — all intermediate results are cached to disk.

## Note on `ensemble.py`

`adlens_pipeline.py` imports only the prompt strings (`PROMPT_SCAREWARE`, `PROMPT_MISLEADING`) from `ensemble.py` and classifies all records directly. It does **not** use the embedding-based cosine-similarity ranking that `ensemble.py` implements as a standalone script. When run standalone, `ensemble.py` first ranks all ads by semantic similarity to reference statements, then runs the LLM classifiers on the top-N candidates — a two-stage approach used during the original large-scale measurement. `adlens_pipeline.py` skips the ranking step and classifies the full labeled dataset directly.
