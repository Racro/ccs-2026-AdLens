# Search Platform

<video src="search_platform.mp4" controls width="100%"></video>

**Demo:** A demonstration of the search platform over the complete crawled dataset.

Gradio web UI for semantic similarity search over the ad corpus. Queries are embedded with a SentenceTransformer model and ranked against pre-computed ad OCR embeddings.

## Requirements

- Python 3.12
- `gradio`, `sentence-transformers`, `torch` (included in top-level `requirements.txt`)
- `sample_data/metadata.json` and `sample_data/cached_translations.json`

## Usage

```bash
# Launch UI with default model (google/embeddinggemma-300m)
python search_platform/sim_search_webui.py

# Pre-select a different embedding model
python search_platform/sim_search_webui.py --model baai

# Pre-build the embedding cache without launching the UI
python search_platform/sim_search_webui.py --build
python search_platform/sim_search_webui.py --model baai --build
```

Open `http://localhost:7860` in your browser after launch.

## Available Models

| Key | Model |
|---|---|
| `gemma` (default) | `google/embeddinggemma-300m` |
| `baai` | `BAAI/bge-m3` |
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` |

## Embedding Cache

Embeddings are cached per model under `embeddings_cache/` so subsequent launches skip recomputation. Delete the cache directory to force a rebuild.
