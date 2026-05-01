"""
sim_search_webui.py — Semantic search over sample ad images.

Similarity is computed between the text query and each ad's translated OCR text
using a SentenceTransformer embedding model. Embeddings are cached per model.

Usage:
  python sim_search_webui.py                        # launch UI (default model: gemma)
  python sim_search_webui.py --model baai           # pre-select model in UI
  python sim_search_webui.py --build                # build cache for default model
  python sim_search_webui.py --model baai --build   # build cache for a specific model

Available models:
  gemma   → google/embeddinggemma-300m
  baai    → BAAI/bge-m3
  minilm  → sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import json
import sys
from pathlib import Path

import gradio as gr
import torch
from sentence_transformers import SentenceTransformer, util

# ── Paths ───────────────────────────────────────────────────────────────────────
SAMPLE_DATA_DIR   = Path(__file__).parent.parent / "sample_data"
IMAGES_DIR        = SAMPLE_DATA_DIR / "images"
METADATA_PATH     = SAMPLE_DATA_DIR / "metadata.json"
TRANSLATIONS_PATH = SAMPLE_DATA_DIR / "cached_translations.json"
EMBEDDINGS_DIR    = Path(__file__).parent / "embeddings_cache"
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# ── Model registry ────────────────────────────────────────────────────────────
MODELS: dict[str, str] = {
    "gemma":  "google/embeddinggemma-300m",
    "baai":   "BAAI/bge-m3",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}
DEFAULT_MODEL = "gemma"
DEFAULT_QUERY = "Your device is severely damaged with viruses"

# ── Load metadata & translations ─────────────────────────────────────────────
print("📖 Loading metadata...")
_raw_meta: list = json.loads(METADATA_PATH.read_text())
METADATA: dict[str, dict] = {r["ad_id"]: r for r in _raw_meta}

print("📖 Loading translation cache...")
TRANSLATIONS: dict[str, str] = json.loads(TRANSLATIONS_PATH.read_text())


def resolve_translation(record: dict) -> str:
    translated = record.get("translated_ocr_text", "")
    if translated:
        return translated
    ocr = record.get("ocr_text", "")
    return TRANSLATIONS.get(ocr, ocr)


def cache_path(model_name: str) -> Path:
    return EMBEDDINGS_DIR / f"{model_name}.pt"


def has_cache(model_name: str) -> bool:
    return cache_path(model_name).exists()


# ── Model & embedding cache ───────────────────────────────────────────────────
_models: dict[str, SentenceTransformer] = {}
_embeddings: dict[str, dict] = {}
_hf_token: str | None = None


def get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _models:
        model_id = MODELS[model_name]
        print(f"📥 Loading model '{model_name}' ({model_id})...")
        _models[model_name] = SentenceTransformer(model_id, token=_hf_token)
    return _models[model_name]


def build_embeddings(model_name: str) -> dict:
    records = [
        r for r in _raw_meta
        if (IMAGES_DIR / f"{r['ad_id']}.png").exists()
    ]

    model = get_model(model_name)
    ids, texts = [], []
    for r in records:
        text = resolve_translation(r)
        ids.append(r["ad_id"])
        texts.append(text if text.strip() else r["ad_id"])

    print(f"  Encoding {len(texts)} texts with '{model_name}'...")
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    data = {"ids": ids, "embeddings": embeddings}

    torch.save(data, cache_path(model_name))
    print(f"  ✅ Cached → {cache_path(model_name)}")
    return data


def get_embeddings(model_name: str) -> dict:
    if model_name in _embeddings:
        return _embeddings[model_name]
    path = cache_path(model_name)
    if path.exists():
        print(f"📂 Loading cached embeddings: {model_name}")
        data = torch.load(path, map_location="cpu", weights_only=False)
    else:
        data = build_embeddings(model_name)
    _embeddings[model_name] = data
    return data


# ── Helpers ───────────────────────────────────────────────────────────────────

def missing_cache_html(model_name: str) -> str:
    cmd = f"python sim_search_webui.py --model {model_name} --build"
    return (
        f'<div class="cache-warn">'
        f'⚠️ No embeddings cache for <b>{model_name}</b>.<br>'
        f'Generate it first:<br><code>{cmd}</code>'
        f'</div>'
    )


# ── Search ────────────────────────────────────────────────────────────────────

def run_query(query: str, model_name: str, k: int, offset: int) -> list[str]:
    model    = get_model(model_name)
    emb_data = get_embeddings(model_name)

    if not emb_data["ids"]:
        return []

    query_emb    = model.encode(query, convert_to_tensor=True)
    scores       = util.cos_sim(query_emb, emb_data["embeddings"])[0]
    sorted_idxs  = torch.argsort(scores, descending=True).tolist()
    selected     = sorted_idxs[offset: offset + k]

    cards = []
    for idx in selected:
        ad_id    = emb_data["ids"][idx]
        record   = METADATA.get(ad_id, {})
        score    = float(scores[idx])

        img_path = IMAGES_DIR / f"{ad_id}.png"
        img_src  = f"/gradio_api/file={img_path.resolve()}"

        creative_url = record.get("creativeURL", "")
        crid         = record.get("crid", ad_id)
        advertiser   = record.get("advertiserName", "")
        ocr_text     = record.get("ocr_text", "")
        translated   = resolve_translation(record)
        violation    = record.get("violation_type", "")
        label        = record.get("label", "").upper()

        crid_html = (
            f'<a href="{creative_url}" target="_blank" class="crid-link">{crid}</a>'
            if creative_url else crid
        )
        advertiser_html = (
            f'<div class="advertiser"><b>Advertiser:</b> {advertiser}</div>'
            if advertiser else ""
        )
        translation_html = (
            f'<div class="translation"><b>Translation:</b> {translated}</div>'
            if translated and translated != ocr_text else ""
        )
        label_cls   = "label-tp" if label == "TP" else "label-tn"
        badge_text  = f"{violation} / {label}" if violation else label
        badge_html  = f'<span class="{label_cls}">{badge_text}</span>' if badge_text else ""

        cards.append(f'''
        <div class="card">
            <img src="{img_src}" />
            <div class="meta">
                <div class="top-row">
                    {badge_html}
                    <span class="score">Sim: {score:.4f}</span>
                </div>
                <div class="crid">{crid_html}</div>
                {advertiser_html}
                <div class="ocr"><b>OCR:</b> {ocr_text}</div>
                {translation_html}
            </div>
        </div>''')

    return cards


def render_grid(cards: list[str]) -> str:
    return f'<div class="grid">{"".join(cards)}</div>'


# ── Handlers ──────────────────────────────────────────────────────────────────

_last_query: str | None = None


def on_model_change(model_name: str):
    if not has_cache(model_name):
        return gr.update(value=missing_cache_html(model_name), visible=True)
    return gr.update(value="", visible=False)


def search_reset(query: str, model_name: str, k: int):
    global _last_query
    if not query.strip():
        return "", 0, []
    if not has_cache(model_name):
        return missing_cache_html(model_name), 0, []
    _last_query = query
    cards = run_query(query, model_name, k, 0)
    return render_grid(cards), k, cards


def search_append(query: str, model_name: str, k: int, offset: int, current_cards: list):
    global _last_query
    if not _last_query or not has_cache(model_name):
        return render_grid(current_cards), offset, current_cards
    new_cards = run_query(_last_query, model_name, k, offset)
    updated = current_cards + new_cards
    return render_grid(updated), offset + k, updated


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.card { border: 1px solid #ddd; border-radius: 10px; overflow: hidden;
        background: white; padding: 8px; }
.card * { color: #000 !important; }
.card img { width: 100%; height: 260px; object-fit: contain; background: #f5f5f5; }
.meta { font-size: 12px; margin-top: 6px; }
.top-row { display: flex; justify-content: space-between; align-items: center;
           margin-bottom: 4px; }
.score { font-size: 11px; color: #555 !important; }
.crid { font-size: 10px; word-break: break-all; margin-bottom: 3px; }
.crid-link { color: #1a73e8 !important; text-decoration: underline; }
.advertiser { font-size: 11px; margin-bottom: 3px; }
.ocr { margin-top: 5px; font-size: 11px; background: #fafafa;
       white-space: pre-wrap; word-break: break-word; }
.translation { margin-top: 4px; font-size: 11px; background: #f0f7ff;
               border-left: 3px solid #4a90d9; padding-left: 4px;
               white-space: pre-wrap; word-break: break-word; }
.label-tp { font-size: 11px; font-weight: bold; background: #fde8e8;
             padding: 1px 6px; border-radius: 4px; border: 1px solid #e57373; }
.label-tn { font-size: 11px; font-weight: bold; background: #e0f0e0;
             padding: 1px 6px; border-radius: 4px; border: 1px solid #66bb6a; }
.cache-warn { padding: 12px 16px; background: #b71c1c; border-left: 4px solid #7f0000;
              border-radius: 4px; font-size: 13px; margin: 8px 0; color: #fff !important; }
.cache-warn * { color: #fff !important; }
.cache-warn code { background: rgba(0,0,0,0.25); padding: 2px 8px; border-radius: 3px;
                   font-family: monospace; color: #ffe082 !important; }
"""


def build_ui(initial_model: str) -> gr.Blocks:
    with gr.Blocks(title="Ad Similarity Search", css=CSS) as demo:
        gallery_state = gr.State([])
        state_offset  = gr.State(0)

        gr.Markdown("## Ad Similarity Search — Sample Data")

        with gr.Row():
            query_input = gr.Textbox(
                label="Semantic Search Query", value=DEFAULT_QUERY, scale=3
            )
            search_btn = gr.Button("Search", variant="primary", scale=1)

        with gr.Row():
            model_drop = gr.Dropdown(
                list(MODELS.keys()),
                value=initial_model,
                label="Embedding Model",
                info="Only models with a cached embeddings file are searchable.",
            )
            n_results = gr.Dropdown([5, 10, 25, 50, 100], value=25, label="Results per page")

        cache_warn = gr.HTML(
            value=missing_cache_html(initial_model) if not has_cache(initial_model) else "",
            visible=not has_cache(initial_model),
        )

        gallery       = gr.HTML(label="Results")
        load_more_btn = gr.Button("Load More")

        search_inputs = [query_input, model_drop, n_results]
        query_input.submit(search_reset, search_inputs, [gallery, state_offset, gallery_state])
        search_btn.click(search_reset,  search_inputs, [gallery, state_offset, gallery_state])
        load_more_btn.click(
            search_append,
            inputs=[query_input, model_drop, n_results, state_offset, gallery_state],
            outputs=[gallery, state_offset, gallery_state],
        )
        model_drop.change(on_model_change, inputs=[model_drop], outputs=[cache_warn])

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    model_list = ", ".join(f"{k} ({v})" for k, v in MODELS.items())
    parser = argparse.ArgumentParser(
        description="Ad similarity search WebUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available models:\n  {model_list}",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Embedding model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build (or rebuild) the embeddings cache for --model, then exit",
    )
    parser.add_argument(
        "--hf-token",
        metavar="TOKEN",
        default=None,
        help="HuggingFace API token for gated/private models",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _hf_token = args.hf_token

    if args.build:
        print(f"⚙️  Building embeddings cache for '{args.model}'...")
        build_embeddings(args.model)
        print("Done.")
        sys.exit(0)

    missing = [m for m in MODELS if not has_cache(m)]
    if missing:
        cmds = "\n".join(
            f"  python sim_search_webui.py --model {m} --build" for m in missing
        )
        print(f"⚠️  Missing caches for: {', '.join(missing)}\nBuild them with:\n{cmds}")

    allowed = [str(IMAGES_DIR.resolve())]
    gr.set_static_paths(allowed)
    build_ui(args.model).launch(allowed_paths=allowed, share=True)
