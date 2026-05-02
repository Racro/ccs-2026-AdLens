"""
detect_misconfigured.py — Misconfigured ad detection via OCR absence + VLM classification

Pipeline (3 steps, each runnable independently):

  Step 1  filter  — Short-char OCR filter
                    Load all ads from 3 metadata files (computer / mobile / software).
                    Apply cached translations. Keep only ads with translated OCR < max_chars.
                    Saves: candidates_filter.json

  Step 2  phash   — pHash deduplication
                    Remove visually identical images (Hamming distance ≤ phash_threshold)
                    before expensive VLM calls.
                    Reads: candidates_filter.json
                    Saves: candidates_phash.json
                    → Feed candidates_phash.json to reocr_retranslate.py to refresh OCR
                      before running the vlm step.

  Step 3  vlm     — VLM classification (Qwen3.5-9B)
                    Classify candidates into: BLANK, QR_CODE, MISCONFIGURED, SAFE.
                    Reads: --vlm-input (default: candidates_phash.json)
                    Saves: all.json, blank.json, qr_code.json, misconfigured.json, safe.json

Typical workflow:
  python detect_misconfigured.py --step filter
  python detect_misconfigured.py --step phash
  python reocr_retranslate.py --candidates results_misconfigured/candidates_phash.json
  python detect_misconfigured.py --step filter   # re-run after metadata update
  python detect_misconfigured.py --step phash
  python detect_misconfigured.py --step vlm

  # Or skip reocr and go straight to VLM:
  python detect_misconfigured.py --step filter phash vlm

Options:
  --step            Steps to run: filter phash vlm (default: all three)
  --max-chars       Max translated OCR char count to include (default: 3)
  --phash-threshold Hamming distance threshold for pHash dedup (default: 2)
  --vlm-input       Input file for vlm step (default: candidates_phash.json)
  --vlm-model       HF model name (default: Qwen/Qwen3.5-9B)
  --hf-token        HuggingFace token for gated models
  --device          cuda | cpu (default: cuda if available)
  --out-dir         Output directory (default: ./results_misconfigured)
  --translations    Path to cached_translations.json
  --limit           Cap images processed (test runs)
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR            = Path(__file__).parent
SAMPLE_DATA_DIR       = SCRIPT_DIR.parent / "sample_data"

METADATA_FILE         = SAMPLE_DATA_DIR / "metadata.json"
IMAGE_DIR             = SAMPLE_DATA_DIR / "images"
DEFAULT_TRANSLATIONS  = SAMPLE_DATA_DIR / "cached_translations.json"
PHASH_CACHE_DIR       = SCRIPT_DIR / "embeddings_cache"

# ── defaults ───────────────────────────────────────────────────────────────────

MAX_OCR_CHARS         = 3
PHASH_DEDUP_THRESHOLD = 2

ALL_STEPS = ["filter", "phash", "vlm"]

# intermediate file names (relative to --out-dir)
FILE_FILTER = "candidates_filter.json"
FILE_PHASH  = "candidates_phash.json"

VALID_LABELS = {"BLANK", "QR_CODE", "MISCONFIGURED", "SAFE"}

# ── VLM prompt ─────────────────────────────────────────────────────────────────

VLM_PROMPT_MISCONFIGURED = """
Role: You are an online advertising quality analyst specialising in technically
broken or misconfigured ad creatives.

Task: Classify the provided advertisement image into exactly one of four labels:
BLANK, QR_CODE, MISCONFIGURED, or SAFE.

Classification Logic:

* BLANK — the ad is completely plain with no visible content whatsoever.
    - A solid white, grey, or black rectangle with nothing on it.
    - A near-blank image with only a tiny artifact (a single pixel, thin line,
      or barely visible dot) — still counts as BLANK.

* QR_CODE — the ad contains a QR code as its primary or only content.
    - A QR code on a blank or near-blank background with no readable text,
      brand name, or other visual content explaining what it links to.
    - A QR code with only a small centred logo but no surrounding context.

* MISCONFIGURED — the ad attempted to render but failed technically. Look for:
    - Broken or partially loaded image: grey placeholder box, broken image icon,
      a single coloured stripe, or a cropped fragment of an image.
    - Garbled text: squares, question marks, mojibake, corrupted character
      encoding, or unreadable symbol sequences where text should be.
    - Store badge only: only an app store badge (Google Play / App Store) on a
      white background with no product context loaded.
    - Webpage error: a browser or server error message in place of ad content.

* SAFE — the ad rendered correctly and contains intentional, visible content.
    - Text is readable and present.
    - Images are fully loaded even if the design is simple.
    - Content is clearly intentional, not the result of a technical failure.

Constraint Rules:
1. If the image is entirely empty or a solid colour with no content, it is BLANK.
2. If a QR code is present alongside readable brand text or a product description,
   classify as SAFE — the QR code is supplementary, not the whole ad.
3. A very simple but fully rendered ad (e.g. a logo on a plain background with
   a button) is SAFE — poor design is not misconfiguration.
4. An ad displaying a YouTube or video player thumbnail with a visible play button
   is SAFE — it is a correctly rendered video ad, not a misconfiguration.
5. If "label" is "SAFE", the "reason" field MUST be an empty string "".

Output format (strict JSON only, no other text):
{
  "label": "BLANK | QR_CODE | MISCONFIGURED | SAFE",
  "reason": "One concise sentence describing the failure type, or empty if SAFE."
}
"""

# ── data structures ────────────────────────────────────────────────────────────

@dataclass
class AdCandidate:
    ad_id:            str
    image_path:       Path
    dataset_category: str
    original_ocr:     str   # translated OCR from the filter step


@dataclass
class AdResult:
    ad_id:            str
    image_path:       str
    dataset_category: str
    original_ocr:     str
    flagged:          bool
    vlm_label:        Optional[str] = None
    vlm_reason:       Optional[str] = None


# ── candidate serialisation ────────────────────────────────────────────────────

def candidates_to_dicts(candidates: list[AdCandidate]) -> list[dict]:
    """Serialise candidates. Uses 'original_ocr' — compatible with reocr_retranslate.py."""
    return [
        {
            "ad_id":            c.ad_id,
            "image_path":       str(c.image_path),
            "dataset_category": c.dataset_category,
            "original_ocr":     c.original_ocr,
        }
        for c in candidates
    ]


def candidates_from_dicts(data: list[dict]) -> list[AdCandidate]:
    return [
        AdCandidate(
            ad_id            = d["ad_id"],
            image_path       = Path(d["image_path"]),
            dataset_category = d["dataset_category"],
            original_ocr     = d.get("original_ocr") or d.get("translated_ocr", ""),
        )
        for d in data
    ]


# ── I/O helpers ────────────────────────────────────────────────────────────────

def save_json(path: Path, data: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("Saved → %s", path)


def load_json(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected intermediate file not found: {path}\n"
            f"Run the preceding step first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Step 1: OCR filter ─────────────────────────────────────────────────────────

def load_translations(path: Path) -> dict[str, str]:
    if not path.exists():
        log.warning("Translations file not found: %s — using raw OCR", path)
        return {}
    with open(path, encoding="utf-8") as f:
        t = json.load(f)
    log.info("Loaded %d cached translations from %s", len(t), path)
    return t


def resolve_image_path(remote_path: str, image_dir: Optional[Path] = None) -> Optional[Path]:
    base = image_dir if image_dir is not None else IMAGE_DIR
    p = base / Path(remote_path).name
    return p if p.exists() else None


def run_filter(
    translations: dict[str, str],
    max_chars: int = MAX_OCR_CHARS,
    limit: Optional[int] = None,
    image_dir: Optional[Path] = None,
) -> list[AdCandidate]:
    """Return all ads whose translated OCR is shorter than max_chars characters."""
    candidates: list[AdCandidate] = []

    if not METADATA_FILE.exists():
        log.warning("Metadata not found: %s", METADATA_FILE)
        return candidates

    with open(METADATA_FILE, encoding="utf-8") as f:
        items = json.load(f)

    cat_count = 0
    for item in items:
        ad_id      = item["ad_id"]
        raw_ocr    = item.get("ocr_text") or ""
        translated = item.get("translated_ocr_text") or translations.get(raw_ocr) or raw_ocr

        if len(translated.strip()) >= max_chars:
            continue

        img_path = (image_dir or IMAGE_DIR) / f"{ad_id}.png"
        if not img_path.exists():
            continue
        candidates.append(AdCandidate(
            ad_id            = ad_id,
            image_path       = img_path,
            dataset_category = item.get("category", "ads"),
            original_ocr     = translated,
        ))
        cat_count += 1

        if limit and len(candidates) >= limit:
            log.info("--limit reached (%d)", limit)
            return candidates

    log.info("Total candidates after filter: %d (OCR < %d chars)", len(candidates), max_chars)
    return candidates


# ── Step 2: pHash dedup ────────────────────────────────────────────────────────

def load_phash_index() -> dict[str, int]:
    index: dict[str, int] = {}
    for category in ["mobile", "computer", "software"]:
        path = PHASH_CACHE_DIR / f"{category}_phash.pt"
        if not path.exists():
            log.warning("pHash cache missing: %s — skipping dedup for %s", path, category)
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        for ad_id, h in zip(data["ids"], data["hashes"].tolist()):
            index[ad_id] = h
    log.info("pHash index loaded: %d entries", len(index))
    return index


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def run_phash(
    candidates: list[AdCandidate],
    phash_index: dict[str, int],
    threshold: int = PHASH_DEDUP_THRESHOLD,
) -> list[AdCandidate]:
    accepted_hashes: list[int] = []
    deduped: list[AdCandidate] = []
    skipped = 0

    for c in candidates:
        h = phash_index.get(c.ad_id)
        if h is None:
            deduped.append(c)
            continue
        if any(hamming(h, seen) <= threshold for seen in accepted_hashes):
            skipped += 1
            continue
        accepted_hashes.append(h)
        deduped.append(c)

    log.info(
        "pHash dedup (threshold=%d): %d → %d (%d duplicates removed)",
        threshold, len(candidates), len(deduped), skipped,
    )
    return deduped


# ── Step 3: VLM ────────────────────────────────────────────────────────────────

VLM_MODEL_NAME = "Qwen/Qwen3.5-9B"

MODEL_ALIASES: dict[str, str] = {
    "qwen":   "Qwen/Qwen3.5-9B",
    "glm":    "zai-org/GLM-4.6V-Flash",
    "gemma3": "google/gemma-3-12b-it",
}

_KNOWN_VL_MODELS: set[str] = {"Qwen/Qwen3.5-9B"}
_GLM_MODELS:      set[str] = {"zai-org/GLM-4.6V-Flash"}
_GEMMA3_MODELS:   set[str] = {"google/gemma-3-12b-it"}

_vlm_model_obj     = None
_vlm_processor_obj = None


def _model_tag(model_name: str) -> str:
    return model_name.split("/")[-1].lower().replace(".", "").replace("-", "_")


def _is_vl_model(name: str) -> bool:
    return "VL" in name or "vl" in name or name in _KNOWN_VL_MODELS


def load_vlm(model_name: str = VLM_MODEL_NAME, token: Optional[str] = None):
    global _vlm_model_obj, _vlm_processor_obj
    if _vlm_model_obj is not None:
        return _vlm_model_obj, _vlm_processor_obj

    log.info("Loading VLM: %s", model_name)

    if model_name in _GLM_MODELS:
        from transformers import AutoProcessor, Glm4vForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_name)
        model = Glm4vForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto",
        )

    elif model_name in _GEMMA3_MODELS:
        from transformers import AutoModelForCausalLM, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )

    else:  # Qwen / generic VL
        from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
        if _is_vl_model(model_name):
            cfg        = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model_type = getattr(cfg, "model_type", "")
            log.info("VL model detected (model_type=%s)", model_type)
            processor  = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            import transformers as _tf
            config_cls_name = type(cfg).__name__
            model_cls_name  = config_cls_name.replace("Config", "ForConditionalGeneration")
            ModelCls = getattr(_tf, model_cls_name, None)
            if ModelCls is None:
                log.warning("%s not found; falling back to Qwen2_5_VLForConditionalGeneration", model_cls_name)
                from transformers import Qwen2_5_VLForConditionalGeneration
                ModelCls = Qwen2_5_VLForConditionalGeneration
        else:
            processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            ModelCls  = AutoModelForCausalLM
        model = ModelCls.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
        )

    model.eval()
    _vlm_model_obj     = model
    _vlm_processor_obj = processor
    return model, processor


def _parse_json_response(text: str) -> dict | None:
    match = re.search(r'\{[^{}]*"label"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def vlm_classify(
    model,
    processor,
    image_path: Path,
    model_name: str = VLM_MODEL_NAME,
    retries: int = 2,
) -> dict:
    # ── GLM-4.6V-Flash ───────────────────────────────────────────────────────
    if model_name in _GLM_MODELS:
        try:
            mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            image_url = f"data:{mime};base64,{b64}"
        except Exception as e:
            log.warning("GLM: failed to encode image %s: %s — text-only.", image_path, e)
            image_url = None

        user_content = (
            [
                {"type": "image", "url": image_url},
                {"type": "text", "text": f"{VLM_PROMPT_MISCONFIGURED}\n\nAnalyse this advertisement image and classify it."},
            ]
            if image_url else
            [{"type": "text", "text": f"{VLM_PROMPT_MISCONFIGURED}\n\nAnalyse this advertisement and classify it."}]
        )
        messages = [{"role": "user", "content": user_content}]

        raw_inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        )
        inputs = (raw_inputs.to(model.device) if hasattr(raw_inputs, "to")
                  else {k: v.to(model.device) if hasattr(v, "to") else v for k, v in raw_inputs.items()})
        inputs.pop("token_type_ids", None)
        input_len = inputs["input_ids"].shape[1]

        for attempt in range(retries + 1):
            try:
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
                raw_out   = processor.decode(output_ids[0][input_len:], skip_special_tokens=False).strip()
                raw_clean = re.sub(r"<think>.*?</think>", "", raw_out, flags=re.DOTALL).strip()
                result    = _parse_json_response(raw_clean)
                if result and result.get("label") in VALID_LABELS:
                    return {"label": result["label"], "reason": result.get("reason", ""),
                            "flagged": result["label"] != "SAFE"}
                log.warning("GLM attempt %d: bad response — %s", attempt + 1, raw_clean[:200])
            except Exception as e:
                log.warning("GLM error attempt %d: %s", attempt + 1, e, exc_info=True)
        return {"label": "SAFE", "reason": "", "flagged": False}

    # ── Gemma-3 ───────────────────────────────────────────────────────────────
    if model_name in _GEMMA3_MODELS:
        pil_image = None
        if image_path is not None:
            try:
                pil_image = Image.open(str(image_path)).convert("RGB")
            except Exception as e:
                log.warning("Gemma3: failed to open image %s: %s — text-only.", image_path, e)

        user_text = f"{VLM_PROMPT_MISCONFIGURED}\n\nAnalyse this advertisement image and classify it."
        content   = ([{"type": "image", "image": pil_image}, {"type": "text", "text": user_text}]
                     if pil_image else [{"type": "text", "text": user_text}])
        messages  = [{"role": "user", "content": content}]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        input_len = inputs["input_ids"].shape[1]

        for attempt in range(retries + 1):
            try:
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                decoded   = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
                raw_clean = decoded.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                result    = _parse_json_response(raw_clean)
                if result and result.get("label") in VALID_LABELS:
                    return {"label": result["label"], "reason": result.get("reason", ""),
                            "flagged": result["label"] != "SAFE"}
                for lbl in VALID_LABELS:
                    if re.search(rf'\b{lbl}\b', decoded, re.IGNORECASE):
                        log.warning("Gemma3 attempt %d: extracted label '%s' from reasoning text", attempt + 1, lbl)
                        return {"label": lbl, "reason": "", "flagged": lbl != "SAFE"}
                log.warning("Gemma3 attempt %d: bad response — %s", attempt + 1, decoded[:200])
            except Exception as e:
                log.warning("Gemma3 error attempt %d: %s", attempt + 1, e, exc_info=True)
        return {"label": "SAFE", "reason": "", "flagged": False}

    # ── Qwen / generic ────────────────────────────────────────────────────────
    is_vl = _is_vl_model(model_name)

    if is_vl:
        user_content = [
            {"type": "image", "image": str(image_path)},
            {"type": "text",  "text": "Analyse this advertisement image and classify it."},
        ]
    else:
        user_content = "Analyse this advertisement and classify it."

    messages = [
        {"role": "system", "content": VLM_PROMPT_MISCONFIGURED},
        {"role": "user",   "content": user_content},
    ]

    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    if is_vl:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)
    else:
        model_inputs = processor(text=[text], return_tensors="pt").to(model.device)

    input_len = model_inputs.input_ids.shape[1]

    for attempt in range(retries + 1):
        try:
            with torch.no_grad():
                output_ids = model.generate(**model_inputs, max_new_tokens=200, do_sample=False)
            raw    = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
            result = _parse_json_response(raw)
            if result and result.get("label") in VALID_LABELS:
                return {
                    "label":   result["label"],
                    "reason":  result.get("reason", ""),
                    "flagged": result["label"] != "SAFE",
                }
            log.warning("Attempt %d: unexpected response — %s", attempt + 1, raw[:200])
        except Exception as e:
            log.warning("VLM error attempt %d: %s", attempt + 1, e, exc_info=True)

    return {"label": "SAFE", "reason": "", "flagged": False}


def run_vlm(candidates: list[AdCandidate], args) -> list[AdResult]:
    vlm_model_name = getattr(args, "vlm_model", VLM_MODEL_NAME)

    cache_path = args.out_dir / f"llm_cache_{_model_tag(vlm_model_name)}.json"
    llm_cache = load_llm_cache(cache_path)
    cache_hits = 0

    vlm_model, vlm_proc = load_vlm(model_name=vlm_model_name, token=args.hf_token)

    results: list[AdResult] = []
    for idx, cand in enumerate(candidates, 1):
        if idx % 100 == 0 or idx == 1:
            log.info("VLM progress: %d / %d", idx, len(candidates))

        if cand.ad_id in llm_cache:
            cached = llm_cache[cand.ad_id]
            out = {"label": cached["label"], "reason": cached["reason"],
                   "flagged": cached["label"] not in ("SAFE",)}
            cache_hits += 1
        else:
            try:
                out = vlm_classify(vlm_model, vlm_proc, cand.image_path, model_name=vlm_model_name)
            except Exception as e:
                log.warning("VLM failed for %s: %s", cand.ad_id, e)
                out = {"label": "SAFE", "reason": "", "flagged": False}
            llm_cache[cand.ad_id] = {"label": out["label"], "reason": out["reason"]}
            save_llm_cache(llm_cache, cache_path)

        results.append(AdResult(
            ad_id            = cand.ad_id,
            image_path       = str(cand.image_path),
            dataset_category = cand.dataset_category,
            original_ocr     = cand.original_ocr,
            flagged          = out["flagged"],
            vlm_label        = out["label"],
            vlm_reason       = out["reason"],
        ))

    log.info("LLM cache: %d hits, %d new calls", cache_hits, len(candidates) - cache_hits)
    del vlm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ── LLM cache helpers ──────────────────────────────────────────────────────────

def load_llm_cache(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_llm_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ── output helpers ─────────────────────────────────────────────────────────────

def results_to_dicts(results: list[AdResult]) -> list[dict]:
    return [asdict(r) for r in results]


def print_summary(results: list[AdResult]):
    from collections import Counter
    label_counts = Counter(r.vlm_label for r in results)
    total = len(results)

    print(f"\n{'='*60}")
    print(f"SUMMARY — {total} candidates through VLM")
    for label in ("BLANK", "QR_CODE", "MISCONFIGURED", "SAFE", None):
        n = label_counts.get(label, 0)
        if n:
            print(f"  {str(label):<16} : {n}")
    print(f"{'='*60}")

    for label in ("BLANK", "QR_CODE", "MISCONFIGURED"):
        hits = [r for r in results if r.vlm_label == label][:5]
        if hits:
            print(f"\n  -- {label} samples --")
            for r in hits:
                reason = r.vlm_reason[:60] if r.vlm_reason else ""
                print(f"    {r.ad_id}  ocr={repr(r.original_ocr[:30])}  {reason}")


# ── arg parsing + main ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--step", nargs="+", default=ALL_STEPS,
                   choices=ALL_STEPS, metavar="STEP",
                   help="Steps to run: filter phash vlm (default: all three)")
    p.add_argument("--max-chars",       type=int,  default=MAX_OCR_CHARS,
                   help=f"Max translated OCR char count for filter step (default: {MAX_OCR_CHARS})")
    p.add_argument("--phash-threshold", type=int,  default=PHASH_DEDUP_THRESHOLD,
                   help=f"Hamming distance threshold for pHash dedup (default: {PHASH_DEDUP_THRESHOLD})")
    p.add_argument("--vlm-input",       type=Path, default=None,
                   help="Input file for vlm step (default: <out-dir>/candidates_phash.json). "
                        "Point at reocr_candidates.json after running reocr_retranslate.py.")
    p.add_argument("--vlm-model",       default=VLM_MODEL_NAME,
                   help=f"VLM model alias or HF ID. Aliases: {', '.join(f'{k}={v}' for k, v in MODEL_ALIASES.items())}. "
                        f"(default: {VLM_MODEL_NAME})")
    p.add_argument("--hf-token",        default=None,
                   help="HuggingFace token for gated models")
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir",         type=Path, default=SCRIPT_DIR / "results" / "misconfigured")
    p.add_argument("--translations",    type=Path, default=DEFAULT_TRANSLATIONS,
                   help="Path to cached_translations.json")
    p.add_argument("--limit",           type=int,  default=None,
                   help="Cap images processed (for test runs)")
    args = p.parse_args()

    # ── Argument validation ────────────────────────────────────────────────────
    if args.max_chars < 1:
        p.error("--max-chars must be >= 1")
    if args.phash_threshold < 0:
        p.error("--phash-threshold must be >= 0")
    if args.limit is not None and args.limit <= 0:
        p.error("--limit must be a positive integer")
    if args.vlm_input is not None and not args.vlm_input.exists():
        p.error(f"--vlm-input file does not exist: {args.vlm_input}")
    if "vlm" in args.step and "phash" not in args.step and args.vlm_input is None:
        pass  # will use default candidates_phash.json; validated at runtime

    args.vlm_model = MODEL_ALIASES.get(args.vlm_model.lower(), args.vlm_model)
    return args


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    steps = args.step
    log.info("Running steps: %s", steps)

    candidates: Optional[list[AdCandidate]] = None

    # ── Step 1: OCR filter ────────────────────────────────────────────────────
    if "filter" in steps:
        translations = load_translations(args.translations)
        candidates   = run_filter(translations, max_chars=args.max_chars, limit=args.limit)
        save_json(args.out_dir / FILE_FILTER, candidates_to_dicts(candidates))
        log.info("Step filter done: %d candidates → %s", len(candidates), FILE_FILTER)

    # ── Step 2: pHash dedup ───────────────────────────────────────────────────
    if "phash" in steps:
        if candidates is None:
            log.info("Loading candidates from %s", FILE_FILTER)
            candidates = candidates_from_dicts(load_json(args.out_dir / FILE_FILTER))
        phash_index = load_phash_index()
        candidates  = run_phash(candidates, phash_index, threshold=args.phash_threshold)
        save_json(args.out_dir / FILE_PHASH, candidates_to_dicts(candidates))
        log.info("Step phash done: %d candidates → %s", len(candidates), FILE_PHASH)
        log.info("  → Feed to reocr_retranslate.py: python reocr_retranslate.py "
                 "--candidates %s", args.out_dir / FILE_PHASH)

    # ── Step 3: VLM ──────────────────────────────────────────────────────────
    if "vlm" in steps:
        vlm_input = args.vlm_input or (args.out_dir / FILE_PHASH)
        if candidates is None or args.vlm_input is not None:
            log.info("Loading VLM input from %s", vlm_input)
            candidates = candidates_from_dicts(load_json(vlm_input))

        results = run_vlm(candidates, args)

        from collections import defaultdict
        by_label: dict[str, list[AdResult]] = defaultdict(list)
        for r in results:
            by_label[r.vlm_label or "no_vlm"].append(r)

        model_tag = _model_tag(args.vlm_model)
        save_json(args.out_dir / f"all_{model_tag}.json", results_to_dicts(results))
        for label, items in by_label.items():
            save_json(args.out_dir / f"{label.lower()}_{model_tag}.json", results_to_dicts(items))

        print_summary(results)
        log.info("Step vlm done: %d results in %s", len(results), args.out_dir)

    log.info("All done.")


if __name__ == "__main__":
    main()
