"""
misleading_design.py — CTA/button-only ad detection via OCR keyword matching + VLM confirm

Pipeline:
  Stage 1 — Short-text OCR filter
              Load all ads from sample_data/metadata.json.
              Keep only ads with translated OCR between 1 and 8 words (inclusive).

  Stage 2 — Lemmatization + fuzzy keyword matching
              Lemmatize each short-text OCR string (spaCy en_core_web_sm).
              Match against 18 CTA-style keywords using:
                (a) lemma-normalized substring match
                (b) rapidfuzz partial_ratio >= fuzzy_thresh
              Score every candidate; rank by (lemma_match DESC, fuzzy_score DESC).
              All keyword-matched candidates are passed to the VLM.

  Stage 3 — VLM confirm (Qwen3-VL-8B or Ollama)
              For each candidate image, ask the VLM whether the ad has a
              CTA button / generic call-to-action with no identifiable advertiser
              information (brand name, product description, or service context).

Usage:
  # Full run via Ollama (recommended)
  python misleading_design.py --ollama --vlm-model gemma3:12b

  # Dry-run — OCR filtering + scoring only, no VLM
  python misleading_design.py --no-vlm

  # Calibrate on known_bad/ad_design images
  python misleading_design.py --mode known_bad

  # Custom Ollama URL or model
  python misleading_design.py --ollama --vlm-model qwen --ollama-url http://localhost:11434

  # Cap records for a quick test
  python misleading_design.py --ollama --vlm-model gemma3:12b --limit 20

Options:
  --mode            full | known_bad               (default: full)
  --ollama          Use Ollama backend instead of HuggingFace
  --vlm-model       Model tag (Ollama) or HF model ID (default: Qwen/Qwen3.5-9B)
                    Ollama aliases: qwen=qwen3.5:9b, gemma3=gemma3:12b
  --translations    Path to cached_translations.json
  --fuzzy-thresh    rapidfuzz partial_ratio cutoff  (default: 82)
  --no-vlm          Skip Stage 3 entirely
  --hf-token        HuggingFace token for gated models
  --device          cuda | cpu                      (default: cuda if available)
  --out-dir         Output directory                (default: results/misleading_design/)
  --limit           Cap images processed (test runs)
  --ollama-url      Ollama server base URL          (default: http://localhost:11434)
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import re
from dataclasses import dataclass, asdict, field
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

SCRIPT_DIR        = Path(__file__).parent
SAMPLE_DATA_DIR   = SCRIPT_DIR.parent / "sample_data"
KNOWN_BAD_DIR     = SCRIPT_DIR / "known_bad"

METADATA_FILE        = SAMPLE_DATA_DIR / "metadata.json"
IMAGE_DIR            = SAMPLE_DATA_DIR / "images"
DEFAULT_TRANSLATIONS = SAMPLE_DATA_DIR / "cached_translations.json"

# ── keyword list + semantic grouping ──────────────────────────────────────────

KEYWORD_GROUPS: dict[str, list[str]] = {
    "cta_generic_action": [
        "start now",
        "start over",
        "activate now",
        "follow up",
        "continue",
    ],
    "cta_access_unlock": [
        "access",
        "access content",
        "access now",
        "unblock content",
    ],
    "cta_register_activate": [
        "register",
        "sign up now",
        "activate this account",
    ],
    "cta_social_chat": [
        "start chatting",
        "let's start a conversation",
        "live stream",
    ],
    "cta_suspicious": [
        "tracking",
        "benign dating",
        "start maintenance",
    ],
}

# Flat list for matching
ALL_KEYWORDS: list[str] = [kw for kws in KEYWORD_GROUPS.values() for kw in kws]

# Reverse map: keyword → group
KW_TO_GROUP: dict[str, str] = {
    kw: grp for grp, kws in KEYWORD_GROUPS.items() for kw in kws
}

MAX_OCR_WORDS = 5  # default; overridden by --max-words at runtime

# ── VLM prompt ─────────────────────────────────────────────────────────────────

VLM_PROMPT_CTA_NO_ADVERTISER = """
Role: You are an online advertising quality analyst specialising in deceptive
and low-quality ad creatives.

Task: Look carefully at the full advertisement image and classify it as
Undisclosed Advertiser or SAFE.

IMPORTANT — check for brand/product context FIRST before labelling anything
as Undisclosed Advertiser. Only flag Undisclosed Advertiser when you are 
certain no identifying advertiser information exists anywhere in the image.

What counts as identifying information (any one of these makes the ad SAFE):
  - A brand name name, even an unfamiliar one (e.g. "Cubbie", "NordVPN")
  - Brand name usually mentioned as a generic product category but it should be present like a brand
    (e.g. "PDF Converter", "VPN Service", "GPS Tracker", "Password Manager", "Payroll Software", "EMR System", "Photo Locator")
  - A visible URL or domain anywhere in the image (e.g. "brand.com",
    "app.example.com") — even a partial domain counts
  - A company logo, app icon, or brand icon
  - Dating / social / live-stream lures with no named platform.
  
Undisclosed Advertiser — ONLY applies when ALL of the following are true:
  1. The ad contains a call-to-action button (CTA) or imperative phrase
     (e.g. "Start Now", "Access", "Register", "Activate", "Continue",
     "Sign Up", "Unblock", "Start Chatting", "Live Stream").
  2. There is NO brand name, product name, URL, logo, or any other identifying information anywhere in the image.
  3. A viewer cannot determine what product or service is being advertised.

SAFE — applies when:
  - The ad contains a CTA alongside any identifying information listed above.
  - The ad has branding even if the design is visually minimal.
  - When in doubt, classify as SAFE — do not flag based on CTA alone.

Output format (strict JSON only, no other text):
{
  "label": "Undisclosed Advertiser" or "SAFE",
  "reason": "One concise sentence identifying the specific label."
}
"""

# ── data structures ────────────────────────────────────────────────────────────

@dataclass
class AdCandidate:
    ad_id:            str
    image_path:       Path
    dataset_category: str        # computer / mobile / software
    translated_ocr:   str
    matched_keywords: list[str]  = field(default_factory=list)
    matched_groups:   list[str]  = field(default_factory=list)
    fuzzy_score:      float      = 0.0
    lemma_match:      bool       = False


@dataclass
class AdResult:
    ad_id:            str
    image_path:       str
    dataset_category: str
    translated_ocr:   str
    matched_keywords: list[str]
    matched_groups:   list[str]
    fuzzy_score:      float
    lemma_match:      bool
    flagged:          bool            # True = Undisclosed Advertiser per VLM
    vlm_label:        Optional[str]   = None
    vlm_reason:       Optional[str]   = None


# ── lemmatization setup ────────────────────────────────────────────────────────

def _load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        log.info("Downloading spaCy en_core_web_sm …")
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True,
        )
        return spacy.load("en_core_web_sm")


_NLP = None  # lazy-loaded


def _nlp():
    global _NLP
    if _NLP is None:
        _NLP = _load_spacy()
    return _NLP


def lemmatize(text: str) -> str:
    """Return space-joined lemmas (lowercased, no spaces/punctuation tokens)."""
    return " ".join(
        t.lemma_.lower()
        for t in _nlp()(text)
        if not t.is_space and not t.is_punct
    )


# Pre-compute lemmatised versions of every keyword once at import time.
# Done lazily on first call to match_keywords() to avoid loading spaCy
# when only the OCR-filter stage is needed.
_KW_LEMMAS: list[tuple[str, str]] | None = None  # [(original_kw, lemmatised_kw), ...]


def _kw_lemmas() -> list[tuple[str, str]]:
    global _KW_LEMMAS
    if _KW_LEMMAS is None:
        log.info("Lemmatising %d keywords …", len(ALL_KEYWORDS))
        _KW_LEMMAS = [(kw, lemmatize(kw)) for kw in ALL_KEYWORDS]
    return _KW_LEMMAS


# ── keyword matching ───────────────────────────────────────────────────────────

def match_keywords(ocr_text: str, fuzzy_thresh: int = 82) -> dict:
    """
    Return a dict with:
      lemma_match     : bool   — at least one keyword matched via lemmatisation
      fuzzy_score     : float  — highest partial_ratio across all keywords
      matched_keywords: list   — keywords that fired (lemma or fuzzy)
      matched_groups  : list   — corresponding semantic groups (deduplicated)
    """
    from rapidfuzz import fuzz

    if not ocr_text or not ocr_text.strip():
        return {
            "lemma_match": False,
            "fuzzy_score": 0.0,
            "matched_keywords": [],
            "matched_groups": [],
        }

    ocr_lower    = ocr_text.lower().replace("\n", " ")
    ocr_lemmatised = lemmatize(ocr_lower)
    ocr_stripped_len = len(ocr_lower.strip())

    matched: list[str] = []
    max_fuzzy: float   = 0.0

    for kw, kw_lemma in _kw_lemmas():
        # Layer 1: lemma substring
        lemma_hit = kw_lemma in ocr_lemmatised

        # Layer 2: fuzzy partial match — skip if OCR is too short to be meaningful
        fscore = fuzz.partial_ratio(kw.lower(), ocr_lower) if ocr_stripped_len >= 5 else 0
        max_fuzzy = max(max_fuzzy, fscore)

        if lemma_hit or fscore >= fuzzy_thresh:
            matched.append(kw)

    matched = list(dict.fromkeys(matched))  # deduplicate, preserve order
    groups  = list(dict.fromkeys(KW_TO_GROUP[kw] for kw in matched))

    return {
        "lemma_match":      bool(matched),
        "fuzzy_score":      round(max_fuzzy, 1),
        "matched_keywords": matched,
        "matched_groups":   groups,
    }


# ── data loading ───────────────────────────────────────────────────────────────

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


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"[ \n]+", text.strip()) if w])


def load_candidates_from_metadata(
    translations: dict[str, str],
    fuzzy_thresh: int,
    min_words: int = 1,
    max_words: int = MAX_OCR_WORDS,
    limit: Optional[int] = None,
    image_dir: Optional[Path] = None,
) -> list[AdCandidate]:
    """
    Scan sample_data/metadata.json. For each screenshotPath variant in an ad:
      1. Translate OCR (from cache or raw).
      2. Skip if translated OCR word count is outside [min_words, max_words].
      3. Run keyword matching.
      4. Keep if any keyword matched (lemma or fuzzy).
    Returns all matched candidates (unsorted).
    """
    candidates: list[AdCandidate] = []

    if not METADATA_FILE.exists():
        log.warning("Metadata not found: %s", METADATA_FILE)
        return candidates

    with open(METADATA_FILE, encoding="utf-8") as f:
        items = json.load(f)

    log.info("%d ads in metadata", len(items))
    short_count = matched_count = 0

    for item in items:
        ad_id      = item["ad_id"]
        raw_ocr    = item.get("ocr_text") or ""
        translated = item.get("translated_ocr_text") or translations.get(raw_ocr) or raw_ocr

        wc = _word_count(translated)
        if wc < min_words or wc > max_words:
            continue
        short_count += 1

        m = match_keywords(translated, fuzzy_thresh)
        if not m["matched_keywords"]:
            continue
        matched_count += 1

        img_path = (image_dir or IMAGE_DIR) / f"{ad_id}.png"
        local = img_path if img_path.exists() else None
        if local is None:
            continue
        candidates.append(AdCandidate(
                ad_id            = ad_id,
                image_path       = local,
                dataset_category = item.get("category", "ads"),
                translated_ocr   = translated,
                matched_keywords = m["matched_keywords"],
                matched_groups   = m["matched_groups"],
                fuzzy_score      = m["fuzzy_score"],
                lemma_match      = m["lemma_match"],
            ))

        if limit and len(candidates) >= limit:
            log.info("--limit reached (%d), stopping early", limit)
            return candidates

    log.info(
        "short-text: %d | keyword-matched: %d | total candidates: %d",
        short_count, matched_count, len(candidates),
    )
    return candidates


def load_single_crid(crid: str, translations: dict[str, str]) -> list[AdCandidate]:
    """Bypass keyword filtering — load one ad by CRID for direct VLM testing."""
    bare = crid[:-3] if re.match(r".*-v\d+$", crid) else crid
    v_idx = int(crid[-1]) if re.match(r".*-v\d+$", crid) else 0

    if not METADATA_FILE.exists():
        log.error("Metadata not found: %s", METADATA_FILE)
        return []
    with open(METADATA_FILE, encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        if item["ad_id"] != crid and item.get("crid") != bare:
            continue
        raw_ocr    = item.get("ocr_text") or ""
        translated = item.get("translated_ocr_text") or translations.get(raw_ocr) or raw_ocr
        img_path   = IMAGE_DIR / f"{item['ad_id']}.png"
        if not img_path.exists():
            log.error("Image not found for %s: %s", crid, img_path)
            return []
        return [AdCandidate(
            ad_id            = item["ad_id"],
            image_path       = img_path,
            dataset_category = item.get("category", "ads"),
            translated_ocr   = translated,
            matched_keywords = [],
            matched_groups   = [],
            fuzzy_score      = 0.0,
            lemma_match      = False,
        )]
    log.error("CRID %s not found in metadata.", bare)
    return []


def load_known_bad_candidates(
    fuzzy_thresh: int,
) -> list[AdCandidate]:
    """Load known_bad/misleading_design PNGs as AdCandidates (OCR matching skipped)."""
    kb_dir = KNOWN_BAD_DIR / "misleading_design"
    if not kb_dir.exists():
        log.warning("known_bad/misleading_design not found: %s", kb_dir)
        return []
    candidates = []
    for f in sorted(kb_dir.glob("*.png")):
        candidates.append(AdCandidate(
            ad_id            = f.stem,
            image_path       = f,
            dataset_category = "misleading_design",
            translated_ocr   = "",
            matched_keywords = [],
            matched_groups   = [],
            fuzzy_score      = 0.0,
            lemma_match      = False,
        ))
    log.info("known_bad/misleading_design: %d images", len(candidates))
    return candidates


def rank_candidates(candidates: list[AdCandidate], top_n: int) -> list[AdCandidate]:
    """
    Sort by: (lemma_match DESC, fuzzy_score DESC).
    Return top_n.
    """
    ranked = sorted(
        candidates,
        key=lambda c: (c.lemma_match, c.fuzzy_score),
        reverse=True,
    )
    log.info(
        "Ranked %d candidates → keeping top %d",
        len(candidates), min(top_n, len(candidates)),
    )
    return ranked[:top_n]


# ── VLM inference ──────────────────────────────────────────────────────────────

def _model_tag(model_name: str) -> str:
    """Filesystem-safe short tag matching the ensemble convention."""
    return model_name.split("/")[-1].lower().replace(".", "").replace("-", "_")


VLM_MODEL_NAME = "Qwen/Qwen3.5-9B"

MODEL_ALIASES: dict[str, str] = {
    "qwen":   "Qwen/Qwen3.5-9B",
    "glm":    "zai-org/GLM-4.6V-Flash",
    "gemma3": "google/gemma-3-12b-it",
}

# Ollama model tag used when --ollama is set (same tag form as classify_image.py)
OLLAMA_DEFAULT_MODEL = "llava"

OLLAMA_MODEL_ALIASES: dict[str, str] = {
    "qwen":    "qwen3.5:9b",
    "glm":     "glm4v:9b",
    "gemma3":  "gemma3:12b",
    "llava":   "llava",
    "minicpm": "minicpm-v",
}

_KNOWN_VL_MODELS: set[str] = {"Qwen/Qwen3.5-9B"}
_GLM_MODELS:      set[str] = {"zai-org/GLM-4.6V-Flash"}
_GEMMA3_MODELS:   set[str] = {"google/gemma-3-12b-it"}

_vlm_model_obj      = None
_vlm_processor_obj  = None


def _is_vl_model(name: str) -> bool:
    return "VL" in name or "vl" in name or name in _KNOWN_VL_MODELS


def load_vlm(model_name: str = VLM_MODEL_NAME, token: Optional[str] = None):
    """Load model + processor, dispatching to the correct loader per model family."""
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
            try:
                cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            except (KeyError, ValueError) as e:
                log.warning("AutoConfig failed (%s) — retrying with PretrainedConfig", e)
                from transformers import PretrainedConfig
                cfg = PretrainedConfig.from_pretrained(model_name, trust_remote_code=True)
            model_type = getattr(cfg, "model_type", "")
            log.info("VL model detected (model_type=%s) — using AutoProcessor", model_type)
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
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
    """Extract first JSON object containing a 'label' key from model output."""
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
    """
    Classify one ad image.  Returns:
      label   : "Undisclosed Advertiser" | "SAFE"
      reason  : str
      flagged : bool
    """
    valid_labels = {"Undisclosed Advertiser", "SAFE"}

    # ── GLM-4.6V-Flash ────────────────────────────────────────────────────────
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
                {"type": "text", "text": f"{VLM_PROMPT_CTA_NO_ADVERTISER}\n\nAnalyse this advertisement image and classify it."},
            ]
            if image_url else
            [{"type": "text", "text": f"{VLM_PROMPT_CTA_NO_ADVERTISER}\n\nAnalyse this advertisement and classify it."}]
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
                if result and result.get("label") in valid_labels:
                    return {"label": result["label"], "reason": result.get("reason", ""),
                            "flagged": result["label"] == "Undisclosed Advertiser"}
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

        user_text = f"{VLM_PROMPT_CTA_NO_ADVERTISER}\n\nAnalyse this advertisement image and classify it."
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
                if result and result.get("label") in valid_labels:
                    return {"label": result["label"], "reason": result.get("reason", ""),
                            "flagged": result["label"] == "Undisclosed Advertiser"}
                for lbl in valid_labels:
                    if re.search(rf'\b{lbl}\b', decoded, re.IGNORECASE):
                        log.warning("Gemma3 attempt %d: extracted label '%s' from reasoning text", attempt + 1, lbl)
                        return {"label": lbl, "reason": "", "flagged": lbl == "Undisclosed Advertiser"}
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
        {"role": "system", "content": VLM_PROMPT_CTA_NO_ADVERTISER},
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
            if result and result.get("label") in valid_labels:
                return {
                    "label":   result["label"],
                    "reason":  result.get("reason", ""),
                    "flagged": result["label"] == "Undisclosed Advertiser",
                }
            log.warning("Attempt %d: unexpected response — %s", attempt + 1, raw[:200])
        except Exception as e:
            log.warning("VLM error attempt %d: %s", attempt + 1, e, exc_info=True)

    return {"label": "SAFE", "reason": "", "flagged": False}


# ── Ollama inference ───────────────────────────────────────────────────────────

def vlm_classify_ollama(
    image_path: Path,
    model_name: str,
    ocr_text: str = "",
    ollama_url: str = "http://localhost:11434",
    retries: int = 2,
) -> dict:
    """Classify one ad image via the Ollama REST API."""
    import base64
    import os
    import ollama as _ollama

    os.environ.setdefault("OLLAMA_HOST", ollama_url)

    valid_labels = {"Undisclosed Advertiser", "SAFE"}

    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        image_data = [b64]
    except Exception as e:
        log.warning("Ollama: failed to read image %s: %s — text-only.", image_path, e)
        image_data = []

    messages = [
        {"role": "system", "content": VLM_PROMPT_CTA_NO_ADVERTISER},
        {
            "role": "user",
            "content": f"Analyse this advertisement image and classify it."
                       + (f" OCR text extracted from the image: '{ocr_text}'" if ocr_text else ""),
            **({"images": image_data} if image_data else {}),
        },
    ]

    last_err = None
    for attempt in range(retries + 1):
        try:
            response = _ollama.chat(model=model_name, messages=messages, format="json",
                                    think=False, options={"temperature": 0, "seed": 42})
            log.info(
                "Ollama inference: %.2fs (%s)",
                response.get("total_duration", 0) / 1e9,
                model_name,
            )
            result = json.loads(response["message"]["content"])
            if result.get("label") in valid_labels:
                return {"label": result["label"], "reason": result.get("reason", ""),
                        "flagged": result["label"] == "Undisclosed Advertiser"}
            log.warning("Ollama attempt %d: unexpected label — %s", attempt + 1, result)
            last_err = result
        except Exception as e:
            log.warning("Ollama error attempt %d: %s", attempt + 1, e)
            last_err = str(e)

    log.error("Ollama: all %d attempts failed. Last: %s", retries + 1, last_err)
    return {"label": "SAFE", "reason": "", "flagged": False}


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

def save_json(path: Path, data: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("Saved → %s", path)


def results_to_dicts(results: list[AdResult]) -> list[dict]:
    return [asdict(r) for r in results]


def print_summary(results: list[AdResult], label: str = ""):
    total   = len(results)
    flagged = sum(1 for r in results if r.flagged)
    vlm_run = sum(1 for r in results if r.vlm_label is not None)
    print(f"\n{'='*60}")
    print(f"SUMMARY{' — ' + label if label else ''}")
    print(f"  Candidates evaluated : {total}")
    print(f"  VLM run              : {vlm_run}")
    print(f"  Undisclosed Advertiser        : {flagged}")
    print(f"  SAFE                 : {vlm_run - flagged}")
    print(f"{'='*60}")
    # Top flagged
    flagged_list = sorted(
        [r for r in results if r.flagged],
        key=lambda r: r.fuzzy_score,
        reverse=True,
    )
    for r in flagged_list[:20]:
        kws = ", ".join(r.matched_keywords[:3])
        print(
            f"  VIOLATION  ocr={repr(r.translated_ocr[:40]):<44} "
            f"kw=[{kws}]  {r.ad_id}"
        )
        if r.vlm_reason:
            print(f"             VLM: {r.vlm_reason[:80]}")


# ── pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(args) -> list[AdResult]:
    translations = load_translations(args.translations)

    # ── Stage 1 + 2: OCR filter + keyword scoring ──────────────────────────
    if getattr(args, "crid", None):
        crid = args.crid if re.match(r".*-v\d+$", args.crid) else args.crid + "-v0"
        candidates = load_single_crid(crid, translations)
    elif args.mode == "known_bad":
        candidates = load_known_bad_candidates(args.fuzzy_thresh)
    else:
        candidates = load_candidates_from_metadata(
            translations, args.fuzzy_thresh,
            min_words=args.min_words, max_words=args.max_words,
            limit=args.limit,
        )

    if not candidates:
        log.warning("No candidates found — check metadata paths and translations.")
        return []

    # ── Rank candidates (all pass to VLM) ─────────────────────────────────
    if getattr(args, "crid", None):
        top_candidates = candidates
    else:
        top_candidates = rank_candidates(candidates, len(candidates))

    log.info(
        "Keyword match stats: %d lemma-matched, %d fuzzy-only",
        sum(1 for c in top_candidates if c.lemma_match),
        sum(1 for c in top_candidates if not c.lemma_match),
    )

    # ── Stage 3: VLM classify ──────────────────────────────────────────────
    results: list[AdResult] = []
    vlm_model = vlm_proc = None

    use_ollama     = getattr(args, "ollama", False)
    vlm_model_name = getattr(args, "vlm_model", VLM_MODEL_NAME)
    ollama_url     = getattr(args, "ollama_url", "http://localhost:11434")

    if use_ollama:
        # When --ollama is set, vlm_model is treated as the Ollama model tag directly.
        ollama_model_tag = vlm_model_name
        log.info("Ollama backend: model=%s url=%s", ollama_model_tag, ollama_url)
        cache_key_name = f"ollama_{ollama_model_tag.replace(':', '_').replace('/', '_')}"
    else:
        cache_key_name = _model_tag(vlm_model_name)

    cache_path = args.out_dir / f"llm_cache_{cache_key_name}.json"
    llm_cache: dict = {}
    cache_hits = 0

    if not args.no_vlm:
        llm_cache = load_llm_cache(cache_path)
        if not use_ollama:
            vlm_model, vlm_proc = load_vlm(model_name=vlm_model_name, token=args.hf_token)

    for idx, cand in enumerate(top_candidates, 1):
        if idx % 100 == 0 or idx == 1:
            log.info("VLM progress: %d / %d", idx, len(top_candidates))

        vlm_label = vlm_reason = None
        flagged   = False

        if not args.no_vlm:
            if cand.ad_id in llm_cache:
                cached = llm_cache[cand.ad_id]
                vlm_label  = cached["label"]
                vlm_reason = cached["reason"]
                flagged    = vlm_label == "Undisclosed Advertiser"
                cache_hits += 1
            else:
                try:
                    if use_ollama:
                        out = vlm_classify_ollama(
                            cand.image_path, ollama_model_tag,
                            ocr_text=cand.translated_ocr, ollama_url=ollama_url,
                        )
                    else:
                        out = vlm_classify(vlm_model, vlm_proc, cand.image_path,
                                           model_name=vlm_model_name)
                    vlm_label  = out["label"]
                    vlm_reason = out["reason"]
                    flagged    = out["flagged"]
                    llm_cache[cand.ad_id] = {"label": vlm_label, "reason": vlm_reason}
                    save_llm_cache(llm_cache, cache_path)
                except Exception as e:
                    log.warning("VLM failed for %s: %s", cand.ad_id, e)

        results.append(AdResult(
            ad_id            = cand.ad_id,
            image_path       = str(cand.image_path),
            dataset_category = cand.dataset_category,
            translated_ocr   = cand.translated_ocr,
            matched_keywords = cand.matched_keywords,
            matched_groups   = cand.matched_groups,
            fuzzy_score      = cand.fuzzy_score,
            lemma_match      = cand.lemma_match,
            flagged          = flagged,
            vlm_label        = vlm_label,
            vlm_reason       = vlm_reason,
        ))

    if not args.no_vlm:
        log.info("LLM cache: %d hits, %d new calls", cache_hits, len(top_candidates) - cache_hits)
        if not use_ollama:
            del vlm_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


# ── arg parsing + main ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode",          choices=["full", "known_bad"], default="full")
    p.add_argument("--translations",  type=Path, default=DEFAULT_TRANSLATIONS,
                   help="Path to cached_translations.json")
    p.add_argument("--fuzzy-thresh",  type=int,  default=82,
                   help="rapidfuzz partial_ratio threshold (default: 82)")
    p.add_argument("--min-words",     type=int,  default=1,
                   help="Min translated OCR word count to include (default: 1)")
    p.add_argument("--max-words",     type=int,  default=MAX_OCR_WORDS,
                   help="Max translated OCR word count to include (default: 5)")
    p.add_argument("--no-vlm",        action="store_true",
                   help="Skip VLM stage — output keyword matches only")
    p.add_argument("--vlm-model",     default=VLM_MODEL_NAME,
                   help=f"VLM model alias or HF ID. Aliases: {', '.join(f'{k}={v}' for k, v in MODEL_ALIASES.items())}. "
                        f"With --ollama this is the Ollama model tag (e.g. llava, qwen2.5vl). "
                        f"(default: {VLM_MODEL_NAME})")
    p.add_argument("--ollama",        action="store_true",
                   help="Use Ollama server for VLM inference instead of HuggingFace")
    p.add_argument("--ollama-url",    default="http://localhost:11434",
                   help="Ollama server base URL (default: http://localhost:11434)")
    p.add_argument("--hf-token",      default=None,
                   help="HuggingFace token for gated models")
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir",       type=Path, default=SCRIPT_DIR / "results" / "misleading_design")
    p.add_argument("--crid",          type=str,  default=None,
                   help="Run on a single ad by creative ID (e.g. CR18077324848429793281-v0).")
    p.add_argument("--limit",         type=int,  default=None,
                   help="Cap images processed (for test runs)")
    args = p.parse_args()

    # ── Argument validation ────────────────────────────────────────────────────
    if args.fuzzy_thresh < 0 or args.fuzzy_thresh > 100:
        p.error(f"--fuzzy-thresh must be between 0 and 100, got {args.fuzzy_thresh}")
    if args.min_words < 1:
        p.error("--min-words must be >= 1")
    if args.max_words < args.min_words:
        p.error(f"--max-words ({args.max_words}) must be >= --min-words ({args.min_words})")
    if args.limit is not None and args.limit <= 0:
        p.error("--limit must be a positive integer")
    if args.ollama_url and not args.ollama_url.startswith(("http://", "https://")):
        p.error(f"--ollama-url must start with http:// or https://, got: {args.ollama_url!r}")

    if not args.ollama:
        args.vlm_model = MODEL_ALIASES.get(args.vlm_model.lower(), args.vlm_model)
    else:
        # When --ollama, resolve alias then default if unchanged from HF default
        args.vlm_model = OLLAMA_MODEL_ALIASES.get(args.vlm_model.lower(), args.vlm_model)
        if args.vlm_model == VLM_MODEL_NAME:
            args.vlm_model = OLLAMA_DEFAULT_MODEL
    return args


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = run_pipeline(args)

    if not results:
        log.warning("No results produced.")
        return

    flagged = [r for r in results if r.flagged]
    flagged.sort(key=lambda r: r.fuzzy_score, reverse=True)

    model_tag = _model_tag(args.vlm_model)
    tag = f"{args.mode}_{model_tag}"
    save_json(args.out_dir / f"{tag}_all.json",     results_to_dicts(results))
    save_json(args.out_dir / f"{tag}_flagged.json", results_to_dicts(flagged))

    if args.no_vlm:
        # Dump a lean candidates file for reocr_retranslate.py
        candidates_out = [
            {"ad_id": r.ad_id, "image_path": r.image_path,
             "dataset_category": r.dataset_category, "original_ocr": r.translated_ocr,
             "matched_keywords": r.matched_keywords, "fuzzy_score": r.fuzzy_score}
            for r in results
        ]
        save_json(args.out_dir / "candidates.json", candidates_out)
        log.info("Candidates file saved (%d entries, word-limit=%d) — ready for reocr_retranslate.py",
                 len(candidates_out), args.max_words)

    print_summary(results, label=tag)
    log.info("Done. Results in %s", args.out_dir)


if __name__ == "__main__":
    main()
