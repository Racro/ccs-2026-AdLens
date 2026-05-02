#!/usr/bin/env python3
"""
adlens_pipeline.py — Classification + judging pipeline for labeled AdLens images.

Reads:  sample_data/metadata.json
        sample_data/<violation>/<label>/<ad_id>.png

Imports prompts from stage-specific scripts:
  • PROMPT_SCAREWARE, PROMPT_MISLEADING  → ensemble.py
  • PROMPT_AD_DESIGN                     → misleading_design.py
  • JUDGE_PROMPTS                        → llm_judge.py

Phase 1 — Classify with qwen3.5:9b and gemma3:12b via Ollama.
Phase 2 — Judge disagreements with gemma4:26b via Ollama.

All results are cached — safe to resume after interruption.

Output (all under --out-dir, default: sample_data/pipeline_results/):
  classify_cache.json       — per-(ad_id, violation, model) cache
  scareware_results.json    — records + per-model + ensemble + judge labels
  misleading_results.json
  ad_design_results.json

Usage:
  python adlens_pipeline.py
  python adlens_pipeline.py --violations scareware misleading
  python adlens_pipeline.py --skip-classify   # judge only (needs cache)
  python adlens_pipeline.py --skip-judge
  python adlens_pipeline.py --limit 50        # cap per violation
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests

# ── Stage-specific imports ────────────────────────────────────────────────────
from ensemble import PROMPT_SCAREWARE, PROMPT_MISLEADING
from misleading_design import VLM_PROMPT_CTA_NO_ADVERTISER as PROMPT_AD_DESIGN
from llm_judge import SYSTEM_PROMPTS as JUDGE_PROMPTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class CallRecord:
    stage: str          # "classify" | "judge"
    model: str
    violation: str
    ad_id: str
    warmup: bool
    wall_ms: float      # total wall time including network
    load_ms: float      # ollama load_duration (model swap cost)
    eval_ms: float      # ollama eval_duration (pure token generation)
    prompt_tokens: int
    output_tokens: int
    label: str          # parsed label or ERROR


_call_log: list[CallRecord] = []


# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR       = Path(__file__).parent
SAMPLE_DATA_DIR  = SCRIPT_DIR.parent / "sample_data"
METADATA_FILE    = SAMPLE_DATA_DIR / "metadata.json"
RESULTS_DIR      = SCRIPT_DIR / "results"
DEFAULT_OUT_DIR  = RESULTS_DIR / "pipeline_results"
OLLAMA_URL       = "http://localhost:11434"

# ── Models ─────────────────────────────────────────────────────────────────────
QWEN_MODEL   = "qwen3.5:9b"
GEMMA_MODEL  = "gemma3:12b"
JUDGE_MODEL  = "gemma4:26b"

QWEN_KEY     = "qwen3_5_9b"
GEMMA_KEY    = "gemma3_12b"


CLASSIFIER_MODELS = [
    (QWEN_MODEL,  QWEN_KEY),
    (GEMMA_MODEL, GEMMA_KEY),
]

ALL_VIOLATIONS = ["scareware", "deceptive_claim", "misleading_design"]

# ── Classification prompts (imported from stage scripts) ──────────────────────
CLASSIFY_PROMPTS = {
    "scareware":         PROMPT_SCAREWARE,
    "deceptive_claim":   PROMPT_MISLEADING,
    "misleading_design": PROMPT_AD_DESIGN,
}

POSITIVE_LABELS = {
    "scareware":         "SCAREWARE",
    "deceptive_claim":   "MISLEADING",
    "misleading_design": "Undisclosed Advertiser",
}

VALID_LABELS = {
    "scareware":         {"SCAREWARE", "SAFE"},
    "deceptive_claim":   {"MISLEADING", "SAFE"},
    "misleading_design": {"Undisclosed Advertiser", "SAFE"},
}

# JUDGE_PROMPTS imported from llm_judge as SYSTEM_PROMPTS


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> object:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    log.info("Saved %d records → %s", len(data) if hasattr(data, "__len__") else 1, path)


def image_to_b64(path: Path) -> Optional[str]:
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception as e:
        log.warning("Cannot read image %s: %s", path, e)
        return None


def resolve_image(record: dict) -> Optional[Path]:
    """Return Path to the labeled image from sample_data/images/<ad_id>.png"""
    path = SAMPLE_DATA_DIR / "images" / f"{record['ad_id']}.png"
    return path if path.exists() else None


def _needs_think_disabled(model: str) -> bool:
    return "qwen3" in model.lower()


def ollama_chat(
    model: str,
    system: str,
    user: str,
    base_url: str,
    image_b64: Optional[str] = None,
    retries: int = 3,
    timeout: int = 180,
) -> str:
    url = f"{base_url.rstrip('/')}/api/chat"
    user_msg: dict = {"role": "user", "content": user}
    if image_b64:
        user_msg["images"] = [image_b64]

    payload: dict = {
        "model":    model,
        "stream":   False,
        "messages": [
            {"role": "system", "content": system},
            user_msg,
        ],
        "options": {"temperature": 0},
    }
    if _needs_think_disabled(model):
        payload["think"] = False

    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as exc:
            wait = 2 ** attempt
            log.warning("Ollama error (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, retries, exc, wait)
            time.sleep(wait)
    return ""


def ollama_chat_timed(
    model: str,
    system: str,
    user: str,
    base_url: str,
    image_b64: Optional[str] = None,
    retries: int = 3,
    timeout: int = 180,
) -> tuple[str, dict]:
    """Like ollama_chat but also returns Ollama timing metadata."""
    url = f"{base_url.rstrip('/')}/api/chat"
    user_msg: dict = {"role": "user", "content": user}
    if image_b64:
        user_msg["images"] = [image_b64]

    payload: dict = {
        "model":    model,
        "stream":   False,
        "messages": [
            {"role": "system", "content": system},
            user_msg,
        ],
        "options": {"temperature": 0},
    }
    if _needs_think_disabled(model):
        payload["think"] = False

    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            resp = requests.post(url, json=payload, timeout=timeout)
            wall_ms = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            data = resp.json()
            meta = {
                "wall_ms":       round(wall_ms, 1),
                "load_ms":       round(data.get("load_duration",        0) / 1e6, 1),
                "eval_ms":       round(data.get("eval_duration",        0) / 1e6, 1),
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count",        0),
            }
            return data["message"]["content"], meta
        except Exception as exc:
            wait = 2 ** attempt
            log.warning("Ollama error (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, retries, exc, wait)
            time.sleep(wait)
    return "", {"wall_ms": 0, "load_ms": 0, "eval_ms": 0, "prompt_tokens": 0, "output_tokens": 0}


def parse_classify_response(raw: str, valid_labels: set[str]) -> dict:
    """Extract {label, reason} from model output. Returns error dict on failure."""
    match = re.search(r'\{[^{}]*"label"[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            label = parsed.get("label", "")
            if label in valid_labels:
                return {"label": label, "reason": parsed.get("reason", "")}
        except json.JSONDecodeError:
            pass
    # Fallback: scan for any valid label string in the output
    for lbl in sorted(valid_labels, key=len, reverse=True):
        if re.search(re.escape(lbl), raw, re.IGNORECASE):
            log.warning("Extracted label '%s' from raw text (no JSON)", lbl)
            return {"label": lbl, "reason": ""}
    log.warning("Could not parse label from: %s", raw[:200])
    return {"label": "ERROR", "reason": raw[:300]}


def parse_judge_response(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            raw_label = parsed.get("judge_label", "ERROR")
            label = (raw_label.upper()
                     if raw_label.upper() in {"SCAREWARE", "DECEPTIVE_CLAIM", "SAFE", "ERROR"}
                     else raw_label)
            return {"judge_label": label, "judge_reason": parsed.get("judge_reason", "")}
        except json.JSONDecodeError:
            pass
    return {"judge_label": "ERROR", "judge_reason": f"Parse failed. Raw: {raw[:200]}"}


# ── Backend dispatcher (Ollama only) ──────────────────────────────────────────

def judge_chat(
    model: str,
    system: str,
    user: str,
    base_url: str,
    image_b64: Optional[str] = None,
    retries: int = 3,
    timeout: int = 180,
) -> str:
    return ollama_chat(model, system, user, base_url,
                       image_b64=image_b64, retries=retries, timeout=timeout)


# ── Phase 1: Classify ──────────────────────────────────────────────────────────

def classify_record(
    record: dict,
    model_name: str,
    violation: str,
    ollama_url: str,
    text_only: bool = False,
    warmup: bool = False,
) -> dict:
    """Classify one record with one model. Returns {label, reason}.
    Also appends a CallRecord to _call_log for latency tracking.
    """
    system_prompt = CLASSIFY_PROMPTS[violation]
    valid         = VALID_LABELS[violation]
    ocr_text      = record.get("translated_ocr_text") or record.get("ocr_text") or ""
    ad_id         = record["ad_id"]

    image_b64 = None
    if not text_only:
        img_path = resolve_image(record)
        image_b64 = image_to_b64(img_path) if img_path else None
        if not image_b64:
            log.warning("No image for %s", ad_id)

    if violation == "misleading_design":
        if text_only:
            user_msg = (
                f"Classify this advertisement based on its text content.\n\nAd text:\n{ocr_text}"
            )
        else:
            user_msg = (
                f"Analyse this advertisement image and classify it."
                + (f" OCR text extracted from the image: '{ocr_text}'" if ocr_text else "")
            )
    else:
        user_msg = (
            f"Classify this advertisement.\n\nAd text:\n{ocr_text}"
            + ("\n\nAnalyse the image as well." if image_b64 else "")
        )

    raw, meta = ollama_chat_timed(model_name, system_prompt, user_msg, ollama_url,
                                  image_b64=image_b64)

    result = parse_classify_response(raw, valid)
    _call_log.append(CallRecord(
        stage="classify", model=model_name, violation=violation, ad_id=ad_id,
        warmup=warmup, wall_ms=meta["wall_ms"], load_ms=meta["load_ms"],
        eval_ms=meta["eval_ms"], prompt_tokens=meta["prompt_tokens"],
        output_tokens=meta["output_tokens"], label=result.get("label", "ERROR"),
    ))
    return result


def run_classify_phase(
    records: list[dict],
    violations: list[str],
    ollama_url: str,
    cache: dict,
    cache_path: Path,
    classifier_models: list[tuple[str, str]] | None = None,
    save_every: int = 25,
    text_only: bool = False,
) -> None:
    """
    Classify all records for the requested violations with both models.
    Updates `cache` in-place and saves periodically.
    """
    if classifier_models is None:
        classifier_models = CLASSIFIER_MODELS
    pending_count = 0
    for violation in violations:
        log.info("Violation=%s: %d records", violation, len(records))

        for model_name, model_key in classifier_models:
            newly_classified = 0
            for idx, rec in enumerate(records, 1):
                cache_key = f"{rec['ad_id']}:{violation}:{model_key}"
                if cache_key in cache:
                    continue
                result = classify_record(rec, model_name, violation, ollama_url, text_only=text_only)
                cache[cache_key] = result
                newly_classified += 1
                pending_count   += 1

                if pending_count % save_every == 0:
                    save_json(cache, cache_path)

                if idx % 50 == 0 or idx == 1:
                    log.info("[%s/%s] %d/%d", violation, model_key, idx, len(records))

            log.info("[%s/%s] done. %d new classifications.", violation, model_key, newly_classified)

    save_json(cache, cache_path)


# ── Phase 2: Build ensemble + Judge ───────────────────────────────────────────

def build_ensemble_records(
    records: list[dict],
    violation: str,
    cache: dict,
    classifier_models: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Merge cache entries into per-record ensemble results."""
    if classifier_models is None:
        classifier_models = CLASSIFIER_MODELS
    positive_label = POSITIVE_LABELS[violation]
    out = []
    for rec in records:
        ad_id = rec["ad_id"]
        per_model: dict[str, dict] = {}
        for _, model_key in classifier_models:
            ck = f"{ad_id}:{violation}:{model_key}"
            if ck in cache:
                per_model[model_key] = cache[ck]

        votes: dict[str, int] = {}
        for md in per_model.values():
            lbl = md.get("label", "")
            if lbl and lbl != "ERROR":
                votes[lbl] = votes.get(lbl, 0) + 1

        total     = sum(votes.values())
        pos_votes = votes.get(positive_label, 0)
        ensemble  = positive_label if (total > 0 and pos_votes * 2 > total) else "SAFE"

        out.append({
            **rec,
            "per_model":      per_model,
            "ensemble_votes": votes,
            "ensemble_label": ensemble,
            "judge_label":    "",
            "judge_reason":   "",
        })
    return out


def is_disagreement(record: dict) -> bool:
    pm = record.get("per_model", {})
    q  = pm.get(QWEN_KEY,  {}).get("label", "")
    g  = pm.get(GEMMA_KEY, {}).get("label", "")
    return bool(q and g and q != g and q != "ERROR" and g != "ERROR")


def build_judge_user_prompt(record: dict, violation: str, text_only: bool = False) -> str:
    pm    = record.get("per_model", {})
    qwen  = pm.get(QWEN_KEY,  {})
    gemma = pm.get(GEMMA_KEY, {})
    ocr   = record.get("translated_ocr_text") or record.get("ocr_text") or "N/A"

    closing = (
        f"One model says {POSITIVE_LABELS[violation]}, the other says SAFE. "
        + ("Carefully analyse the ad text and both reasoning chains, "
           if text_only else
           "Carefully analyse the ad image, the ad text, and both reasoning chains, ")
        + "then output your JSON."
    )
    lines = [
        f"### Ad Text",
        f"{ocr}",
        "",
        f"### Model assessments",
        f"Qwen ({QWEN_KEY}):",
        f"  Verdict : {qwen.get('label', 'N/A')}",
        f"  Reason  : {qwen.get('reason', 'N/A')}",
        "",
        f"Gemma ({GEMMA_KEY}):",
        f"  Verdict : {gemma.get('label', 'N/A')}",
        f"  Reason  : {gemma.get('reason', 'N/A')}",
        "",
        closing,
    ]
    return "\n".join(lines)


def run_judge_phase(
    ensemble_records: list[dict],
    violation: str,
    judge_model: str,
    ollama_url: str,
    judge_cache: dict,
    save_every: int = 25,
    text_only: bool = False,
) -> list[dict]:
    """Judge disagreements. Returns the full record list with judge fields filled."""
    disagreements = [r for r in ensemble_records if is_disagreement(r)]
    log.info("[%s] %d disagreements to judge", violation, len(disagreements))

    judged_count = 0
    for idx, rec in enumerate(disagreements, 1):
        ad_id = rec["ad_id"]
        if ad_id in judge_cache:
            verdict = judge_cache[ad_id]
        else:
            image_b64 = None
            if not text_only:
                img_path  = resolve_image(rec)
                image_b64 = image_to_b64(img_path) if img_path else None
            user_prompt = build_judge_user_prompt(rec, violation, text_only=text_only)
            raw, meta = ollama_chat_timed(judge_model, JUDGE_PROMPTS[violation],
                                          user_prompt, ollama_url, image_b64=image_b64)
            verdict = parse_judge_response(raw)
            judge_cache[ad_id] = verdict
            judged_count += 1
            _call_log.append(CallRecord(
                stage="judge", model=judge_model, violation=violation, ad_id=ad_id,
                warmup=False, wall_ms=meta["wall_ms"], load_ms=meta["load_ms"],
                eval_ms=meta["eval_ms"], prompt_tokens=meta["prompt_tokens"],
                output_tokens=meta["output_tokens"], label=verdict.get("judge_label", "ERROR"),
            ))

        # Write back into the record in-place
        rec["judge_label"]  = verdict["judge_label"]
        rec["judge_reason"] = verdict["judge_reason"]

        q = rec["per_model"].get(QWEN_KEY,  {}).get("label", "?")
        g = rec["per_model"].get(GEMMA_KEY, {}).get("label", "?")
        log.info("[%s] [%d/%d] %s | qwen=%s gemma=%s → judge=%s",
                 violation, idx, len(disagreements), ad_id, q, g, verdict["judge_label"])

        if judged_count % save_every == 0 and judged_count > 0:
            log.info("Checkpoint: %d records judged", judged_count)

    log.info("[%s] Judged %d new, %d from cache", violation, judged_count,
             len(disagreements) - judged_count)
    return ensemble_records


def run_judge_sample_phase(
    records: list[dict],
    judge_model: str,
    ollama_url: str,
    judge_sample_cache: dict,
    n: int = 50,
    text_only: bool = False,
) -> None:
    """Run the judge on a random sample of N records regardless of disagreement status."""
    import random
    sample = random.sample(records, min(n, len(records)))
    log.info("=== Judge sample: %d records via %s ===", len(sample), judge_model)
    for rec in sample:
        ad_id     = rec["ad_id"]
        violation = rec.get("violation_type", ALL_VIOLATIONS[0])
        if ad_id in judge_sample_cache:
            continue
        ocr = rec.get("translated_ocr_text") or rec.get("ocr_text") or ""
        image_b64 = None
        if not text_only:
            img_path  = resolve_image(rec)
            image_b64 = image_to_b64(img_path) if img_path else None
        user_prompt = (
            f"Ad text:\n{ocr}\n\n"
            f"Model A: {POSITIVE_LABELS[violation]}\n"
            f"Model B: SAFE\n"
            f"One model flagged this ad, the other did not. Output your JSON verdict."
        )
        raw, meta = ollama_chat_timed(judge_model, JUDGE_PROMPTS[violation],
                                      user_prompt, ollama_url, image_b64=image_b64)
        verdict = parse_judge_response(raw)
        judge_sample_cache[ad_id] = {**verdict, "violation": violation}
        _call_log.append(CallRecord(
            stage="judge_sample", model=judge_model, violation=violation, ad_id=ad_id,
            warmup=False, wall_ms=meta["wall_ms"], load_ms=meta["load_ms"],
            eval_ms=meta["eval_ms"], prompt_tokens=meta["prompt_tokens"],
            output_tokens=meta["output_tokens"], label=verdict.get("judge_label", "ERROR"),
        ))
        log.info("[judge_sample] %s | violation=%s → %s", ad_id, violation, verdict.get("judge_label"))
    log.info("Judge sample done. %d new calls.", sum(1 for c in _call_log if c.stage == "judge_sample"))


def print_summary(records: list[dict], violation: str) -> None:
    total  = len(records)
    pos    = POSITIVE_LABELS[violation]
    ensemb = Counter(r["ensemble_label"] for r in records)
    disag  = sum(1 for r in records if is_disagreement(r))
    judged = sum(1 for r in records if r.get("judge_label") and r["judge_label"] != "ERROR" and r["judge_label"] != "")

    print(f"\n{'='*64}")
    print(f"SUMMARY — {violation.upper()}")
    print(f"  Total records    : {total}")
    print(f"  Ensemble {pos:<22}: {ensemb.get(pos, 0)}")
    print(f"  Ensemble SAFE    : {ensemb.get('SAFE', 0)}")
    print(f"  Disagreements    : {disag}")
    print(f"  Judged           : {judged}")

    if judged:
        judge_counts = Counter(r["judge_label"] for r in records if r.get("judge_label") and r["judge_label"] != "")
        for lbl, cnt in sorted(judge_counts.items()):
            print(f"  Judge {lbl:<24}: {cnt}")

    print(f"{'='*64}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--violations",     nargs="+", default=ALL_VIOLATIONS,
                        choices=ALL_VIOLATIONS, help="Which violations to process")
    parser.add_argument("--classifier-models", nargs="+",
                        default=[QWEN_MODEL, GEMMA_MODEL],
                        help="Ollama model tags for classification")
    parser.add_argument("--judge-model",    default=JUDGE_MODEL,
                        help="Ollama model tag for judging (default: %(default)s). "
                             "Options: gemma4:26b, qwen3.5:27b, mistral-small3.2:24b")
    parser.add_argument("--ollama-url",     default=OLLAMA_URL,
                        help="Ollama base URL (default: %(default)s)")
    parser.add_argument("--out-dir",        type=Path, default=DEFAULT_OUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--limit",          type=int, default=None,
                        help="Cap total number of records to process")
    parser.add_argument("--skip-classify",  action="store_true",
                        help="Skip Phase 1 — use existing cache only")
    parser.add_argument("--skip-judge",     action="store_true",
                        help="Skip Phase 2 — save ensemble results without judging")
    parser.add_argument("--judge-sample",   type=int, default=None, metavar="N",
                        help="Also run the judge on N random records (regardless of disagreement)")
    parser.add_argument("--no-images",      action="store_true",
                        help="Text-only mode — do not pass images to classifiers or judge")
    args = parser.parse_args()

    # ── Argument validation ────────────────────────────────────────────────────
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if args.judge_sample is not None and args.judge_sample <= 0:
        parser.error("--judge-sample must be a positive integer")
    if args.skip_classify and args.skip_judge:
        parser.error("--skip-classify and --skip-judge together leave nothing to do")
    if not args.ollama_url.startswith(("http://", "https://")):
        parser.error(f"--ollama-url must start with http:// or https://, got: {args.ollama_url!r}")
    if not args.classifier_models:
        parser.error("--classifier-models requires at least one model tag")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metadata ──────────────────────────────────────────────────────────
    if not METADATA_FILE.exists():
        log.error("Metadata not found: %s", METADATA_FILE)
        log.error("Expected: sample_data/metadata.json at project root")
        sys.exit(1)
    records: list[dict] = load_json(METADATA_FILE)
    log.info("Loaded %d records from %s", len(records), METADATA_FILE)

    # ── Total record cap ───────────────────────────────────────────────────────
    if args.limit:
        records = records[: args.limit]
        log.info("--limit %d applied: %d records", args.limit, len(records))

    # ── Load classify cache ────────────────────────────────────────────────────
    cache_path = args.out_dir / "classify_cache.json"
    cache: dict = load_json(cache_path) if cache_path.exists() else {}
    log.info("Classify cache: %d entries", len(cache))

    # ── Phase 1: Classify ──────────────────────────────────────────────────────
    if not args.skip_classify:
        log.info("=== Phase 1: Classification ===")
        classifier_models = [
            (m, re.sub(r"[^a-zA-Z0-9]", "_", m.split("/")[-1].lower()))
            for m in args.classifier_models
        ]
        run_classify_phase(
            records, args.violations, args.ollama_url,
            cache, cache_path,
            classifier_models=classifier_models,
            text_only=args.no_images,
        )
    else:
        log.info("Skipping Phase 1 (--skip-classify)")

    # ── Phase 2: Ensemble + Judge ──────────────────────────────────────────────
    classifier_models = [
        (m, re.sub(r"[^a-zA-Z0-9]", "_", m.split("/")[-1].lower()))
        for m in args.classifier_models
    ]
    for violation in args.violations:
        log.info("=== Phase 2: %s ===", violation)

        ensemble_records = build_ensemble_records(records, violation, cache,
                                                   classifier_models=classifier_models)
        if not ensemble_records:
            log.warning("No records for violation=%s", violation)
            continue

        judge_cache_path = args.out_dir / f"judge_cache_{violation}.json"
        judge_cache: dict = load_json(judge_cache_path) if judge_cache_path.exists() else {}

        if not args.skip_judge:
            run_judge_phase(
                ensemble_records, violation, args.judge_model,
                args.ollama_url, judge_cache,
                text_only=args.no_images,
            )
            save_json(judge_cache, judge_cache_path)

        out_path = args.out_dir / f"{violation}_results.json"
        save_json(ensemble_records, out_path)
        print_summary(ensemble_records, violation)

    # ── Judge sample (independent of disagreements) ───────────────────────────
    if args.judge_sample:
        judge_sample_cache_path = args.out_dir / "judge_sample_cache.json"
        judge_sample_cache: dict = load_json(judge_sample_cache_path) if judge_sample_cache_path.exists() else {}
        run_judge_sample_phase(
            records, args.judge_model, args.ollama_url,
            judge_sample_cache, n=args.judge_sample,
            text_only=args.no_images,
        )
        save_json(judge_sample_cache, judge_sample_cache_path)

    # ── Save per-call latency log ──────────────────────────────────────────────
    calls_path = args.out_dir / "latency_calls.json"
    real_calls = [c for c in _call_log if not c.warmup]
    calls_path.write_text(json.dumps([asdict(c) for c in real_calls], indent=2, ensure_ascii=False))
    log.info("Latency log saved → %s (%d calls)", calls_path, len(real_calls))
    log.info("Done. Results in %s", args.out_dir)


if __name__ == "__main__":
    main()
