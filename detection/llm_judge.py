"""
llm_judge.py — LLM judge for model disagreement cases
======================================================
Supports three violation types:
  scareware   — Qwen/Gemma disagree on SCAREWARE vs SAFE
  misleading  — Qwen/Gemma disagree on MISLEADING vs SAFE
  ad_design   — any two of Gemma/GLM/Qwen disagree on Undisclosed Advertiser vs SAFE

For all types the judge receives the ad image + OCR text + per-model reasons.

Two inference backends are supported:
  • Ollama  — default; requires `ollama serve` + model pulled locally
  • GLM     — uses HuggingFace transformers directly (no Ollama needed)
              activated automatically when --judge-model is a GLM HF repo name
              e.g. zai-org/GLM-4.6V-Flash

Available judge models (Ollama):
  qwen3.5:27b            — default; strong reasoning, supports vision
  gemma4:27b             — Google Gemma 4, multimodal
  mistral-small3.2:24b   — Mistral Small 3.2, 24B

Prerequisites (Ollama backend):
  ollama serve
  ollama pull qwen3.5:27b   # or whichever judge model you want
  ollama pull gemma4:27b
  ollama pull mistral-small3.2:24b

Usage:
  python llm_judge.py [--violation scareware|misleading|ad_design]
                      [--judge-model qwen3.5:27b]              # Ollama (default)
                      [--judge-model gemma4:27b]               # Gemma4 via Ollama
                      [--judge-model mistral-small3.2:24b]     # Mistral Small via Ollama
                      [--judge-model zai-org/GLM-4.6V-Flash]  # GLM (HF)
                      [--input-dir results]
                      [--output results/judge_results.json]
                      [--ollama-url http://localhost:11434]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
RESULTS_DIR  = SCRIPT_DIR / "results"
OLLAMA_URL   = "http://localhost:11434"

QWEN35_JUDGE_MODEL  = "qwen3.5:27b"
GEMMA4_JUDGE_MODEL  = "gemma4:27b"
MISTRAL_JUDGE_MODEL = "mistral-small3.2:24b"
JUDGE_MODEL         = QWEN35_JUDGE_MODEL  # default

SAMPLE_DATA_DIR = SCRIPT_DIR.parent / "sample_data"
IMAGE_DIR       = SAMPLE_DATA_DIR / "images"

QWEN_KEY   = "qwen35_9b"
GEMMA_KEY  = "gemma_3_12b_it"
GLM_KEY    = "glm_46v_flash"

AD_DESIGN_MODELS = [GEMMA_KEY, QWEN_KEY]


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    print(f"Saved {len(data)} records → {path}")


def load_image_b64(image_path: Path) -> str | None:
    try:
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")
    except Exception as e:
        print(f"  [warn] Could not read image {image_path}: {e}")
        return None


_image_dir_override: Path | None = None

def resolve_image(ad_id: str) -> Path | None:
    """Return Path to <ad_id>.png in sample_data/images/, or None if not found."""
    base = _image_dir_override if _image_dir_override is not None else IMAGE_DIR
    p = base / f"{ad_id}.png"
    return p if p.exists() else None


# ── Ensemble collation ────────────────────────────────────────────────────────

def _pick_model_files(input_dir: Path, violation: str, model_key: str) -> list[Path]:
    """Return per-bin classified files for one model, preferring non-crid_dedup."""
    all_files = sorted(input_dir.glob(f"{violation}_*{model_key}_ranked_top*_classified.json"))
    non_crid = [f for f in all_files if "crid_dedup" not in f.name]
    return non_crid if non_crid else all_files


def build_ensemble_collated(input_dir: Path, violation: str) -> Path:
    """Build ensemble collated file from individual qwen + gemma classified files."""
    viol_label = violation.upper()

    qwen_files  = _pick_model_files(input_dir, violation, "qwen35_9b")
    gemma_files = _pick_model_files(input_dir, violation, "gemma_3_12b_it")

    if not qwen_files or not gemma_files:
        sys.exit(f"Cannot build ensemble: missing qwen or gemma classified files in {input_dir}")

    gemma_idx = {}
    for f in gemma_files:
        for rec in load_json(f):
            gemma_idx[rec["id"]] = rec

    stem = qwen_files[0].name.split("_qwen35_9b_")[0]
    out_path = input_dir / f"{stem}_ensemble_qwen35_9b_gemma_3_12b_it_ranked_collated_classified.json"

    ensemble = []
    for qf in qwen_files:
        for q in load_json(qf):
            g  = gemma_idx.get(q["id"])
            ql = q["label"]
            gl = g["label"] if g else ""

            if ql == viol_label and gl == viol_label:
                ens_label = viol_label
            elif gl and ql != gl:
                ens_label = "SPLIT"
            else:
                ens_label = "SAFE"

            votes: dict = {}
            for lbl in [ql, gl]:
                if lbl:
                    votes[lbl] = votes.get(lbl, 0) + 1

            rec = {k: v for k, v in q.items() if k not in ("label", "reason")}
            rec["ensemble_label"] = ens_label
            rec["ensemble_votes"] = votes
            rec["per_model"] = {
                QWEN_KEY:  {"label": ql, "reason": q.get("reason", "")},
                GEMMA_KEY: {"label": gl, "reason": g.get("reason", "") if g else ""},
            }
            ensemble.append(rec)

    ensemble.sort(key=lambda x: x["sim_rank"])
    save_json(ensemble, out_path)

    labels = {k: sum(1 for r in ensemble if r["ensemble_label"] == k) for k in set(r["ensemble_label"] for r in ensemble)}
    print(f"Ensemble built: {len(ensemble)} records | {labels}")
    return out_path


# ── Input file collection ─────────────────────────────────────────────────────

def collect_input_files(input_dir: Path, violation: str) -> list[Path]:
    pattern = f"{violation}_*ensemble*_classified.json"
    files = sorted(input_dir.glob(pattern))
    if not files:
        print(f"No ensemble files found — building from individual model files...")
        build_ensemble_collated(input_dir, violation)
        files = sorted(input_dir.glob(pattern))
    if not files:
        sys.exit(f"No files matching '{pattern}' in {input_dir}")
    return files


AD_DESIGN_MODEL_FILES = {
    GEMMA_KEY: "full_gemma3:12b_all.json",
    QWEN_KEY:  "full_qwen35:9b_all.json",
}


def collect_ad_design_records(ad_design_dir: Path) -> list[dict]:
    """Build ensemble records directly from per-model _all.json files."""
    records: dict[str, dict] = {}

    for model_key, fname in AD_DESIGN_MODEL_FILES.items():
        path = ad_design_dir / fname
        if not path.exists():
            print(f"  [warn] Missing: {path}")
            continue
        for rec in load_json(path):
            ad_id = rec["ad_id"]
            if ad_id not in records:
                records[ad_id] = {
                    "ad_id":            ad_id,
                    "image_path":       rec.get("image_path", ""),
                    "dataset_category": rec.get("dataset_category", ""),
                    "translated_ocr":   rec.get("translated_ocr", ""),
                    "matched_keywords": rec.get("matched_keywords", []),
                    "matched_groups":   rec.get("matched_groups", []),
                    "fuzzy_score":      rec.get("fuzzy_score", 0.0),
                    "lemma_match":      rec.get("lemma_match", False),
                    "flagged":          rec.get("flagged", False),
                    "per_model":        {},
                    "ensemble_votes":   {},
                }
            records[ad_id]["per_model"][model_key] = {
                "label":  rec.get("vlm_label", ""),
                "reason": rec.get("vlm_reason", ""),
            }

    # compute ensemble votes + label
    pos = "Undisclosed Advertiser"
    for rec in records.values():
        votes: dict[str, int] = {}
        for md in rec["per_model"].values():
            lbl = md.get("label", "")
            if lbl:
                votes[lbl] = votes.get(lbl, 0) + 1
        rec["ensemble_votes"] = votes
        total = sum(votes.values())
        pos_votes = votes.get(pos, 0)
        rec["ensemble_label"] = pos if pos_votes * 2 > total else "SAFE"

    return list(records.values())


# ── Disagreement filters ──────────────────────────────────────────────────────

def is_disagreement(record: dict) -> bool:
    """Qwen and Gemma give different verdicts (scareware/misleading)."""
    per_model = record.get("per_model", {})
    q = per_model.get(QWEN_KEY, {}).get("label", "").upper()
    g = per_model.get(GEMMA_KEY, {}).get("label", "").upper()
    return bool(q and g and q != g)


def is_ad_design_disagreement(record: dict) -> bool:
    """Qwen and Gemma give different verdicts."""
    per_model = record.get("per_model", {})
    q = per_model.get(QWEN_KEY, {}).get("label", "")
    g = per_model.get(GEMMA_KEY, {}).get("label", "")
    return bool(q and g and q != g)


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_SCAREWARE = (
    "You are an expert cybersecurity analyst specialising in malvertising and scareware. "
    "You will be given an ad image along with its text and the reasoning from two AI models "
    "that disagree on whether it is scareware. "
    "Following directives were provided to the models:\n\n"
    "SCAREWARE — text makes unsubstantiated, declarative claims of existing threats or immediate consequences to induce panic:\n"
    "  • Account Deactivation/Suspension - Examples: \"Your account has been suspended\", \"Update now or you will lose your data\"\n"
    "  • Outdated Version - Examples: \"Your PDF version has expired\", \"The PDF reader toolbar is outdated or too old\"\n"
    "  • Storage & Device Performance - Examples: \"Your phone is no longer working\", \"Your device is X% occupied\"\n"
    "  • Virus / Malware Infection - Examples: \"Viruses found\", \"Infection detected\", \"System corrupted\", \"Your storage is full and your phone will stop functioning\"\n"
    "  • Hacker / Data Breach - Examples: \"Hackers are watching you\", \"Your passwords have been exposed in a data breach\"\n"
    "  • Legal / Law Enforcement - Examples: \"Your IP has been flagged for illegal activity\", \"Unauthorized transactions detected on your card\"\n"
    "  • Privacy Exposure - Examples: \"Your data is being sold on the dark web right now\", \"Your personal photos are visible to everyone on the internet\"\n\n"
    "SAFE — text is suggestive or preventive WITHOUT asserting a current compromise, or has a maybe/question mark (?):\n"
    "  • \"You should scan your device for viruses\"\n"
    "  • \"Use a VPN to stay safe online\"\n"
    "  • \"Does your phone need cleaning?\"\n"
    "  • Surveillance/tracking offers: \"Track your partner's location\", \"Find phone by number\"\n"
    "  • Relationship anxiety: \"Is your partner honest?\", infidelity statistics\n"
    "  • Service descriptions: \"I can detect malware\", \"We analyze phones\", \"Remove it for free\"\n"
    "  • Claims with a question mark (?) or maybe claims — are SAFE because they propose a narrative rather than asserting a current fact (e.g. \"Maybe your phone has viruses?\", \"Could your device be at risk?\")\n"
    "  • Uncertain or tentative language (\"maybe\", \"could\", \"might\", \"possibly\") makes a claim suggestive rather than declarative → SAFE\n\n"
    "Respond ONLY with a valid JSON object using this exact schema:\n"
    '{"judge_reason": "<step-by-step reasoning>", "judge_label": "SCAREWARE" or "SAFE"}'
)

SYSTEM_PROMPT_MISLEADING = (
    "You are an expert digital advertising analyst specialising in misleading ad creatives. "
    "You will be given an ad image along with its text and the reasoning from two AI models "
    "that disagree on whether it is misleading. "
    "Following directives were provided to the models:\n\n"
    "MISLEADING — text makes false or unverifiable claims designed to deceive:\n"
    "  • Fake recovery bait: \"347 deleted files found on your device\", \"Photos deleted 4 years ago recovered\"\n"
    "  • Impossible surveillance: \"Track anyone's location by phone number\", \"Find spirits or spells around you\"\n"
    "  • Social manipulation: \"Find out who secretly hates you\", \"Install this app to find out who has a secret crush on you.\"\n\n"
    "SAFE — honest, verifiable, or genuinely suggestive claims:\n"
    "  • Legitimate promotions with realistic discounts\n"
    "  • Honest app descriptions\n"
    "  • General advice without false premises\n"
    "  • Hyperbolic but generic marketing (\"amazing results\", \"best app\")\n"
    "  • Family/GPS tracking app offers: \"Track your loved ones in real-time\", \"Family Locator\"\n"
    "  • Claims with a question mark (?) or maybe claims — are SAFE because they propose a narrative and not make claims\n\n"
    "Respond ONLY with a valid JSON object using this exact schema:\n"
    '{"judge_reason": "<step-by-step reasoning>", "judge_label": "MISLEADING" or "SAFE"}'
)

SYSTEM_PROMPT_AD_DESIGN = (
    "You are an expert online advertising quality analyst specialising in deceptive "
    "and low-quality ad creatives. "
    "You will be given an ad image and the verdicts from multiple AI models that disagree "
    "on whether the ad is a Undisclosed advertiser (an ad usually with a button or a clickable entity with no identifiable "
    "advertiser information — no brand name, product name, URL, or logo). "
    "Look carefully at the image before deciding.\n"
    "Respond ONLY with a valid JSON object using this exact schema:\n"
    '{"judge_reason": "<step-by-step reasoning>", "judge_label": "Undisclosed Advertiser" or "SAFE"}'
)

SYSTEM_PROMPTS = {
    "scareware":         SYSTEM_PROMPT_SCAREWARE,
    "deceptive_claim":   SYSTEM_PROMPT_MISLEADING,
    "misleading_design": SYSTEM_PROMPT_AD_DESIGN,
}


def build_user_prompt(record: dict, violation: str) -> str:
    per_model = record.get("per_model", {})
    qwen  = per_model.get(QWEN_KEY, {})
    gemma = per_model.get(GEMMA_KEY, {})
    violation_label = violation.upper()

    return (
        f"### Ad Text\n"
        f"OCR text: {record.get('ocr_text', 'N/A')}\n"
        f"Translated/embedded text: {record.get('embed_text', 'N/A')}\n\n"
        f"### Reference match\n"
        f"Closest known {violation} pattern: {record.get('matched_ref', 'N/A')} "
        f"(similarity={record.get('score', 0):.3f})\n\n"
        f"### Model assessments\n"
        f"Qwen ({QWEN_KEY}):\n"
        f"  Verdict : {qwen.get('label', 'N/A')}\n"
        f"  Reason  : {qwen.get('reason', 'N/A')}\n\n"
        f"Gemma ({GEMMA_KEY}):\n"
        f"  Verdict : {gemma.get('label', 'N/A')}\n"
        f"  Reason  : {gemma.get('reason', 'N/A')}\n\n"
        f"One model says {violation_label}, the other says SAFE. "
        f"Carefully analyse the ad image, the ad text, and both reasoning chains, "
        f"then output your JSON."
    )


def build_user_prompt_ad_design(record: dict) -> str:
    per_model = record.get("per_model", {})
    lines = ["### Ad OCR Text", record.get("translated_ocr", "N/A"), "", "### Model assessments"]
    for key in AD_DESIGN_MODELS:
        md = per_model.get(key, {})
        lines.append(f"{key}:")
        lines.append(f"  Verdict : {md.get('label', 'N/A')}")
        lines.append(f"  Reason  : {md.get('reason', 'N/A')}")
    lines.append("")
    lines.append(
        "The models disagree. Look carefully at the image and determine whether the ad "
        "has a CTA button with NO identifying advertiser information (brand, product, URL, logo). "
        "Then output your JSON."
    )
    return "\n".join(lines)


# ── Ollama inference ──────────────────────────────────────────────────────────

def _needs_think_disabled(model: str) -> bool:
    return "qwen3" in model.lower()


def ollama_chat(
    model: str,
    system: str,
    user: str,
    base_url: str,
    image_b64: str | None = None,
    retries: int = 3,
    timeout: int = 120,
) -> str:
    url = f"{base_url.rstrip('/')}/api/chat"

    user_message: dict = {"role": "user", "content": user}
    if image_b64:
        user_message["images"] = [image_b64]

    payload: dict = {
        "model":    model,
        "stream":   False,
        "messages": [
            {"role": "system", "content": system},
            user_message,
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
            print(f"  [warn] Ollama error (attempt {attempt+1}/{retries}): {exc} — retrying in {wait}s")
            time.sleep(wait)
    return ""


def parse_judge_output(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            raw_label = parsed.get("judge_label", "ERROR")
            # Normalise only all-caps labels (SCAREWARE, MISLEADING, SAFE, ERROR)
            # but preserve mixed-case labels like "Undisclosed Advertiser".
            label = raw_label.upper() if raw_label.upper() in {"SCAREWARE", "DECEPTIVE_CLAIM", "SAFE", "ERROR"} else raw_label
            return {
                "judge_label":  label,
                "judge_reason": parsed.get("judge_reason", ""),
            }
        except json.JSONDecodeError:
            pass
    return {
        "judge_label":  "ERROR",
        "judge_reason": f"Failed to parse JSON. Raw: {raw[:200]}",
    }


# ── GLM inference (HuggingFace transformers, no Ollama) ───────────────────────

_GLM_JUDGE_MODELS: set[str] = {
    "zai-org/GLM-4.6V-Flash",
    "zai-org/GLM-4.1V-9B-Thinking",
}
_glm_judge_cache: dict[str, tuple] = {}  # model_name → (model, processor)


def _is_glm_judge(model_name: str) -> bool:
    return model_name in _GLM_JUDGE_MODELS


def _get_glm_judge(model_name: str) -> tuple:
    """Load (and cache) a GLM judge model + processor."""
    if model_name not in _glm_judge_cache:
        from transformers import AutoProcessor, Glm4vForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_name)
        model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()
        _glm_judge_cache[model_name] = (model, processor)
    return _glm_judge_cache[model_name]


def glm_chat(
    model_name: str,
    system: str,
    user: str,
    image_b64: str | None = None,
    retries: int = 3,
) -> str:
    """Run GLM inference locally via transformers. Returns raw decoded text."""
    import torch

    model, processor = _get_glm_judge(model_name)

    image_url = f"data:image/png;base64,{image_b64}" if image_b64 else None

    if image_url:
        user_content = [
            {"type": "image", "url": image_url},
            {"type": "text", "text": f"{system}\n\n{user}"},
        ]
    else:
        user_content = [{"type": "text", "text": f"{system}\n\n{user}"}]

    messages = [{"role": "user", "content": user_content}]

    raw_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    if hasattr(raw_inputs, "to"):
        inputs = raw_inputs.to(model.device)
    else:
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in raw_inputs.items()}
    inputs.pop("token_type_ids", None)

    input_len = inputs["input_ids"].shape[1]

    for attempt in range(retries):
        try:
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            new_tokens = generated_ids[0][input_len:]
            raw_out = processor.decode(new_tokens, skip_special_tokens=False).strip()
            # strip thinking blocks before JSON parsing downstream
            return re.sub(r"<think>.*?</think>", "", raw_out, flags=re.DOTALL).strip()
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [warn] GLM error (attempt {attempt+1}/{retries}): {exc} — retrying in {wait}s")
            time.sleep(wait)
    return ""


# ── Backend dispatcher ─────────────────────────────────────────────────────────

def judge_chat(
    model: str,
    system: str,
    user: str,
    base_url: str,
    image_b64: str | None = None,
    retries: int = 3,
    timeout: int = 120,
) -> str:
    """Route to GLM (local transformers) or Ollama depending on the model name."""
    if _is_glm_judge(model):
        return glm_chat(model, system, user, image_b64=image_b64, retries=retries)
    return ollama_chat(model, system, user, base_url, image_b64=image_b64, retries=retries, timeout=timeout)


# ── Pipelines ─────────────────────────────────────────────────────────────────

def run_judge_scareware_misleading(
    violation: str,
    judge_model: str,
    input_dir: Path,
    output_path: Path,
    ollama_url: str,
) -> None:
    files = collect_input_files(input_dir, violation)
    print(f"Found {len(files)} input files for violation='{violation}'")

    all_records: dict[str, dict] = {}
    for f in files:
        for rec in load_json(f):
            all_records[rec["id"]] = rec
    print(f"Total records loaded: {len(all_records)}")

    disagreements = [r for r in all_records.values() if is_disagreement(r)]
    print(f"Disagreements (Qwen vs Gemma): {len(disagreements)}")

    existing: dict[str, dict] = {}
    if output_path.exists():
        for rec in load_json(output_path):
            existing[rec["id"]] = rec
        print(f"Resuming — {len(existing)} already judged")

    results = list(existing.values())
    judged_ids = set(existing.keys())
    pending = [r for r in disagreements if r["id"] not in judged_ids]
    print(f"Pending: {len(pending)}")

    system_prompt = SYSTEM_PROMPTS[violation]

    for i, record in enumerate(pending, 1):
        img_path  = resolve_image(record["id"])
        image_b64 = load_image_b64(img_path) if img_path else None
        if not image_b64:
            print(f"  [warn] No image for {record['id']}")

        user_prompt = build_user_prompt(record, violation)
        raw     = judge_chat(judge_model, system_prompt, user_prompt, ollama_url,
                             image_b64=image_b64)
        verdict = parse_judge_output(raw)

        enriched = {**record, **verdict, "judge_model": judge_model}
        results.append(enriched)

        q = record["per_model"].get(QWEN_KEY, {}).get("label", "?")
        g = record["per_model"].get(GEMMA_KEY, {}).get("label", "?")
        print(f"  [{i}/{len(pending)}] {record['id']} | qwen={q} gemma={g} → judge={verdict['judge_label']}")

        if i % 25 == 0:
            save_json(results, output_path)

    save_json(results, output_path)
    _print_summary(results)


def run_judge_ad_design(
    judge_model: str,
    ad_design_dir: Path,
    output_path: Path,
    ollama_url: str,
) -> None:
    all_records = collect_ad_design_records(ad_design_dir)
    print(f"Total ad_design records: {len(all_records)}")

    disagreements = [r for r in all_records if is_ad_design_disagreement(r)]
    print(f"Disagreements (non-unanimous ensemble): {len(disagreements)}")

    existing: dict[str, dict] = {}
    if output_path.exists():
        for rec in load_json(output_path):
            existing[rec["ad_id"]] = rec
        print(f"Resuming — {len(existing)} already judged")

    results = list(existing.values())
    judged_ids = set(existing.keys())
    pending = [r for r in disagreements if r["ad_id"] not in judged_ids]
    print(f"Pending: {len(pending)}")

    for i, record in enumerate(pending, 1):
        img_path  = resolve_image(record["ad_id"])
        image_b64 = load_image_b64(img_path) if img_path else None
        if not image_b64:
            print(f"  [warn] No image for {record['ad_id']}")

        user_prompt = build_user_prompt_ad_design(record)
        raw     = judge_chat(judge_model, SYSTEM_PROMPT_AD_DESIGN, user_prompt, ollama_url,
                             image_b64=image_b64)
        verdict = parse_judge_output(raw)

        enriched = {**record, **verdict, "judge_model": judge_model}
        results.append(enriched)

        votes = record.get("ensemble_votes", {})
        print(f"  [{i}/{len(pending)}] {record['ad_id']} | votes={votes} → judge={verdict['judge_label']}")

        if i % 25 == 0:
            save_json(results, output_path)

    save_json(results, output_path)
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    from collections import Counter
    labels = Counter(r.get("judge_label") for r in results)
    print("\n── Judge summary ──")
    for label, count in sorted(labels.items()):
        print(f"  {label}: {count}")


def judge_one_record(
    record: dict,
    violation: str,
    judge_model: str = JUDGE_MODEL,
    ollama_url: str = OLLAMA_URL,
) -> dict:
    """Judge a single record in-process. Returns enriched dict with judge_label/judge_reason."""
    system_prompt = SYSTEM_PROMPTS.get(violation, SYSTEM_PROMPT_AD_DESIGN)
    if violation == "misleading_design":
        user_prompt = build_user_prompt_ad_design(record)
    else:
        user_prompt = build_user_prompt(record, violation)

    ad_id     = record.get("ad_id") or record.get("id", "")
    img_path  = resolve_image(ad_id)
    image_b64 = load_image_b64(img_path) if img_path else None

    raw     = judge_chat(judge_model, system_prompt, user_prompt, ollama_url, image_b64=image_b64)
    verdict = parse_judge_output(raw)
    return {**record, **verdict, "judge_model": judge_model}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM judge for model disagreements")
    parser.add_argument("--violation",       default="scareware",
                        choices=["scareware", "deceptive_claim", "misleading_design"],
                        help="Which violation type to process (default: scareware)")
    parser.add_argument("--judge-model",          default=JUDGE_MODEL,
                        help="Ollama model tag to use as judge (default: %(default)s). "
                             "Options: qwen3.5:27b, gemma4:27b, mistral-small3.2:24b")
    parser.add_argument("--results-dir",          default=str(RESULTS_DIR),
                        help="Base results directory; violation files are read from "
                             "<results-dir>/<violation>/ (default: detection/results/)")
    parser.add_argument("--output",               default=None,
                        help="Output JSON path (default: <results-dir>/<violation>/<violation>_judge_results.json)")
    parser.add_argument("--ollama-url",           default=OLLAMA_URL,
                        help="Ollama base URL (default: %(default)s)")
    parser.add_argument("--images-dir",           default=None,
                        help="Override image lookup directory (default: sample_data/images/)")
    parser.add_argument("--crid",                 default=None,
                        help="Judge a single creative ID, e.g. CR18077324848429793281 or CR18077324848429793281-v0")
    args = parser.parse_args()

    # ── Argument validation ────────────────────────────────────────────────────
    if not args.ollama_url.startswith(("http://", "https://")):
        parser.error(f"--ollama-url must start with http:// or https://, got: {args.ollama_url!r}")
    if args.images_dir is not None and not Path(args.images_dir).exists():
        parser.error(f"--images-dir does not exist: {args.images_dir}")

    global _image_dir_override
    if args.images_dir:
        _image_dir_override = Path(args.images_dir)

    violation_dir = Path(args.results_dir) / args.violation

    if args.crid:
        crid = args.crid if re.match(r".*-v\d+$", args.crid) else args.crid + "-v0"

        if args.violation == "misleading_design":
            all_records = collect_ad_design_records(violation_dir)
            record = next((r for r in all_records if r["ad_id"] == crid), None)
            src = str(violation_dir)
        else:
            files = collect_input_files(violation_dir, args.violation)
            all_records = {r["id"]: r for f in files for r in load_json(f)}
            record = all_records.get(crid)
            src = str(violation_dir)

        if record is None:
            sys.exit(f"No record found for '{crid}' in {src}")

        result = judge_one_record(
            record=record,
            violation=args.violation,
            judge_model=args.judge_model,
            ollama_url=args.ollama_url,
        )
        print(f"\nad_id       : {result.get('ad_id') or result.get('id')}")
        print(f"judge_label : {result['judge_label']}")
        print(f"judge_reason: {result['judge_reason']}")
        print(f"judge_model : {result['judge_model']}")
        return

    model_slug = re.sub(r"[^a-zA-Z0-9]", "_", args.judge_model)

    if args.violation == "misleading_design":
        output_path = Path(args.output) if args.output else violation_dir / f"{args.violation}_judge_results_{model_slug}.json"
        run_judge_ad_design(
            judge_model=args.judge_model,
            ad_design_dir=violation_dir,
            output_path=output_path,
            ollama_url=args.ollama_url,
        )
    else:
        output_path = Path(args.output) if args.output else violation_dir / f"{args.violation}_judge_results_{model_slug}.json"
        run_judge_scareware_misleading(
            violation=args.violation,
            judge_model=args.judge_model,
            input_dir=violation_dir,
            output_path=output_path,
            ollama_url=args.ollama_url,
        )


if __name__ == "__main__":
    main()
