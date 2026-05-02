#!/usr/bin/env python3
"""
run_all.py  —  AdLens end-to-end pipeline.

Crawl → OCR → Translate → Detect → one consolidated JSON.

Modes:
  python run_all.py --image path/to/ad.png          # single local image
  python run_all.py --csv   ads.csv [--workers 2]   # bulk crawl from CSV

Output:
  <out-dir>/screenshots/        ad screenshots (crawl mode only)
  <out-dir>/ads_after_ocr.json        intermediate: OCR added
  <out-dir>/ads_after_translate.json  intermediate: translations added
  <out-dir>/ads_final.json            final: all stages merged

Each ad record in ads_final.json has:
  creativeID, advertiserID, advertiserName, screenshotPath   (from crawler)
  variants[]:
    screenshot_path, ocr_status, ocr_text, target_urls      (from OCR)
    translated_ocr_text                                      (from translate)
    classification: {scareware, deceptive_claim,
                     misleading_design}                      (from detect)
    malicious, detected_violations                           (from detect)
  malicious, detected_violations                             (top-level)
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ── repo root on sys.path so submodule imports work ──────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))               # utils/
sys.path.insert(0, str(ROOT / "ocr"))       # ocr_engine
sys.path.insert(0, str(ROOT / "detection")) # ensemble, translate_ocr, etc.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_all")

# ── Violation config ──────────────────────────────────────────────────────────
ALL_VIOLATIONS = ["scareware", "deceptive_claim", "misleading_design"]

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


# ── JSON helpers ──────────────────────────────────────────────────────────────
def load_json(path: Path) -> object:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    log.info("Saved → %s", path)


def model_key(model_name: str) -> str:
    """'qwen3.5:9b' → 'qwen3_5_9b'"""
    return re.sub(r"[^a-zA-Z0-9]", "_", model_name.split("/")[-1].lower())


# ── Ollama helper ─────────────────────────────────────────────────────────────
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
) -> tuple[str, dict]:
    import requests
    url = f"{base_url.rstrip('/')}/api/chat"
    user_msg: dict = {"role": "user", "content": user}
    if image_b64:
        user_msg["images"] = [image_b64]

    payload: dict = {
        "model":    model,
        "stream":   False,
        "messages": [{"role": "system", "content": system}, user_msg],
        "options":  {"temperature": 0},
    }
    if _needs_think_disabled(model):
        payload["think"] = False

    for attempt in range(retries):
        try:
            import requests as _req
            t0 = time.perf_counter()
            resp = _req.post(url, json=payload, timeout=timeout)
            wall_ms = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"], {"wall_ms": round(wall_ms, 1)}
        except Exception as exc:
            wait = 2 ** attempt
            log.warning("Ollama attempt %d/%d: %s — retry in %ds", attempt + 1, retries, exc, wait)
            time.sleep(wait)
    return "", {}


def _img_b64(path_str: str) -> Optional[str]:
    try:
        return base64.b64encode(Path(path_str).read_bytes()).decode()
    except Exception as e:
        log.warning("Cannot read image %s: %s", path_str, e)
        return None


# ── Response parsers ──────────────────────────────────────────────────────────
def parse_classify(raw: str, valid_labels: set) -> dict:
    match = re.search(r'\{[^{}]*"label"[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            label = parsed.get("label", "")
            if label in valid_labels:
                return {"label": label, "reason": parsed.get("reason", "")}
        except json.JSONDecodeError:
            pass
    for lbl in sorted(valid_labels, key=len, reverse=True):
        if re.search(re.escape(lbl), raw, re.IGNORECASE):
            return {"label": lbl, "reason": ""}
    return {"label": "ERROR", "reason": raw[:300]}


def parse_judge(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            raw_label = parsed.get("judge_label", "ERROR")
            # Normalise known all-caps labels; preserve casing for misleading_design ("Undisclosed Advertiser")
            if raw_label.upper() in {"SCAREWARE", "MISLEADING", "SAFE", "ERROR"}:
                label = raw_label.upper()
            else:
                label = raw_label
            return {"judge_label": label, "judge_reason": parsed.get("judge_reason", "")}
        except json.JSONDecodeError:
            pass
    return {"judge_label": "ERROR", "judge_reason": f"parse failed: {raw[:200]}"}


# ── Stage 1: Crawl ────────────────────────────────────────────────────────────
def stage_crawl(csv_file: Path, out_dir: Path, workers: int, limit: Optional[int] = None) -> list[dict]:
    log.info("=== Stage 1: Crawl ===")
    screenshots_dir = out_dir / "screenshots"
    crawl_out       = out_dir / "ads_crawled.json"

    cmd = [
        sys.executable, str(ROOT / "crawler" / "run.py"),
        str(csv_file),
        "--workers",        str(workers),
        "--output",         str(crawl_out),
        "--screenshots-dir", str(screenshots_dir),
    ]
    if limit is not None:
        cmd += ["--max", str(limit)]
    log.info("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        log.error("Crawler exited with code %d", result.returncode)
        sys.exit(result.returncode)

    records = load_json(crawl_out)
    log.info("Crawled %d ad records", len(records))
    return records


def single_image_record(image_path: Path) -> list[dict]:
    """Wrap a single local image as a synthetic crawler record."""
    return [{
        "creativeID":     image_path.stem,
        "advertiserID":   None,
        "advertiserName": None,
        "screenshotPath": [str(image_path.resolve())],
        "status":         "success",
    }]


# ── Stage 2: OCR ──────────────────────────────────────────────────────────────
def stage_ocr(records: list[dict], out_dir: Path, device: str) -> list[dict]:
    log.info("=== Stage 2: OCR ===")
    from ocr_engine import OCREngine
    from utils import ImgCacheController

    ocr_engine = OCREngine(lang="en", device=device)
    cache_dir  = str(out_dir / "ocr_cache")

    # Build a flat task list for all images across all records
    all_tasks: list[dict] = []
    for i, rec in enumerate(records):
        for path_str in rec.get("screenshotPath", []):
            all_tasks.append({"file_path": path_str, "_rec_idx": i})

    log.info("Running OCR on %d image(s)", len(all_tasks))

    ocr_by_path: dict[str, dict] = {}
    pending: list[dict] = []

    with ImgCacheController(cache_dir, save_interval=10) as cache:
        for task in all_tasks:
            p = task["file_path"]
            if not Path(p).exists():
                log.warning("Image not found: %s", p)
                ocr_by_path[p] = {"ocr_text": "", "target_urls": [], "ocr_status": "error_file_not_found"}
                continue
            img_bytes = Path(p).read_bytes()
            task["_img_bytes"] = img_bytes
            hit = cache.get_image_data(img_bytes)
            if hit:
                ocr_by_path[p] = {**hit, "ocr_status": "completed_from_cache"}
            else:
                pending.append(task)

        if pending:
            results = ocr_engine.run_batch_inference(pending)
            result_by_path = {r["task"]["file_path"]: r["text"] for r in results}
            for task in pending:
                p = task["file_path"]
                if p in result_by_path:
                    text = result_by_path[p]
                    urls = ocr_engine.extract_links(text)
                    cache.set_image_data(task["_img_bytes"], {"ocr_text": text, "target_urls": urls})
                    ocr_by_path[p] = {"ocr_text": text, "target_urls": urls, "ocr_status": "completed"}
                else:
                    ocr_by_path[p] = {"ocr_text": "", "target_urls": [], "ocr_status": "error_ocr_failed"}

    for rec in records:
        variants = []
        for p in rec.get("screenshotPath", []):
            r = ocr_by_path.get(p, {"ocr_text": "", "target_urls": [], "ocr_status": "not_processed"})
            variants.append({
                "screenshot_path": p,
                "ocr_status":      r["ocr_status"],
                "ocr_text":        r["ocr_text"],
                "target_urls":     r["target_urls"],
            })
        rec["variants"] = variants

    save_json(records, out_dir / "ads_after_ocr.json")
    return records


# ── Stage 3: Translate ────────────────────────────────────────────────────────
def stage_translate(
    records: list[dict],
    out_dir: Path,
    translate_model: str,
    ollama_url: str,
) -> list[dict]:
    log.info("=== Stage 3: Translate ===")
    from translate_ocr import get_translation

    cache_path = out_dir / "translation_cache.json"
    cache: dict = load_json(cache_path) if cache_path.exists() else {}

    for rec in records:
        for variant in rec.get("variants", []):
            ocr = variant.get("ocr_text", "")
            variant["translated_ocr_text"] = get_translation(ocr, cache, translate_model, ollama_url)
        # persist cache after each ad so interruptions don't lose work
        save_json(cache, cache_path)

    save_json(records, out_dir / "ads_after_translate.json")
    return records


# ── Stage 4: Detect ───────────────────────────────────────────────────────────
def stage_detect(
    records: list[dict],
    out_dir: Path,
    classifier_models: list[tuple[str, str]],
    judge_model: str,
    ollama_url: str,
) -> list[dict]:
    log.info("=== Stage 4: Detect ===")

    from ensemble import PROMPT_SCAREWARE, PROMPT_MISLEADING
    from misleading_design import VLM_PROMPT_CTA_NO_ADVERTISER as PROMPT_AD_DESIGN
    from llm_judge import SYSTEM_PROMPTS as JUDGE_PROMPTS

    CLASSIFY_PROMPTS = {
        "scareware":         PROMPT_SCAREWARE,
        "deceptive_claim":   PROMPT_MISLEADING,
        "misleading_design": PROMPT_AD_DESIGN,
    }

    cache_path = out_dir / "classify_cache.json"
    cache: dict = load_json(cache_path) if cache_path.exists() else {}

    def _classify(vid: str, variant: dict, violation: str, m_name: str, m_key: str) -> dict:
        ck = f"{vid}:{violation}:{m_key}"
        if ck in cache:
            return cache[ck]
        ocr     = variant.get("translated_ocr_text") or variant.get("ocr_text") or ""
        img_b64 = _img_b64(variant["screenshot_path"])
        if violation == "misleading_design":
            user_msg = ("Analyse this advertisement image and classify it."
                        + (f" OCR text extracted from the image: '{ocr}'" if ocr else ""))
        else:
            user_msg = (f"Classify this advertisement.\n\nAd text:\n{ocr}"
                        + ("\n\nAnalyse the image as well." if img_b64 else ""))
        raw, _ = ollama_chat(m_name, CLASSIFY_PROMPTS[violation], user_msg, ollama_url, image_b64=img_b64)
        result = parse_classify(raw, VALID_LABELS[violation])
        cache[ck] = result
        save_json(cache, cache_path)
        return result

    def _judge(vid: str, variant: dict, violation: str, per_model: dict) -> dict:
        ck = f"{vid}:{violation}:judge"
        if ck in cache:
            return cache[ck]
        ocr     = variant.get("translated_ocr_text") or variant.get("ocr_text") or ""
        img_b64 = _img_b64(variant["screenshot_path"])
        pos     = POSITIVE_LABELS[violation]
        keys    = list(per_model)
        a, b    = per_model[keys[0]], per_model[keys[1]]
        user_msg = "\n".join([
            "### Ad Text", ocr, "",
            "### Model assessments",
            f"Model A ({keys[0]}):",
            f"  Verdict : {a.get('label', 'N/A')}",
            f"  Reason  : {a.get('reason', 'N/A')}",
            "",
            f"Model B ({keys[1]}):",
            f"  Verdict : {b.get('label', 'N/A')}",
            f"  Reason  : {b.get('reason', 'N/A')}",
            "",
            f"One model says {pos}, the other says SAFE. "
            "Carefully analyse the ad image, the ad text, and both reasoning chains, "
            "then output your JSON.",
        ])
        raw, _ = ollama_chat(judge_model, JUDGE_PROMPTS[violation], user_msg, ollama_url, image_b64=img_b64)
        result = parse_judge(raw)
        cache[ck] = result
        save_json(cache, cache_path)
        return result

    for rec in records:
        creative_id = rec.get("creativeID") or rec.get("id", "unknown")
        for vi, variant in enumerate(rec.get("variants", [])):
            vid = f"{creative_id}_v{vi}"
            classification: dict[str, dict] = {}

            for violation in ALL_VIOLATIONS:
                per_model: dict[str, dict] = {}
                for m_name, m_key in classifier_models:
                    per_model[m_key] = _classify(vid, variant, violation, m_name, m_key)

                # Majority-vote ensemble
                pos = POSITIVE_LABELS[violation]
                votes: dict[str, int] = {}
                for r in per_model.values():
                    lbl = r.get("label", "")
                    if lbl and lbl != "ERROR":
                        votes[lbl] = votes.get(lbl, 0) + 1
                total     = sum(votes.values())
                pos_votes = votes.get(pos, 0)
                ensemble  = pos if (total > 0 and pos_votes * 2 > total) else "SAFE"

                # Judge on disagreement
                valid_labels_seen = [r.get("label", "") for r in per_model.values()
                                     if r.get("label") not in ("", "ERROR")]
                disagreement = len(set(valid_labels_seen)) > 1 and len(valid_labels_seen) >= 2
                judge: Optional[dict] = _judge(vid, variant, violation, per_model) if disagreement else None

                final = (judge.get("judge_label") if judge else ensemble) or ensemble
                classification[violation] = {
                    "per_model":    per_model,
                    "ensemble":     ensemble,
                    "judge":        judge,
                    "final":        final,
                    "is_violation": final == pos,
                }

            variant["classification"]      = classification
            variant["malicious"]           = any(classification[v]["is_violation"] for v in ALL_VIOLATIONS)
            variant["detected_violations"] = [v for v in ALL_VIOLATIONS if classification[v]["is_violation"]]

        rec["malicious"] = any(v.get("malicious", False) for v in rec.get("variants", []))
        rec["detected_violations"] = list({
            viol
            for v in rec.get("variants", [])
            for viol in v.get("detected_violations", [])
        })

    save_json(records, out_dir / "ads_final.json")
    return records


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, metavar="IMG",
                     help="Single local image to classify (skips crawl)")
    src.add_argument("--csv",   type=Path, metavar="CSV",
                     help="CSV of advertiser/creative IDs to crawl then classify")

    parser.add_argument("--workers",           type=int,  default=2,
                        help="Parallel crawl workers (CSV mode only, default: 2)")
    parser.add_argument("--out-dir",           type=Path, default=Path("run_all_output"),
                        help="Output directory (default: run_all_output/)")
    parser.add_argument("--device",            default="cpu",
                        help="OCR device: cpu | gpu:0 (default: cpu)")
    parser.add_argument("--ollama-url",        default="http://localhost:11434",
                        help="Ollama base URL (default: %(default)s)")
    parser.add_argument("--classifier-models", nargs="+",
                        default=["qwen3.5:9b", "gemma3:12b"],
                        help="Classifier model tags (default: qwen3.5:9b gemma3:12b)")
    parser.add_argument("--judge-model",       default="gemma4:26b",
                        help="Judge model for disagreements (default: %(default)s)")
    parser.add_argument("--translate-model",   default="translategemma:4b",
                        help="Translation model (default: %(default)s)")

    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Cap the number of ads processed (applied after crawl/load)")

    # Skip flags for resuming after partial runs
    parser.add_argument("--skip-crawl",     action="store_true",
                        help="Skip Stage 1 — load ads_crawled.json from --out-dir")
    parser.add_argument("--skip-ocr",       action="store_true",
                        help="Skip Stage 2 — load ads_after_ocr.json from --out-dir")
    parser.add_argument("--skip-translate", action="store_true",
                        help="Skip Stage 3 — load ads_after_translate.json from --out-dir")
    parser.add_argument("--skip-detect",    action="store_true",
                        help="Skip Stage 4 — save collated JSON without detection")

    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image and not args.image.exists():
        parser.error(f"--image path does not exist: {args.image}")
    if args.csv and not args.csv.exists():
        parser.error(f"--csv path does not exist: {args.csv}")

    c_models = [(m, model_key(m)) for m in args.classifier_models]

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if args.image:
        records = single_image_record(args.image)
        log.info("Single-image mode: %s", args.image.name)
    elif args.skip_crawl:
        p = out_dir / "ads_crawled.json"
        if not p.exists():
            parser.error(f"--skip-crawl requires {p}")
        records = load_json(p)
        log.info("Loaded %d records (crawl skipped)", len(records))
    else:
        records = stage_crawl(args.csv, out_dir, args.workers, args.limit)

    if args.limit is not None:
        records = records[: args.limit]
        log.info("--limit %d applied: processing %d record(s)", args.limit, len(records))

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    if args.skip_ocr:
        p = out_dir / "ads_after_ocr.json"
        if not p.exists():
            parser.error(f"--skip-ocr requires {p}")
        records = load_json(p)
        log.info("Loaded %d records (OCR skipped)", len(records))
    else:
        records = stage_ocr(records, out_dir, args.device)

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    if args.skip_translate:
        p = out_dir / "ads_after_translate.json"
        if not p.exists():
            parser.error(f"--skip-translate requires {p}")
        records = load_json(p)
        log.info("Loaded %d records (translate skipped)", len(records))
    else:
        records = stage_translate(records, out_dir, args.translate_model, args.ollama_url)

    # ── Stage 4 ──────────────────────────────────────────────────────────────
    if not args.skip_detect:
        records = stage_detect(records, out_dir, c_models, args.judge_model, args.ollama_url)
    else:
        save_json(records, out_dir / "ads_final.json")
        log.info("Detection skipped.")

    malicious = sum(1 for r in records if r.get("malicious"))
    log.info("Done. %d / %d ad(s) flagged as malicious → %s",
             malicious, len(records), out_dir / "ads_final.json")


if __name__ == "__main__":
    main()
