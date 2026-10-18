#!/usr/bin/env python3
"""
eval_pipeline_results.py — Recompute Table 2 from `adlens_pipeline.py` output.

Run the pipeline in evaluation mode first, then score it:

    # Table 2(a)/(b): labeled golden set (1,283 ads)
    python detection/adlens_pipeline.py --eval
    python scripts/eval_pipeline_results.py

    # Table 2(c): held-out look-forward test set (see datasets/golden_test_data/README.md)
    python detection/adlens_pipeline.py \\
        --metadata datasets/golden_test_data/metadata.json \\
        --images-dir datasets/golden_test_data/images \\
        --out-dir detection/results/test_set_results
    python scripts/eval_pipeline_results.py \\
        --metadata datasets/golden_test_data/metadata.json \\
        --results-dir detection/results/test_set_results

Reported:
  (a) each classifier on every labeled ad of the violation type
  (b) the judge on the classifier-disagreement cases only
  (c) the full pipeline: unanimous ensemble label, judge label on disagreements
      (this is the "Gemma4:26b (judge)" row of Table 2(c) when run on the test set)

Predictions that are ERROR / missing count as "not flagged". Positive class =
the violation label; ground truth is the record's `label` (`tp` -> positive,
`tn` -> negative). Only the standard library is needed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA = ROOT / "datasets" / "golden_main_data" / "metadata.json"
DEFAULT_RESULTS = ROOT / "detection" / "results" / "pipeline_results"

# pipeline violation -> (column title, label used by the model for "violation")
VIOLATIONS = {
    "deceptive_claim":   ("Deceptive Claims",   "MISLEADING"),
    "scareware":         ("Scareware",          "SCAREWARE"),
    "misleading_design": ("Misleading Design",  "Undisclosed Advertiser"),
}
ALIASES = {"misleading": "deceptive_claim", "ad_design": "misleading_design"}
RESULT_FILES = {
    "deceptive_claim":   "misleading_results.json",
    "scareware":         "scareware_results.json",
    "misleading_design": "ad_design_results.json",
}


def norm(violation_type: str) -> str:
    return ALIASES.get(violation_type, violation_type)


def prf(pairs):
    """pairs: iterable of (truth_positive, pred_positive) -> (P, R, F1, tp, fp, fn, tn)."""
    tp = sum(t and p for t, p in pairs)
    fp = sum((not t) and p for t, p in pairs)
    fn = sum(t and (not p) for t, p in pairs)
    tn = sum((not t) and (not p) for t, p in pairs)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F, tp, fp, fn, tn


def is_pos(label: str, positive: str) -> bool:
    return (label or "").strip().lower() == positive.lower()


def load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def model_key(tag: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", tag.split("/")[-1].lower())


def print_table(title: str, rows: dict[str, dict[str, list]], note: str = "") -> None:
    cols = [v for v in VIOLATIONS if any(v in r for r in rows.values())]
    print(f"\n{'=' * 118}\n  {title}")
    if note:
        print(f"  {note}")
    head = f"  {'Model':<26}" + "".join(f"{VIOLATIONS[v][0]:<30}" for v in cols) + "Macro P / R / F1"
    print(head)
    print(f"  {'':<26}" + "".join(f"{'P':>6} {'R':>6} {'F1':>6} {'n':>5}   " for _ in cols))
    print("  " + "-" * 114)
    for name, per_v in rows.items():
        parts, ms = [], []
        for v in cols:
            if v not in per_v:
                parts.append(" " * 30)
                continue
            P, R, F, tp, fp, fn, tn = prf(per_v[v])
            ms.append((P, R, F))
            parts.append(f"{P:6.3f} {R:6.3f} {F:6.3f} {len(per_v[v]):5d}   ")
        macro = ""
        if len(ms) == len(cols) and ms:
            macro = "  ".join(f"{sum(m[i] for m in ms) / len(ms):.3f}" for i in range(3))
        print(f"  {name:<26}" + "".join(parts) + macro)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=DEFAULT_METADATA,
                    help="labeled metadata with violation_type + label (default: %(default)s)")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                    help="adlens_pipeline.py --out-dir (default: %(default)s)")
    ap.add_argument("--classifier-models", nargs="+", default=["qwen3.5:9b", "gemma3:12b"],
                    help="classifier tags used in the run (default: %(default)s)")
    ap.add_argument("--judge-name", default="Gemma4:26b (judge)")
    ap.add_argument("--json-out", type=Path, default=None, help="also write the numbers to this JSON file")
    args = ap.parse_args()

    if not args.metadata.exists():
        sys.exit(f"metadata not found: {args.metadata}")
    meta = load(args.metadata)
    if not any("label" in r and "violation_type" in r for r in meta):
        sys.exit("metadata has no `violation_type`/`label` fields — use datasets/golden_main_data/metadata.json")
    truth: dict[str, dict[str, bool]] = {}      # violation -> ad_id -> is_positive
    for r in meta:
        truth.setdefault(norm(r["violation_type"]), {})[r["ad_id"]] = (r["label"] == "tp")

    cache_path = args.results_dir / "classify_cache.json"
    if not cache_path.exists():
        sys.exit(f"{cache_path} not found — run adlens_pipeline.py first (see --help)")
    cache = load(cache_path)

    clf_rows: dict[str, dict] = {}
    for tag in args.classifier_models:
        keys = (model_key(tag), tag.lower().replace(".", "").replace(":", "_"))  # current / legacy spelling
        for v, (_, pos) in VIOLATIONS.items():
            pairs = []
            for ad_id, t in truth.get(v, {}).items():
                entry = next((cache[f"{ad_id}:{v}:{k}"] for k in keys if f"{ad_id}:{v}:{k}" in cache), None)
                if entry is None:
                    continue                      # not run (e.g. --limit / --violations)
                pairs.append((t, is_pos(entry.get("label", ""), pos)))
            if pairs:
                clf_rows.setdefault(tag, {})[v] = pairs

    judge_rows: dict[str, dict] = {}
    pipe_rows: dict[str, dict] = {}
    ens_rows: dict[str, dict] = {}
    for v, (_, pos) in VIOLATIONS.items():
        path = args.results_dir / RESULT_FILES[v]
        if not path.exists():
            continue
        jp, pp, ep = [], [], []
        for r in load(path):
            t = truth.get(v, {}).get(r["ad_id"])
            if t is None:
                continue
            ens = is_pos(r.get("ensemble_label", ""), pos)
            ep.append((t, ens))
            if r.get("judge_label") and r["judge_label"] != "ERROR":
                j = is_pos(r["judge_label"], pos)
                jp.append((t, j))
                pp.append((t, j))
            else:
                pp.append((t, ens))
        if ep:
            ens_rows.setdefault("Majority-vote ensemble", {})[v] = ep
        if jp:
            judge_rows.setdefault(args.judge_name, {})[v] = jp
        if pp:
            pipe_rows.setdefault("Full pipeline", {})[v] = pp

    print_table("(a) Ensemble classifiers — all labeled ads of each violation type", {**clf_rows, **ens_rows})
    print_table("(b) Judge — classifier-disagreement cases only", judge_rows,
                "n = number of disagreements the judge resolved")
    print_table("(c) Full pipeline (ensemble + judge on disagreements)", pipe_rows)
    print()

    if args.json_out:
        out = {}
        for sec, rows in (("classifiers", clf_rows), ("ensemble", ens_rows), ("judge", judge_rows), ("pipeline", pipe_rows)):
            out[sec] = {m: {v: dict(zip(("precision", "recall", "f1", "tp", "fp", "fn", "tn"), prf(p)))
                            for v, p in per.items()} for m, per in rows.items()}
        args.json_out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
