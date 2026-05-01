"""
f1_score.py — Self-contained reproduction of Table 1 (combined results).

All data is defined inline as (y_true, y_pred) label arrays derived from
the confusion-matrix counts of each model × violation-type × evaluation split.
No file I/O or external dependencies beyond the standard library.

Sections:
  (a) Ensemble Classifiers  — golden set, 1283 items (all items evaluated)
  (b) Judge Models          — golden set, disagreement cases only
  (c) Golden Test Dataset   — held-out test set (seed-42 sample, max 50/folder)
"""

# ── Data builder ───────────────────────────────────────────────────────────────

def make_labels(tp, fn, fp, tn):
    """Return (y_true, y_pred) lists from confusion-matrix counts."""
    y_true = [1] * tp + [1] * fn + [0] * fp + [0] * tn
    y_pred = [1] * tp + [0] * fn + [1] * fp + [0] * tn
    return y_true, y_pred


# ══════════════════════════════════════════════════════════════════════════════
# (a)  GOLDEN SET — per-model classifiers (all 1283 items)
#
#      misleading : 225 positives, 224 negatives  (449 total)
#      scareware  : 198 positives, 202 negatives  (400 total)
#      ad_design  : 239 positives, 195 negatives  (434 total)
# ══════════════════════════════════════════════════════════════════════════════

GOLDEN_SET = {
    # ── Deceptive Claims (misleading) ─────────────────────────────────────────
    "misleading": {
        "Qwen3.5:9b":          make_labels(tp=177, fn=48,  fp=5,  tn=219),
        "Gemma3:12b":          make_labels(tp=185, fn=40,  fp=27, tn=197),
        "GLM-4.6V-Flash:9b":   make_labels(tp=178, fn=47,  fp=1,  tn=223),
    },
    # ── Scareware ─────────────────────────────────────────────────────────────
    "scareware": {
        "Qwen3.5:9b":          make_labels(tp=188, fn=10,  fp=2,  tn=200),
        "Gemma3:12b":          make_labels(tp=159, fn=39,  fp=1,  tn=201),
        "GLM-4.6V-Flash:9b":   make_labels(tp=198, fn=0,   fp=15, tn=187),
    },
    # ── Misleading Design (ad_design) ─────────────────────────────────────────
    "ad_design": {
        "Qwen3.5:9b":          make_labels(tp=226, fn=7,   fp=37, tn=158),
        "Gemma3:12b":          make_labels(tp=140, fn=99,  fp=5,  tn=190),
        "GLM-4.6V-Flash:9b":   make_labels(tp=26,  fn=213, fp=1,  tn=194),
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# (b)  GOLDEN SET — judge models on disagreement cases only
#
#      misleading : 50  disagreement items  (28 pos, 22 neg)
#      scareware  : 34  disagreement items  (31 pos,  3 neg)
#      ad_design  : 124 disagreement items  (98 pos, 26 neg — approx)
# ══════════════════════════════════════════════════════════════════════════════

DISAGREEMENTS = {
    # ── Deceptive Claims (misleading) ─────────────────────────────────────────
    "misleading": {
        "Gemma4:26b":            make_labels(tp=26, fn=2,  fp=3,  tn=19),
        "Qwen3.5:27b":           make_labels(tp=27, fn=1,  fp=16, tn=6),
        "Mistral-small3.2:24b":  make_labels(tp=20, fn=8,  fp=5,  tn=17),
    },
    # ── Scareware ─────────────────────────────────────────────────────────────
    "scareware": {
        "Gemma4:26b":            make_labels(tp=29, fn=2,  fp=1,  tn=2),
        "Qwen3.5:27b":           make_labels(tp=31, fn=0,  fp=2,  tn=1),
        "Mistral-small3.2:24b":  make_labels(tp=30, fn=1,  fp=1,  tn=2),
    },
    # ── Misleading Design (ad_design) ─────────────────────────────────────────
    "ad_design": {
        "Gemma4:26b":            make_labels(tp=87, fn=5,  fp=14, tn=18),
        "Qwen3.5:27b":           make_labels(tp=87, fn=5,  fp=14, tn=18),
        "Mistral-small3.2:24b":  make_labels(tp=78, fn=14, fp=9,  tn=23),
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# (c)  GOLDEN TEST SET — classifiers + standalone judge (seed=42, max 50/folder)
#
#      misleading : 50 tp sampled + 50 tn sampled = 100 items (classifiers)
#                   51 tp + 62 tn = 113 items (judge, full test set)
#      scareware  : 48 tp (all) + 50 tn sampled  = 98 items (classifiers)
#                   48 tp + 50 tn = 98 items (judge, full test set)
#      ad_design  : 3  tp (all) + 50 tn sampled  = 53 items (classifiers + judge)
# ══════════════════════════════════════════════════════════════════════════════

GOLDEN_TEST = {
    # ── Deceptive Claims (misleading) ─────────────────────────────────────────
    "misleading": {
        "Qwen3.5:9b":          make_labels(tp=44, fn=6, fp=1, tn=49),
        "Gemma3:12b":          make_labels(tp=48, fn=2, fp=7, tn=43),
        "Gemma4:26b (judge)":  make_labels(tp=51, fn=0, fp=0, tn=62),
    },
    # ── Scareware ─────────────────────────────────────────────────────────────
    "scareware": {
        "Qwen3.5:9b":          make_labels(tp=46, fn=2, fp=0, tn=50),
        "Gemma3:12b":          make_labels(tp=45, fn=3, fp=1, tn=49),
        "Gemma4:26b (judge)":  make_labels(tp=48, fn=0, fp=1, tn=49),
    },
    # ── Misleading Design (ad_design) ─────────────────────────────────────────
    "ad_design": {
        "Qwen3.5:9b":          make_labels(tp=3, fn=0, fp=0, tn=50),
        "Gemma3:12b":          make_labels(tp=3, fn=0, fp=0, tn=50),
        "Gemma4:26b (judge)":  make_labels(tp=3, fn=0, fp=0, tn=50),
    },
}


# ── Metrics & printing ─────────────────────────────────────────────────────────

def prf(y_true, y_pred):
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


VTYPES = ("misleading", "scareware", "ad_design")
COL_LABELS = ("Deceptive Claims", "Scareware", "Misleading Design")

def print_header(title):
    print(f"\n{'='*108}")
    print(f"  {title}")
    print(f"  {'Model':<28}  " + "  ".join(f"{c:<28}" for c in COL_LABELS) + "  Macro F1")
    print(f"  {'':28}  " + "  ".join(f"{'P':>6} {'R':>6} {'F1':>6}   " for _ in COL_LABELS))
    print(f"  {'-'*104}")


def print_row(name, data):
    parts, f1s = [], []
    for vtype in VTYPES:
        p, r, f = prf(*data[vtype][name])
        parts.append(f"{p:.3f}  {r:.3f}  {f:.3f}")
        f1s.append(f)
    mf1 = sum(f1s) / len(f1s)
    print(f"  {name:<28}  " + "    ".join(parts) + f"    {mf1:.3f}")


def section(title, data, models):
    print_header(title)
    for m in models:
        print_row(m, data)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    section(
        "(a) Ensemble Classifiers — golden set (1283 items)",
        GOLDEN_SET,
        ["Qwen3.5:9b", "Gemma3:12b", "GLM-4.6V-Flash:9b"],
    )
    section(
        "(b) Judge Models — golden set, disagreement cases only",
        DISAGREEMENTS,
        ["Gemma4:26b", "Qwen3.5:27b", "Mistral-small3.2:24b"],
    )
    section(
        "(c) Golden Test Dataset",
        GOLDEN_TEST,
        ["Qwen3.5:9b", "Gemma3:12b", "Gemma4:26b (judge)"],
    )
    print()
