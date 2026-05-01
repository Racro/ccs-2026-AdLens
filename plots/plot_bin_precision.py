import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FONT_SIZE = 12
plt.rcParams.update({"font.size": FONT_SIZE})

# Models in the ensemble:  Qwen3.5-9B  and  Gemma3-12B
# Judge used for tie-breaking (gemma_judge dir): Gemma4-26B
# "Ensemble"      = both Qwen3.5-9B and Gemma3-12B agree on violation
# "Gemma4-26B judge" = Ensemble + Gemma4-26B resolves splits

BIN_INFO    = "result_adlens/bin_info.json"
SCARE_VIOLS = "result_adlens/gemma_judge/scareware_violations.json"
MIS_VIOLS   = "result_adlens/gemma_judge/misleading_violations.json"
SCARE_COLLATED = "results/scareware_google_embeddinggemma-300m_fulltext_no_short_text_dedup_ensemble_qwen35_9b_gemma_3_12b_it_ranked_collated_classified.json"
MIS_COLLATED   = "results/misleading_google_embeddinggemma-300m_fulltext_no_short_text_dedup_ensemble_qwen35_9b_gemma_3_12b_it_ranked_collated_classified.json"

with open(BIN_INFO) as f:
    bin_info = json.load(f)
with open(SCARE_VIOLS) as f:
    scare_viol_ids = set(v["id"] for v in json.load(f))
with open(MIS_VIOLS) as f:
    mis_viol_ids = set(v["id"] for v in json.load(f))
with open(SCARE_COLLATED) as f:
    scare_records = {r["sim_rank"]: r for r in json.load(f)}
with open(MIS_COLLATED) as f:
    mis_records = {r["sim_rank"]: r for r in json.load(f)}


def judge_bins(ref_bins, records, viol_ids):
    """Precision per bin from violations file (ensemble + Gemma4-26B judge on splits).
    Stops at the last bin that contains at least one violation."""
    result = []
    for b in ref_bins:
        start, end = b["rank_range"]
        chunk = [records[r] for r in range(start, end + 1) if r in records]
        if not chunk:
            continue
        n   = len(chunk)
        pos = sum(1 for r in chunk if r["id"] in viol_ids)
        result.append({"rank_range": b["rank_range"], "classified": n,
                        "positive": pos, "precision": pos / n})
    return result


def aggregate_500(bins):
    merged = []
    for i in range(0, len(bins), 5):
        chunk = bins[i:i+5]
        total_c = sum(b["classified"] for b in chunk)
        total_p = sum(b["positive"]   for b in chunk)
        merged.append({
            "rank_range": [chunk[0]["rank_range"][0], chunk[-1]["rank_range"][1]],
            "classified": total_c, "positive": total_p,
            "precision":  total_p / total_c if total_c else 0,
        })
    return merged


# Scareware — 100-unit bins
sw = {
    "Qwen3.5-9B":      bin_info["scareware_qwen3_5_9b_bin"],
    "Gemma3-12B":      bin_info["scareware_gemma3_12b_bin"],
    "Ensemble (Agreement)":        bin_info["scareware_ensemble_bin"],
    "Gemma4-26B judge": judge_bins(bin_info["scareware_gemma3_12b_bin"], scare_records, scare_viol_ids),
}

# Misleading — aggregate 100-unit → 500-unit
ml = {
    "Qwen3.5-9B":      aggregate_500(bin_info["misleading_qwen3_5_9b_bin"]),
    "Gemma3-12B":      aggregate_500(bin_info["misleading_gemma3_12b_bin"]),
    "Ensemble (Agreement)":        aggregate_500(bin_info["misleading_ensemble_bin"]),
    "Gemma4-26B judge": aggregate_500(judge_bins(bin_info["misleading_gemma3_12b_bin"], mis_records, mis_viol_ids)),
}

COLORS = {"Qwen3.5-9B": "#E87B35", "Gemma3-12B": "#4C8AC8", "Ensemble (Agreement)": "#5EAD6F", "Gemma4-26B judge": "#9B59B6"}

def plot_series(ax, bins, label, max_rank=None):
    filtered = [b for b in bins if max_rank is None or b["rank_range"][0] <= max_rank]
    xs = [b["rank_range"][1] for b in filtered]
    ps = [b["precision"]     for b in filtered]
    ax.plot(xs, ps, marker="o", markersize=4, linewidth=1.8, color=COLORS[label], label=label)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.subplots_adjust(wspace=0.35)

ax = axes[0]
for label, bins in sw.items():
    plot_series(ax, bins, label, max_rank=1000)
ax.set_title("Scareware (100-rank bins)", fontsize=20, fontweight="bold")
ax.set_xlabel("Ads (Ranked by similarity)", fontsize=20)
ax.set_ylabel("Positive Rate (violations)", fontsize=20)
ax.set_xlim(1, 1000)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.xaxis.set_major_locator(mticker.MultipleLocator(200))
ax.legend(fontsize=FONT_SIZE)
ax.grid(axis="y", linestyle="--", alpha=0.4)

ax = axes[1]
for label, bins in ml.items():
    plot_series(ax, bins, label, max_rank=10000)
ax.set_title("Deceptive Claims (500-rank bins)", fontsize=20, fontweight="bold")
ax.set_xlabel("Ads (Ranked by similarity)", fontsize=20)
ax.set_ylabel("Positive Rate (violations)", fontsize=20)
ax.set_xlim(1, 10000)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.xaxis.set_major_locator(mticker.MultipleLocator(2000))
ax.legend(fontsize=FONT_SIZE)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
OUT = "result_adlens/bin_precision.png"
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")
