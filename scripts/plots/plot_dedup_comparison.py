import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FONT_SIZE = 12
plt.rcParams.update({"font.size": FONT_SIZE})

BASE = "./results/gemma_judge"  # path to directory containing violation JSON files

# pool sizes: (original pool, dedup pool)
POOLS = {
    "Scareware (Top 1000)":  (1200,  903),
    "Deceptive Claim (Top 10000)": (11998, 7458),
    "Misleading Ad design (OCR length filtered)":  (689,   522),
}

data = {
    "Scareware (Top 1000)":  ("scareware_violations.json",  "scareware_violations_dedup.json"),
    "Deceptive Claim (Top 10000)": ("misleading_violations.json", "misleading_violations_dedup.json"),
    "Misleading Ad design (OCR length filtered)":  ("ad_design_violations.json",  "ad_design_violations_dedup.json"),
}

labels, orig_counts, dedup_counts = [], [], []
for vtype, (orig_file, dedup_file) in data.items():
    with open(os.path.join(BASE, orig_file))  as f: orig  = json.load(f)
    with open(os.path.join(BASE, dedup_file)) as f: dedup = json.load(f)
    labels.append(vtype)
    orig_counts.append(len(orig))
    dedup_counts.append(len(dedup))

fig, ax = plt.subplots(figsize=(10, 4))

bar_h   = 0.35
y       = range(len(labels))
y_orig  = [i + bar_h / 2 for i in y]
y_dedup = [i - bar_h / 2 for i in y]

bars_orig  = ax.barh(y_orig,  orig_counts,  height=bar_h, label="Original",   color="#7EB3D8", edgecolor="white")
bars_dedup = ax.barh(y_dedup, dedup_counts, height=bar_h, label="AdID dedup", color="#E87B35", edgecolor="white")

offset = max(orig_counts) * 0.01
for bar, vtype, val in zip(bars_orig, labels, orig_counts):
    pool = POOLS[vtype][0]
    pct  = val / pool * 100
    ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:,}  ({pct:.1f}%)", va="center", ha="left", fontsize=FONT_SIZE)

for bar, vtype, val in zip(bars_dedup, labels, dedup_counts):
    pool = POOLS[vtype][1]
    pct  = val / pool * 100
    ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:,}  ({pct:.1f}%)", va="center", ha="left", fontsize=FONT_SIZE)

ax.set_yticks(list(y))
ax.set_yticklabels([l.replace(" (", "\n(") for l in labels], fontsize=16)
ax.set_xlabel("Violation Count", fontsize=16)
ax.set_title("Total violation counts - Original & AdID dedup", fontsize=16, fontweight="bold")
ax.legend(fontsize=FONT_SIZE)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(right=max(orig_counts) * 1.25)

plt.tight_layout()
OUT = "result_adlens/dedup_comparison.png"
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")
