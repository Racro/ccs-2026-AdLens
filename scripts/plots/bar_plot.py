import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

months = [
    '2023-03','2023-04','2023-05','2023-07','2023-08','2023-09','2023-10',
    '2023-11','2023-12','2024-01','2024-02','2024-03','2024-04','2024-05',
    '2024-06','2024-07','2024-08','2024-09','2024-10','2024-11','2024-12',
    '2025-01','2025-02','2025-03','2025-04','2025-05','2025-06','2025-07',
    '2025-08','2025-09','2025-10','2025-11','2025-12','2026-01','2026-02',
    '2026-03',
]

all_mis = np.array([54,1,3,4,5,10,10,8,10,5,6,5,9,8,7,17,52,26,116,37,31,22,21,58,65,101,129,91,111,139,180,156,149,219,342,997])
all_ad  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,24,10,11,36,16,9,14,15,10,24,50])
all_sc  = np.array([2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,4,5,10,7,6,8,10,12,6,14,32,104])

act_mis = np.array([19,0,3,1,3,4,6,4,4,2,0,2,4,7,1,4,48,19,91,20,19,5,6,14,11,7,41,27,20,22,35,30,40,106,342,997])
act_ad  = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,2,16,9,2,8,9,6,24,50])
act_sc  = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,2,1,0,1,5,6,4,9,32,104])

act_total = act_mis + act_ad + act_sc

x = np.arange(len(months))
w = 0.6

MIS_FULL  = '#378ADD'
AD_FULL   = '#1D9E75'
SC_FULL   = '#D85A30'

fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(7, 5),
                                      gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05})

for ax in (ax_top, ax_bot):
    ax.bar(x, all_mis,           width=w, color=MIS_FULL, label='_nolegend_')
    ax.bar(x, all_ad,            width=w, color=AD_FULL,  bottom=all_mis, label='_nolegend_')
    ax.bar(x, all_sc,            width=w, color=SC_FULL,  bottom=all_mis+all_ad, label='_nolegend_')
    ax.plot(x, act_total, color='#2C2C2A', linewidth=1.5, marker='o', markersize=2.5, zorder=5, label='_nolegend_')
    ax.grid(axis='y', color='#e0e0e0', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['right'].set_visible(False)

ax_top.set_ylim(990, 1100)
ax_top.set_yticks([1000, 1100])
ax_top.spines['bottom'].set_visible(False)
ax_top.spines['top'].set_visible(False)
ax_top.tick_params(axis='x', which='both', bottom=False)
ax_top.tick_params(axis='y', labelsize=9)

ax_bot.set_ylim(0, 420)
ax_bot.spines['top'].set_visible(False)
ax_bot.tick_params(axis='both', labelsize=9)
ax_bot.set_xticks(x[::3])
ax_bot.set_xticklabels([months[i] for i in range(0, len(months), 3)], rotation=45, ha='right', fontsize=9)
ax_bot.set_xlabel('First shown date', fontsize=12)

fig.text(0.01, 0.45, 'Ad count', va='center', rotation='vertical', fontsize=12)

d = 0.015
kwargs = dict(transform=ax_top.transAxes, color='#888780', clip_on=False, linewidth=1)
ax_top.plot((-d, +d), (-d*3, +d*3), **kwargs)
ax_top.plot((1-d, 1+d), (-d*3, +d*3), **kwargs)
kwargs.update(transform=ax_bot.transAxes)
ax_bot.plot((-d, +d), (1-d*3, 1+d*3), **kwargs)
ax_bot.plot((1-d, 1+d), (1-d*3, 1+d*3), **kwargs)

legend_handles = [
    mpatches.Patch(color=MIS_FULL, label='Deceptive claim'),
    mpatches.Patch(color=AD_FULL, label='Misleading design'),
    mpatches.Patch(color=SC_FULL, label='Scareware'),
    plt.Line2D([0], [0], color='#2C2C2A', linewidth=1.5, marker='o', markersize=4, label='Recently shown'),
]
ax_top.legend(handles=legend_handles, fontsize=12, ncol=2, loc='center',
              bbox_to_anchor=(0.5, 0.5), frameon=False, columnspacing=1, handlelength=1.5)

plt.savefig('./result_adlens/bar_plot.png', dpi=150, bbox_inches='tight')
print("Saved to result_adlens/bar_plot.png")
