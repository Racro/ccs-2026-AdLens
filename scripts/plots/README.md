# Plots

Python scripts that generate the figures used in the paper.

## Requirements

- Python 3.12
- `matplotlib`, `numpy` (included in top-level `requirements.txt`)

## Scripts

| Script | Output | Description |
|---|---|---|
| `bar_plot.py` | `result_adlens/bar_plot.png` | Stacked bar chart of violation counts by month (2023–2026), split by active vs. all detected |
| `plot_bin_precision.py` | `result_adlens/bin_precision.png` | Precision-by-bin plot for Scareware (100-rank bins) and Deceptive Claims (500-rank bins) across models |
| `plot_dedup_comparison.py` | `result_adlens/dedup_comparison.png` | Horizontal bar chart comparing original vs. AdID-dedup violation counts across all three categories |
| `plots.ipynb` | — | Jupyter notebook with exploratory versions of the above figures |

## Usage

Run from the repo root:

```bash
python scripts/plots/bar_plot.py
python scripts/plots/plot_bin_precision.py
python scripts/plots/plot_dedup_comparison.py
```

Output figures are saved as `.png` files under `result_adlens/`.
