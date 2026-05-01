# Plots

Python scripts that generate the figures used in the paper.

## Requirements

- Python 3.12
- `matplotlib`, `numpy` (included in top-level `requirements.txt`)

## Scripts

| Script | Figure |
|---|---|
| `bar_plot_4.py` | Stacked bar chart of violation counts by month (2023–2026), split by active vs. all detected |
| `plot_bin_precision.py` | Precision-by-bin plot for classifier calibration |

## Usage

Run from the repo root:

```bash
python plots/bar_plot_4.py
python plots/plot_bin_precision.py
```

Output figures are saved to the current directory as `.pdf` / `.png` files.
