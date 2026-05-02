# Crawler

Puppeteer-based crawler for the **Google Ad Transparency Center** (`adstransparency.google.com`). `run.py` is the main entrypoint — it reads a CSV of ad creatives, splits them across parallel workers, and calls `main.js` internally for each chunk.

## Requirements

- Python 3.12
- `pandas`
- Node.js 18+
- Google Chrome (path configurable via `BROWSER_EXECUTABLE_PATH`)
- Optional: Xvfb for headless servers

```bash
npm install
```

## Usage

```bash
# Basic crawl — 2 parallel workers
python run.py ads.csv --workers 2

# Cap to 100 rows and save to a specific file
python run.py ads.csv --max 100 --output results/ads.json

# Filter to a specific advertiser
python run.py ads.csv --filter "advertiser_id == 'AR09188314108603138049'" --workers 4

# Resume after a crash using previously saved progress files
python run.py ads.csv --resume-from "results/progress" --workers 4

# Set up a persistent browser profile (one-time authenticated login)
python run.py --setup-profile

# Retry ads that failed in a previous run
python run.py ads.csv --retry-failures results/ads_previous.json
```

### Key Arguments

| Argument | Description |
|---|---|
| `csv_file` | CSV with `advertiser_id`, `creative_id`, and optionally `creative_page_url`. See `sql_queries/create_90k.csv` for an example generated via the BigQuery queries in `sql_queries/`. |
| `--workers N` | Number of parallel crawl workers (default: 1) |
| `--max N` | Maximum rows to process |
| `--start N` | Start from row N (0-indexed) |
| `--filter EXPR` | Pandas query string to filter the CSV |
| `--output FILE` | Output JSON file (default: `ads_<timestamp>.json`) |
| `--screenshots-dir DIR` | Screenshot output directory (default: `./results/screenshots`) |
| `--progress-dir DIR` | Directory for per-worker progress checkpoints |
| `--resume-from PATH` | Resume from a progress directory or glob pattern |
| `--retry-failures FILE` | Re-crawl ads with `error`/`not_found` status from a previous run |
| `--setup-profile` | Interactive one-time login to create a saved browser profile |
| `--dry-run` | Preview URLs that would be crawled without running |

## Configuration

Defaults live in `config.js` and can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `HEADLESS` | `true` | Run browser headlessly |
| `XVFB_SWITCH` | `0` | `1` = use Xvfb display (headless servers) |
| `BROWSER_EXECUTABLE_PATH` | `/usr/bin/google-chrome` | Chrome binary path |
| `PROFILE_DIR` | `./browser_profile` | Saved login session directory |
| `PAGE_LOAD_WAIT` | `4000` | Wait after page load (ms) |
| `BROWSER_RESTART_INTERVAL` | `50` | Restart browser every N ads |
| `SCREENSHOT_DIR` | `./results/screenshots` | Screenshot output directory |
| `PROGRESS_DIR` | `./results/progress` | Progress checkpoint directory |

## Output

Results are written to `results/ads_<timestamp>.json`. Each record contains:
- `creativeID` — unique creative ID
- `advertiserID` / `advertiserName`
- `screenshotPath` — path(s) to saved screenshot(s)
- Screenshots are stored in `results/screenshots/worker_<id>/`
- Ad metadata (topic, region, dates, etc.)

Per-worker progress checkpoints in `results/progress/` allow resuming interrupted crawls.
