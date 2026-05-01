# Crawler

Puppeteer-based crawler for the **Google Ad Transparency Center** (`adstransparency.google.com`). Given a list of advertiser IDs or ad URLs it fetches ad creatives, screenshots, and metadata, saving results as JSON.

## Requirements

- Node.js 18+
- Google Chrome (path configurable via `BROWSER_EXECUTABLE_PATH`)
- Optional: Xvfb for headless servers

```bash
npm install
```

## Usage

```bash
# Crawl a single advertiser by ID
node main.js AR16735076323512287233 50

# Crawl from a URL list file
node main.js --urls-file urls.txt --output results/ads.json

# Set up a persistent browser profile (one-time login)
node main.js --setup-profile

# Authenticated crawl using saved profile
node main.js --force-auth AR16735076323512287233 100
```

### Key Arguments

| Argument | Description |
|---|---|
| `--urls-file FILE` | File with ad URLs to crawl (one per line) |
| `--urls JSON` | JSON array of URLs passed directly |
| `--output FILE` | Output JSON file (default: `results/ads_<timestamp>.json`) |
| `--screenshots-dir DIR` | Override screenshot output directory |
| `--progress-dir DIR` | Override progress checkpoint directory |
| `--worker-id ID` | Worker identifier for parallel runs |
| `--force-auth` | Use saved browser profile for authentication |
| `--setup-profile` | Interactive one-time profile setup (opens visible browser) |

## Configuration

All defaults live in `config.js` and can be overridden via environment variables:

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
- `ad_id` — unique creative ID
- `advertiser_id` / `advertiser_name`
- `screenshotPath` — path(s) to saved screenshot(s)
- Ad metadata (topic, region, dates, etc.)

Progress checkpoints in `results/progress/` allow resuming interrupted crawls.
