# Scripts

Utility scripts for supplementary data collection tasks.

## pdns/

Passive DNS (pDNS) lookup tooling for ad landing domains.

| File | Description |
|---|---|
| `query_pdns.sh` | Shell script that queries a pDNS API for each domain in the input list |
| `all_ad_link_domains.txt` | Deduplicated list of ad landing domains to look up |

### Usage

```bash
cd scripts/pdns
bash query_pdns.sh
```

Results are written to `progress/` and `screenshots/` subdirectories.
