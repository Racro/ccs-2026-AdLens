# SQL Queries

BigQuery SQL queries and Python client for querying the Google Ad Transparency public dataset.

## Requirements

- Python 3.12
- `google-cloud-bigquery`, `google-auth`, `pandas`, `db-dtypes` (included in top-level `requirements.txt`)
- Google Cloud credentials (`gcloud auth application-default login` or a service account key)

## Files

| File | Description |
|---|---|
| `bq_client.py` | Python BigQuery client wrapper — daily diff pipeline (new/removed ads) |
| `advertiser_violations.sql` | Query advertisers ranked by violation count |
| `format_violations.sql` | Violation breakdown by ad format |
| `create_90k.sql` | Build the 90k-ad labeled sample table |
| `create_90k.csv` | Sample output from `create_90k.sql` |

## Daily Pipeline (bq_client.py)

The client runs two scans per day against the Google Ad Transparency `creative_stats` table:

- **Scan 1** (~52 GB) — full details for tracked topics; used for new-ad diffs and crawl input
- **Scan 2** (~52 GB) — `(advertiser_id, creative_id)` only from all topics; used to distinguish truly-removed ads from topic-changed ads

All diff queries run against your own project tables (free). Total cost ~$13/month.

```bash
python sql_queries/bq_client.py
```

Credentials are loaded via `google.auth.default()` (ADC) or a service account JSON key.
