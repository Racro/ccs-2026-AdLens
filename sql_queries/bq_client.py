"""
BigQuery client wrapper for the daily ads pipeline.

Two scans per day (both necessary):

  Scan 1 — build_today_table()   ~52 GB
    Fetches full details for our 3 tracked topics.
    Used for: added diff, crawl input, snapshot.

  Scan 2 — build_all_ids_table() ~52 GB
    Fetches ONLY (advertiser_id, creative_id) from ALL topics in creative_stats.
    Used for: removed verification — to distinguish truly-removed ads from
    ads that simply moved to an untracked topic (Banking, Healthcare, etc.)

  All diffs (get_new_ads, get_removed_ads) query your own tables → FREE.

Daily cost: ~104 GB → ~$13/month  (vs original ~$23/month with 3 separate scans)

BQ tables in your project:
  bq_temp_all   — today's 3-topic details; dropped after promote
  bq_all_ids    — today's full creative_stats IDs (all topics); dropped after promote
  bq_snapshot_all — yesterday's 3-topic details; persists as diff baseline

Removal types produced by get_removed_ads():
  'moved_to_tracked_category' — moved within our 3 topics  (seen in bq_temp_all)
  'topic_change_untracked'    — moved to a topic we don't track (in bq_all_ids, not bq_temp_all)
  'truly_removed'             — gone from ALL of creative_stats (not in bq_all_ids)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from google.auth import default as google_auth_default
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

_SELECT_TEMPLATE = """\
SELECT
  topic                        AS category,
  advertiser_id,
  creative_id,
  advertiser_legal_name,
  STRING_AGG(DISTINCT r.region_code, ', ' ORDER BY r.region_code) AS all_regions,
  MIN(r.first_shown)           AS earliest_first_shown,
  MAX(r.last_shown)            AS latest_last_shown
FROM `bigquery-public-data.google_ads_transparency_center.creative_stats`,
UNNEST(region_stats) AS r
WHERE (
  {conditions}
)
GROUP BY
  category, advertiser_id, creative_id,
  advertiser_legal_name
"""


def _build_query(categories: dict) -> str:
    conditions = [
        f"(topic = '{cfg['topic']}' AND r.first_shown >= '{cfg['first_shown_cutoff']}')"
        for cfg in categories.values()
    ]
    return _SELECT_TEMPLATE.format(conditions="\n  OR ".join(conditions))


class BQClient:
    def __init__(
        self,
        project: str,
        dataset: str,
        credentials_path: Optional[str] = None,
    ):
        self.project = project
        self.dataset = dataset

        self.client = bigquery.Client(project=project, credentials=self._resolve_credentials(credentials_path))

    @staticmethod
    def _resolve_credentials(credentials_path: Optional[str]):
        """
        Credential resolution order:
          1. Application Default Credentials (ADC) — ~/.config/gcloud/... or GOOGLE_APPLICATION_CREDENTIALS env var
          2. Service account key at credentials_path (from GCP_CREDENTIALS_PATH in .env)
          3. Raise with a clear message if neither is available
        """
        # 1. Try ADC first
        try:
            creds, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/bigquery"])
            logger.info("GCP auth: using Application Default Credentials")
            return creds
        except DefaultCredentialsError:
            pass

        # 2. Fall back to service account key
        if credentials_path:
            key_path = Path(credentials_path)
            if not key_path.exists():
                raise FileNotFoundError(
                    f"GCP_CREDENTIALS_PATH set but file not found: {key_path}\n"
                    "  Fix: check the path in cron/.env"
                )
            creds = service_account.Credentials.from_service_account_file(
                str(key_path),
                scopes=["https://www.googleapis.com/auth/bigquery"],
            )
            logger.info(f"GCP auth: using service account key {key_path}")
            return creds

        # 3. Neither worked
        raise RuntimeError(
            "No GCP credentials found. Tried:\n"
            "  1. Application Default Credentials (not configured)\n"
            "  2. GCP_CREDENTIALS_PATH in cron/.env (not set)\n"
            "Fix with one of:\n"
            "  a) gcloud auth application-default login\n"
            "  b) Set GCP_CREDENTIALS_PATH=/path/to/service-account.json in cron/.env"
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _ref(self, table_name: str) -> str:
        return f"{self.project}.{self.dataset}.{table_name}"

    def table_exists(self, table_name: str) -> bool:
        try:
            self.client.get_table(self._ref(table_name))
            return True
        except NotFound:
            return False

    def _run(self, query: str, job_config=None) -> bigquery.QueryJob:
        job = self.client.query(query, job_config=job_config)
        job.result()
        return job

    # ── scan 1: 3-topic details ───────────────────────────────────────────────

    def build_today_table(self, categories: dict) -> int:
        """
        SCAN 1 (~52 GB billed).
        Fetches full ad details for our 3 tracked topics → bq_temp_all.
        Used for the added diff and as crawl input.
        """
        dest = self._ref("bq_temp_all")
        job_config = bigquery.QueryJobConfig(
            destination=dest,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
        )
        logger.info(f"[Scan 1] 3-topic details → {dest}")
        job = self._run(_build_query(categories), job_config=job_config)
        gb = (job.total_bytes_processed or 0) / 1e9
        logger.info(f"[Scan 1] done: {gb:.1f} GB processed")
        return job.total_bytes_processed or 0

    # ── scan 2: all-topics IDs ────────────────────────────────────────────────

    def build_all_ids_table(self) -> int:
        """
        SCAN 2 (~52 GB billed).
        Fetches ONLY (advertiser_id, creative_id) from the ENTIRE creative_stats
        table — all topics, no cutoff filter — and saves to bq_all_ids.

        This is the equivalent of QUERY 1 from sql_queries.md, but done
        automatically every day so removed verification doesn't need a
        manual step.

        An ad that is NOT in bq_all_ids is confirmed gone from Google entirely.
        An ad that IS in bq_all_ids but not in bq_temp_all just changed topics.
        """
        dest = self._ref("bq_all_ids")
        query = """
        SELECT DISTINCT advertiser_id, creative_id
        FROM `bigquery-public-data.google_ads_transparency_center.creative_stats`
        """
        job_config = bigquery.QueryJobConfig(
            destination=dest,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
        )
        logger.info(f"[Scan 2] All-topics ID scan → {dest}")
        job = self._run(query, job_config=job_config)
        gb = (job.total_bytes_processed or 0) / 1e9
        logger.info(f"[Scan 2] done: {gb:.1f} GB processed")
        return job.total_bytes_processed or 0

    # ── diffs (free — your own tables) ────────────────────────────────────────

    def get_new_ads(self) -> pd.DataFrame:
        """
        ADDED diff: creative IDs in bq_temp_all that are NOT in bq_snapshot_all.

        Joined on (advertiser_id, creative_id) regardless of category — so an ad
        that moved from Software → MobileApp is NOT flagged as new (it existed yesterday).

        First run (no snapshot): returns ALL ads from bq_temp_all so the initial
        crawl covers every ad since each category's cutoff date. Free.
        """
        temp_ref = self._ref("bq_temp_all")
        snap_ref = self._ref("bq_snapshot_all")

        if not self.table_exists("bq_snapshot_all"):
            logger.info("No snapshot yet — first run. Returning all ads from bq_temp_all.")
            df = self.client.query(
                f"SELECT * FROM `{temp_ref}` ORDER BY category, earliest_first_shown DESC"
            ).to_dataframe()
            logger.info(f"First run: {len(df)} total ads")
            if not df.empty and "category" in df.columns:
                for cat, grp in df.groupby("category"):
                    logger.info(f"  {cat}: {len(grp)} ads")
            return df

        query = f"""
        SELECT t.*
        FROM `{temp_ref}` t
        LEFT JOIN `{snap_ref}` s
          ON  t.advertiser_id = s.advertiser_id
          AND t.creative_id   = s.creative_id
        WHERE s.creative_id IS NULL
        ORDER BY t.category, t.earliest_first_shown DESC
        """
        logger.info("Computing added-diff")
        df = self.client.query(query).to_dataframe()
        logger.info(f"Added diff: {len(df)} new ads total")
        if not df.empty and "category" in df.columns:
            for cat, grp in df.groupby("category"):
                logger.info(f"  {cat}: {len(grp)} added")
        return df

    def get_removed_ads(self) -> pd.DataFrame:
        """
        REMOVED diff with full topic verification (uses bq_all_ids from Scan 2).

        Finds ads that were in bq_snapshot_all (for a given category) but are
        no longer in bq_temp_all for that SAME category today.

        Each removed ad gets a 'removal_type' column:
          'moved_to_tracked_category'
              — still in bq_temp_all under a different tracked topic.
                No further action needed.
          'topic_change_untracked'
              — still in bq_all_ids (present in creative_stats under some
                other topic like Banking). NOT truly removed from Google.
          'truly_removed'
              — NOT in bq_all_ids at all. Confirmed gone from ALL of
                creative_stats. Cross-check with weekly QUERY 2
                (removed_creative_stats) for Google's removal reason.

        Returns empty DataFrame on first run. Free.
        """
        snap_ref = self._ref("bq_snapshot_all")
        temp_ref = self._ref("bq_temp_all")
        ids_ref  = self._ref("bq_all_ids")

        if not self.table_exists("bq_snapshot_all"):
            logger.info("No snapshot yet — first run. Empty removed-diff.")
            return pd.DataFrame()

        query = f"""
        SELECT
          s.*,
          t_diff.category AS moved_to_tracked,
          CASE
            WHEN t_diff.creative_id  IS NOT NULL THEN 'moved_to_tracked_category'
            WHEN all_ids.creative_id IS NOT NULL THEN 'topic_change_untracked'
            ELSE                                      'truly_removed'
          END AS removal_type
        FROM `{snap_ref}` s
        -- still in the same tracked category today?
        LEFT JOIN `{temp_ref}` t_same
          ON  s.advertiser_id = t_same.advertiser_id
          AND s.creative_id   = t_same.creative_id
          AND s.category      = t_same.category
        -- moved to a different tracked category today?
        LEFT JOIN `{temp_ref}` t_diff
          ON  s.advertiser_id = t_diff.advertiser_id
          AND s.creative_id   = t_diff.creative_id
          AND s.category     != t_diff.category
        -- still exists anywhere in creative_stats (any topic)?
        LEFT JOIN `{ids_ref}` all_ids
          ON  s.advertiser_id = all_ids.advertiser_id
          AND s.creative_id   = all_ids.creative_id
        WHERE t_same.creative_id IS NULL
        ORDER BY s.category, s.earliest_first_shown DESC
        """
        logger.info("Computing removed-diff (with full topic verification)")
        df = self.client.query(query).to_dataframe()
        logger.info(f"Removed diff: {len(df)} removed ads total")

        if not df.empty and "category" in df.columns:
            for cat, grp in df.groupby("category"):
                counts = grp["removal_type"].value_counts().to_dict()
                moved    = counts.get("moved_to_tracked_category", 0)
                untrack  = counts.get("topic_change_untracked",    0)
                gone     = counts.get("truly_removed",             0)
                logger.info(
                    f"  {cat}: {len(grp)} removed — "
                    f"{moved} moved to tracked cat | "
                    f"{untrack} topic change (untracked) | "
                    f"{gone} truly removed from Google"
                )
        return df

    # ── promote + cleanup ────────────────────────────────────────────────────

    def promote_snapshot(self):
        """
        Overwrite bq_snapshot_all with today's bq_temp_all, then drop both
        temp tables (bq_temp_all and bq_all_ids).

        Called AFTER both diff CSVs are saved locally. Free.
        """
        temp_ref = self._ref("bq_temp_all")
        snap_ref = self._ref("bq_snapshot_all")
        ids_ref  = self._ref("bq_all_ids")

        logger.info("Promoting bq_temp_all → bq_snapshot_all")
        self._run(f"CREATE OR REPLACE TABLE `{snap_ref}` AS SELECT * FROM `{temp_ref}`")
        self.client.delete_table(temp_ref, not_found_ok=True)
        self.client.delete_table(ids_ref,  not_found_ok=True)
        logger.info("Snapshot updated, temp tables dropped")

    # ── optional helpers ──────────────────────────────────────────────────────

    def download_full_csv(self, output_path: Path) -> int:
        """Download bq_temp_all as a combined CSV. Optional (--save-full-csv)."""
        temp_ref = self._ref("bq_temp_all")
        logger.info(f"Downloading full snapshot CSV → {output_path}")
        df = self.client.query(f"SELECT * FROM `{temp_ref}`").to_dataframe()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"{len(df)} rows → {output_path}")
        return len(df)

    def get_snapshot_row_count(self) -> dict:
        """Sanity check: row counts per category in the current snapshot."""
        snap_ref = self._ref("bq_snapshot_all")
        if not self.table_exists("bq_snapshot_all"):
            return {}
        rows = self.client.query(
            f"SELECT category, COUNT(*) AS n FROM `{snap_ref}` GROUP BY category"
        ).result()
        return {row.category: row.n for row in rows}


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Run a BigQuery SQL file and save results to CSV")
    parser.add_argument("--key", "-k", required=True, help="Path to GCP service account JSON key file")
    parser.add_argument("--sql", "-s", required=True, help="Path to SQL file to execute")
    parser.add_argument("--output", "-o", help="Output CSV path (default: <sql_file>.csv)")
    parser.add_argument("--project", "-p", help="GCP project ID (default: read from key file)")
    args = parser.parse_args()

    key_path = Path(args.key)
    sql_path = Path(args.sql)

    if not key_path.exists():
        sys.exit(f"Key file not found: {key_path}")
    if not sql_path.exists():
        sys.exit(f"SQL file not found: {sql_path}")

    project = args.project
    if not project:
        with open(key_path) as f:
            project = json.load(f).get("project_id")
        if not project:
            sys.exit("Could not determine project ID. Use --project or ensure key file contains 'project_id'.")

    output_path = Path(args.output) if args.output else sql_path.with_suffix(".csv")

    creds = service_account.Credentials.from_service_account_file(
        str(key_path),
        scopes=["https://www.googleapis.com/auth/bigquery"],
    )
    client = bigquery.Client(project=project, credentials=creds)

    sql = sql_path.read_text()
    logger.info(f"Running {sql_path} ...")
    df = client.query(sql).to_dataframe()
    logger.info(f"{len(df)} rows returned")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved → {output_path}")
