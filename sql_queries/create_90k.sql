WITH base AS (
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
  WHERE topic IN ('Software', 'Mobile App Utilities', 'Computers & Consumer Electronics')
    AND r.first_shown < '2026-03-26'
  GROUP BY
    category, advertiser_id, creative_id, advertiser_legal_name
),

ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY category
      ORDER BY earliest_first_shown DESC
    ) AS rank_closest
  FROM base
),

closest_10k AS (
  SELECT *, 'closest_to_mar26' AS sample_bucket
  FROM ranked
  WHERE rank_closest <= 10000
),

-- Re-rank randomly only within the non-closest pool so we get exactly
-- min(20000, pool_size) rows per category.
random_pool AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY category
      ORDER BY RAND()
    ) AS rank_random_pool
  FROM ranked
  WHERE rank_closest > 10000
),

random_20k AS (
  SELECT *, 'random' AS sample_bucket
  FROM random_pool
  WHERE rank_random_pool <= 20000
)

SELECT
  category,
  advertiser_id,
  creative_id,
  advertiser_legal_name,
  all_regions,
  earliest_first_shown,
  latest_last_shown,
  sample_bucket
FROM closest_10k

UNION ALL

SELECT
  category,
  advertiser_id,
  creative_id,
  advertiser_legal_name,
  all_regions,
  earliest_first_shown,
  latest_last_shown,
  sample_bucket
FROM random_20k

ORDER BY category, sample_bucket, earliest_first_shown DESC;
