WITH base AS (
  SELECT
    topic AS category,
    advertiser_id,
    creative_id,
    MIN(r.first_shown) AS earliest_first_shown
  FROM `bigquery-public-data.google_ads_transparency_center.creative_stats`,
  UNNEST(region_stats) AS r
  WHERE topic IN ('Software', 'Mobile App Utilities', 'Computers & Consumer Electronics')
    AND r.first_shown < '2026-03-26'
  GROUP BY category, advertiser_id, creative_id
),

ranked AS (
  SELECT
    category,
    ROW_NUMBER() OVER (PARTITION BY category ORDER BY earliest_first_shown DESC) AS rank_closest
  FROM base
)

SELECT
  category,
  COUNT(*)                                         AS total_rows,
  COUNTIF(rank_closest <= 10000)                   AS closest_10k_actual,
  COUNTIF(rank_closest > 10000 AND rank_closest <= 30000) AS random_pool_size,
  LEAST(COUNTIF(rank_closest > 10000), 20000)      AS random_20k_actual,
  COUNTIF(rank_closest <= 10000)
    + LEAST(COUNTIF(rank_closest > 10000), 20000)  AS contributed_rows
FROM ranked
GROUP BY category
ORDER BY category;
