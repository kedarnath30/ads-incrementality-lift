-- Conversion Lift style readout (intent-to-treat): rates by randomized arm, absolute and relative lift,
-- 95% Wald CI on the absolute difference, and incremental outcomes per 1M users.
WITH arm AS (
  SELECT treatment,
         COUNT(*)        AS n,
         AVG(visit)      AS visit_rate,
         AVG(conversion) AS conv_rate
  FROM criteo GROUP BY treatment
), w AS (
  SELECT t.n AS n_t, c.n AS n_c,
         t.visit_rate AS vt, c.visit_rate AS vc,
         t.conv_rate  AS ct, c.conv_rate  AS cc
  FROM arm t JOIN arm c ON t.treatment = 1 AND c.treatment = 0
)
SELECT metric, rate_treatment, rate_control,
       rate_treatment - rate_control                         AS abs_lift,
       (rate_treatment - rate_control) / rate_control        AS rel_lift,
       (rate_treatment - rate_control) - 1.96 * se           AS abs_lift_ci_low,
       (rate_treatment - rate_control) + 1.96 * se           AS abs_lift_ci_high,
       (rate_treatment - rate_control) / se                  AS z,
       (rate_treatment - rate_control) * 1e6                 AS incremental_per_1m_users
FROM (
  SELECT 'visit' AS metric, vt AS rate_treatment, vc AS rate_control,
         SQRT(vt*(1-vt)/n_t + vc*(1-vc)/n_c) AS se FROM w
  UNION ALL
  SELECT 'conversion', ct, cc, SQRT(ct*(1-ct)/n_t + cc*(1-cc)/n_c) FROM w
);
