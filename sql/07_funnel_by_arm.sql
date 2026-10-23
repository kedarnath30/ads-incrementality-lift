-- Visit -> conversion funnel by arm: is the lift driven by more visits, or better conversion per visit?
SELECT treatment,
       AVG(visit)                                         AS visit_rate,
       SUM(conversion) * 1.0 / NULLIF(SUM(visit), 0)      AS conv_per_visit,
       AVG(conversion)                                    AS conv_rate,
       SUM(CASE WHEN conversion = 1 AND visit = 0 THEN 1 ELSE 0 END) AS conversions_without_visit
FROM criteo GROUP BY treatment ORDER BY treatment;
