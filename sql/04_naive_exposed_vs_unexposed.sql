-- The tempting-but-biased comparison: exposed vs unexposed users, ignoring randomization.
-- Also splits the treatment arm so you can see unexposed treated users look nothing like control.
SELECT CASE WHEN treatment = 0 THEN 'control (never exposed)'
            WHEN exposure = 1  THEN 'treatment, exposed'
            ELSE 'treatment, not exposed' END AS grp,
       COUNT(*)        AS users,
       AVG(visit)      AS visit_rate,
       AVG(conversion) AS conv_rate
FROM criteo GROUP BY 1 ORDER BY 1;
