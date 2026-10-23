-- Arm sizes and treatment share (design is ~85% treatment / 15% control). Used for a sample-ratio sanity check.
SELECT treatment,
       COUNT(*)                                    AS users,
       COUNT(*) * 1.0 / SUM(COUNT(*)) OVER ()      AS share
FROM criteo
GROUP BY treatment
ORDER BY treatment;
