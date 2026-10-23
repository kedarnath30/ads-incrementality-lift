-- Compliance: how many assigned users were actually exposed to an ad, by arm.
-- One-sided non-compliance means control users can never be exposed.
SELECT treatment,
       COUNT(*)                 AS users,
       SUM(exposure)            AS exposed_users,
       AVG(exposure)            AS exposure_rate
FROM criteo GROUP BY treatment ORDER BY treatment;
