-- Randomization check: feature means by arm (should match) vs by exposure within treatment (they will not).
SELECT 'arm' AS split, CAST(treatment AS VARCHAR) AS grp,
       AVG(f0) f0, AVG(f1) f1, AVG(f2) f2, AVG(f3) f3, AVG(f4) f4, AVG(f5) f5,
       AVG(f6) f6, AVG(f7) f7, AVG(f8) f8, AVG(f9) f9, AVG(f10) f10, AVG(f11) f11
FROM criteo GROUP BY treatment
UNION ALL
SELECT 'exposure_within_treatment', CAST(exposure AS VARCHAR),
       AVG(f0), AVG(f1), AVG(f2), AVG(f3), AVG(f4), AVG(f5),
       AVG(f6), AVG(f7), AVG(f8), AVG(f9), AVG(f10), AVG(f11)
FROM criteo WHERE treatment = 1 GROUP BY exposure
ORDER BY split, grp;
