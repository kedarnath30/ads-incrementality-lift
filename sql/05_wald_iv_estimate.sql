-- Wald / IV estimator of the complier average causal effect (CACE = effect of treatment on the treated,
-- because control cannot be exposed): ITT on outcome divided by ITT on exposure.
WITH a AS (
  SELECT AVG(CASE WHEN treatment=1 THEN visit END)      - AVG(CASE WHEN treatment=0 THEN visit END)      AS itt_visit,
         AVG(CASE WHEN treatment=1 THEN conversion END) - AVG(CASE WHEN treatment=0 THEN conversion END) AS itt_conv,
         AVG(CASE WHEN treatment=1 THEN exposure END)   - AVG(CASE WHEN treatment=0 THEN exposure END)   AS itt_exposure
  FROM criteo
)
SELECT itt_visit, itt_conv, itt_exposure,
       itt_visit / itt_exposure AS cace_visit,
       itt_conv  / itt_exposure AS cace_conv
FROM a;
