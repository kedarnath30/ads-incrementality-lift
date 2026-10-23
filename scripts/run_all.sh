#!/usr/bin/env bash
# End-to-end pipeline. ~6-8 minutes on a 2-core / 8 GB machine.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/download_data.py            # full 13.98M rows -> data/criteo_uplift.parquet
python src/run_sql.py                      # DuckDB metric queries in sql/ -> results/sql/*.csv
cd src
python lift_readout.py                     # ITT lift, CIs, SRM, balance, CACE/TOT, regression adjustment
python uplift_models.py split              # 30% train / 70% test, stratified by treatment
for y in visit conversion; do
  python uplift_models.py fit_t  "$y"      # T-learner (two HistGradientBoosting models)
  python uplift_models.py fit_ct "$y"      # class-transformation model
  python uplift_models.py predict "$y"     # score the held-out test set
done
python evaluate_uplift.py                  # uplift / Qini curves, deciles, targeting trade-off
python make_figures.py                     # figures/*.png
