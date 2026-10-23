-- Base view over the Parquet produced by scripts/download_data.py
CREATE OR REPLACE VIEW criteo AS SELECT * FROM read_parquet('data/criteo_uplift.parquet');
