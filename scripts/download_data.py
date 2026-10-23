"""Download the Criteo Uplift Prediction dataset (v2.1) and convert it to Parquet.

Source: Criteo AI Lab, "A Large Scale Benchmark for Uplift Modeling" (Diemert et al., 2018).
Official page: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
The original go.criteo.net link currently returns 404, so this script pulls the identical file
from the public Hugging Face mirror published by Criteo (no login required):
    https://huggingface.co/datasets/criteo/criteo-uplift
License: CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike). Raw data is NOT committed.

Usage:  python scripts/download_data.py            # full 13.98M rows
        python scripts/download_data.py --sample 2000000   # optional random sample (seed 42)
"""
import argparse, pathlib, urllib.request
import duckdb

URL = "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="rows to sample (0 = use all rows)")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    gz = DATA / "criteo-research-uplift-v2.1.csv.gz"
    if not gz.exists():
        print(f"Downloading {URL} (~311 MB) ...")
        urllib.request.urlretrieve(URL, gz)
    out = DATA / "criteo_uplift.parquet"
    con = duckdb.connect()
    con.execute("SELECT setseed(0.42)")
    src = f"read_csv_auto('{gz.as_posix()}')"
    sel = f"SELECT * FROM {src}"
    if args.sample:
        sel = f"SELECT * FROM {src} USING SAMPLE reservoir({args.sample} ROWS) REPEATABLE (42)"
    con.execute(f"""COPY (
        SELECT CAST(f0 AS FLOAT) f0, CAST(f1 AS FLOAT) f1, CAST(f2 AS FLOAT) f2, CAST(f3 AS FLOAT) f3,
               CAST(f4 AS FLOAT) f4, CAST(f5 AS FLOAT) f5, CAST(f6 AS FLOAT) f6, CAST(f7 AS FLOAT) f7,
               CAST(f8 AS FLOAT) f8, CAST(f9 AS FLOAT) f9, CAST(f10 AS FLOAT) f10, CAST(f11 AS FLOAT) f11,
               CAST(treatment AS TINYINT) treatment, CAST(conversion AS TINYINT) conversion,
               CAST(visit AS TINYINT) visit, CAST(exposure AS TINYINT) exposure
        FROM ({sel})) TO '{out.as_posix()}' (FORMAT PARQUET)""")
    n = con.execute(f"SELECT COUNT(*) FROM '{out.as_posix()}'").fetchone()[0]
    print(f"Wrote {out} with {n:,} rows")


if __name__ == "__main__":
    main()
