"""Execute every query in sql/ with DuckDB against data/criteo_uplift.parquet; save CSVs to results/sql/."""
import pathlib, duckdb, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sql"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET file_search_path='{ROOT.as_posix()}'")
    con.execute((ROOT / "sql" / "00_view.sql").read_text())
    pd.set_option("display.width", 200, "display.max_columns", 20)
    for q in sorted((ROOT / "sql").glob("[0-9]*.sql")):
        if q.name.startswith("00_"):
            continue
        df = con.execute(q.read_text()).df()
        df.to_csv(OUT / (q.stem + ".csv"), index=False)
        print(f"\n== {q.name}\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
