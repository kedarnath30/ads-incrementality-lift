import pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "criteo_uplift.parquet"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
FEATURES = [f"f{i}" for i in range(12)]
SEED = 42


def load(columns=None):
    df = pd.read_parquet(DATA, columns=columns)
    for c in ("treatment", "conversion", "visit", "exposure"):
        if c in df:
            df[c] = df[c].astype(np.int8)
    return df
