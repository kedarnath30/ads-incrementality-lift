"""Heterogeneous treatment effects: T-learner (two-model) and class-transformation uplift models.

Split: random 30% train / 70% test (seed 42), stratified by treatment. Models are scikit-learn
HistGradientBoostingClassifier. Each step is a separate CLI call so it can be run in short chunks:
    python uplift_models.py split
    python uplift_models.py fit_t <visit|conversion>
    python uplift_models.py fit_ct <visit|conversion>
    python uplift_models.py predict <visit|conversion>
"""
import sys, time
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from common import load, RESULTS, FEATURES, SEED

MODELS = RESULTS / "models"
SPLIT = RESULTS / "split_is_test.npy"


def hgb():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                          min_samples_leaf=500, l2_regularization=1.0,
                                          early_stopping=False, random_state=SEED)


def get(part):
    df = load(FEATURES + ["treatment", "visit", "conversion"])
    is_test = np.load(SPLIT)
    return df[is_test] if part == "test" else df[~is_test]


def split():
    t = load(["treatment"]).treatment.values
    idx = np.arange(len(t))
    _, te = train_test_split(idx, test_size=0.7, stratify=t, random_state=SEED)
    mask = np.zeros(len(t), bool); mask[te] = True
    np.save(SPLIT, mask)
    print(f"train={(~mask).sum():,} test={mask.sum():,}")


def fit_t(y):
    tr = get("train"); MODELS.mkdir(exist_ok=True)
    for arm in (1, 0):
        d = tr[tr.treatment == arm]; t0 = time.time()
        m = hgb().fit(d[FEATURES].values, d[y].values)
        joblib.dump(m, MODELS / f"t_{y}_arm{arm}.joblib")
        print(f"T-learner {y} arm{arm}: n={len(d):,} {time.time()-t0:.0f}s")


def fit_ct(y):
    """Class transformation (Jaskowski & Jaroszewicz 2012): Z = 1 if (T=1,Y=1) or (T=0,Y=0).
    Treatment is 85/15, so rows are reweighted to an effective 50/50 split; then uplift = 2*P(Z=1) - 1."""
    tr = get("train"); MODELS.mkdir(exist_ok=True)
    p = tr.treatment.mean()
    z = ((tr.treatment == 1) & (tr[y] == 1)) | ((tr.treatment == 0) & (tr[y] == 0))
    w = np.where(tr.treatment == 1, 0.5 / p, 0.5 / (1 - p))
    t0 = time.time()
    m = hgb().fit(tr[FEATURES].values, z.values.astype(int), sample_weight=w)
    joblib.dump(m, MODELS / f"ct_{y}.joblib")
    print(f"Class-transformation {y}: n={len(tr):,} {time.time()-t0:.0f}s")


def predict(y):
    te = get("test")
    X = te[FEATURES].values
    m1 = joblib.load(MODELS / f"t_{y}_arm1.joblib"); m0 = joblib.load(MODELS / f"t_{y}_arm0.joblib")
    mc = joblib.load(MODELS / f"ct_{y}.joblib")
    p1 = m1.predict_proba(X)[:, 1]; p0 = m0.predict_proba(X)[:, 1]
    out = pd.DataFrame({"treatment": te.treatment.values, "visit": te.visit.values,
                        "conversion": te.conversion.values,
                        "t_learner": (p1 - p0).astype(np.float32),
                        "class_transform": (2 * mc.predict_proba(X)[:, 1] - 1).astype(np.float32),
                        "response_model": p1.astype(np.float32)})   # "target likely converters" baseline
    out.to_parquet(RESULTS / f"test_scores_{y}.parquet", index=False)
    print(out.describe().T)


if __name__ == "__main__":
    step = sys.argv[1]
    {"split": lambda: split(), "fit_t": lambda: fit_t(sys.argv[2]), "fit_ct": lambda: fit_ct(sys.argv[2]),
     "predict": lambda: predict(sys.argv[2])}[step]()
