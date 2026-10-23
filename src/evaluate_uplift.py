"""Evaluate uplift models on the 70% held-out test set: uplift/Qini curves, Qini coefficient,
observed uplift by predicted decile, and the targeting trade-off (share of incremental outcomes kept vs share of
users targeted, used as the proxy for spend)."""
import json
import numpy as np, pandas as pd
from common import RESULTS

MODELS = ["t_learner", "class_transform", "response_model", "random"]
GRID = np.linspace(0.01, 1.0, 100)


def curves(df, score, y, seed=42):
    # exposure (if present) gives a second spend proxy: share of the treated arm's ad exposures in the targeted group
    s = df[score].values if score != "random" else np.random.default_rng(seed).random(len(df))
    order = np.argsort(-s, kind="stable")
    t = df.treatment.values[order]; yy = df[y].values[order]
    ex = np.cumsum(df.exposure.values[order] * t) if "exposure" in df else None
    ct = np.cumsum(t); cc = np.cumsum(1 - t); yt = np.cumsum(yy * t); yc = np.cumsum(yy * (1 - t))
    n = len(df); rows = []
    for g in GRID:
        k = int(round(g * n)) - 1
        nt, nc = ct[k], cc[k]
        rt, rc = yt[k] / nt, yc[k] / nc
        rows.append(dict(frac=g, n_t=nt, n_c=nc, rate_t=rt, rate_c=rc,
                         uplift=rt - rc,                       # avg uplift inside targeted group
                         gain=(rt - rc) * (nt + nc),           # incremental outcomes if targeted group is treated
                         qini=yt[k] - yc[k] * nt / nc,
                         share_exposures=(ex[k] / ex[-1]) if ex is not None else np.nan))
    c = pd.DataFrame(rows)
    total = c.gain.iloc[-1]
    c["share_incremental_kept"] = c.gain / total
    return c


def qini_coef(c):
    """Normalised area between model gain curve and the random line (fraction of total incremental outcomes)."""
    x = np.r_[0, c.frac.values]; g = np.r_[0, c.share_incremental_kept.values]
    return float(np.trapezoid(g, x) - 0.5)


def deciles(df, score, y):
    d = df.copy()
    d["decile"] = pd.qcut(d[score].rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
    g = d.groupby(["decile", "treatment"])[y].agg(["mean", "count", "var"]).unstack()
    out = pd.DataFrame({"rate_t": g[("mean", 1)], "rate_c": g[("mean", 0)]})
    out["uplift"] = out.rate_t - out.rate_c
    se = np.sqrt(g[("var", 1)] / g[("count", 1)] + g[("var", 0)] / g[("count", 0)])
    out["ci_low"] = out.uplift - 1.96 * se; out["ci_high"] = out.uplift + 1.96 * se
    out["n"] = g[("count", 1)] + g[("count", 0)]
    return out.reset_index()


def main():
    summary = {}
    for y in ("conversion", "visit"):
        df = pd.read_parquet(RESULTS / f"test_scores_{y}.parquet")
        from common import load
        df["exposure"] = load(["exposure"]).exposure.values[np.load(RESULTS / "split_is_test.npy")]
        models = list(MODELS)
        if y == "conversion":
            # Proxy: rank conversions by the visit-uplift T-learner (visits are ~16x denser, so less noisy).
            # Same 70% test rows in the same order (both files come from the same saved split mask).
            df["t_learner_visit_proxy"] = pd.read_parquet(RESULTS / "test_scores_visit.parquet", columns=["t_learner"]).t_learner.values
            models.insert(2, "t_learner_visit_proxy")
        overall = df.loc[df.treatment == 1, y].mean() - df.loc[df.treatment == 0, y].mean()
        summary[y] = {"test_rows": len(df), "test_overall_uplift": overall, "models": {}}
        allc = []
        for m in models:
            c = curves(df, m, y); c["model"] = m; allc.append(c)
            at = lambda f: c.loc[np.isclose(c.frac, f)].iloc[0]
            info = {"qini_coefficient": qini_coef(c)}
            for f in (0.1, 0.2, 0.3, 0.5):
                r = at(f)
                info[f"top{int(f*100)}"] = dict(uplift=r.uplift, uplift_vs_overall=r.uplift / overall,
                                               share_incremental_kept=r.share_incremental_kept,
                                               share_exposures=r.share_exposures)
            info["peak_share_kept"] = float(c.share_incremental_kept.max())
            info["peak_at_frac"] = float(c.frac[c.share_incremental_kept.idxmax()])
            # smallest share of users that keeps >= 80% / 90% of incremental outcomes
            for q in (0.8, 0.9):
                hit = c[c.share_incremental_kept >= q]
                info[f"frac_for_{int(q*100)}pct"] = float(hit.frac.iloc[0]) if len(hit) else None
            if m != "random":
                dec = deciles(df, m, y); dec.to_csv(RESULTS / f"deciles_{y}_{m}.csv", index=False)
                info["top_decile"] = dec.iloc[0][["uplift", "ci_low", "ci_high"]].to_dict()
                info["bottom_decile"] = dec.iloc[-1][["uplift", "ci_low", "ci_high"]].to_dict()
            summary[y]["models"][m] = info
        pd.concat(allc).to_csv(RESULTS / f"uplift_curves_{y}.csv", index=False)
    (RESULTS / "uplift_eval.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
