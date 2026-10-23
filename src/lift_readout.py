"""Conversion-Lift style readout on the full Criteo uplift data.
ITT lift with CIs, SRM check, covariate balance, CACE / treatment-on-treated via assignment-as-instrument,
and a decomposition of why naive exposed-vs-unexposed comparisons are biased."""
import json
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from common import load, RESULTS, FEATURES

Z = stats.norm.ppf(0.975)


def itt(df, y):
    t, c = df.loc[df.treatment == 1, y], df.loc[df.treatment == 0, y]
    xt, nt, xc, nc = int(t.sum()), len(t), int(c.sum()), len(c)
    pt, pc = xt / nt, xc / nc
    zstat, p = proportions_ztest([xt, xc], [nt, nc])
    lo, hi = confint_proportions_2indep(xt, nt, xc, nc, compare="diff", method="wald")
    rlo, rhi = confint_proportions_2indep(xt, nt, xc, nc, compare="ratio", method="log")
    return dict(metric=y, n_t=nt, n_c=nc, conv_t=xt, conv_c=xc, rate_t=pt, rate_c=pc,
                abs_lift=pt - pc, abs_ci=[lo, hi], rel_lift=pt / pc - 1, rel_ci=[rlo - 1, rhi - 1],
                z=zstat, p_value=p, incremental_per_1m=(pt - pc) * 1e6,
                incremental_per_1m_ci=[lo * 1e6, hi * 1e6],
                total_incremental_in_treated_arm=(pt - pc) * nt)


def cace(df, y):
    """Wald IV: (E[Y|Z=1]-E[Y|Z=0]) / (E[D|Z=1]-E[D|Z=0]), delta-method SE incl. Y-D covariance."""
    g1, g0 = df[df.treatment == 1], df[df.treatment == 0]
    n1, n0 = len(g1), len(g0)
    num = g1[y].mean() - g0[y].mean()
    den = g1.exposure.mean() - g0.exposure.mean()
    v_num = g1[y].var() / n1 + g0[y].var() / n0
    v_den = g1.exposure.var() / n1 + g0.exposure.var() / n0
    cov = np.cov(g1[y], g1.exposure)[0, 1] / n1 + (np.cov(g0[y], g0.exposure)[0, 1] / n0 if g0.exposure.var() > 0 else 0)
    est = num / den
    se = np.sqrt(v_num / den**2 + num**2 * v_den / den**4 - 2 * num * cov / den**3)
    exposed = g1.loc[g1.exposure == 1, y].mean()
    # Under one-sided non-compliance, compliers = exposed treated users; their untreated baseline:
    complier_y0 = exposed - est
    return dict(metric=y, itt=num, first_stage=den, cace=est, cace_ci=[est - Z * se, est + Z * se],
                exposed_rate=exposed, complier_baseline_y0=complier_y0,
                cace_rel_lift=est / complier_y0)


def naive(df, y):
    exp = df.loc[df.exposure == 1, y].mean()
    unexp = df.loc[df.exposure == 0, y].mean()
    ctrl = df.loc[df.treatment == 0, y].mean()
    t_unexp = df.loc[(df.treatment == 1) & (df.exposure == 0), y].mean()
    return dict(metric=y, exposed=exp, unexposed_all=unexp, naive_diff=exp - unexp,
                naive_rel=exp / unexp - 1, control=ctrl, treated_unexposed=t_unexp)


def smd(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


def main():
    df = load()
    out = {"n_rows": len(df)}
    # SRM against the 85/15 design
    n1 = int(df.treatment.sum()); n = len(df)
    chi = stats.chisquare([n1, n - n1], [0.85 * n, 0.15 * n])
    out["srm"] = dict(treat_share=n1 / n, chi2=chi.statistic, p_value=chi.pvalue)
    out["itt"] = [itt(df, y) for y in ("visit", "conversion")]
    out["cace"] = [cace(df, y) for y in ("visit", "conversion")]
    out["naive"] = [naive(df, y) for y in ("visit", "conversion")]
    t, c = df[df.treatment == 1], df[df.treatment == 0]
    te, tu = t[t.exposure == 1], t[t.exposure == 0]
    bal = pd.DataFrame({"smd_treat_vs_control": [smd(t[f], c[f]) for f in FEATURES],
                        "smd_exposed_vs_unexposed_in_treatment": [smd(te[f], tu[f]) for f in FEATURES]},
                       index=FEATURES)
    bal.to_csv(RESULTS / "covariate_balance_smd.csv")
    out["balance"] = dict(max_abs_smd_arm=float(bal.iloc[:, 0].abs().max()),
                          max_abs_smd_exposure=float(bal.iloc[:, 1].abs().max()))
    # Regression-adjusted ITT (CUPED-style variance reduction with pre-treatment features).
    # OLS with HC1 robust SEs computed in chunks to keep memory flat on 14M rows.
    res = {}
    mu, sd = df[FEATURES].mean(), df[FEATURES].std()
    for y in ("visit", "conversion"):
        res[y] = {}
        for label, cols in (("adjusted", FEATURES), ("unadjusted", [])):
            k = 2 + len(cols)
            def design(part):
                Xp = np.empty((len(part), k))
                Xp[:, 0] = 1.0; Xp[:, 1] = part.treatment.values
                if cols:
                    Xp[:, 2:] = ((part[cols] - mu[cols]) / sd[cols]).values
                return Xp
            xtx = np.zeros((k, k)); xty = np.zeros(k)
            chunks = range(0, len(df), 1_000_000)
            for s0 in chunks:
                part = df.iloc[s0:s0 + 1_000_000]; Xp = design(part); yp = part[y].values.astype(float)
                xtx += Xp.T @ Xp; xty += Xp.T @ yp
            beta = np.linalg.solve(xtx, xty); meat = np.zeros((k, k))
            for s0 in chunks:
                part = df.iloc[s0:s0 + 1_000_000]; Xp = design(part)
                e = part[y].values - Xp @ beta; meat += (Xp * (e ** 2)[:, None]).T @ Xp
            inv = np.linalg.inv(xtx); n_ = len(df)
            V = inv @ meat @ inv * n_ / (n_ - k)
            res[y][label] = dict(lift=beta[1], se=float(np.sqrt(V[1, 1])))
        a, u = res[y]["adjusted"], res[y]["unadjusted"]
        res[y]["variance_reduction"] = 1 - (a["se"] / u["se"]) ** 2
    out["regression_adjusted"] = res
    (RESULTS / "lift_readout.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))
    print(bal.round(3))


if __name__ == "__main__":
    main()
