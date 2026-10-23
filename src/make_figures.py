"""Render all charts in figures/ from results/ (run after lift_readout.py and evaluate_uplift.py)."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS, FIGS

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
BLUE, GREY, ORANGE, GREEN, RED = "#1877F2", "#8A8D91", "#F28C28", "#2E9E5B", "#C0392B"
LABELS = {"t_learner": "T-learner (two-model)", "class_transform": "Class transformation",
          "t_learner_visit_proxy": "T-learner on visits (proxy)", "response_model": "Response model P(y|treated)",
          "random": "Random targeting"}
COLORS = {"t_learner": BLUE, "class_transform": ORANGE, "t_learner_visit_proxy": GREEN,
          "response_model": "#6C3483", "random": GREY}


def readout(lr):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, r in zip(axes, lr["itt"]):
        rates = [r["rate_c"] * 100, r["rate_t"] * 100]
        se = [np.sqrt(p / 100 * (1 - p / 100) / n) * 196 for p, n in zip(rates, [r["n_c"], r["n_t"]])]
        ax.bar(["Control\n(holdout, 15%)", "Treatment\n(eligible for ads, 85%)"], rates, yerr=se, capsize=6,
               color=[GREY, BLUE], width=0.55)
        lo, hi = r["rel_ci"]
        ax.set_title(f"{r['metric'].capitalize()} rate: +{r['rel_lift']*100:.1f}% lift "
                     f"(95% CI {lo*100:.1f}% to {hi*100:.1f}%)", fontsize=10)
        ax.set_ylabel("% of users")
        for i, v in enumerate(rates):
            ax.text(i, v * 1.02 + max(rates) * 0.02, f"{v:.3f}%", ha="center")
        ax.set_ylim(0, max(rates) * 1.25)
    fig.suptitle("Intent-to-treat lift readout, full Criteo data (13.98M users)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGS / "01_lift_readout.png"); plt.close(fig)


def naive_vs_causal(lr):
    c = [x for x in lr["cace"] if x["metric"] == "conversion"][0]
    n = [x for x in lr["naive"] if x["metric"] == "conversion"][0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    groups = ["Treated,\nnot exposed", "Control\n(all)", "Exposed users'\ncounterfactual\n(no ad, IV est.)", "Treated,\nexposed"]
    vals = [n["treated_unexposed"], n["control"], c["complier_baseline_y0"], c["exposed_rate"]]
    ax.bar(groups, np.array(vals) * 100, color=[GREY, GREY, ORANGE, BLUE], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v * 100 + 0.08, f"{v*100:.2f}%", ha="center")
    ax.set_ylabel("Conversion rate (%)"); ax.set_ylim(0, c["exposed_rate"] * 100 * 1.2)
    ax.set_title("Exposed users were already ~11x likelier to convert", fontsize=10)
    ax = axes[1]
    labels = ["Naive: exposed\nvs unexposed", "CACE / TOT\n(assignment as IV)", "ITT\n(per assigned user)"]
    vals = [n["naive_diff"], c["cace"], c["itt"]]
    errs = [0, (c["cace_ci"][1] - c["cace_ci"][0]) / 2, 0]
    ax.barh(labels, np.array(vals) * 100, xerr=np.array(errs) * 100, color=[RED, BLUE, GREEN], capsize=5)
    for i, v in enumerate(vals):
        ax.text((v + errs[i]) * 100 + 0.12, i, f"{v*100:.2f} pp", va="center")
    ax.invert_yaxis(); ax.set_xlabel("Effect on conversion rate (percentage points)")
    ax.set_xlim(0, n["naive_diff"] * 100 * 1.3)
    ax.set_title(f"Naive comparison overstates the per-exposed effect by {n['naive_diff']/c['cace']:.1f}x", fontsize=10)
    fig.suptitle("Why exposed-vs-unexposed is biased: selection into exposure", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGS / "02_naive_vs_causal.png"); plt.close(fig)


def balance():
    b = pd.read_csv(RESULTS / "covariate_balance_smd.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(b))
    ax.scatter(b.iloc[:, 0], y, color=BLUE, label="Treatment vs control (randomized)", zorder=3)
    ax.scatter(b.iloc[:, 1], y, color=RED, marker="s", label="Exposed vs unexposed (within treatment)", zorder=3)
    ax.axvline(0, color="k", lw=0.8); ax.axvspan(-0.1, 0.1, color=GREEN, alpha=0.1, label="|SMD| < 0.1")
    ax.set_yticks(y, b.index); ax.set_xlabel("Standardized mean difference")
    ax.set_title("Covariate balance: randomized arms vs self-selected exposure", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "03_covariate_balance.png"); plt.close(fig)


def gain_curves(y, ev, fname, title):
    c = pd.read_csv(RESULTS / f"uplift_curves_{y}.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m, g in c.groupby("model", sort=False):
        q = ev[y]["models"][m]["qini_coefficient"]
        ax.plot(np.r_[0, g.frac] * 100, np.r_[0, g.share_incremental_kept] * 100, color=COLORS[m],
                lw=2.2 if m != "random" else 1.2, ls="--" if m == "random" else "-",
                label=f"{LABELS[m]} (Qini coef {q:+.3f})")
    ax.set_xlabel("% of users targeted (ranked by model score)")
    ax.set_ylabel(f"% of incremental {y}s captured")
    ax.set_title(title, fontweight="bold", fontsize=10); ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 100); fig.tight_layout(); fig.savefig(FIGS / fname); plt.close(fig)


def deciles(ev):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (y, m) in zip(axes, [("visit", "t_learner"), ("conversion", "response_model")]):
        d = pd.read_csv(RESULTS / f"deciles_{y}_{m}.csv")
        ax.bar(d.decile, d.uplift * 100, yerr=[(d.uplift - d.ci_low) * 100, (d.ci_high - d.uplift) * 100],
               color=BLUE, capsize=3)
        ax.axhline(ev[y]["test_overall_uplift"] * 100, color=RED, ls="--", lw=1, label="Average uplift (test set)")
        ax.set_xticks(d.decile); ax.set_xlabel("Predicted-score decile (1 = highest)")
        ax.set_ylabel(f"Observed {y} uplift (pp)"); ax.legend(fontsize=8)
        ax.set_title(f"{y.capitalize()}: {LABELS[m]}", fontsize=10)
    fig.suptitle("Observed uplift by model decile on the 70% held-out test set (95% CIs)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGS / "06_uplift_by_decile.png"); plt.close(fig)


def tradeoff(ev):
    c = pd.read_csv(RESULTS / "uplift_curves_conversion.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, (col, xl) in zip(axes, [("frac", "% of users targeted (reach / budget proxy)"),
                                    ("share_exposures", "% of observed ad exposures (impression proxy)")]):
        for m in ("response_model", "t_learner_visit_proxy", "t_learner", "random"):
            g = c[c.model == m]
            ax.plot(np.r_[0, g[col]] * 100, np.r_[0, g.share_incremental_kept] * 100, color=COLORS[m],
                    lw=2 if m != "random" else 1.2, ls="--" if m == "random" else "-", label=LABELS[m])
        ax.plot([0, 100], [0, 100], color="k", lw=0.6, alpha=0.4)
        ax.set_xlabel(xl); ax.set_xlim(0, 100); ax.set_ylim(0, 105)
    axes[0].set_ylabel("% of incremental conversions kept")
    r = ev["conversion"]["models"]["response_model"]["top20"]
    axes[0].annotate(f"Top 20% of users\n-> {r['share_incremental_kept']*100:.0f}% of incremental conv.",
                     xy=(20, r["share_incremental_kept"] * 100), xytext=(35, 55),
                     arrowprops=dict(arrowstyle="->"), fontsize=9)
    axes[1].annotate(f"...but those users hold\n{r['share_exposures']*100:.0f}% of exposures",
                     xy=(r["share_exposures"] * 100, r["share_incremental_kept"] * 100), xytext=(58, 22),
                     arrowprops=dict(arrowstyle="->"), fontsize=9)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Targeting trade-off for conversions: incremental conversions kept vs share of spend",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGS / "07_targeting_tradeoff.png"); plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    lr = json.loads((RESULTS / "lift_readout.json").read_text())
    ev = json.loads((RESULTS / "uplift_eval.json").read_text())
    readout(lr); naive_vs_causal(lr); balance()
    gain_curves("visit", ev, "04_uplift_curve_visit.png", "Cumulative uplift (gain) curve: visits, held-out test set")
    gain_curves("conversion", ev, "05_uplift_curve_conversion.png",
                "Cumulative uplift (gain) curve: conversions, held-out test set")
    deciles(ev); tradeoff(ev)
    print(sorted(p.name for p in FIGS.glob("*.png")))


if __name__ == "__main__":
    main()
