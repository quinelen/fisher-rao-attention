"""Results figure of the paper (Figure 4), plotted from the result JSONs of this folder.

Usage (from verificacion_claude/):
    python fig_results.py            # writes fig_results.png here and in ../arxiv/

Panels: (a) mean learned gate weights of the mixed-curvature model on Dyck-2 and copy
(resultados_mixta.json); (b) Dyck-2 closing accuracy beyond the training length for the
five single-geometry arms (resultados_tres_geometrias.json, resultados_ricci_addendum.json,
resultados_fast_gaussian.json); (c) kernel ablation with Gaussian tokens held fixed
(resultados_wasserstein_vs_fr.json). Error bars: population standard deviation over the
5 seeds, as in the tables of the paper.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def pct(xs):
    a = 100 * np.asarray(xs, dtype=float)
    return a.mean(), a.std()          # population sd (ddof = 0)


GRAY, BLUE, PURPLE, GREEN, RED, ORANGE = "#999999", "#4C72B0", "#8172B3", "#55A868", "#C44E52", "#DD8452"
INK, MUTED = "#000000", "#444444"

mixta = load("resultados_mixta.json")
tres = load("resultados_tres_geometrias.json")
ricci = load("resultados_ricci_addendum.json")
fast = load("resultados_fast_gaussian.json")
wfr = load("resultados_wasserstein_vs_fr.json")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1.0, 1.25, 1.0]})

# (a) gate allocation: mean over seeds, layers and heads of the [layers, heads, 3] gate tensor
ax = axes[0]
gates = {t: np.array([r["gates"] for r in mixta[t]["mix"]]).reshape(-1, 3).mean(0) for t in ("dyck", "copy")}
x = np.arange(3)
w = 0.36
for off, task, color, lab in ((-w / 2, "dyck", BLUE, "Dyck-2 (hierarchical)"), (w / 2, "copy", ORANGE, "Copy (flat)")):
    bars = ax.bar(x + off, gates[task], w, color=color, label=lab)
    for b, v in zip(bars, gates[task]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(["Euclidean\n(K=0)", "Spherical\n(K=+1/4)", "Hyperbolic\n(K=-1/2)"], fontsize=9)
ax.set_ylabel("Mean learned gate weight")
ax.set_ylim(0, 0.75)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("(a) Gate allocation tracks data structure", fontsize=11)

# (b) the five single-geometry arms
ax = axes[1]
arms = [
    ("Euclidean", tres["euclidiana (K=0)"]["out_acc"], GRAY),
    ("Spherical\nFR exact", tres["esferica (K=+1/4)"]["out_acc"], BLUE),
    ("Ricci-\nMahalanobis", ricci["out_acc"], PURPLE),
    ("Hyperbolic\naniso.", tres["hiperbolica (K=-1/2)"]["out_acc"], GREEN),
    ("Hyperbolic\nISO (ours)", [r["out_acc"] for r in fast["isotropica"]], RED),
]
for i, (lab, vals, color) in enumerate(arms):
    m, s = pct(vals)
    ax.bar(i, m, 0.8, color=color, yerr=s, capsize=4, error_kw={"elinewidth": 1.6, "capthick": 1.6})
    ax.text(i, m + s + 0.12, f"{m:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
ax.set_xticks(range(len(arms)))
ax.set_xticklabels([a[0] for a in arms], fontsize=8)
ax.set_ylabel("OUT bracket-closing accuracy (%)")
ax.set_ylim(84, 90)
ax.set_title("(b) Dyck-2 accuracy beyond the training length (5 seeds)", fontsize=11)

# (c) same Gaussian tokens, different score
ax = axes[2]
abl = [
    ("Euclidean\n(point tokens)", wfr["euclidiana"], GRAY),
    ("Wasserstein\n(Gaussian tokens)", wfr["wasserstein (Bures)"], ORANGE),
    ("Fisher-Rao\n(Gaussian tokens)", wfr["fisher-rao (hiperb)"], RED),
]
means = []
for i, (lab, vals, color) in enumerate(abl):
    m, s = pct(vals)
    means.append(m)
    ax.bar(i, m, 0.8, color=color, yerr=s, capsize=4, error_kw={"elinewidth": 1.6, "capthick": 1.6})
    ax.text(i, m + s + 0.12, f"{m:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
ax.text(0.03, 0.965,
        f"Wasserstein \u2212 Euclidean: +{means[1] - means[0]:.2f} pp\nFisher-Rao \u2212 Wasserstein: +{means[2] - means[1]:.2f} pp",
        transform=ax.transAxes, ha="left", va="top", fontsize=8, color=MUTED, linespacing=1.5)
ax.set_xticks(range(len(abl)))
ax.set_xticklabels([a[0] for a in abl], fontsize=8)
ax.set_ylabel("OUT bracket-closing accuracy (%)")
ax.set_ylim(84, 90)
ax.set_title("(c) Kernel ablation with Gaussian\ntokens held fixed (§4.6)", fontsize=11)

fig.tight_layout()
for folder in (HERE, os.path.join(HERE, "..", "arxiv")):
    fig.savefig(os.path.join(folder, "fig_results.png"), dpi=200)
print("gates (euc, sph, hyp):", {t: gates[t].round(3).tolist() for t in gates})
print("arms:", [(a[0].replace("\n", " "), tuple(round(v, 2) for v in pct(a[1]))) for a in arms])
print("ablation:", [(a[0].replace("\n", " "), tuple(round(v, 2) for v in pct(a[1]))) for a in abl],
      "| steps:", round(means[1] - means[0], 2), round(means[2] - means[1], 2))
