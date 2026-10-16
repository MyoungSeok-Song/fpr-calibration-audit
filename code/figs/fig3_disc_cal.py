"""Fig. 3 (v2): calibration error versus discrimination, three regimes.

Source: results/unit_rank_unseen.csv, reference "unseen4" (Lyria 3, HeartMuLa, MiniMax 2.6,
Mureka v9; generators weighted equally, songs 1/n_f; cluster bootstrap B = 10,000, τ re-estimated).
For every detector and human cohort c: x = FPR_c at the deployed window-level Jamendo-A threshold
(calibration), y = oracle TPR = reference TPR at cohort c's own 2% threshold (discrimination).
Regimes (colour): calibration-dominated = EAT families (orange), calibration + discrimination = C1 and
M0 on speech-over-music cohorts (teal), coverage failure = public detectors (grey). Hulls enclose the
points of each regime; horizontal bands span each group's deployed TPR (TPR at the Jamendo-A τ).
Every number written in the figure is read from the CSV.
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (TAB, NAME, EXCLUDE, ORDER, COL_W_IN, ORANGE, ORANGE_FILL, TEAL, TEAL_FILL, GREY, GREY_L, GREY_FILL,  # noqa: E402
                    INK, style, save, fpr_axis, alpha_band)

style()
FLOOR = 0.02
d = pd.read_csv(TAB / "unit_rank_unseen.csv")
d = d[d["ref"] == "unseen4"].copy()
missing = sorted({NAME.get(x, x) for x in d.det} - set(ORDER))
if missing:
    print("WARNING unmapped detectors:", missing)
d["paper"] = d.det.map(lambda x: NAME.get(x, x))
d = d.dropna(subset=["paper"])
d = d[~d.paper.isin(EXCLUDE)]
SPEECH = {"ava_sm5", "audioset_sm", "openbmat_human_musicbearing"}
EAT = {"B0", "B0′", "B0+0", "C0", "C0+0"}
fam = lambda p: "eat" if p in EAT else ({"C1": "c1", "M0": "m0"}.get(p, "pub"))
d["fam"] = d.paper.map(fam)
d["speech"] = d.cohort.isin(SPEECH)
d["x"] = d.fpr_tauJamA.clip(lower=FLOOR)
# regime of each point
def regime(r):
    if r.fam == "pub":
        return "coverage"
    if r.fam in ("c1", "m0") and r.speech:
        return "disc"
    return "calib"
d["regime"] = d.apply(regime, axis=1)

fig, ax = plt.subplots(figsize=(COL_W_IN, 2.15))
fig.subplots_adjust(left=0.125, right=0.885, top=0.91, bottom=0.18)
XLO, XHI = FLOOR * 0.8, 24
alpha_band(ax, axis="x")
fpr_axis(ax, lo=XLO, hi=XHI, ticks=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20], label_floor=FLOOR, axis="x")
ax.set_ylim(-3, 103)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.grid(axis="y", which="major", zorder=0)
ax.set_xlabel("FPR (%) at deployed Jamendo-A τ  (→ calibration error)", labelpad=2)
ax.set_ylabel("Oracle TPR (%), 4 new families", labelpad=2)

# ---- regime hulls (convex hull in log-x space, padded by a wide round-joined outline of the fill colour)
def hull(points, fc, ec):
    P = np.array([[np.log10(x), y] for x, y in points])
    V = P[ConvexHull(P).vertices] if len(P) >= 3 else P
    V = np.column_stack([10 ** V[:, 0], V[:, 1]])
    ax.add_patch(Polygon(V, closed=True, fc=fc, ec=fc, lw=7, joinstyle="round", zorder=0.5))
    # thin dotted outline in the regime color: keeps the three regions apart in grayscale print
    ax.add_patch(Polygon(V, closed=True, fc="none", ec=ec, lw=0.6, ls=(0, (1.5, 1.5)), alpha=0.55, joinstyle="round", zorder=0.55))
for reg, fc, ec in (("calib", ORANGE_FILL, ORANGE), ("disc", TEAL_FILL, TEAL), ("coverage", GREY_FILL, GREY)):
    s = d[d.regime == reg]
    hull(list(zip(s.x, s.tpr_oracle)), fc, ec)

# ---- deployed-TPR bands (TPR at τ) per group
bands = {}
for key, members in (("ft", ("eat", "c1", "m0")), ("pub", ("pub",))):
    s = d[d.fam.isin(members)]
    lo, hi = s.tpr_tauJamA.min(), s.tpr_tauJamA.max(); bands[key] = (lo, hi)
    ax.axhspan(lo, hi, color="#DADADA", zorder=0.3, lw=0)
    ax.axhline(lo, color="#9A9A9A", lw=0.4, ls=(0, (1, 1.5)), zorder=0.4); ax.axhline(hi, color="#9A9A9A", lw=0.4, ls=(0, (1, 1.5)), zorder=0.4)
for key in ("ft", "pub"):
    lo, hi = bands[key]
    ax.text(XHI * 1.05, (lo + hi) / 2, "TPR@τ", fontsize=7, color="#555555", ha="left", va="center", clip_on=False)

# ---- points
mk = {"eat": "o", "c1": "D", "m0": "s", "pub": "^"}
col = {"eat": ORANGE, "c1": TEAL, "m0": TEAL, "pub": GREY}
for r in d.itertuples():
    c = col[r.fam]
    ax.plot(r.x, r.tpr_oracle, marker=mk[r.fam], ms=3.6 if r.fam != "eat" else 3.3, mfc=c if r.speech else "white",
            mec=c, mew=0.9, ls="none", zorder=3)

# ---- annotations (all numbers from the CSV)
from common import COHORT, ORDER
b0a = d[(d.paper == "B0") & (d.cohort == "audioset_sm")].iloc[0]
ax.annotate("", xy=(b0a.x, b0a.tpr_oracle + 1.5), xytext=(b0a.x, b0a.tpr_tauJamA - 0.3),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.8, mutation_scale=7, shrinkA=0, shrinkB=0), zorder=4)
ax.annotate(f"B0 · AudioSet-SM\n{b0a.tpr_tauJamA - b0a.tpr_oracle:.1f} pp lower", xy=(b0a.x * 0.93, b0a.tpr_oracle + 1.0), xytext=(0.22, 64),
            fontsize=7, color=ORANGE, ha="center", va="center", linespacing=1.0,
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.5, shrinkA=0, shrinkB=1, alpha=0.8), zorder=4)
c1 = d[(d.paper == "C1") & (d.cohort == "ava_sm5")]
if len(c1):   # C1 is not released with this repository; the label is drawn only when its rows are present
    c1 = c1.iloc[0]
    ax.annotate(f"C1 · AVA-SM: FPR {c1.fpr_tauJamA:.1f}%", xy=(c1.x, c1.tpr_oracle + 1.5), xytext=(XHI * 0.95, 96), fontsize=7, color=TEAL,
                ha="right", va="center", arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.5, shrinkA=0, shrinkB=1, alpha=0.8), zorder=4)
# regime labels (three, in empty areas)
calib = d[d.regime == "calib"]; disc = d[d.regime == "disc"]; cov = d[d.regime == "coverage"]
ax.text(0.024, 101.5, f"Calibration-dominated · AUROC {calib.auroc.min():.3f}–{calib.auroc.max():.3f}", fontsize=7.5, color=ORANGE,
        fontweight="bold", va="bottom", ha="left")
ax.text(XHI * 0.95, 52, f"Calibration +\ndiscrimination\noracle TPR {disc.tpr_oracle.min():.1f}–{disc.tpr_oracle.max():.1f}%",
        fontsize=7.5, color=TEAL, fontweight="bold", va="top", ha="right", linespacing=1.0)
ax.text(0.024, 36.5, f"Coverage failure\nTPR ≤ {cov.tpr_oracle.max():.1f}%", fontsize=7.5, color=GREY, fontweight="bold", va="bottom", ha="left", linespacing=1.0)
ax.text(2.0 * 1.08, 60, "α = 2%", fontsize=7, color=GREY, va="center", ha="left")

# ---- legend below the axes (shape = family, fill = cohort type)
leg = [Line2D([], [], marker="o", ls="none", mfc=ORANGE, mec=ORANGE, ms=3.6, label="B0, C0"),
       Line2D([], [], marker="D", ls="none", mfc=TEAL, mec=TEAL, ms=3.4, label="C1"),
       Line2D([], [], marker="s", ls="none", mfc=TEAL, mec=TEAL, ms=3.4, label="M0"),
       Line2D([], [], marker="^", ls="none", mfc=GREY, mec=GREY, ms=3.8, label="public"),
       Line2D([], [], marker="o", ls="none", mfc=INK, mec=INK, ms=3.6, label="filled: speech+music"),
       Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, ms=3.6, label="open: music only")]
ax.legend(handles=leg, fontsize=7, frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 0.10),
          handletextpad=0.2, columnspacing=0.7, labelspacing=0.15, borderaxespad=0.0)
save(fig, "fig3_disc_cal")
