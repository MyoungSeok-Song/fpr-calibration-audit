"""Fig. 2 (v2): decision-unit mismatch — calibration unit k_cal × deployment unit k_dep.

Source: results/unit_matrix.csv (Jamendo-A calibration, α = 2%, evaluated on Jamendo-B;
cluster bootstrap B = 10,000 with τ re-estimated). A k-window unit is the upper-20%-trimmed mean of k
consecutive windows. ArtifactNet scores at most 7 windows, so its largest unit is k = 7.
(a) B0 heat-map: diverging color map centred on the nominal 2% (log scale; teal below, orange above),
    values printed in every cell, matched units (diagonal) outlined.
(b) every detector's single-window FPR as k_cal grows (log axes, dense ticks, labels at the line ends).
"""
import pathlib
import sys

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (TAB, NAME, EXCLUDE, ORDER, FAMILY, COL_W_IN, ORANGE, TEAL, GREY, GREY_L, INK,  # noqa: E402
                    style, save, fpr_axis, alpha_band)

style()
d = pd.read_csv(TAB / "unit_matrix.csv")
missing = sorted({NAME.get(x, x) for x in d.det} - set(ORDER))
if missing:
    print("WARNING unmapped detectors:", missing)
d["paper"] = d.det.map(lambda x: NAME.get(x, x))
d = d[(d.cohort == "jamendo_B")].dropna(subset=["paper"])
d = d[~d.paper.isin(EXCLUDE)]

fig = plt.figure(figsize=(COL_W_IN, 2.05))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], left=0.115, right=0.775, top=0.90, bottom=0.215, wspace=0.42)
a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# ---------------------------------------------------------------- (a) B0 matrix
b0 = d[d.paper == "B0"]
kc = sorted(b0.k_cal.unique())
kd = sorted(b0.k_dep.unique())
M = np.full((len(kc), len(kd)), np.nan)
for r in b0.itertuples():
    M[kc.index(r.k_cal), kd.index(r.k_dep)] = r.rate
# lighter teal end: the below-2% range (0.4–2) is narrow, and a light end keeps the grey-scale ordering monotone
cmap = mc.LinearSegmentedColormap.from_list("teal_white_orange", ["#6FB0B8", "#FFFFFF", ORANGE])
vmin, vcenter, vmax = np.log10(0.35), np.log10(2.0), np.log10(16.0)
norm = mc.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
a1.imshow(np.log10(M), cmap=cmap, norm=norm, aspect="auto")
for i in range(len(kc)):
    for j in range(len(kd)):
        v = M[i, j]
        dark = (np.log10(v) - vcenter) > 0.55
        a1.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if dark else INK,
                fontweight="bold" if kc[i] == kd[j] else "normal")
        if kc[i] == kd[j]:
            a1.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.3, zorder=3))
a1.set_xticks(range(len(kd)))
a1.set_xticklabels([str(k) for k in kd])
a1.set_yticks(range(len(kc)))
a1.set_yticklabels([str(k) for k in kc])
a1.tick_params(length=0, pad=2)
a1.set_xlabel("deployment unit $k_{dep}$", labelpad=2)
a1.set_ylabel("calibration unit $k_{cal}$ (windows)", labelpad=2)
for sp in a1.spines.values():
    sp.set_visible(True); sp.set_color("#444444"); sp.set_linewidth(0.6)
m_match = M[kc.index(1), kd.index(1)]; m_mis = M[kc.index(12), kd.index(1)]; m_rev = M[kc.index(1), kd.index(12)]
a1.text(0.0, 1.03, "(a) B0, Jamendo-B", transform=a1.transAxes, fontsize=7.5, fontweight="bold", color=INK, va="bottom")

# ---------------------------------------------------------------- (b) all detectors, single-window deployment
alpha_band(a2)
mk = {"B0": "o", "C0": "s", "C1": "D", "M0": "P", "pub": "^"}
dets = [n for n in ORDER if n in set(d.paper)]
ends = []
for det in dets:
    s = d[(d.paper == det) & (d.k_dep == 1)].sort_values("k_cal")
    fam = FAMILY[det]
    is_b0 = det == "B0"
    col = TEAL if is_b0 else ("#777777" if fam in ("B0", "C0") else ("#3A3A3A" if fam in ("C1", "M0") else "#9A9A9A"))
    a2.plot(s.k_cal, s.rate, marker=mk[fam], ls="-", color=col, ms=3.0, lw=1.3 if is_b0 else 0.8,
            mfc=col if fam != "pub" else "white", mew=0.8, zorder=4 if is_b0 else 3, clip_on=False)
    ends.append((det, float(s.k_cal.iloc[-1]), float(s.rate.iloc[-1]), col, is_b0))
YLO, YHI = 1.3, 22
fpr_axis(a2, lo=YLO, hi=YHI, ticks=[1, 2, 5, 10, 20])
a2.set_xscale("log", base=2)
a2.set_xlim(0.9, 13.5)
a2.set_xticks([1, 2, 4, 8, 12])
a2.set_xticklabels(["1", "2", "4", "8", "12"])
a2.xaxis.set_minor_locator(plt.NullLocator())
a2.set_xlabel("calibration unit $k_{cal}$", labelpad=2)
a2.set_ylabel("single-window FPR (%)", labelpad=2)
k1 = d[(d.k_dep == 1) & (d.k_cal == 1)].rate
kmax = d[(d.k_dep == 1)].sort_values("k_cal").groupby("paper").tail(1).rate
a2.text(0.0, 1.03, "(b) all, $k_{dep}$ = 1", transform=a2.transAxes, fontsize=7.5, fontweight="bold", color=INK, va="bottom")
a2.text(2.9, 1.93, "α = 2%", fontsize=7, color=GREY, va="top", ha="left")
# end labels spread over the axis height (log space) with leader lines
ends.sort(key=lambda t: t[2])
ylog = [np.log10(e[2]) for e in ends]
lo_l, hi_l = np.log10(YLO) + 0.03, np.log10(YHI) - 0.03
n = len(ends); min_gap = min(0.095, (hi_l - lo_l) / max(n - 1, 1))
pos = ylog[:]
pos[0] = max(pos[0], lo_l)
for i in range(1, n):
    pos[i] = max(pos[i], pos[i - 1] + min_gap)
if pos[-1] > hi_l:
    shift = pos[-1] - hi_l; pos = [p - shift for p in pos]
    for i in range(n - 2, -1, -1):
        pos[i] = min(pos[i], pos[i + 1] - min_gap)
for (det, kx, y, col, is_b0), p in zip(ends, pos):
    a2.plot([kx, 14.0, 14.9], [y, 10 ** p, 10 ** p], color=col, lw=0.5, alpha=0.9, clip_on=False, zorder=2)
    a2.text(15.3, 10 ** p, det, fontsize=7, color=TEAL if is_b0 else "#333333", va="center", ha="left",
            fontweight="bold" if is_b0 else "normal", clip_on=False)
save(fig, "fig2_unit_matrix")
