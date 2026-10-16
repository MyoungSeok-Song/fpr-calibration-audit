"""Fig. 1 (v2): realized FPR at a nominal 2% for calibration source × decision unit.

Source: results/unit_prov.csv (two-stage cluster bootstrap, B = 10,000,
threshold re-estimated in every replicate). Colour = calibration source (orange FMA-dev,
teal Jamendo-A); fill = decision unit (open track-level, filled window-level); the filled
teal circle is the protocol. Every number written in the figure is read from the CSV.
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (TAB, NAME, EXCLUDE, ORDER, FAMILY, TEXT_W_IN, ORANGE, TEAL, GREY, GREY_L, INK,  # noqa: E402
                    style, save, fpr_axis, alpha_band)

style()
d = pd.read_csv(TAB / "unit_prov.csv")
missing = sorted({NAME.get(x, x) for x in d.det} - set(ORDER))
if missing:
    print("WARNING unmapped detectors:", missing)
d["paper"] = d.det.map(lambda x: NAME.get(x, x))
d = d.dropna(subset=["paper"])
d = d[~d.paper.isin(EXCLUDE)]
dets = [n for n in ORDER if n in set(d.paper)]

panels = [("jamendo_B", "(a) Jamendo-B", "web, unseen artists"),
          ("ava_sm5", "(b) AVA-SM", "film speech + music"),
          ("audioset_sm", "(c) AudioSet-SM", "speech + music")]
conds = [  # (source, unit, marker, facecolor, edgecolor, x-offset)
    ("fma_dev800", "file", "s", "white", ORANGE, -0.30),
    ("fma_dev800", "win", "s", ORANGE, ORANGE, -0.10),
    ("jamendo_A", "file", "o", "white", TEAL, 0.10),
    ("jamendo_A", "win", "o", TEAL, TEAL, 0.30),
]
FLOOR = 0.04   # zeros are drawn at the bottom edge and labelled "0" (well below the 0.1 tick)


def val(coh, src, unit, det):
    s = d[(d.cohort == coh) & (d.source == src) & (d.unit == unit) & (d.paper == det)]
    return None if s.empty else s.iloc[0]


fig, axes = plt.subplots(1, 3, figsize=(TEXT_W_IN, 2.1), sharey=True)
fig.subplots_adjust(left=0.068, right=0.962, top=0.715, bottom=0.255, wspace=0.12)
x = np.arange(len(dets))
for ax, (coh, title, sub) in zip(axes, panels):
    alpha_band(ax)
    fam = [FAMILY[n] for n in dets]
    for b in [i for i in range(1, len(dets)) if fam[i] != fam[i - 1]]:
        ax.axvline(b - 0.5, color="#DDDDDD", lw=0.6, zorder=0)
    for src, unit, mk, fc, ec, off in conds:
        for i, det in enumerate(dets):
            r = val(coh, src, unit, det)
            if r is None:
                continue
            y, lo, hi = max(r.rate, FLOOR), max(r.ci_lo, FLOOR), max(r.ci_hi, FLOOR)
            ax.plot([x[i] + off] * 2, [lo, hi], color=ec, lw=0.7, alpha=0.9, zorder=2, solid_capstyle="butt")
            ax.plot(x[i] + off, y, marker=mk, ms=3.5, mfc=fc, mec=ec, mew=0.8, ls="none", zorder=3)
    fpr_axis(ax, lo=FLOOR * 0.72, hi=110, label_floor=FLOOR)
    ax.set_xlim(-0.6, len(dets) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(dets, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=2)
    # title + two short take-away lines (all numbers from the CSV)
    fma_win = {det: val(coh, "fma_dev800", "win", det).rate for det in dets if val(coh, "fma_dev800", "win", det) is not None}
    jam_win = {det: val(coh, "jamendo_A", "win", det).rate for det in dets if val(coh, "jamendo_A", "win", det) is not None}
    worst = max(fma_win, key=fma_win.get)
    line1 = f"FMA-dev τ up to {fma_win[worst]:.1f}% ({worst})"
    if coh == "jamendo_B":   # same ranges as the manuscript: EAT families and C1, window-level τ from FMA-dev vs Jamendo-A
        ft = [k for k in ("B0", "B0′", "B0+0", "C0", "C0+0", "C1") if k in fma_win and k in jam_win]
        f_lo, f_hi = min(fma_win[k] for k in ft), max(fma_win[k] for k in ft)
        j_lo, j_hi = min(jam_win[k] for k in ft), max(jam_win[k] for k in ft)
        line2 = f"EAT & C1: {f_lo:.1f}–{f_hi:.1f}% vs {j_lo:.1f}–{j_hi:.1f}%"
    else:
        b0f, b0w = val(coh, "jamendo_A", "file", "B0").rate, val(coh, "jamendo_A", "win", "B0").rate
        line2 = f"track τ on windows: B0 {b0f:.1f}→{b0w:.1f}% ({b0f / b0w:.1f}×)"
    ax.text(0.0, 1.25, f"{title} · {sub}", transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=INK, va="bottom")
    ax.text(0.0, 1.14, line1, transform=ax.transAxes, fontsize=7, color="#444444", va="bottom")
    ax.text(0.0, 1.03, line2, transform=ax.transAxes, fontsize=7, color="#444444", va="bottom")
    if coh == "audioset_sm":   # α / 3α labels in the right margin, outside the last panel
        ax.text(len(dets) - 0.25, 2.0, "α", fontsize=7, color=GREY, va="center", ha="left", clip_on=False)
        ax.text(len(dets) - 0.25, 6.0, "3α", fontsize=7, color=GREY, va="center", ha="left", clip_on=False)
axes[0].set_ylabel("Realized FPR (%) at nominal α = 2%")
for ax in axes[1:]:
    ax.tick_params(axis="y", length=0)
leg = [Line2D([], [], marker="s", ls="none", mfc="white", mec=ORANGE, mew=0.9, ms=4, label="FMA-dev τ, track"),
       Line2D([], [], marker="s", ls="none", mfc=ORANGE, mec=ORANGE, ms=4, label="FMA-dev τ, window"),
       Line2D([], [], marker="o", ls="none", mfc="white", mec=TEAL, mew=0.9, ms=4, label="Jamendo-A τ, track"),
       Line2D([], [], marker="o", ls="none", mfc=TEAL, mec=TEAL, ms=4, label="Jamendo-A τ, window (protocol)")]
fig.legend(handles=leg, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.998),
           handletextpad=0.3, columnspacing=1.2, fontsize=7)
save(fig, "fig1_prov_unit")
