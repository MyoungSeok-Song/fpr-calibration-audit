"""Fig. 1: realized FPR at a nominal 2% for calibration source x decision unit.

Source: results/unit_prov.csv (code/unit_provenance.py; two-stage cluster
bootstrap, B = 10,000, threshold re-estimated in every replicate).
Track-level threshold = upper-20%-trimmed mean over all windows of a file
(Jamendo-A ~12 windows, FMA-dev 5); window-level threshold = one window.
Each threshold is applied at the unit it was calibrated on.
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import TAB, NAME, EXCLUDE, ORDER, FINETUNED, TEXT_W_IN, style, save  # noqa: E402

style()
d = pd.read_csv(TAB / "unit_prov.csv")
missing = sorted(set(d.det) - set(NAME))
if missing:
    print("WARNING unmapped detectors:", missing)
d["paper"] = d.det.map(NAME)
d = d.dropna(subset=["paper"])
d = d[~d.paper.isin(EXCLUDE)]
dets = [n for n in ORDER if n in set(d.paper)]

panels = [("jamendo_B", "(a) Jamendo-B: web, unseen artists"),
          ("ava_sm5", "(b) AVA-SM: film speech+music"),
          ("audioset_sm", "(c) AudioSet-SM: speech+music")]
conds = [  # (source, unit, label, marker, facecolor, edgecolor, offset)
    ("fma_dev800", "file", "FMA-dev, track-level τ", "s", "white", "0.45", -0.27),
    ("fma_dev800", "win", "FMA-dev, window-level τ", "s", "0.45", "0.45", -0.09),
    ("jamendo_A", "file", "Jamendo-A, track-level τ", "o", "white", "black", 0.09),
    ("jamendo_A", "win", "Jamendo-A, window-level τ (protocol)", "o", "black", "black", 0.27),
]
FLOOR = 0.07

fig, axes = plt.subplots(1, 3, figsize=(TEXT_W_IN, 1.92), sharey=True)
for ax, (coh, title) in zip(axes, panels):
    x = np.arange(len(dets))
    for src, unit, lab, mk, fc, ec, off in conds:
        s = d[(d.cohort == coh) & (d.source == src) & (d.unit == unit)].set_index("paper")
        for i, det in enumerate(dets):
            if det not in s.index:
                continue
            r = s.loc[det]
            y = max(r.rate, FLOOR)
            lo = max(r.ci_lo, FLOOR)
            hi = max(r.ci_hi, FLOOR)
            ax.plot([x[i] + off] * 2, [lo, hi], color=ec, lw=0.7, zorder=1)
            ax.plot(x[i] + off, y, marker=mk, ms=3.6, mfc=fc, mec=ec, mew=0.8, ls="none", zorder=2,
                    label=lab if i == 0 else None)
    ax.axhline(2.0, color="0.3", lw=0.7, ls="--", zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.8, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(dets, rotation=50, ha="right")
    ax.set_title(title.replace("\n", " "), loc="left", pad=3)
    # separate fine-tuned and public detectors
    nft = sum(1 for n in dets if n in FINETUNED)
    ax.axvline(nft - 0.5, color="0.75", lw=0.6)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["≤0.1", "1", "10", "100"])
    ax.grid(axis="y", color="0.9", lw=0.5, zorder=0)
axes[0].set_ylabel("Realized FPR (%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0),
           handletextpad=0.2, columnspacing=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=0.5)
save(fig, "fig1_prov_unit")
