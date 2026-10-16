"""Fig. 2: decision-unit mismatch. Calibration unit k_cal x deployment unit k_dep.

Source: results/unit_matrix.csv (code/unit_provenance.py; Jamendo-A calibration, alpha = 2%,
evaluated on Jamendo-B; cluster bootstrap B = 10,000 with tau re-estimation).
A k-window unit is the upper-20%-trimmed mean of k consecutive windows.
ArtifactNet scores at most 7 windows, so its largest unit is k = 7.
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import TAB, NAME, EXCLUDE, ORDER, FINETUNED, COL_W_IN, style, save  # noqa: E402

style()
d = pd.read_csv(TAB / "unit_matrix.csv")
missing = sorted(set(d.det) - set(NAME))
if missing:
    print("WARNING unmapped detectors:", missing)
d["paper"] = d.det.map(NAME)
d = d[(d.cohort == "jamendo_B")].dropna(subset=["paper"])
d = d[~d.paper.isin(EXCLUDE)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W_IN, 2.05), gridspec_kw={"width_ratios": [1.0, 1.15]})

# (a) B0 matrix
b0 = d[d.paper == "B0"]
kc = sorted(b0.k_cal.unique())
kd = sorted(b0.k_dep.unique())
M = np.full((len(kc), len(kd)), np.nan)
for r in b0.itertuples():
    M[kc.index(r.k_cal), kd.index(r.k_dep)] = r.rate
im = a1.imshow(M, cmap="Greys", vmin=0, vmax=16, aspect="auto")
for i in range(len(kc)):
    for j in range(len(kd)):
        v = M[i, j]
        diag = kc[i] == kd[j]
        a1.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                color="white" if v > 8 else "black", fontweight="bold" if diag else "normal")
a1.set_xticks(range(len(kd)))
a1.set_xticklabels([str(k) for k in kd])
a1.set_yticks(range(len(kc)))
a1.set_yticklabels([str(k) for k in kc])
a1.set_xlabel("deployment unit $k_{dep}$")
a1.set_ylabel("calibration unit $k_{cal}$")
a1.set_title("(a) B0, FPR (%)", loc="left", pad=3)
for sp in a1.spines.values():
    sp.set_visible(True)

# (b) all detectors, single-window deployment
styles = {  # grayscale-safe: marker + line style
    "B0": ("o", "-", "black"), "B0′": ("o", ":", "0.35"), "B0+0": ("o", "--", "0.55"),
    "C0": ("s", "-", "black"), "C0+0": ("s", "--", "0.55"), "C1": ("D", "-", "0.25"), "M0": ("P", "-", "0.4"),
    "SpecTTTra": ("^", "-", "0.45"), "ArtifactNet": ("v", "-", "0.6"), "Fakeprint LR": ("x", "-", "0.3"),
}
for det in [n for n in ORDER if n in set(d.paper)]:
    s = d[(d.paper == det) & (d.k_dep == 1)].sort_values("k_cal")
    mk, ls, c = styles.get(det, ("o", "-", "0.5"))
    a2.plot(s.k_cal, s.rate, marker=mk, ls=ls, color=c, ms=3.2, lw=0.9, mfc="white" if det not in FINETUNED else c,
            label=det)
a2.axhline(2.0, color="0.3", lw=0.7, ls="--")
a2.set_xscale("log", base=2)
a2.set_xticks([1, 2, 4, 8, 12])
a2.set_xticklabels(["1", "2", "4", "8", "12"])
a2.set_xlabel("calibration unit $k_{cal}$")
a2.set_ylabel("FPR (%), $k_{dep}=1$")
a2.set_ylim(0, 16)
a2.set_title("(b) single-window use", loc="left", pad=3)
h, l = a2.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=7,
           handlelength=1.8, columnspacing=0.8, handletextpad=0.3, labelspacing=0.2)
fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.4)
save(fig, "fig2_unit_matrix")
