"""Fig. 4: paired AUROC on AI-OpenBMAT falls as music sinks under speech.

Source: results/openbmat_window_auroc.csv (code/openbmat_window_auroc.py; window level;
a window is assigned to a class when >= 80% of it carries that class; human
and AI windows are paired by position in the same mix; cluster bootstrap over
source recordings, B = 10,000). Class levels follow the AI-OpenBMAT mixing
configuration: "music" and "fg-music" segments contain no speech; "similar",
"bg-music" and "bgvl-music" put music about 0, -5 and -10 dB relative to speech.
M0 is read from variants_m0_auroc.csv (code/agg_window_variants.py, same
definition). C1 is not released, so its line is absent.
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import TAB, COL_W_IN, EXCLUDE, style, save  # noqa: E402

style()
d = pd.read_csv(TAB / "openbmat_window_auroc.csv")
d = d.rename(columns={"ci_lo": "lo", "ci_hi": "hi"})
rename = {"B0": "B0", "B0'": "B0′", "B0+0": "B0+0", "C0": "C0", "C0+0": "C0+0", "SpecTTTra": "SpecTTTra"}
d["paper"] = d.name.map(rename)
# Optional extra detectors (window-level paired AUROC, same class rule).
for tag, paper, fn in [("M0", "M0", "variants_m0_auroc.csv")]:
    f = TAB / fn
    if f.exists():
        e = pd.read_csv(f)
        e = e[(e.unit == "window") & (e.model == tag)]
        e["paper"] = paper
        d = pd.concat([d, e[["cls", "auroc", "lo", "hi", "paper"]]], ignore_index=True)
        print("added", paper, "from", f.name)

classes = ["mus", "fg", "sim", "bg", "bgvl"]
xl = ["music", "fg music", "similar\n0 dB", "bg\n\u22125 dB", "bg very low\n\u221210 dB"]
styles = {"B0": ("o", "-", "black"), "B0′": ("o", ":", "0.35"), "B0+0": ("o", "--", "0.55"),
          "C0": ("s", "-", "black"), "C0+0": ("s", "--", "0.55"), "C1": ("D", "-", "0.25"),
          "M0": ("P", "-", "0.4"), "SpecTTTra": ("^", "-", "0.45")}
order = ["B0", "B0′", "B0+0", "C0", "C0+0", "C1", "M0", "SpecTTTra"]
fig, ax = plt.subplots(figsize=(COL_W_IN, 1.82))
present = [p for p in order if p in set(d.paper.dropna()) and p not in EXCLUDE]
for k, p in enumerate(present):
    s = d[d.paper == p].set_index("cls")
    xs = [i + (k - (len(present) - 1) / 2) * 0.05 for i in range(len(classes))]
    ys = [s.loc[c, "auroc"] for c in classes]
    lo = [s.loc[c, "auroc"] - s.loc[c, "lo"] for c in classes]
    hi = [s.loc[c, "hi"] - s.loc[c, "auroc"] for c in classes]
    mk, ls, col = styles[p]
    ax.errorbar(xs, ys, yerr=[lo, hi], marker=mk, ls=ls, color=col, ms=3.2, lw=0.9, elinewidth=0.6,
                capsize=0, mfc="white" if p == "SpecTTTra" else col, label=p)
nomus = d[(d.cls == "nomus")].auroc
ax.axhline(0.5, color="0.6", lw=0.7, ls=":")
ax.text(-0.25, 0.485, f"no-music windows (identical audio): {nomus.min():.2f}\u2013{nomus.max():.2f}",
        ha="left", va="top", fontsize=7, color="0.35")
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(xl)
ax.set_xlabel("no speech          music under speech (level vs. speech)", fontsize=7)
ax.set_ylabel("Paired window AUROC")
ax.set_ylim(0.40, 1.02)
ax.grid(axis="y", color="0.92", lw=0.5)
ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower left", handlelength=1.8, columnspacing=0.7,
          handletextpad=0.3, labelspacing=0.15, borderaxespad=0.2, bbox_to_anchor=(0.0, 0.16))
fig.tight_layout()
save(fig, "fig4_openbmat_auroc")
