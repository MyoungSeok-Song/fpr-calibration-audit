"""Fig. 3: calibration error versus discrimination, against unseen generator families.

Source: results/unit_rank_unseen.csv (code/unit_provenance.py), reference "unseen4" (Lyria 3,
HeartMuLa, MiniMax 2.6, Mureka v9; generators weighted equally, songs 1/n_f;
cluster bootstrap B = 10,000 with tau re-estimated). For every detector and
human cohort c:
  x = FPR_c at the deployed window-level Jamendo-A threshold (calibration),
  y = oracle TPR = reference TPR at cohort c's own 2% threshold (discrimination).
The deployed TPR (reference TPR at the Jamendo-A threshold) is the oracle TPR on
the calibration cohort; its range per detector group is shaded. The vertical
gap between a point and its group's band is the discrimination loss.
ArtifactNet's and Fakeprint LR's training data are unknown, so "unseen" is not
guaranteed for them. C1 is not released, so its markers are absent.
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import TAB, NAME, EXCLUDE, COL_W_IN, style, save  # noqa: E402

style()
FLOOR = 0.02
src = TAB / "unit_rank_unseen.csv"
d = pd.read_csv(src)
d = d[d["ref"] == "unseen4"].copy()
missing = sorted(set(d.det) - set(NAME))
if missing:
    print("WARNING unmapped detectors:", missing)
d["paper"] = d.det.map(NAME)
d = d.dropna(subset=["paper"])
d = d[~d.paper.isin(EXCLUDE)]
SPEECH = {"ava_sm5", "audioset_sm", "openbmat_human_musicbearing"}
EAT = {"B0", "B0′", "B0+0", "C0", "C0+0"}
group = lambda p: "eat" if p in EAT else ({"C1": "c1", "M0": "m0"}.get(p, "pub"))
d["grp"] = d.paper.map(group)
mk = {"eat": "o", "c1": "D", "m0": "s", "pub": "^"}
col = {"eat": "black", "c1": "0.35", "m0": "0.2", "pub": "0.55"}

fig, ax = plt.subplots(figsize=(COL_W_IN, 1.98))
# deployed-TPR bands (TPR at the Jamendo-A threshold) per group
for members in [("eat", "c1", "m0"), ("pub",)]:
    s = d[d.grp.isin(members)]
    if s.empty:
        continue
    lo, hi = s.tpr_tauJamA.min(), s.tpr_tauJamA.max()
    ax.axhspan(lo, hi, color="0.88", zorder=0, lw=0)
for r in d.itertuples():
    sp = r.cohort in SPEECH
    ax.plot(max(r.fpr_tauJamA, FLOOR), r.tpr_oracle, marker=mk[r.grp], ms=3.4,
            mfc=col[r.grp] if sp else "white", mec=col[r.grp], mew=0.8, ls="none", zorder=3)
ax.axvline(2.0, color="0.3", lw=0.7, ls="--", zorder=1)
ax.set_xscale("log")
ax.set_xlim(FLOOR * 0.8, 20)
ax.set_xticks([FLOOR, 0.1, 1, 2, 10])
ax.set_xticklabels(["0", "0.1", "1", "2", "10"])
ax.set_ylim(-3, 103)
ax.set_xlabel("Realized FPR (%) at the Jamendo-A threshold")
ax.set_ylabel("Oracle TPR (%)")
ax.grid(color="0.93", lw=0.5, zorder=0)
ax.text(0.024, 89.0, "TPR at τ", fontsize=7, color="0.35", va="bottom")
ax.text(5.0, 5.5, "TPR at τ", fontsize=7, color="0.35", va="bottom")
leg = [Line2D([], [], marker="o", ls="none", mfc="black", mec="black", ms=3.6, label="fine-tuned (EAT)"),
       Line2D([], [], marker="s", ls="none", mfc="0.2", mec="0.2", ms=3.2, label="M0"),
       Line2D([], [], marker="^", ls="none", mfc="0.55", mec="0.55", ms=3.6, label="public"),
       Line2D([], [], marker="h", ls="none", mfc="0.3", mec="0.3", ms=3.6, label="speech+music cohort"),
       Line2D([], [], marker="h", ls="none", mfc="white", mec="0.3", ms=3.6, label="other cohort")]
ax.legend(handles=leg, fontsize=7, frameon=False, ncol=2, loc="center left", bbox_to_anchor=(0.2, 0.40),
          handletextpad=0.2, columnspacing=0.6, labelspacing=0.2, borderaxespad=0.1)
fig.tight_layout()
save(fig, "fig3_disc_cal")
