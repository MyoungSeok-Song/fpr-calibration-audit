"""Derived numbers quoted in the paper, each with its source table.

Run:  python code/figs/derived_numbers.py   (FIG_SRC=out to use regenerated tables)
Writes out/derived_numbers.md. Ratios, ranges and means below are arithmetic on
values from the result tables; nothing is re-estimated here. C1 is not released,
so ranges that include C1 in the paper are computed here without it.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import TAB, NAME, ORDER, BB_NAME, ROOT, OUT  # noqa: E402

OUTMD = OUT.parent / "derived_numbers.md"
lines = []


def out(s=""):
    lines.append(s)
    print(s)


def fmt(x, d=1):
    return "nan" if pd.isna(x) else f"{x:.{d}f}"


def ordered(dets):
    return sorted(dets, key=lambda n: ORDER.index(n) if n in ORDER else 99)


# ---------------------------------------------------------------- 1. provenance x unit
out("# Derived numbers for the paper (auto-generated)")
out()
out("## 1. Calibration source x decision unit (unit_prov.csv), alpha = 2%")
prov = pd.read_csv(TAB / "unit_prov.csv")
unknown = sorted(set(prov.det) - set(NAME))
if unknown:
    out(f"WARNING: unmapped detector ids in unit_prov.csv: {unknown}")
prov["paper"] = prov.det.map(NAME)
for coh in ["jamendo_B", "ava_sm5", "audioset_sm", "openbmat_human_musicbearing"]:
    out(f"### {coh}")
    out("| detector | FMA file | FMA win | JamA file | JamA win | FMA/JamA (win) | IF FMA win | IF JamA win |")
    out("|---|---|---|---|---|---|---|---|")
    for det in ordered(prov.paper.dropna().unique()):
        s = prov[(prov.paper == det) & (prov.cohort == coh)]
        if s.empty:
            continue

        def g(src, unit):
            r = s[(s.source == src) & (s.unit == unit)]
            return r.iloc[0] if len(r) else None
        a, b, c, d = g("fma_dev800", "file"), g("fma_dev800", "win"), g("jamendo_A", "file"), g("jamendo_A", "win")
        cell = lambda r: f"{fmt(r.rate)} [{fmt(r.ci_lo)}-{fmt(r.ci_hi)}]" if r is not None else "-"
        ratio = b.rate / d.rate if (b is not None and d is not None and d.rate > 0) else np.nan
        out(f"| {det} | {cell(a)} | {cell(b)} | {cell(c)} | {cell(d)} | {fmt(ratio)} | "
            f"{fmt(b.rate / 2 if b is not None else np.nan)} | {fmt(d.rate / 2 if d is not None else np.nan)} |")
    out()

ft = prov[(prov.cohort == "jamendo_B") & (prov.unit == "win") & (prov.source == "fma_dev800")]
ftj = prov[(prov.cohort == "jamendo_B") & (prov.unit == "win") & (prov.source == "jamendo_A")]
out("Ranges, Jamendo-B, window unit:")
for grp, members in [("fine-tuned", ["B0", "B0′", "B0+0", "C0", "C0+0", "C1", "M0"]),
                     ("public", ["SpecTTTra", "ArtifactNet", "Fakeprint LR"])]:
    f1 = ft[ft.paper.isin(members)].rate
    f2 = ftj[ftj.paper.isin(members)].rate
    if len(f1):
        out(f"- {grp}: FMA-dev tau {fmt(f1.min())}-{fmt(f1.max())}% (IF {fmt(f1.min()/2)}-{fmt(f1.max()/2)}); "
            f"Jamendo-A tau {fmt(f2.min())}-{fmt(f2.max())}% (IF {fmt(f2.min()/2,2)}-{fmt(f2.max()/2,2)})")
out()

# ---------------------------------------------------------------- 2. unit matrix
out("## 2. Decision-unit matrix (unit_matrix.csv), Jamendo-B, alpha = 2%")
mat = pd.read_csv(TAB / "unit_matrix.csv")
mat["paper"] = mat.det.map(NAME)
out("| detector | matched k=1 | kcal=max -> kdep=1 | ratio to matched | ratio to nominal | kcal=1 -> kdep=max | diag (all k) |")
out("|---|---|---|---|---|---|---|")
for det in ordered(mat.paper.dropna().unique()):
    s = mat[(mat.paper == det) & (mat.cohort == "jamendo_B")]
    kmax = s.k_cal.max()
    kdmax = s.k_dep.max()
    m11 = s[(s.k_cal == 1) & (s.k_dep == 1)].iloc[0]
    mk1 = s[(s.k_cal == kmax) & (s.k_dep == 1)].iloc[0]
    m1k = s[(s.k_cal == 1) & (s.k_dep == kdmax)].iloc[0]
    diag = s[s.k_cal == s.k_dep].sort_values("k_cal")
    out(f"| {det} | {fmt(m11.rate)} [{fmt(m11.ci_lo)}-{fmt(m11.ci_hi)}] | k{kmax}->1: {fmt(mk1.rate)} [{fmt(mk1.ci_lo)}-{fmt(mk1.ci_hi)}] | "
        f"{fmt(mk1.rate / m11.rate)} | {fmt(mk1.rate / 2)} | 1->k{kdmax}: {fmt(m1k.rate)} [{fmt(m1k.ci_lo)}-{fmt(m1k.ci_hi)}] | "
        + ", ".join(f"k{int(r.k_cal)}={fmt(r.rate)}" for r in diag.itertuples()) + " |")
out()
for coh in ["musdb18", "gtzan", "openbmat_human_musicbearing"]:
    s = mat[(mat.paper == "B0") & (mat.cohort == coh)]
    m11 = s[(s.k_cal == 1) & (s.k_dep == 1)].iloc[0]
    m121 = s[(s.k_cal == 12) & (s.k_dep == 1)].iloc[0]
    out(f"- B0 {coh}: matched k1 {fmt(m11.rate)}, k12->1 {fmt(m121.rate)} [{fmt(m121.ci_lo)}-{fmt(m121.ci_hi)}]")
out()

# ---------------------------------------------------------------- 3. padding (public_window_rates)
out("## 3. Padded vs unpadded inputs (public_window_rates.csv), tau_JamA@2%, window unit")
pwr = pd.read_csv(TAB / "public_window_rates.csv")
pwr2 = pwr[pwr.thr == "tau_JamA@2%"]
for model in ["B0", "C0", "SpecTTTra"]:
    s = pwr2[pwr2.model == model]
    for base in ["ava_cs", "librispeech_tc", "ava_sm5", "esc50", "null_white", "null_dsil", "null_pink"]:
        r = {k: s[s.cohort == (base if k == "all" else f"{base}[{k}]")] for k in ["all", "long", "short"]}
        cells = []
        for k, v in r.items():
            if len(v):
                v = v.iloc[0]
                cells.append(f"{k} {fmt(v.rate)} [{fmt(v.ci_lo)}-{fmt(v.ci_hi)}] n={int(v.n)} G={int(v.clusters)}")
        if cells:
            out(f"- {NAME.get(model, model)} {base}: " + "; ".join(cells))
out()

# ---------------------------------------------------------------- 4. excess ratios over Jamendo-B
out("## 4. In-the-wild excess over Jamendo-B (public_window_g0.csv), tau_JamA@2%")
g0 = pd.read_csv(TAB / "public_window_g0.csv")
g0 = g0[(g0.thr == "tau_JamA@2%") & (g0.role == "real") & (~g0.cohort.str.contains(r"\[")) &
        (g0.n >= 30) & (g0.clusters >= 10)]
for model in g0.model.unique():
    s = g0[g0.model == model].sort_values("ratio", ascending=False)
    top = s.iloc[0]
    out(f"- {NAME.get(model, model)}: max ratio {top.cohort} {fmt(top.fpr)} "
        f"ratio {fmt(top.ratio,2)} [{fmt(top.ratio_lo,2)}-{fmt(top.ratio_hi,2)}]; pass strict/orig/ci_sep any: "
        f"{bool(s.pass_strict.any())}/{bool(s.pass_orig.any())}/{bool(s.pass_ci_sep.any())}")
    for coh in ["audioset_sm", "ava_sm5", "openbmat_human_musicbearing"]:
        r = s[s.cohort == coh]
        if len(r):
            r = r.iloc[0]
            out(f"    {coh}: ratio {fmt(r.ratio,2)} [{fmt(r.ratio_lo,2)}-{fmt(r.ratio_hi,2)}]")
out()

# ---------------------------------------------------------------- 5. head contrasts
out("## 5. Head contrasts (heads_contrast_all.csv)")
con = pd.read_csv(TAB / "heads_contrast_all.csv")
cols = ["bb", "base", "method", "kind", "worst_base", "worst_method", "worst_base_cohort", "worst_method_cohort",
        "d_worst", "d_worst_lo", "d_worst_hi", "p_worst", "oot_base", "oot_method", "d_oot", "d_oot_lo", "d_oot_hi"]
extra = [c for c in con.columns if c.startswith("holm") or c.startswith("p_holm") or c == "macro_rel" or c.startswith("macr")]
for r in con.itertuples():
    rr = r._asdict()
    out(f"- {BB_NAME.get(r.bb, r.bb)} {r.base}->{r.method} ({r.kind}): worst {fmt(r.worst_base)} ({r.worst_base_cohort}) -> "
        f"{fmt(r.worst_method)} ({r.worst_method_cohort}); d {fmt(r.d_worst)} [{fmt(r.d_worst_lo)}, {fmt(r.d_worst_hi)}] p={fmt(r.p_worst,3)}; "
        f"OOT {fmt(r.oot_base)} -> {fmt(r.oot_method)} d {fmt(r.d_oot)} [{fmt(r.d_oot_lo)}, {fmt(r.d_oot_hi)}]; "
        + "; ".join(f"{c}={rr.get(c)}" for c in extra))
out()

# ---------------------------------------------------------------- 6. evidence types
out("## 6. Evidence types of false-alarm windows, four real-world cohorts (heads_fa_types.csv)")
fa = pd.read_csv(TAB / "heads_fa_types.csv")
rw = ["openbmat_human_musicbearing", "musdb18", "gtzan", "ava_sm5"]
s = fa[fa.cohort.isin(rw)].groupby(["bb", "head"])[["n_fp_windows", "el_5", "ep_5", "mx_5"]].sum()
for (bb, head), r in s.iterrows():
    n = r.n_fp_windows
    if head in ("orig", "free_split", "pos_split") and n > 0:
        out(f"- {BB_NAME.get(bb, bb)} {head}: n={int(n)} evidence-loss {100*r.el_5/n:.0f}% / E+ {100*r.ep_5/n:.0f}% / mixed {100*r.mx_5/n:.0f}%")
out()

# ---------------------------------------------------------------- 7. null inputs and sample-rate provenance
out("## 7. Null inputs (heads_rates.csv, alpha 0.02, subset all)")
wr = pd.read_csv(TAB / "heads_rates.csv")
wr2 = wr[(wr.alpha == 0.02) & (wr.subset == "all")]
nulls = ["ava_cs_long", "ava_amb_long", "librispeech_tc", "null_white", "null_dsil", "null_pink",
         "null_white10", "null_dither10", "null_white30", "null_dither30", "esc50"]
for bb in ["B0", "C0", "EAT (frozen)", "EAT-AS2M (frozen)", "MERT (frozen)", "CLAP (frozen)"]:
    for head in ["orig", "free_split", "pos_split"]:
        s2 = wr2[(wr2.bb == bb) & (wr2["head"] == head)]
        if s2.empty:
            continue
        cells = []
        for c in nulls:
            v = s2[s2.cohort == c]
            if len(v):
                cells.append(f"{c}={fmt(v.iloc[0].rate)}")
        out(f"- {BB_NAME.get(bb, bb)} {head}: " + ", ".join(cells))
nl = pd.read_csv(TAB / "heads_null.csv")
out("heads_null.csv (unpadded long nulls):")
for r in nl[nl["head"].isin(["orig", "free_split", "pos_split"])].itertuples():
    out(f"  {BB_NAME.get(r.bb, r.bb)} {r.head} {r.cohort}: fpr={fmt(r.fpr)} margin_sd={fmt(r.margin_sd,2)}")
out()

# ---------------------------------------------------------------- 8. provenance in SD units
out("## 8. Threshold shift in SD units (heads_provenance.csv), alpha 2%")
wp = pd.read_csv(TAB / "heads_provenance.csv")
for r in wp[wp["head"].isin(["orig", "free_split", "pos_split"])].itertuples():
    if r.calib == "jamendo_A":
        continue
    out(f"- {BB_NAME.get(r.bb, r.bb)} {r.head} {r.calib}: dT={fmt(r.dT,2)} ({fmt(r.dT_sd,2)} SD); Jamendo-B {fmt(r.jamendo_B)} "
        f"[{fmt(r.jamendo_B_lo)}-{fmt(r.jamendo_B_hi)}]; rw_worst {fmt(r.rw_worst)} ({r.rw_worst_cohort})")
probe = wp[(wp.bb.isin(["EAT (frozen)", "EAT-AS2M (frozen)", "MERT (frozen)", "CLAP (frozen)"])) & (wp["head"] == "free_split")]
pv = probe.pivot_table(index="bb", columns="calib", values="dT_sd")
pv["ov_minus_test"] = (pv["fma_ov_probe800"] - pv["fma_test_disjoint800"]).abs()
out("Frozen probes, UC heads (dT_sd): ")
out(pv.round(2).to_string())
out(f"|ov - test| range: {pv.ov_minus_test.min():.2f}-{pv.ov_minus_test.max():.2f} SD; "
    f"FMA (ov,test) below Jamendo-A by {(-pv[['fma_ov_probe800','fma_test_disjoint800']]).min().min():.2f}-"
    f"{(-pv[['fma_ov_probe800','fma_test_disjoint800']]).max().max():.2f} SD")
pj = probe.pivot_table(index="bb", columns="calib", values="jamendo_B")
out("Frozen probes, UC heads, Jamendo-B FPR by calibration source:")
out(pj.round(1).to_string())
out()

# ---------------------------------------------------------------- 9b. unseen-family reference
out("## 9. Discrimination vs calibration against unseen families (unit_rank_unseen.csv)")
ru = pd.read_csv(TAB / "unit_rank_unseen.csv")
ru["paper"] = ru.det.map(NAME)
for ref in ["unseen4"]:
    s = ru[ru["ref"] == ref]
    out(f"### reference {ref}")
    out("| detector | AUROC range | FPR@JamA range | TPR@JamA | oracle TPR range | max calib. error (pp) | max discr. loss (pp) |")
    out("|---|---|---|---|---|---|---|")
    for det in ordered(s.paper.dropna().unique()):
        t = s[s.paper == det]
        ce = t.fpr_tauJamA - 2.0
        dl = t.tpr_tauJamA - t.tpr_oracle
        out(f"| {det} | {t.auroc.min():.3f}-{t.auroc.max():.3f} | {fmt(t.fpr_tauJamA.min())}-{fmt(t.fpr_tauJamA.max())} | "
            f"{fmt(t.tpr_tauJamA.iloc[0])} | {fmt(t.tpr_oracle.min())}-{fmt(t.tpr_oracle.max())} | "
            f"{fmt(ce.max())} ({t.cohort.iloc[ce.values.argmax()]}) | {fmt(dl.max())} ({t.cohort.iloc[dl.values.argmax()]}) |")
    sp = s[s.cohort.isin(["ava_sm5", "audioset_sm", "openbmat_human_musicbearing"])]
    ot = s[~s.cohort.isin(["ava_sm5", "audioset_sm", "openbmat_human_musicbearing"])]
    for det in ["C1", "B0"]:
        a, b = sp[sp.paper == det], ot[ot.paper == det]
        if len(a):
            out(f"- {det}: speech+music AUROC {a.auroc.min():.3f}-{a.auroc.max():.3f}, oracle {fmt(a.tpr_oracle.min())}-{fmt(a.tpr_oracle.max())}; "
                f"other AUROC {b.auroc.min():.3f}-{b.auroc.max():.3f}, oracle {fmt(b.tpr_oracle.min())}-{fmt(b.tpr_oracle.max())}")
    eat = s[s.paper.isin(["B0", "B0\u2032", "B0+0", "C0", "C0+0"])]
    out(f"- EAT family pooled: AUROC {eat.auroc.min():.3f}-{eat.auroc.max():.3f}; FPR {fmt(eat.fpr_tauJamA.min())}-{fmt(eat.fpr_tauJamA.max())}")
    pub = s[s.paper.isin(["SpecTTTra", "ArtifactNet", "Fakeprint LR"])]
    out(f"- public pooled: AUROC {pub.auroc.min():.3f}-{pub.auroc.max():.3f}; TPR@JamA {fmt(pub.tpr_tauJamA.min())}-{fmt(pub.tpr_tauJamA.max())}")
out()
# ---------------------------------------------------------------- 10. AI-OpenBMAT classes
out("## 10. Paired window AUROC by AI-OpenBMAT class (openbmat_window_auroc.csv)")
ob = pd.read_csv(TAB / "openbmat_window_auroc.csv")
piv = ob.pivot_table(index="name", columns="cls", values="auroc")
out(piv.round(3).to_string())
nper = ob.pivot_table(index="name", columns="cls", values="n_chunks_human")
out(nper.to_string())
out()

# ---------------------------------------------------------------- 11. generators and exposure
out("## 11. Generator TPR (unit_gen.csv, tau_JamA, k = 1) and exposure (unit_exposure.csv)")
gen = pd.read_csv(TAB / "unit_gen.csv")
gen["paper"] = gen.det.map(NAME)
newfam = ["lyria3", "heartmula", "minimax_2.6", "mureka_v9"]
out("| detector | " + " | ".join(newfam) + " | mean new-family | riffusion | udio_hf | suno_ext | sonics all |")
out("|---|" + "---|" * (len(newfam) + 5))
for det in ordered(gen.paper.dropna().unique()):
    s = gen[gen.paper == det]
    vals = [s[s.generator == g].tpr.iloc[0] if (s.generator == g).any() else np.nan for g in newfam]
    get = lambda coh, g="all": (s[(s.cohort == coh) & (s.generator == g)].tpr.iloc[0]
                               if ((s.cohort == coh) & (s.generator == g)).any() else np.nan)
    out(f"| {det} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(np.nanmean(vals))} | "
        f"{fmt(get('aime_sa', 'riffusion_aime'))} | {fmt(get('udio_hf_dev'))} | {fmt(get('suno_ext_dev'))} | {fmt(get('sonics_fake'))} |")
nf = pwr2[(pwr2.model == "ArtifactNet (native rate)")]
nat = {g: nf[nf.cohort.str.endswith("::" + g)].rate.iloc[0] for g in newfam}
out(f"ArtifactNet native band (public_window_rates.csv): " + ", ".join(f"{k} {fmt(v)}" for k, v in nat.items())
    + f"; mean {fmt(np.mean(list(nat.values())))}; riffusion {fmt(nf[nf.cohort == 'aime_sa::riffusion_aime'].rate.iloc[0])}; "
    f"sonics all {fmt(nf[nf.cohort == 'sonics_fake'].rate.iloc[0])}")
ns = gen[(gen.det == "B0")].set_index("generator").n_files
out("n files: " + ", ".join(f"{g}={int(ns[g])}" for g in newfam + ["riffusion_aime"] if g in ns.index))
out()


# ---------------------------------------------------------------- 13. M0 from the head-analysis pipeline
out("## 13. M0 (fine-tuned MERT-330M), head-analysis pipeline (heads_*.csv, family M0)")
m0p = TAB / "heads_rates.csv"
if m0p.exists():
    r = pd.read_csv(m0p)
    r = r[r.bb == "M0"]
    o = r[(r["head"] == "orig") & (r.alpha == 0.02) & (r.subset == "all")].set_index("cohort")
    onp = r[(r["head"] == "orig") & (r.alpha == 0.02) & (r.subset == "nonpad")].set_index("cohort")
    jb = o.loc["jamendo_B", "rate"]
    for c in ["jamendo_B", "openbmat_human_musicbearing", "ava_sm5", "audioset_sm", "musdb18", "gtzan", "esc50", "librispeech_tc",
              "openbmat_ai_musicbearing", "sonics_fake", "music8k", "aime_sa"]:
        if c in o.index:
            extra = f"; ratio to Jamendo-B {o.loc[c, 'rate'] / jb:.2f} (point)" if c in ("openbmat_human_musicbearing", "ava_sm5", "audioset_sm", "musdb18", "gtzan") else ""
            npv = f"; unpadded {onp.loc[c, 'rate']:.2f}" if c in onp.index else ""
            out(f"- M0 orig {c}: {o.loc[c, 'rate']:.2f} [{o.loc[c, 'ci_lo']:.2f}-{o.loc[c, 'ci_hi']:.2f}]{extra}{npv}")
    pv = pd.read_csv(TAB / "heads_provenance.csv")
    pv = pv[pv.bb == "M0"]
    for rr in pv[pv["head"].isin(["orig", "free_split", "pos_split"])].itertuples():
        out(f"- M0 {rr.head} calib {rr.calib}: dT_sd {rr.dT_sd:.2f}; Jamendo-B {rr.jamendo_B:.2f} [{rr.jamendo_B_lo:.2f}-{rr.jamendo_B_hi:.2f}]")
    ca = pd.read_csv(TAB / "heads_contrast_all.csv")
    out("- Holm (confirmatory PE-UC; nine families incl. C1 in results/, families present in out/): " + ", ".join(
        f"{BB_NAME.get(x.bb, x.bb)} {x.p_holm:.3f}" for x in ca[ca.kind == "confirmatory"].itertuples() if pd.notna(x.p_holm)))
    c = ca[ca.bb == "M0"]
    for x in c.itertuples():
        out(f"- M0 {x.base}->{x.method}: worst {x.worst_base:.2f} -> {x.worst_method:.2f}; d {x.d_worst:.2f} [{x.d_worst_lo:.2f}, {x.d_worst_hi:.2f}]; "
            f"OOT {x.oot_base:.2f} -> {x.oot_method:.2f}; d {x.d_oot:.2f} [{x.d_oot_lo:.2f}, {x.d_oot_hi:.2f}]")
    fa = pd.read_csv(TAB / "heads_fa_types.csv")
    fa = fa[fa.bb == "M0"]
    t = fa[fa.cohort.isin(rw)].groupby("head")[["n_fp_windows", "el_5", "ep_5", "mx_5"]].sum()
    for h in ["free_split", "pos_split"]:
        n = t.loc[h, "n_fp_windows"]
        out(f"- M0 {h}: n={int(n)} evidence-loss {100*t.loc[h,'el_5']/n:.0f}% / E+ {100*t.loc[h,'ep_5']/n:.0f}% / mixed {100*t.loc[h,'mx_5']/n:.0f}%")
out()
# ---------------------------------------------------------------- 14. pre-specified replication (Section 5.6)
xs = TAB / "extra_sonics_variants.csv"
if xs.exists():
    out("## 14. Pre-specified replication with SpecTTTra-alpha and -beta (extra_sonics_variants.csv)")
    x = pd.read_csv(xs)
    for det in ["SpecTTTra", "SpecTTTra-alpha", "SpecTTTra-beta"]:
        c = x[(x.section == "context") & (x.detector == det)].set_index("quantity").value
        sp = c[[m for m in c.index if any(k in m for k in ("ava_cs", "librispeech_tc", "openbmat_nomusic"))]]
        mu = c[[m for m in c.index if any(k in m for k in ("ava_mo", "audioset_mo"))]]
        v = x[(x.section == "verdict") & (x.detector == det)].set_index("quantity")
        out(f"- {det}: speech alone {fmt(sp.min())}-{fmt(sp.max())}%, film and AudioSet music alone {fmt(mu.min())}-{fmt(mu.max())}%; "
            + "; ".join(f"{e} {v.loc[e, 'text']} ({v.loc[e, 'verdict']})" for e in ("E1", "E2", "E3", "E4")))
    out()
# ---------------------------------------------------------------- 15. reference size and oracle alpha' (Section 5.1)
if (TAB / "unit_refsize_summary.csv").exists() and (TAB / "unit_multiref_vs_strict.csv").exists():
    out("## 15. Reference size (unit_refsize*.csv) and oracle alpha' (unit_multiref_vs_strict.csv)")
    rs = pd.read_csv(TAB / "unit_refsize_summary.csv")
    for r in rs.itertuples():
        out(f"- {NAME.get(r.det, r.det)} {r.size} tracks: worst FPR {fmt(r.med)} [{fmt(r.p05)}-{fmt(r.p95)}]")
    dr = pd.read_csv(TAB / "unit_refsize.csv")
    n = dr.assign(ok=dr.worst_all8 < 6).groupby(["size", "draw"]).ok.sum()
    for k in (150, 322):
        out(f"- {k}-track draws: detectors below 6% on all eight cohorts, 5th-95th percentile "
            f"{np.percentile(n.loc[k], 5):.0f}-{np.percentile(n.loc[k], 95):.0f} of {dr.det.nunique()}")
    vs = pd.read_csv(TAB / "unit_multiref_vs_strict.csv").set_index("det")
    gain = -vs.loc[["B0", "B0'", "B0+0"], "dTPR"]
    out(f"- oracle alpha' {vs.alpha_prime.min():.2f}-{vs.alpha_prime.max():.2f}%; TPR of the single threshold minus MAX, B0 family "
        f"{gain.min():.1f}-{gain.max():.1f} points; readings: " + ", ".join(f"{k} {v}" for k, v in vs.verdict_i.value_counts().items()))
    out()
# ---------------------------------------------------------------- 12. cohort sizes
out("## 12. Cohort sizes (lists/cohorts.csv)")
reg = pd.read_csv(ROOT / "lists" / "cohorts.csv")
for r in reg.itertuples():
    out(f"- {r.name}: n={r.n}, clusters={r.n_clusters} ({r.cluster}), role={r.role}")

OUTMD.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", OUTMD)
