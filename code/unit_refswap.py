"""Reference-swap matrix (Section 5.1, "A platform-disjoint reference is necessary but not sufficient").

Definitions (window unit = the detector's native window, alpha = 2%, track-weighted (1/n_f) quantile as for tau_JamA):
  for every reference R in {Jamendo-A (baseline), MUSDB18, GTZAN, AVA-MO (evaluation films), AudioSet-MO, FMA-dev (training
  platform, control)} the threshold tau_R is calibrated, and the realised window FPR is computed on the other human cohorts
  {Jamendo-B, AVA-SM (evaluation films), AudioSet-SM, AI-OpenBMAT-H (music-bearing), MUSDB18, GTZAN, AVA-MO, AudioSet-MO}.
  Excluded pairs: R itself; AVA-SM when R = AVA-MO (same films); AudioSet-SM when R = AudioSet-MO (same source).
  Summary per (R, detector): worst cohort FPR, worst ratio FPR/alpha, whether every cohort stays below 3 alpha = 6% (point
  estimate; cohorts whose CI lower bound exceeds 6% are counted as significant excesses).
Intervals: cluster bootstrap with B replicates (default 2,000, seed 0; the reference clusters are resampled in every replicate
to re-estimate tau, evaluation clusters independently). GTZAN clusters are files (no artist metadata).
Outputs (out/): unit_refswap.csv (per cohort), unit_refswap_summary.csv
usage: python code/unit_refswap.py [--B 2000] [--dets "B0,C0"]
  The replication detectors SpecTTTra-alpha and SpecTTTra-beta run only when named in --dets (code/extra_sonics.sh).
"""
import os, sys, argparse, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from agg_window import wilson  # noqa: E402
from unit_provenance import FileSet, load_det, calib, deploy  # noqa: E402

R = C.R; TB = C.TB
REFS = [('jamendo_A', 'Jamendo-A'), ('musdb18', 'MUSDB18'), ('gtzan', 'GTZAN'), ('ava_mo', 'AVA-MO'), ('audioset_mo', 'AudioSet-MO'), ('fma_dev800', 'FMA-dev')]
TARGETS = [('jamendo_B', 'Jamendo-B'), ('ava_sm5', 'AVA-SM'), ('audioset_sm', 'AudioSet-SM'), ('openbmat_human_musicbearing', 'OBM-H'),
           ('musdb18', 'MUSDB18'), ('gtzan', 'GTZAN'), ('ava_mo', 'AVA-MO'), ('audioset_mo', 'AudioSet-MO')]
EXCLUDE = {('ava_mo', 'ava_sm5'), ('audioset_mo', 'audioset_sm')}  # same films / same source
DISJOINT = ['musdb18', 'gtzan', 'ava_mo', 'audioset_mo']  # platform-disjoint alternatives to Jamendo-A
RS_DETS = [('B0', 'B0'), ("B0'", "B0'"), ('B0+0', 'B0+0'), ('C0', 'C0'), ('C0+0', 'C0+0'), ('M0', 'M0'),
           ('ArtifactNet', 'ArtifactNet'), ('SpecTTTra', 'SpecTTTra'), ('Fakeprint LR', 'Fakeprint LR')]
EXTRA = [(n, n) for n in C.REPLICATION]  # replication detectors, only when named in --dets
A2 = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=2000); ap.add_argument('--dets', default='', help='paper or directory names, comma-separated')
    a = ap.parse_args(); B = a.B; t0 = time.time()
    reg = C.load_registry().set_index('name')
    metas = {p: C.read_list(f'{R}/{p}').assign(_row=lambda m: np.arange(len(m))) for p in reg.list.unique()}

    def fileset(name, c):
        r = reg.loc[name]; m = metas[r['list']]; sub = m.query(r['subset']) if r['subset'] else m
        return FileSet(c, sub._row.values, sub[r['cluster']].astype(str).values, None)
    rows = []
    names_all = sorted({n for n, _ in REFS} | {n for n, _ in TARGETS})
    for tag, dname in C.select(RS_DETS, EXTRA, a.dets):
        ch = load_det(tag, {reg.loc[n, 'dump'] for n in names_all})
        if not {reg.loc[n, 'dump'] for n in names_all} <= set(ch):
            print(f'[skip] {tag}: missing scores {sorted({reg.loc[n, "dump"] for n in names_all} - set(ch))}', flush=True); continue
        rng = np.random.default_rng(0)
        FS = {n: fileset(n, ch[reg.loc[n, 'dump']]) for n in names_all}
        cal = {}
        for rn, rl in REFS:
            T, Tb, ncal = calib(FS[rn], 1, rng, B); cal[rn] = (T, Tb, ncal)
        for tn, tl in TARGETS:
            dp = deploy(FS[tn], 1)
            if dp is None: continue
            Co, nf, nw, ncl = dp
            refs_ok = [rn for rn, _ in REFS if rn != tn and (rn, tn) not in EXCLUDE]
            Tpt = np.array([cal[rn][0] for rn in refs_ok]); Tmat = np.stack([cal[rn][1] for rn in refs_ok])
            pt = Co.rate(Tpt); bt = Co.boot(Tmat, rng, B)
            for i, rn in enumerate(refs_ok):
                lo, hi = wilson(pt[i], ncl)
                rows.append(dict(det=tag, name=dname, ref=rn, ref_label=dict(REFS)[rn], ref_n=cal[rn][2], tau=Tpt[i], cohort=tn, cohort_label=tl,
                                 n_files=nf, n_windows=nw, clusters=ncl, fpr=100 * pt[i], ci_lo=100 * np.percentile(bt[i], 2.5),
                                 ci_hi=100 * np.percentile(bt[i], 97.5), ratio_alpha=pt[i] / A2, wil_hi_clusters=100 * hi))
        print(f'[{tag}] done {time.time() - t0:.0f}s', flush=True)
    d = pd.DataFrame(rows); d.to_csv(f'{TB}/unit_refswap.csv', index=False)
    S = []
    for (tag, nm, rn), g in d.groupby(['det', 'name', 'ref'], sort=False):
        w = g.loc[g.fpr.idxmax()]
        S.append(dict(det=tag, name=nm, ref=rn, ref_label=g.ref_label.iloc[0], n_cohorts=len(g), worst_cohort=w.cohort_label, worst_fpr=w.fpr,
                      worst_ci_lo=w.ci_lo, worst_ci_hi=w.ci_hi, worst_ratio_alpha=w.fpr / (100 * A2),
                      all_below_3x=bool((g.fpr < 300 * A2).all()), n_exceed_point=int((g.fpr >= 300 * A2).sum()), n_exceed_sig=int((g.ci_lo > 300 * A2).sum())))
    S = pd.DataFrame(S); S.to_csv(f'{TB}/unit_refswap_summary.csv', index=False)
    print('Detectors keeping every other human cohort below 6% (3 alpha), per reference:')
    for rn, rl in REFS:
        g = S[S.ref == rn]
        if len(g): print(f'  {rl:12s} {int(g.all_below_3x.sum())} / {len(g)}')
    print('REFSWAP_DONE', len(d), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
