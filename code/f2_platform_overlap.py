"""Intervention on platform overlap between training data and reference (, "An intervention confirms the mechanism").

B0+broad repeats B0+0's extra epoch with about 12,000 added human tracks by other Jamendo artists (disjoint from the A and B
halves). Control = B0+0, noise floor = B0+0 (seed 2).
Hypothesis: adding the Jamendo platform to training turns Jamendo-A from a platform-disjoint reference into a training-platform
reference, so its threshold understates the FPR on other platforms.
Thresholds (window unit, alpha = 2%, track-weighted 1/n_f quantile as in unit_refswap and unit_multiref):
  (a) tau_JamA  (b) other disjoint references tau_R, R in {MUSDB18, GTZAN, AVA-MO (evaluation films), AudioSet-MO}
  (c) MAX4 = the largest tau of the four disjoint references other than Jamendo-A (leave-one-cohort-out: references from the
      same source as the evaluation cohort are dropped, AVA-SM -> AVA-MO, AudioSet-SM -> AudioSet-MO)
  (d) tau_FMAov (FMA-dev, training platform)
Evaluation cohorts: Jamendo-B, AVA-SM (evaluation films), AudioSet-SM, AI-OpenBMAT-H (music-bearing), MUSDB18, GTZAN, AVA-MO,
AudioSet-MO, plus AVA-SM regions >= 10.24 s and LibriSpeech utterances >= 10.24 s (no padding).
Intervals: cluster bootstrap with B replicates (default 2,000, seed 0). The three detectors share files and window grids, so the
same seed and call order give the same resampling, and paired differences (B0+broad - B0+0, B0+0 (seed 2) - B0+0) get intervals.
TPR: the four unseen generator families (generators and files weighted equally, as in unit_rank_unseen) at (a) and at MAX over all
four disjoint references.
Outputs (out/): f2_platform_overlap.csv, f2_platform_overlap_delta.csv, f2_platform_overlap_tpr.csv
usage: python code/f2_platform_overlap.py [--B 2000]
"""
import os, sys, argparse, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from unit_provenance import FileSet, load_det, calib, deploy, ref_set, ref_boot_w, unseen_gens, BATCH  # noqa: E402

R = C.R; TB = C.TB
CTRL = 'B0+0'
DETS = [('B0+0', 'B0+0'), ('B0+broad', 'B0+broad'), ('B0+0 (seed 2)', 'B0+0 (seed 2)')]
REFS = ['jamendo_A', 'musdb18', 'gtzan', 'ava_mo', 'audioset_mo', 'fma_dev800']
DISJ = ['musdb18', 'gtzan', 'ava_mo', 'audioset_mo']
LAB = {'jamendo_A': 'JamA', 'musdb18': 'MUSDB18', 'gtzan': 'GTZAN', 'ava_mo': 'AVA-MO', 'audioset_mo': 'AudioSet-MO', 'fma_dev800': 'FMAov'}
TARGETS = [('jamendo_B', 'Jamendo-B'), ('ava_sm5', 'AVA-SM'), ('ava_sm_long', 'AVA-SM long'), ('audioset_sm', 'AudioSet-SM'),
           ('openbmat_human_musicbearing', 'OBM-H'), ('musdb18', 'MUSDB18'), ('gtzan', 'GTZAN'), ('ava_mo', 'AVA-MO'), ('audioset_mo', 'AudioSet-MO'),
           ('librispeech_long', 'LibriSpeech long')]
SAME = {'ava_sm5': 'ava_mo', 'ava_sm_long': 'ava_mo', 'audioset_sm': 'audioset_mo'}
THRS = ['JamA', 'MUSDB18', 'GTZAN', 'AVA-MO', 'AudioSet-MO', 'MAX4', 'FMAov']
ALPHA = 0.02


def thr_set(c):
    """Threshold name -> references used for evaluation cohort c."""
    out = {'JamA': ['jamendo_A'], 'FMAov': ['fma_dev800']}
    drop = {c, SAME.get(c)}
    for r in DISJ:
        if r not in drop: out[LAB[r]] = [r]
    out['MAX4'] = [r for r in DISJ if r not in drop]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--B', type=int, default=2000); a = ap.parse_args(); B = a.B; t0 = time.time()
    reg = C.load_registry().set_index('name')
    metas = {p: C.read_list(f'{R}/{p}').assign(_row=lambda m: np.arange(len(m))) for p in reg.list.unique()}

    def fileset(name, c, with_strata=False):
        if name == 'librispeech_long':  # long LibriSpeech utterances (no padding)
            r = reg.loc['librispeech_tc']; m = metas[r['list']]
            longid = set(c[c.dur >= 10.24].ID.unique()); sub = m[m._row.isin(longid)]
            return FileSet(c, sub._row.values, sub[r['cluster']].astype(str).values, None)
        r = reg.loc[name]; m = metas[r['list']]; sub = m.query(r['subset']) if r['subset'] else m
        st = sub['group'].astype(str).values if with_strata and 'group' in sub else None
        return FileSet(c, sub._row.values, sub[r['cluster']].astype(str).values, st)
    dumpname = lambda n: reg.loc['librispeech_tc', 'dump'] if n == 'librispeech_long' else reg.loc[n, 'dump']
    names = sorted(set(REFS) | {n for n, _ in TARGETS})
    res, boots, tpr, tprb = [], {}, [], {}
    nf_ref = {}  # paired resampling requires the same files and windows per cohort for the three detectors
    for tag, dn in DETS:
        ch = load_det(tag, {dumpname(n) for n in names} | {'newgen', 'music8k'})
        rng = np.random.default_rng(0)  # same seed and call order for the three detectors -> paired resampling
        FS = {n: fileset(n, ch[dumpname(n)]) for n in names}
        for r in REFS:
            sc_, fi_ = FS[r].units(1)
            assert nf_ref.setdefault('ref_' + r, (len(sc_), len(FS[r].ids))) == (len(sc_), len(FS[r].ids)), (tag, r)
        cal = {r: calib(FS[r], 1, rng, B) for r in REFS}  # (tau, tau_b, n)
        for c, cl in TARGETS:
            ts = thr_set(c); Co, nf, nw, ncl = deploy(FS[c], 1)
            assert nf_ref.setdefault(c, (nf, nw, ncl)) == (nf, nw, ncl), (tag, c, nf_ref[c], (nf, nw, ncl))
            keys = [k for k in THRS if k in ts]
            Tpt = np.array([max(cal[r][0] for r in ts[k]) for k in keys])
            Tb = np.stack([np.max(np.stack([cal[r][1] for r in ts[k]]), axis=0) for k in keys])
            pt = Co.rate(Tpt); bt = Co.boot(Tb, rng, B)
            for i, k in enumerate(keys):
                res.append(dict(det=tag, name=dn, cohort=c, cohort_label=cl, thr=k, refs='+'.join(LAB[r] for r in ts[k]), tau=Tpt[i], n_files=nf, clusters=ncl,
                                fpr=100 * pt[i], ci_lo=100 * np.percentile(bt[i], 2.5), ci_hi=100 * np.percentile(bt[i], 97.5)))
                boots[(tag, c, k)] = (100 * pt[i], 100 * bt[i])
        ref = ref_set(ch, lambda n, c_: fileset(n, c_, True), reg, unseen_gens())
        rs = ref['s']; TJ, TJb = cal['jamendo_A'][0], cal['jamendo_A'][1]
        TM = max(cal[r][0] for r in DISJ); TMb = np.max(np.stack([cal[r][1] for r in DISJ]), axis=0)
        tb = {'JamA': [], 'MAX4': []}
        for b0 in range(0, B, BATCH):
            m = min(B, b0 + BATCH) - b0; wr = ref_boot_w(ref, m, rng); den = wr.sum(1)
            tb['JamA'].append((wr * (rs[None, :] >= TJb[b0:b0 + m, None])).sum(1) / den)
            tb['MAX4'].append((wr * (rs[None, :] >= TMb[b0:b0 + m, None])).sum(1) / den)
        for k, T in (('JamA', TJ), ('MAX4', TM)):
            v = 100 * np.concatenate(tb[k]); p = 100 * float((ref['base'] * (rs >= T)).sum())
            tpr.append(dict(det=tag, name=dn, thr=k, tau=T, tpr=p, ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5)))
            tprb[(tag, k)] = (p, v)
        print(f'[{tag}] done {time.time() - t0:.0f}s', flush=True)
    d = pd.DataFrame(res); d.to_csv(f'{TB}/f2_platform_overlap.csv', index=False)
    t = pd.DataFrame(tpr); t.to_csv(f'{TB}/f2_platform_overlap_tpr.csv', index=False)
    D = []
    others = [x for x, _ in DETS if x != CTRL]
    for (tag, c, k), (p, v) in boots.items():
        if tag != CTRL: continue
        for other in others:
            if (other, c, k) not in boots: continue
            po, vo = boots[(other, c, k)]
            dv = vo - v
            D.append(dict(cohort=c, thr=k, comp=f'{other} - {CTRL}', delta=po - p, ci_lo=np.percentile(dv, 2.5), ci_hi=np.percentile(dv, 97.5)))
    for k in ('JamA', 'MAX4'):
        p, v = tprb[(CTRL, k)]
        for other in others:
            po, vo = tprb[(other, k)]; dv = vo - v
            D.append(dict(cohort='UNSEEN4_TPR', thr=k, comp=f'{other} - {CTRL}', delta=po - p, ci_lo=np.percentile(dv, 2.5), ci_hi=np.percentile(dv, 97.5)))
    D = pd.DataFrame(D); D.to_csv(f'{TB}/f2_platform_overlap_delta.csv', index=False)
    x = d[(d.cohort == 'openbmat_human_musicbearing') & (d.thr == 'JamA')].set_index('det').tau
    print('tau_JamA: ' + ', '.join(f'{k} {v:.3f}' for k, v in x.items()))
    print('F2_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
