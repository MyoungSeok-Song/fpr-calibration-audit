"""Multi-reference thresholds, leave-one-cohort-out (, "most conservative of five references").

References R in {Jamendo-A, MUSDB18, GTZAN, AVA-MO (evaluation films), AudioSet-MO}. Window unit, alpha = 2%, track-weighted
quantile (the definition of tau_JamA). For every evaluation cohort c in {Jamendo-B, AVA-SM, AudioSet-SM, AI-OpenBMAT-H
(music-bearing), MUSDB18, GTZAN, AVA-MO, AudioSet-MO} the references from the same source are left out: AVA-SM -> AVA-MO,
AudioSet-SM -> AudioSet-MO, and c itself when c is a reference.
  (Sensitivity) Jamendo-B is also evaluated with Jamendo-A (same platform) left out; this row is not counted in the summary.
  (i)  MAX : tau_max(c)  = the largest tau_R(2%) of the remaining references
  (ii) POOL: tau_pool(c) = the 98th percentile of the pooled windows of the remaining references (equal total weight per
       reference, 1/n_f per track within a reference)
Reported: realised window FPR of c at both thresholds, the worst cohort per detector and whether every cohort stays below 6%,
the number of detectors within 3 alpha (MAX, POOL, single Jamendo-A), and the TPR on the four unseen generator families
(generators and files weighted equally, as in unit_rank_unseen) at tau_max and tau_pool over all five references versus tau_JamA.
Intervals: cluster bootstrap with B replicates (default 2,000, seed 0). In every replicate the clusters of every reference are
resampled (one draw serves tau_R, tau_max and tau_pool) and the thresholds re-estimated; evaluation clusters and the unseen
reference (stratified by generator) are resampled independently.
Outputs (out/): unit_multiref.csv (per cohort), unit_multiref_summary.csv, unit_multiref_tpr.csv
usage: python code/unit_multiref.py [--B 2000] [--dets "B0,C0"] [--part] [--merge]
  The replication detectors SpecTTTra-alpha and SpecTTTra-beta run only when named in --dets (code/extra_sonics.sh).
"""
import os, sys, argparse, time, glob
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from unit_provenance import FileSet, load_det, deploy, wq_tau, wilson, ref_set, ref_boot_w, unseen_gens, BATCH  # noqa: E402

R = C.R; TB = C.TB
REFS = [('jamendo_A', 'Jamendo-A'), ('musdb18', 'MUSDB18'), ('gtzan', 'GTZAN'), ('ava_mo', 'AVA-MO'), ('audioset_mo', 'AudioSet-MO')]
TARGETS = [('jamendo_B', 'Jamendo-B'), ('ava_sm5', 'AVA-SM'), ('audioset_sm', 'AudioSet-SM'), ('openbmat_human_musicbearing', 'OBM-H'),
           ('musdb18', 'MUSDB18'), ('gtzan', 'GTZAN'), ('ava_mo', 'AVA-MO'), ('audioset_mo', 'AudioSet-MO')]
SAME = {'ava_sm5': 'ava_mo', 'audioset_sm': 'audioset_mo'}  # same-source pairs; the Jamendo-B row without Jamendo-A is a separate sensitivity row
SENS = [('jamendo_B', 'Jamendo-B (Jamendo-A left out, sensitivity)', 'jamendo_A')]
DETS = [('B0', 'B0'), ("B0'", "B0'"), ('B0+0', 'B0+0'), ('C0', 'C0'), ('C0+0', 'C0+0'), ('M0', 'M0'),
        ('ArtifactNet', 'ArtifactNet'), ('SpecTTTra', 'SpecTTTra'), ('Fakeprint LR', 'Fakeprint LR')]
EXTRA = [(n, n) for n in C.REPLICATION]  # replication detectors, only when named in --dets
ALPHA = 0.02


def ref_subset(c, extra=None):
    drop = {c, SAME.get(c), extra}
    return tuple(rn for rn, _ in REFS if rn not in drop)


def wtau_rows(ss, W, alpha):
    """Scores ss sorted in descending order, weights W (m x n, same order) -> per row the weighted (1 - alpha) quantile
    (the last score whose cumulative weight <= alpha * total)."""
    cum = np.cumsum(W, axis=1); cnt = (cum <= alpha * cum[:, -1:] + 1e-12).sum(1)
    return np.where(cnt > 0, ss[np.maximum(cnt - 1, 0)], np.inf)


class RefPack:
    """Windows of one reference (descending), their tracks and clusters."""

    def __init__(self, fs):
        sc, fi = fs.units(1); o = np.argsort(-sc, kind='stable')
        self.s = sc[o]; self.fi = fi[o]; self.n_f = np.bincount(fi)[self.fi].astype(float)
        self.ccode = pd.factorize(pd.Series(fs.clus))[0][self.fi]; self.G = int(self.ccode.max()) + 1
        self.base = 1.0 / self.n_f  # track weight


def thresholds(packs, subsets, rng, B):
    """Point estimates and B bootstrap replicates of tau_R per reference and of tau_max / tau_pool per reference subset,
    all from the same reference draws."""
    names = list(packs)
    pt_R = {r: wq_tau(packs[r].s, packs[r].base, ALPHA)[0] for r in names}
    # pooled windows per subset: equal total weight per reference, 1/n_f per track within a reference
    pool_pt, pool_meta = {}, {}
    for S in subsets:
        s = np.concatenate([packs[r].s for r in S]); w = np.concatenate([packs[r].base / packs[r].base.sum() / len(S) for r in S])
        o = np.argsort(-s, kind='stable'); pool_meta[S] = (s[o], o)
        pool_pt[S] = wq_tau(s, w, ALPHA)[0]
    bt_R = {r: np.empty(B) for r in names}; bt_pool = {S: np.empty(B) for S in subsets}
    for a in range(0, B, BATCH):
        m = min(B, a + BATCH) - a
        U = {}
        for r in names:
            p = packs[r]; cw = rng.multinomial(p.G, np.full(p.G, 1.0 / p.G), size=m).astype(np.float64)
            U[r] = cw[:, p.ccode] * p.base[None, :]
            bt_R[r][a:a + m] = wtau_rows(p.s, U[r], ALPHA)
        for S in subsets:
            ss, o = pool_meta[S]
            W = np.concatenate([U[r] / np.maximum(U[r].sum(1, keepdims=True), 1e-12) / len(S) for r in S], axis=1)[:, o]
            bt_pool[S][a:a + m] = wtau_rows(ss, W, ALPHA)
    out = {}
    for S in subsets:
        out[S] = dict(max_pt=max(pt_R[r] for r in S), pool_pt=pool_pt[S],
                      max_bt=np.max(np.stack([bt_R[r] for r in S]), axis=0), pool_bt=bt_pool[S])
    out['_R'] = (pt_R, bt_R)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--B', type=int, default=2000); ap.add_argument('--dets', default='')
    ap.add_argument('--part', action='store_true'); ap.add_argument('--merge', action='store_true')
    a = ap.parse_args(); B = a.B; t0 = time.time()
    if a.merge:
        order = {t_: i for i, (t_, _) in enumerate(DETS + EXTRA)}
        cat = lambda k: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{TB}/multiref_parts/*_{k}.csv'))], ignore_index=True).sort_values('det', key=lambda c: c.map(order), kind='stable')
        finish(cat('rows'), cat('tpr')); return
    reg = C.load_registry().set_index('name')
    metas = {p: C.read_list(f'{R}/{p}').assign(_row=lambda m: np.arange(len(m))) for p in reg.list.unique()}

    def fileset(name, c, with_strata=False):
        r = reg.loc[name]; m = metas[r['list']]; sub = m.query(r['subset']) if r['subset'] else m
        st = sub['group'].astype(str).values if with_strata and 'group' in sub else None
        return FileSet(c, sub._row.values, sub[r['cluster']].astype(str).values, st)
    rows, trows = [], []
    allnames = sorted({n for n, _ in REFS} | {n for n, _ in TARGETS} | {'newgen', 'music8k'})
    subsets = sorted({ref_subset(c) for c, _ in TARGETS} | {ref_subset(c, e) for c, _, e in SENS} | {tuple(rn for rn, _ in REFS)}, key=len)
    ALL5 = tuple(rn for rn, _ in REFS)
    dets = C.select(DETS, EXTRA, a.dets)
    for tag, dname in dets:
        ch = load_det(tag, {reg.loc[n, 'dump'] for n in allnames})
        need = {reg.loc[n, 'dump'] for n in allnames}
        if not need <= set(ch): print(f'[skip] {tag}: missing scores {sorted(need - set(ch))}', flush=True); continue
        rng = np.random.default_rng(0)
        FS = {n: fileset(n, ch[reg.loc[n, 'dump']]) for n, _ in TARGETS + REFS}
        packs = {rn: RefPack(FS[rn]) for rn, _ in REFS}
        th = thresholds(packs, subsets, rng, B)
        pt_R, bt_R = th['_R']
        for c, cl, extra, sens in [(c, cl, None, False) for c, cl in TARGETS] + [(c, cl, e, True) for c, cl, e in SENS]:
            S = ref_subset(c, extra); Co, nf, nw, ncl = deploy(FS[c], 1)
            T = np.array([th[S]['max_pt'], th[S]['pool_pt'], pt_R['jamendo_A']])
            Tb = np.stack([th[S]['max_bt'], th[S]['pool_bt'], bt_R['jamendo_A']])
            pt = Co.rate(T); bt = Co.boot(Tb, rng, B)
            for i, kind in enumerate(['MAX', 'POOL', 'JamA']):
                if kind == 'JamA' and c == 'jamendo_A': continue
                lo, hi = wilson(pt[i], ncl)
                if sens and kind == 'JamA': continue
                rows.append(dict(det=tag, name=dname, cohort=c, cohort_label=cl, sensitivity=int(sens), refs='+'.join(dict(REFS)[r] for r in S) if kind != 'JamA' else 'Jamendo-A',
                                 kind=kind, tau=T[i], n_files=nf, n_windows=nw, clusters=ncl, fpr=100 * pt[i],
                                 ci_lo=100 * np.percentile(bt[i], 2.5), ci_hi=100 * np.percentile(bt[i], 97.5), wil_hi_clusters=100 * hi))
        # TPR on the unseen generator families: tau_max and tau_pool over all five references versus tau_JamA
        ref = ref_set(ch, lambda n, c_: fileset(n, c_, True), reg, unseen_gens())
        if ref is not None:
            rs = ref['s']; Tt = [('MAX(5)', th[ALL5]['max_pt'], th[ALL5]['max_bt']), ('POOL(5)', th[ALL5]['pool_pt'], th[ALL5]['pool_bt']),
                                 ('JamA', pt_R['jamendo_A'], bt_R['jamendo_A'])]
            tb = {k: [] for k, _, _ in Tt}
            for b0 in range(0, B, BATCH):
                m = min(B, b0 + BATCH) - b0; wr = ref_boot_w(ref, m, rng); den = wr.sum(1)
                for k, _, tbk in Tt:
                    tb[k].append((wr * (rs[None, :] >= tbk[b0:b0 + m, None])).sum(1) / den)
            for k, tp, _ in Tt:
                v = np.concatenate(tb[k])
                trows.append(dict(det=tag, name=dname, kind=k, tau=tp, tpr=100 * float((ref['base'] * (rs >= tp)).sum()),
                                  ci_lo=100 * np.percentile(v, 2.5), ci_hi=100 * np.percentile(v, 97.5)))
        print(f'[{tag}] done {time.time() - t0:.0f}s', flush=True)
    d = pd.DataFrame(rows); t = pd.DataFrame(trows)
    if a.part:
        os.makedirs(f'{TB}/multiref_parts', exist_ok=True); tg = '_'.join(C.DETECTORS[x][0] for x, _ in dets) or 'all'
        d.to_csv(f'{TB}/multiref_parts/{tg}_rows.csv', index=False); t.to_csv(f'{TB}/multiref_parts/{tg}_tpr.csv', index=False)
        print('MULTIREF_PART_DONE', tg, f'{time.time() - t0:.0f}s'); return
    finish(d, t)


def finish(d, t):
    d.to_csv(f'{TB}/unit_multiref.csv', index=False); t.to_csv(f'{TB}/unit_multiref_tpr.csv', index=False)
    S = []
    for (tag, nm, kind), g in d[d.sensitivity == 0].groupby(['det', 'name', 'kind'], sort=False):
        w = g.loc[g.fpr.idxmax()]
        S.append(dict(det=tag, name=nm, kind=kind, worst_cohort=w.cohort_label, worst_fpr=w.fpr, worst_ci_lo=w.ci_lo, worst_ci_hi=w.ci_hi,
                      worst_ratio_alpha=w.fpr / (100 * ALPHA), all_below_3x=bool((g.fpr < 300 * ALPHA).all()),
                      n_exceed=int((g.fpr >= 300 * ALPHA).sum()), n_exceed_sig=int((g.ci_lo > 300 * ALPHA).sum())))
    S = pd.DataFrame(S); S.to_csv(f'{TB}/unit_multiref_summary.csv', index=False)
    print('Detectors keeping all eight human cohorts below 6% (3 alpha):')
    for kind in ('MAX', 'POOL', 'JamA'):
        g = S[S.kind == kind]
        if len(g): print(f'  {kind:5s} {int(g.all_below_3x.sum())} / {len(g)}  (significant excess in {int((g.n_exceed_sig > 0).sum())})')
    sens = d[(d.sensitivity == 1) & (d.kind == 'MAX')]
    if len(sens):
        ok = S[(S.kind == 'MAX') & S.all_below_3x].det
        print(f'  MAX with Jamendo-A also left out for Jamendo-B: {int(sum((sens.set_index("det").fpr < 6).reindex(ok).fillna(False)))} / {len(S[S.kind == "MAX"])}')
    print('MULTIREF_DONE', len(d))


if __name__ == '__main__':
    main()
