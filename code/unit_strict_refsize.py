"""Two checks on the multi-reference result (, "A platform-disjoint reference is necessary but not sufficient").

Same definitions as unit_refswap.py and unit_multiref.py: window unit, alpha = 2%, track-weighted (1/n_f) quantile, the eight human
cohorts of unit_multiref.py and the four unseen generator families of unit_rank_unseen.
(1) --mode refsize: reference size. Jamendo-A is subsampled by artist to 150 and 322 tracks (the sizes of MUSDB18 and AudioSet-MO),
    500 draws each (artists in random order until the target is passed; the last artist's tracks are then subsampled at random so
    that the size is exact; seed 0; the draws index Jamendo-A files and are the same for every detector). For every draw: tau and
    the worst window FPR over the evaluation cohorts (point estimates), for the eight cohorts of the Jamendo-A reference and, for
    comparison, for the cohort sets of the MUSDB18 reference (seven, MUSDB18 left out) and of the AudioSet-MO reference (six,
    AudioSet-MO and AudioSet-SM left out). Summary: median and 5th-95th percentiles over the draws; detectors below 6% (3 alpha).
(2) --mode strict: the most conservative of five references (MAX) versus a stricter single Jamendo-A threshold at an oracle level
    alpha'. MAX as in unit_multiref.py: for cohort c, tau_max(c) = the largest tau_R over the references other than c and its
    source, W_MAX = max_c FPR_c(tau_max(c)); the deployed MAX threshold is the largest tau_R over all five references (tau_all5),
    with TPR on the unseen families at tau_all5. alpha' = the Jamendo-A track-weighted window share at the smallest single Jamendo-A
    threshold whose worst FPR over the eight cohorts equals W_MAX; its TPR is compared with the MAX TPR (dTPR = TPR_MAX - TPR_alpha').
    Intervals: cluster bootstrap with B replicates (default 1,000, seed 0); every replicate resamples the references, the cohorts and
    the unseen reference and re-fits tau_R, tau_max, alpha' and the single threshold. Reading: interval above 0 -> MAX better,
    below 0 -> single stricter threshold better, otherwise equivalent.
Outputs (out/): unit_refsize.csv (one row per draw), unit_refsize_summary.csv, unit_refsize.md; unit_multiref_vs_strict.{csv,md}
usage: python code/unit_strict_refsize.py --mode refsize|strict [--draws 500] [--B 1000] [--dets "B0,C0"] [--part] [--merge]
"""
import os, sys, argparse, time, glob
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from unit_provenance import FileSet, load_det, wq_tau, ref_set, ref_boot_w, unseen_gens  # noqa: E402
from unit_multiref import REFS, TARGETS, ref_subset, RefPack, wtau_rows  # noqa: E402

R = C.R; TB = C.TB
DETS = [('B0', 'B0'), ("B0'", "B0'"), ('B0+0', 'B0+0'), ('C0', 'C0'), ('C0+0', 'C0+0'), ('M0', 'M0'),
        ('ArtifactNet', 'ArtifactNet'), ('SpecTTTra', 'SpecTTTra'), ('Fakeprint LR', 'Fakeprint LR')]
ALPHA = 0.02; BATCH = 250; LIM = 3 * ALPHA * 100  # 6%
VERDICT = lambda lo, hi: 'MAX better' if lo > 0 else ('single stricter threshold better' if hi < 0 else 'equivalent')


class CW:
    """Evaluation cohort: window scores in descending order and the track-weighted cumulative FPR (point and bootstrap)."""

    def __init__(self, fs):
        sc, fi = fs.units(1); o = np.argsort(-sc, kind='stable')
        self.ss = sc[o]; self.fi = fi[o]; self.inv_n = 1.0 / np.bincount(fi)[self.fi]; self.nf = len(fs.ids)
        self.clus = pd.factorize(pd.Series(fs.clus))[0]; self.G = int(self.clus.max()) + 1

    def cum_point(self):
        return np.r_[0.0, np.cumsum(self.inv_n)] / self.nf

    def cum_boot(self, m, rng):
        cw = rng.multinomial(self.G, np.full(self.G, 1.0 / self.G), size=m).astype(np.float64); Wf = cw[:, self.clus]
        cum = np.concatenate([np.zeros((m, 1)), np.cumsum(Wf[:, self.fi] * self.inv_n[None, :], axis=1)], axis=1)
        return cum / np.maximum(Wf.sum(1, keepdims=True), 1e-12)

    def pos(self, T):
        return np.searchsorted(-self.ss, -np.asarray(T, float), side='right')

    def tmin(self, cumrow, w):
        """The smallest threshold (between window scores) with FPR <= w."""
        p = int(np.searchsorted(cumrow, w + 1e-12, side='right')) - 1
        return -np.inf if p >= len(self.ss) else float(np.nextafter(self.ss[p], np.inf))


def load_all(tag, reg, metas):
    names = sorted({n for n, _ in REFS} | {n for n, _ in TARGETS} | {'newgen', 'music8k'})
    dump = lambda n: reg.loc[n, 'dump']
    ch = load_det(tag, {dump(n) for n in names})
    if not {dump(n) for n in names} <= set(ch): return None

    def fileset(name, c, with_strata=False):
        r = reg.loc[name]; m = metas[r['list']]; sub = m.query(r['subset']) if r['subset'] else m
        st = sub['group'].astype(str).values if with_strata and 'group' in sub else None
        return FileSet(c, sub._row.values, sub[r['cluster']].astype(str).values, st)
    FS = {n: fileset(n, ch[dump(n)]) for n, _ in REFS + TARGETS}
    unseen = ref_set(ch, lambda n, c_: fileset(n, c_, True), reg, unseen_gens())
    return FS, unseen


def strict_one(tag, nm, FS, unseen, B):
    rng = np.random.default_rng(0)
    refs = [rn for rn, _ in REFS]; packs = {r: RefPack(FS[r]) for r in refs}; Co = {c: CW(FS[c]) for c, _ in TARGETS}
    subs = {c: ref_subset(c) for c, _ in TARGETS}; jA = packs['jamendo_A']; rs = unseen['s']

    def evaluate(tauR, cums, wref, wjam):
        """One replicate (point estimate: cums = point cumulatives, wref = unseen-reference weights, wjam = Jamendo-A weights)."""
        tmax = {c: max(tauR[r] for r in subs[c]) for c, _ in TARGETS}; tall = max(tauR.values())
        fmax = {c: cums[c][Co[c].pos(tmax[c])] for c, _ in TARGETS}
        Wmax = max(fmax.values())
        Ti = max(Co[c].tmin(cums[c], Wmax) for c, _ in TARGETS)
        tpr = lambda T: float((wref * (rs >= T)).sum() / wref.sum())
        alpha_p = float((wjam * (jA.s >= Ti)).sum() / wjam.sum())
        return dict(Wmax=100 * Wmax, Ti=Ti, alpha_p=100 * alpha_p, tpr_strict=100 * tpr(Ti), tpr_max=100 * tpr(tall),
                    worst_max=max(fmax, key=fmax.get), tall=tall)
    tauR = {r: wq_tau(packs[r].s, packs[r].base, ALPHA)[0] for r in refs}
    pt = evaluate(tauR, {c: Co[c].cum_point() for c, _ in TARGETS}, unseen['base'], jA.base)
    bs = []
    for a in range(0, B, BATCH):
        m = min(B, a + BATCH) - a
        U = {}; tb = {}
        for r in refs:
            p = packs[r]; cw = rng.multinomial(p.G, np.full(p.G, 1.0 / p.G), size=m).astype(np.float64)
            U[r] = cw[:, p.ccode] * p.base[None, :]; tb[r] = wtau_rows(p.s, U[r], ALPHA)
        cums = {c: Co[c].cum_boot(m, rng) for c, _ in TARGETS}
        wr = ref_boot_w(unseen, m, rng)
        for i in range(m):
            bs.append(evaluate({r: tb[r][i] for r in refs}, {c: cums[c][i] for c, _ in TARGETS}, wr[i], U['jamendo_A'][i]))
    bdf = pd.DataFrame(bs)
    dT = bdf.tpr_max - bdf.tpr_strict
    q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    lab = dict(TARGETS)
    return dict(det=tag, name=nm, W_max=pt['Wmax'], W_max_lo=q(bdf.Wmax)[0], W_max_hi=q(bdf.Wmax)[1], worst_cohort_max=lab[pt['worst_max']],
                tau_all5=pt['tall'], tpr_max=pt['tpr_max'], tpr_max_lo=q(bdf.tpr_max)[0], tpr_max_hi=q(bdf.tpr_max)[1],
                alpha_prime=pt['alpha_p'], alpha_prime_lo=q(bdf.alpha_p)[0], alpha_prime_hi=q(bdf.alpha_p)[1], tau_strict=pt['Ti'],
                tpr_strict=pt['tpr_strict'], tpr_strict_lo=q(bdf.tpr_strict)[0], tpr_strict_hi=q(bdf.tpr_strict)[1],
                dTPR=pt['tpr_max'] - pt['tpr_strict'], dTPR_lo=q(dT)[0], dTPR_hi=q(dT)[1], verdict_i=VERDICT(*q(dT)))


def refsize_one(tag, nm, FS, draws):
    jA = FS['jamendo_A']; sc, fi = jA.units(1)
    Co = {c: CW(FS[c]) for c, _ in TARGETS}; cum = {c: Co[c].cum_point() for c, _ in TARGETS}
    sets = {'all8': [c for c, _ in TARGETS], 'as_MUSDB': [c for c, _ in TARGETS if c != 'musdb18'],
            'as_AudioSetMO': [c for c, _ in TARGETS if c not in ('audioset_mo', 'audioset_sm')]}
    rows = []
    for size, dl in draws.items():
        for k, files in enumerate(dl):
            sel = np.isin(fi, files); s = sc[sel]; f = fi[sel]
            w = 1.0 / np.bincount(f)[f]
            T = wq_tau(s, w, ALPHA)[0]
            fpr = {c: 100 * cum[c][Co[c].pos(T)] for c, _ in TARGETS}
            r = dict(det=tag, name=nm, size=size, draw=k, tau=T)
            for sn, cs in sets.items():
                wc = max(cs, key=lambda c: fpr[c]); r[f'worst_{sn}'] = fpr[wc]; r[f'worstc_{sn}'] = wc
            rows.append(r)
    return rows


def make_draws(FS, n_draws, seed=0):
    jA = FS['jamendo_A']; ids = np.arange(len(jA.ids)); art = pd.factorize(pd.Series(jA.clus))[0]
    by = {a: ids[art == a] for a in np.unique(art)}; rng = np.random.default_rng(seed); out = {}
    for size in (150, 322):
        dl = []
        for _ in range(n_draws):
            order = rng.permutation(list(by)); got = []
            for a in order:
                fl = list(by[a]); need = size - len(got)
                if len(fl) >= need:
                    got += list(rng.choice(fl, need, replace=False)); break
                got += fl
            dl.append(np.array(got))
        out[size] = dl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['strict', 'refsize']); ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--draws', type=int, default=500); ap.add_argument('--dets', default='', help='paper or directory names, comma-separated')
    ap.add_argument('--part', action='store_true'); ap.add_argument('--merge', action='store_true')
    a = ap.parse_args(); t0 = time.time()
    if a.merge: return merge(a)
    reg = C.load_registry().set_index('name')
    metas = {p: C.read_list(f'{R}/{p}').assign(_row=lambda m: np.arange(len(m))) for p in reg.list.unique()}
    dets = C.select(DETS, [], a.dets)
    rows, draws = [], None
    for tag, nm in dets:
        ld = load_all(tag, reg, metas)
        if ld is None: print(f'[skip] {tag}: missing scores', flush=True); continue
        FS, unseen = ld
        if a.mode == 'strict':
            rows.append(strict_one(tag, nm, FS, unseen, a.B))
        else:
            if draws is None: draws = make_draws(FS, a.draws)  # the draws index Jamendo-A files: the same for every detector
            rows += refsize_one(tag, nm, FS, draws)
        print(f'[{a.mode} {tag}] {time.time() - t0:.0f}s', flush=True)
    d = pd.DataFrame(rows)
    os.makedirs(f'{TB}/strict_refsize_parts', exist_ok=True)
    tg = '_'.join(C.DETECTORS[t][0] for t, _ in dets) if a.dets else 'all'
    d.to_csv(f'{TB}/strict_refsize_parts/{a.mode}_{tg}.csv', index=False)
    if not a.part: merge(a)
    print('DONE', a.mode, f'{time.time() - t0:.0f}s')


def merge(a):
    order = {t: i for i, (t, _) in enumerate(DETS)}
    fs = sorted(glob.glob(f'{TB}/strict_refsize_parts/{a.mode}_*.csv'))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).sort_values('det', key=lambda c: c.map(order), kind='stable')
    nd = d.det.nunique()
    if a.mode == 'strict':
        d.to_csv(f'{TB}/unit_multiref_vs_strict.csv', index=False)
        L = ['# Most conservative of five references (MAX) versus a stricter single Jamendo-A threshold (oracle alpha\')', '',
             f'Window unit, track-weighted quantile, {nd} detectors, eight human cohorts. MAX = the leave-one-cohort-out protocol of unit_multiref.py. '
             f'Intervals: cluster bootstrap, B = {a.B:,} (all thresholds re-fitted in every replicate). TPR on the four unseen generator families '
             '(generators and tracks weighted equally); TPR_MAX at the deployed MAX threshold (largest tau over all five references).', '',
             '| detector | W_MAX (worst cohort) | alpha\' (%) [95% CI] | TPR at alpha\' | TPR at MAX | dTPR = MAX - single [95% CI] | reading |',
             '|---|---|---|---|---|---|---|']
        for _, x in d.iterrows():
            L.append(f'| {x["name"]} | {x.W_max:.1f} [{x.W_max_lo:.1f}-{x.W_max_hi:.1f}] {x.worst_cohort_max} | {x.alpha_prime:.2f} '
                     f'[{x.alpha_prime_lo:.2f}-{x.alpha_prime_hi:.2f}] | {x.tpr_strict:.1f} | {x.tpr_max:.1f} | {x.dTPR:+.1f} [{x.dTPR_lo:+.1f}, {x.dTPR_hi:+.1f}] | '
                     f'{x.verdict_i} |')
        cnt = lambda v: int((d.verdict_i == v).sum())
        L += ['', f'MAX better {cnt("MAX better")}, equivalent {cnt("equivalent")}, single stricter threshold better {cnt("single stricter threshold better")}; '
              f'alpha\' {d.alpha_prime.min():.2f}-{d.alpha_prime.max():.2f}% (median {d.alpha_prime.median():.2f}%).',
              'alpha\' is set by looking at the worst FPR of the deployment cohorts, so it is an oracle: MAX reaches the same worst case without deployment data.', '']
        open(f'{TB}/unit_multiref_vs_strict.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    else:
        d.to_csv(f'{TB}/unit_refsize.csv', index=False)
        p5 = lambda v: np.percentile(v, 5); p95 = lambda v: np.percentile(v, 95)
        g = d.groupby(['det', 'name', 'size'], sort=False)
        Sm = g.agg(med=('worst_all8', 'median'), p05=('worst_all8', p5), p95=('worst_all8', p95),
                   medM=('worst_as_MUSDB', 'median'), p05M=('worst_as_MUSDB', p5), p95M=('worst_as_MUSDB', p95),
                   medA=('worst_as_AudioSetMO', 'median'), p05A=('worst_as_AudioSetMO', p5), p95A=('worst_as_AudioSetMO', p95)).reset_index()
        Sm.to_csv(f'{TB}/unit_refsize_summary.csv', index=False)
        sw = pd.read_csv(f'{TB}/unit_refswap_summary.csv') if os.path.exists(f'{TB}/unit_refswap_summary.csv') else None
        ref = lambda det, rn: '' if sw is None or sw[(sw.det == det) & (sw.ref == rn)].empty else f'{sw[(sw.det == det) & (sw.ref == rn)].worst_fpr.iloc[0]:.1f}'
        L = [f'# Reference size: artist-level draws of 150 and 322 Jamendo-A tracks ({d.draw.max() + 1} draws each)', '',
             'Window unit, alpha = 2%, track-weighted quantile. Worst window FPR over the evaluation cohorts per draw (point estimates), median '
             '[5th-95th percentile] over the draws; full Jamendo-A, MUSDB18 and AudioSet-MO from unit_refswap_summary.csv.', '',
             '| detector | Jamendo-A, all tracks (8 cohorts) | 150 tracks (8) | 322 tracks (8) | MUSDB18 (7 cohorts) | 150 tracks (same 7) | '
             'AudioSet-MO (6 cohorts) | 322 tracks (same 6) |', '|---|---|---|---|---|---|---|---|']
        for tag, nm in DETS:
            x = Sm[Sm.det == tag]
            if x.empty: continue
            s1 = x[x['size'] == 150].iloc[0]; s3 = x[x['size'] == 322].iloc[0]
            L.append(f'| {nm} | {ref(tag, "jamendo_A")} | {s1.med:.1f} [{s1.p05:.1f}-{s1.p95:.1f}] | {s3.med:.1f} [{s3.p05:.1f}-{s3.p95:.1f}] | '
                     f'{ref(tag, "musdb18")} | {s1.medM:.1f} [{s1.p05M:.1f}-{s1.p95M:.1f}] | {ref(tag, "audioset_mo")} | {s3.medA:.1f} [{s3.p05A:.1f}-{s3.p95A:.1f}] |')
        per_draw = d.assign(ok=d.worst_all8 < LIM).groupby(['size', 'draw']).ok.sum()
        L += ['', f'Detectors below 6% (3 alpha) on all eight cohorts (of {nd}):', '']
        if sw is not None:
            L.append(f"- Jamendo-A, all tracks: {int(sw[sw.ref == 'jamendo_A'].all_below_3x.astype(str).str.lower().isin(['true', '1']).sum())}")
        for size in (150, 322):
            pdr = per_draw.loc[size]
            L.append(f'- {size} tracks: median {np.median(pdr):.0f}, 5th-95th percentile {np.percentile(pdr, 5):.0f}-{np.percentile(pdr, 95):.0f} over the draws')
        if sw is not None:
            for rn, rl in (('musdb18', 'MUSDB18'), ('audioset_mo', 'AudioSet-MO')):
                L.append(f"- {rl}: {int(sw[sw.ref == rn].all_below_3x.astype(str).str.lower().isin(['true', '1']).sum())}")
        L += ['', 'Reading: if small Jamendo-A draws stay near the full set while MUSDB18 does not, the disagreement between references comes from '
              'platform and content, not from size.', '']
        open(f'{TB}/unit_refsize.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('MERGED', a.mode, len(d))


if __name__ == '__main__':
    main()
