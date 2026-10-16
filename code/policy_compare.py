"""Threshold policies P1-P4 compared on the same cohorts (post hoc; multi-reference thresholds).

Window unit, alpha = 2%, track-weighted (1/n_f) quantiles. Every policy is fixed before evaluation.
  P1     = single Jamendo-A tau (the main policy of the paper).
  P2     = pooled tau: weighted (1 - alpha) quantile of the pooled windows of the five references (Jamendo-A, MUSDB18, GTZAN, AVA-MO,
           AudioSet-MO), total weight 1/5 per reference and equal weight per track within a reference (1/n_f per window)
           = unit_multiref POOL over all five references, applied to every cohort.
  P2-LSO = (sensitivity) leave-source-out pooled tau = unit_multiref POOL (AVA-SM without AVA-MO, AudioSet-SM without
           AudioSet-MO; Jamendo-B and OBM-H use all five, i.e. P2). The deployed threshold is the pooled tau of all five, so its
           TPR equals that of P2.
  P3     = the largest of the five reference taus, fixed once and applied to every cohort (tau_all5 = unit_multiref MAX(5)).
  P4     = leave-source-out max (unit_multiref MAX, leave one cohort out): AVA-SM without AVA-MO, AudioSet-SM without AudioSet-MO.
           Jamendo-B and OBM-H leave nothing out, so they equal P3; the deployed threshold is tau_all5, so its TPR equals P3's.
Reported per detector and policy: window FPR of the four evaluation cohorts (Jamendo-B, AVA-SM, AudioSet-SM, OBM-H) with cluster
  bootstrap 95% intervals, the worst of the four, whether the worst point FPR and its upper bounds stay below 6% (3 alpha), the
  window TPR on the four unseen generator families (lyria3, heartmula, minimax_2.6, mureka_v9; generators weighted equally,
  files weighted equally within a generator, as in unit_multiref_tpr) and the TPR cost relative to P1 with a paired interval.
  Two intervals for the worst FPR: (a) simultaneous (sim): per replicate the maximum over the four cohorts (as W_max of
  unit_multiref_vs_strict), used to count upper bounds below 6%; (b) cohort (coh): the interval of the cohort with the worst
  point estimate (as unit_multiref_summary).
  P2-LSO and P4 also give the Jamendo-B sensitivity row with Jamendo-A left out (sens_jamB_noJamA_*).
Intervals: the cluster bootstrap of unit_multiref.py (B = 2,000, seed 0 per detector). In every replicate the clusters of every
  reference are resampled and tau_R, tau_max and tau_pool re-estimated together; evaluation clusters and the unseen reference
  (stratified by generator) are resampled independently. The random draws are consumed in the order of unit_multiref.main
  (8 cohorts + 1 sensitivity row, then the TPR), so the P1, P4 and P2-LSO cohort rows and the TPR equal unit_multiref.csv and
  unit_multiref_tpr.csv.
Output (out/): policy_compare.csv (detector x policy)
usage: python code/policy_compare.py [--B 2000] [--dets "B0,C0"]
"""
import os, sys, argparse, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from unit_provenance import FileSet, load_det, deploy, wilson, ref_set, ref_boot_w, unseen_gens, BATCH  # noqa: E402
from unit_multiref import REFS, TARGETS, SENS, ALPHA, ref_subset, RefPack, thresholds  # noqa: E402

R = C.R; TB = C.TB
DETS = [('B0', 'B0'), ("B0'", "B0'"), ('B0+0', 'B0+0'), ('C0', 'C0'), ('C0+0', 'C0+0'), ('M0', 'M0'),
        ('SpecTTTra', 'SpecTTTra'), ('ArtifactNet', 'ArtifactNet'), ('Fakeprint LR', 'Fakeprint LR')]  # C1 is not released
FINETUNED = ['B0', "B0'", 'B0+0', 'C0', 'C0+0', 'M0']
FOUR = [('jamendo_B', 'Jamendo-B', 'JamB'), ('ava_sm5', 'AVA-SM', 'AVASM'), ('audioset_sm', 'AudioSet-SM', 'ASSM'),
        ('openbmat_human_musicbearing', 'OBM-H', 'OBMH')]
POL = [('P1', 'single Jamendo-A tau'), ('P2', 'pooled tau of the five references (weight 1/5 per reference)'),
       ('P2-LSO', '(sensitivity) leave-source-out pooled tau = unit_multiref POOL'), ('P3', 'max of the five reference taus, fixed once (tau_all5)'),
       ('P4', 'leave-source-out max (leave one cohort out)')]
ALL5 = tuple(rn for rn, _ in REFS)
LIM = 300 * ALPHA  # 6 %


def run(B, dets):
    reg = C.load_registry().set_index('name')
    metas = {p: C.read_list(f'{R}/{p}').assign(_row=lambda m: np.arange(len(m))) for p in reg.list.unique()}

    def fileset(name, c, with_strata=False):  # the same definition as in unit_multiref.main
        r = reg.loc[name]; m = metas[r['list']]; sub = m.query(r['subset']) if r['subset'] else m
        st = sub['group'].astype(str).values if with_strata and 'group' in sub else None
        return FileSet(c, sub._row.values, sub[r['cluster']].astype(str).values, st)

    allnames = sorted({n for n, _ in REFS} | {n for n, _ in TARGETS} | {'newgen', 'music8k'})
    subsets = sorted({ref_subset(c) for c, _ in TARGETS} | {ref_subset(c, e) for c, _, e in SENS} | {ALL5}, key=len)
    coh_rows, wide = [], []
    t0 = time.time()
    for det, name in DETS:
        if dets and det not in dets: continue
        need = {reg.loc[n, 'dump'] for n in allnames}
        ch = load_det(det, need)
        if not need <= set(ch): print(f'[skip] {det}: missing scores {sorted(need - set(ch))}', flush=True); continue
        # ---- the random order of unit_multiref.main ----
        rng = np.random.default_rng(0)
        FS = {n: fileset(n, ch[reg.loc[n, 'dump']]) for n, _ in TARGETS + REFS}
        packs = {rn: RefPack(FS[rn]) for rn, _ in REFS}
        th = thresholds(packs, subsets, rng, B)
        pt_R, bt_R = th['_R']
        boots = {}  # (cohort, policy) -> B replicates of the FPR (the four cohorts, no sensitivity row)
        for c, cl, extra, sens in [(c, cl, None, False) for c, cl in TARGETS] + [(c, cl, e, True) for c, cl, e in SENS]:
            S = ref_subset(c, extra); Co, nf, nw, ncl = deploy(FS[c], 1)
            # the first three as in unit_multiref (MAX_S, POOL_S, JamA), the last two over all five references (MAX5 = P3, POOL5 = P2)
            kinds = ['MAX', 'POOL', 'JamA', 'MAX5', 'POOL5']
            T = np.array([th[S]['max_pt'], th[S]['pool_pt'], pt_R['jamendo_A'], th[ALL5]['max_pt'], th[ALL5]['pool_pt']])
            Tb = np.stack([th[S]['max_bt'], th[S]['pool_bt'], bt_R['jamendo_A'], th[ALL5]['max_bt'], th[ALL5]['pool_bt']])
            pt = Co.rate(T); bt = Co.boot(Tb, rng, B)  # the draws depend only on the clusters and B -> the same as in unit_multiref
            for i, kind in enumerate(kinds):
                lo, hi = wilson(pt[i], ncl)
                coh_rows.append(dict(det=det, name=name, cohort=c, cohort_label=cl, sensitivity=int(sens), kind=kind,
                                     refs='+'.join(dict(REFS)[r] for r in (S if kind in ('MAX', 'POOL') else (ALL5 if kind != 'JamA' else ('jamendo_A',)))),
                                     tau=T[i], n_files=nf, n_windows=nw, clusters=ncl, fpr=100 * pt[i],
                                     ci_lo=100 * np.percentile(bt[i], 2.5), ci_hi=100 * np.percentile(bt[i], 97.5), wil_hi_clusters=100 * hi))
            if not sens and c in [x[0] for x in FOUR]:
                for pol, i in (('P1', 2), ('P2', 4), ('P2-LSO', 1), ('P3', 3), ('P4', 0)):
                    boots[(c, pol)] = 100 * bt[i]
        # ---- TPR: the order and definitions of unit_multiref (MAX(5) = P3 and P4, POOL(5) = P2 and P2-LSO, JamA = P1) ----
        ref = ref_set(ch, lambda n, c_: fileset(n, c_, True), reg, unseen_gens())
        rs = ref['s']
        Tt = [('MAX(5)', th[ALL5]['max_pt'], th[ALL5]['max_bt']), ('POOL(5)', th[ALL5]['pool_pt'], th[ALL5]['pool_bt']),
              ('JamA', pt_R['jamendo_A'], bt_R['jamendo_A'])]
        tb = {k: [] for k, _, _ in Tt}
        for b0 in range(0, B, BATCH):
            m = min(B, b0 + BATCH) - b0; wr = ref_boot_w(ref, m, rng); den = wr.sum(1)
            for k, _, tbk in Tt:
                tb[k].append((wr * (rs[None, :] >= tbk[b0:b0 + m, None])).sum(1) / den)
        tb = {k: 100 * np.concatenate(v) for k, v in tb.items()}
        tpt = {k: 100 * float((ref['base'] * (rs >= tp)).sum()) for k, tp, _ in Tt}
        gen_tpr = {}  # per-generator TPR (point)
        for k, tp, _ in Tt:
            for gi, it in enumerate(ref['info']):
                sel = ref['fgen'][ref['fidx']] == gi
                gen_tpr[(k, it['generator'])] = 100 * float((ref['base'][sel] * (rs[sel] >= tp)).sum() / ref['base'][sel].sum())
        # ---- one row per policy ----
        cr = pd.DataFrame([r for r in coh_rows if r['det'] == det])
        pol_kind = {'P1': 'JamA', 'P2': 'POOL5', 'P2-LSO': 'POOL', 'P3': 'MAX5', 'P4': 'MAX'}
        pol_tpr = {'P1': 'JamA', 'P2': 'POOL(5)', 'P2-LSO': 'POOL(5)', 'P3': 'MAX(5)', 'P4': 'MAX(5)'}
        pol_tau = {'P1': pt_R['jamendo_A'], 'P2': th[ALL5]['pool_pt'], 'P2-LSO': th[ALL5]['pool_pt'], 'P3': th[ALL5]['max_pt'], 'P4': th[ALL5]['max_pt']}
        for pol, pdesc in POL:
            w = dict(det=det, name=name, policy=pol, policy_desc=pdesc, B=B, tau_deployed=pol_tau[pol])
            vals = {}
            for c, cl, key in FOUR:
                x = cr[(cr.cohort == c) & (cr.kind == pol_kind[pol]) & (cr.sensitivity == 0)].iloc[0]
                w.update({f'tau_{key}': x.tau, f'refs_{key}': x.refs, f'fpr_{key}': x.fpr, f'lo_{key}': x.ci_lo, f'hi_{key}': x.ci_hi,
                          f'wilhi_{key}': x.wil_hi_clusters, f'clusters_{key}': int(x.clusters), f'files_{key}': int(x.n_files)})
                vals[c] = x
            wc = max(vals, key=lambda c: vals[c].fpr)  # worst point estimate (ties: the first in FOUR)
            sim = np.max(np.stack([boots[(c, pol)] for c, _, _ in FOUR]), axis=0)
            w.update(worst_cohort=dict((c, cl) for c, cl, _ in FOUR)[wc], worst_fpr=vals[wc].fpr,
                     worst_coh_lo=vals[wc].ci_lo, worst_coh_hi=vals[wc].ci_hi,
                     worst_sim_lo=np.percentile(sim, 2.5), worst_sim_hi=np.percentile(sim, 97.5),
                     below6_point=bool(vals[wc].fpr < LIM), below6_upper_sim=bool(np.percentile(sim, 97.5) < LIM),
                     below6_upper_coh=bool(vals[wc].ci_hi < LIM), all4_upper_below6=bool(all(vals[c].ci_hi < LIM for c in vals)),
                     n_cohorts_ge6=int(sum(vals[c].fpr >= LIM for c in vals)))
            if pol in ('P2-LSO', 'P4'):  # sensitivity: Jamendo-B with Jamendo-A also left out (the sensitivity row of unit_multiref)
                xs = cr[(cr.cohort == 'jamendo_B') & (cr.kind == pol_kind[pol]) & (cr.sensitivity == 1)].iloc[0]
                others = max(vals[c].fpr for c, _, _ in FOUR if c != 'jamendo_B')
                w.update(sens_jamB_noJamA_tau=xs.tau, sens_jamB_noJamA_fpr=xs.fpr, sens_jamB_noJamA_lo=xs.ci_lo, sens_jamB_noJamA_hi=xs.ci_hi,
                         below6_point_sensJamB=bool(max(others, xs.fpr) < LIM))
            k = pol_tpr[pol]; d = tb[k] - tb['JamA']
            w.update(tpr=tpt[k], tpr_lo=np.percentile(tb[k], 2.5), tpr_hi=np.percentile(tb[k], 97.5),
                     tpr_cost_vs_P1=tpt[k] - tpt['JamA'], cost_lo=np.percentile(d, 2.5), cost_hi=np.percentile(d, 97.5))
            for (kk, g), v in gen_tpr.items():
                if kk == k: w[f'tpr_{g}'] = v
            wide.append(w)
        print(f'[{det}] done {time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(wide)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--B', type=int, default=2000); ap.add_argument('--dets', default='', help='paper or directory names, comma-separated')
    a = ap.parse_args(); dets = set(C.parse_names(a.dets))
    wide = run(a.B, dets)
    wide.to_csv(f'{TB}/policy_compare.csv', index=False)
    ft = wide.det.isin(FINETUNED)
    print('policy: detectors with worst point FPR < 6% / simultaneous upper bound < 6%; TPR cost vs P1, median (range) [fine-tuned median]')
    for pol, pdesc in POL:
        g = wide[wide.policy == pol]; c = g.tpr_cost_vs_P1; cf = wide[(wide.policy == pol) & ft].tpr_cost_vs_P1
        print(f'  {pol:6s} {int(g.below6_point.sum())}/{len(g)}  {int(g.below6_upper_sim.sum())}/{len(g)}  '
              f'{c.median():+.1f} ({c.min():+.1f} to {c.max():+.1f}) [{cf.median():+.1f}]  {pdesc}')
    print('POLICY_COMPARE_DONE', len(wide))


if __name__ == '__main__':
    main()
