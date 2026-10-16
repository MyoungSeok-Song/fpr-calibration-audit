"""Exposure-specificity crossover of the training additions (post hoc analysis of the calibration-source intervention).

Question: does the effect of B0+broad come from exposing the source of the reference (the Jamendo platform) in training, or
from any change in the amount or composition of the human training data? Predictions, recorded before the computation:
  (1) B0+broad (about 12,000 human tracks by other Jamendo artists) raises the external FPR mainly at tau_JamA, not at the
      other references;
  (2) B0+AVA (80 AVA films outside the evaluation films, about 5.5 h) raises it mainly at tau_AVA-MO, not at tau_JamA;
  (3) B0+SU (Suno/Udio compilations, AI side) shows no reference-specific pattern.
Control = B0+0, noise floor = B0+0 (seed 2). Single-seed one-epoch variants and a post hoc analysis: descriptive evidence.
Definitions are those of f2_platform_overlap.py (its constants and thr_set are imported): window unit (k = 1), alpha = 2%,
track-weighted (1/n_f) quantile tau_R for R in {Jamendo-A (JamA), MUSDB18, GTZAN, AVA-MO, AudioSet-MO}, MAX4 and FMAov as in
f2, the same cohorts, B (2,000) and seed (default_rng(0) restarted for every detector, same call order), so the detectors are
resampled in pairs. Delta = FPR_T(tau_R^T) - FPR_B0+0(tau_R^B0+0); interval = 2.5 / 97.5 percentiles of the same-replicate
differences.
  The same-source cells left out by f2 (AVA-SM and AVA-SM long at tau_AVA-MO, AudioSet-SM at tau_AudioSet-MO) are computed in
  the same call; the random draws of Cohort.boot do not depend on the number of thresholds, so the f2 cells are unchanged.
  They are flagged same_source = True (in_f2 = False). A reference cohort is not evaluated at its own tau.
TPR: the four unseen generator families (lyria3, heartmula, minimax_2.6, mureka_v9; generators and files weighted equally, as
  in f2 and unit_multiref) at every tau, in the random stream of f2. SONICS fake (2,000 files, resampled within generators,
  track-weighted window TPR as unit_gen 'all') after the f2 stream with a separate seed (101), the same for every detector.
Threshold-free discrimination: (a) paired AI-OpenBMAT window AUROC on music-bearing windows (code path, seed and B of
  agg_window_variants.py --parts auroc) and (b) window AUROC of each human cohort versus the four unseen families (weights as in
  unit_rank_unseen; seed 202, B), both with paired-difference intervals against B0+0.
Summary per (T, R): external cohorts = human cohorts that share no source with R, from U = {Jamendo-B, AVA-SM, AudioSet-SM,
  OBM-H, MUSDB18, GTZAN, AVA-MO, AudioSet-MO, LibriSpeech long} (AVA-SM long is a subset of AVA-SM and is not in U).
  variant as_specified = all external cohorts; strict_no_T_source = also without the cohorts that share a source with the
  training addition of T (B0+broad: Jamendo-B; B0+AVA: AVA-SM, AVA-MO); music_only = without LibriSpeech long;
  same_source_cell = one row per same-source cell (Jamendo-B at JamA, AVA-SM and AVA-SM long at AVA-MO, AudioSet-SM at
  AudioSet-MO). Mean Delta with a paired interval (cohort mean per replicate), the largest Delta (its cohort and that cell's
  interval), and the number of cohorts whose interval lies above / below 0.
Crossover 2x2: on the common set Cx = {AudioSet-SM, OBM-H, MUSDB18, GTZAN, AudioSet-MO, LibriSpeech long}, which shares no
  source with Jamendo-A, AVA-MO, Jamendo or AVA: mean Delta at tau_JamA and at tau_AVA-MO, their difference, and the interaction
  [Delta(B0+broad, JamA) - Delta(B0+broad, AVA-MO)] - [Delta(B0+AVA, JamA) - Delta(B0+AVA, AVA-MO)] with a paired interval.
Inputs: scores/windows/<detector>/, out/openbmat_window_classes.csv (code/openbmat_window_classes.py)
Outputs (out/): f2x_exposure_crossover.csv (detector x cohort x threshold, with Delta), f2x_exposure_crossover_summary.csv,
  f2x_exposure_crossover_cross2x2.csv, f2x_exposure_crossover_tpr.csv, f2x_exposure_crossover_auroc.csv
usage: python code/f2x_exposure_crossover.py [--B 2000] [--B_obm 10000]
"""
import os, sys, argparse, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from f2_platform_overlap import REFS, DISJ, LAB, TARGETS, SAME, THRS, thr_set  # noqa: E402  (main is not run)
from unit_provenance import FileSet, load_det, calib, deploy, ref_set, ref_boot_w, unseen_gens, BATCH  # noqa: E402
from agg_window_variants import seed_of, cohort  # noqa: E402  (paired AI-OpenBMAT AUROC)
from auroc import wauc  # noqa: E402

R = C.R; TB = C.TB
DETS = [('B0+0', 'B0+0'), ('B0+broad', 'B0+broad'), ('B0+0 (seed 2)', 'B0+0 (seed 2)'), ('B0+AVA', 'B0+AVA'), ('B0+SU', 'B0+SU')]
CTRL = 'B0+0'; TREAT = ['B0+broad', 'B0+AVA', 'B0+SU']; NOISE = 'B0+0 (seed 2)'
SINGLE = ['JamA', 'MUSDB18', 'GTZAN', 'AVA-MO', 'AudioSet-MO']  # single references R
TAUKEYS = SINGLE + ['MAX4', 'FMAov']
REF_OF = {'JamA': 'jamendo_A', 'MUSDB18': 'musdb18', 'GTZAN': 'gtzan', 'AVA-MO': 'ava_mo', 'AudioSet-MO': 'audioset_mo', 'FMAov': 'fma_dev800'}
U = ['jamendo_B', 'ava_sm5', 'audioset_sm', 'openbmat_human_musicbearing', 'musdb18', 'gtzan', 'ava_mo', 'audioset_mo', 'librispeech_long']
SRC_R = {'JamA': {'jamendo_B'}, 'MUSDB18': {'musdb18'}, 'GTZAN': {'gtzan'}, 'AVA-MO': {'ava_mo', 'ava_sm5', 'ava_sm_long'},
         'AudioSet-MO': {'audioset_mo', 'audioset_sm'}}  # cohorts that share a source with reference R
EXP_T = {'B0+broad': {'jamendo_B'}, 'B0+AVA': {'ava_sm5', 'ava_sm_long', 'ava_mo'}, 'B0+SU': set(), 'B0+0 (seed 2)': set()}  # ... with T's addition
CROSS = ['audioset_sm', 'openbmat_human_musicbearing', 'musdb18', 'gtzan', 'audioset_mo', 'librispeech_long']
SAMESRC = {('jamendo_B', 'JamA'), ('ava_sm5', 'AVA-MO'), ('ava_sm_long', 'AVA-MO'), ('audioset_sm', 'AudioSet-MO')}
AU_COH = ['jamendo_B', 'openbmat_human_musicbearing', 'ava_sm5', 'audioset_sm', 'musdb18', 'gtzan', 'ava_mo', 'audioset_mo']
INTERACTION = ('B0+broad vs B0+AVA', 'interaction [(B0+broad: JamA - AVA-MO) - (B0+AVA: JamA - AVA-MO)]')
pc = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--B', type=int, default=2000); ap.add_argument('--B_obm', type=int, default=10000)
    a = ap.parse_args(); B = a.B; t0 = time.time()
    reg = C.load_registry().set_index('name')
    metas = {p: C.read_list(f'{R}/{p}').assign(_row=lambda m: np.arange(len(m))) for p in reg.list.unique()}

    def fileset(name, c, with_strata=False):  # the same definition as in f2_platform_overlap.main
        if name == 'librispeech_long':  # long LibriSpeech utterances (no padding)
            r = reg.loc['librispeech_tc']; m = metas[r['list']]
            longid = set(c[c.dur >= 10.24].ID.unique()); sub = m[m._row.isin(longid)]
            return FileSet(c, sub._row.values, sub[r['cluster']].astype(str).values, None)
        r = reg.loc[name]; m = metas[r['list']]; sub = m.query(r['subset']) if r['subset'] else m
        st = sub['group'].astype(str).values if with_strata and 'group' in sub else None
        return FileSet(c, sub._row.values, sub[r['cluster']].astype(str).values, st)
    dumpname = lambda n: reg.loc['librispeech_tc', 'dump'] if n == 'librispeech_long' else reg.loc[n, 'dump']
    names = sorted(set(REFS) | {n for n, _ in TARGETS})
    res, boots, tpr, tprb, aurows, aub = [], {}, [], {}, [], {}
    nf_ref = {}  # paired resampling requires the same files and windows per cohort for every detector
    for tag, dn in DETS:
        ch = load_det(tag, {dumpname(n) for n in names} | {'newgen', 'music8k', 'sonics_fake'})
        # ---------- the random stream of f2_platform_overlap (seed 0, same call order) ----------
        rng = np.random.default_rng(0)
        FS = {n: fileset(n, ch[dumpname(n)]) for n in names}
        for r in REFS:
            sc_, fi_ = FS[r].units(1)
            assert nf_ref.setdefault('ref_' + r, (len(sc_), len(FS[r].ids))) == (len(sc_), len(FS[r].ids)), (tag, r)
        cal = {r: calib(FS[r], 1, rng, B) for r in REFS}  # (tau, tau_b, n)
        for c, cl in TARGETS:
            ts = thr_set(c); Co, nf, nw, ncl = deploy(FS[c], 1)
            assert nf_ref.setdefault(c, (nf, nw, ncl)) == (nf, nw, ncl), (tag, c, nf_ref[c], (nf, nw, ncl))
            keys = [k for k in THRS if k in ts]
            refs_of = dict(ts); extra = []
            if c in SAME:  # the same-source single reference that f2 leaves out (reported separately)
                extra = [LAB[SAME[c]]]; refs_of[LAB[SAME[c]]] = [SAME[c]]
            allk = keys + extra
            Tpt = np.array([max(cal[r][0] for r in refs_of[k]) for k in allk])
            Tb = np.stack([np.max(np.stack([cal[r][1] for r in refs_of[k]]), axis=0) for k in allk])
            pt = Co.rate(Tpt); bt = Co.boot(Tb, rng, B)  # one multinomial draw per batch, whatever the number of thresholds
            for i, k in enumerate(allk):
                res.append(dict(det=tag, name=dn, cohort=c, cohort_label=cl, thr=k, refs='+'.join(LAB[r] for r in refs_of[k]), tau=Tpt[i], n_files=nf,
                                clusters=ncl, fpr=100 * pt[i], ci_lo=100 * np.percentile(bt[i], 2.5), ci_hi=100 * np.percentile(bt[i], 97.5),
                                in_f2=k in keys, same_source=(c, k) in SAMESRC))
                boots[(tag, c, k)] = (100 * pt[i], 100 * bt[i])
        ref = ref_set(ch, lambda n, c_: fileset(n, c_, True), reg, unseen_gens())
        rs = ref['s']
        TT = {k: (cal[REF_OF[k]][0], cal[REF_OF[k]][1]) for k in SINGLE + ['FMAov']}
        TT['MAX4'] = (max(cal[r][0] for r in DISJ), np.max(np.stack([cal[r][1] for r in DISJ]), axis=0))
        tb = {k: [] for k in TAUKEYS}
        for b0 in range(0, B, BATCH):
            m = min(B, b0 + BATCH) - b0; wr = ref_boot_w(ref, m, rng); den = wr.sum(1)
            for k in TAUKEYS:
                tb[k].append((wr * (rs[None, :] >= TT[k][1][b0:b0 + m, None])).sum(1) / den)
        for k in TAUKEYS:
            v = 100 * np.concatenate(tb[k]); p = 100 * float((ref['base'] * (rs >= TT[k][0])).sum())
            per = {it['generator']: 100 * float((ref['base'][ref['fgen'][ref['fidx']] == g] * (rs[ref['fgen'][ref['fidx']] == g] >= TT[k][0])).sum()) * ref['G']
                   for g, it in enumerate(ref['info'])}
            tpr.append(dict(det=tag, name=dn, set='unseen4', thr=k, tau=TT[k][0], tpr=p, ci_lo=pc(v)[0], ci_hi=pc(v)[1],
                            **{f'tpr_{g}': x for g, x in per.items()}))
            tprb[(tag, 'unseen4', k)] = (p, v)
        # ---------- outside the f2 stream (separate seeds, the same for every detector -> paired) ----------
        rng_s = np.random.default_rng(101)
        fsS = fileset('sonics_fake', ch[dumpname('sonics_fake')], True)
        Cs, n_fl, nws, ncls = deploy(fsS, 1)
        Tp_all = np.array([TT[k][0] for k in TAUKEYS]); Tb_all = np.stack([TT[k][1] for k in TAUKEYS])
        ps = Cs.rate(Tp_all); bs = Cs.boot(Tb_all, rng_s, B)
        for i, k in enumerate(TAUKEYS):
            tpr.append(dict(det=tag, name=dn, set='sonics_fake', thr=k, tau=Tp_all[i], tpr=100 * ps[i], ci_lo=100 * pc(bs[i])[0], ci_hi=100 * pc(bs[i])[1]))
            tprb[(tag, 'sonics_fake', k)] = (100 * ps[i], 100 * bs[i])
        # threshold-free discrimination (b): window AUROC of a human cohort versus the four unseen families (rank_unseen weights)
        rng_a = np.random.default_rng(202)
        for cn in AU_COH:
            hs = FS[cn]; hsc, hfi = hs.units(1); hw = 1.0 / np.bincount(hfi)[hfi]
            s = np.r_[hsc, rs]; y = np.r_[np.zeros(len(hsc)), np.ones(len(rs))]
            au = float(wauc(s, y, np.r_[hw, ref['base']][None, :])[0])
            ccode = pd.factorize(pd.Series(hs.clus[hfi]))[0]; Gh = int(ccode.max()) + 1; vals = []
            for b0 in range(0, B, BATCH):
                m = min(B, b0 + BATCH) - b0
                wh = rng_a.multinomial(Gh, np.full(Gh, 1.0 / Gh), size=m)[:, ccode] * hw[None, :]
                wr = ref_boot_w(ref, m, rng_a)
                vals.append(wauc(s, y, np.concatenate([wh, wr], axis=1)))
            vals = np.concatenate(vals)
            aurows.append(dict(det=tag, name=dn, measure='window AUROC human cohort vs unseen4', cohort=cn, B=B, auroc=au, lo=pc(vals)[0], hi=pc(vals)[1]))
            aub[(tag, 'unseen4', cn)] = (au, vals)
        print(f'[{tag}] done {time.time() - t0:.0f}s', flush=True)
    # threshold-free discrimination (a): paired AI-OpenBMAT AUROC (music-bearing windows)
    obm_auc(reg, metas, a.B_obm, aurows, aub)
    # ---------- tables ----------
    d = pd.DataFrame(res)
    D = []
    for (tag, c, k), (p, v) in boots.items():
        if tag == CTRL: continue
        p0, v0 = boots[(CTRL, c, k)]; dv = v - v0
        D.append(dict(det=tag, cohort=c, thr=k, delta=p - p0, d_lo=pc(dv)[0], d_hi=pc(dv)[1]))
    d = d.merge(pd.DataFrame(D), on=['det', 'cohort', 'thr'], how='left')
    d.to_csv(f'{TB}/f2x_exposure_crossover.csv', index=False)
    t = pd.DataFrame(tpr); TD = []
    for (tag, st, k), (p, v) in tprb.items():
        if tag == CTRL: continue
        p0, v0 = tprb[(CTRL, st, k)]; dv = v - v0
        TD.append(dict(det=tag, set=st, thr=k, delta=p - p0, d_lo=pc(dv)[0], d_hi=pc(dv)[1]))
    t = t.merge(pd.DataFrame(TD), on=['det', 'set', 'thr'], how='left'); t.to_csv(f'{TB}/f2x_exposure_crossover_tpr.csv', index=False)
    au = pd.DataFrame(aurows); AD = []
    for (tag, st, cn), (p, v) in aub.items():
        if tag == CTRL or (CTRL, st, cn) not in aub: continue
        p0, v0 = aub[(CTRL, st, cn)]; dv = v - v0
        AD.append(dict(det=tag, set=st, cohort=cn, delta=p - p0, d_lo=pc(dv)[0], d_hi=pc(dv)[1]))
    au['set'] = np.where(au.measure.str.contains('OpenBMAT'), 'obm_paired', 'unseen4')
    au = au.merge(pd.DataFrame(AD), on=['det', 'set', 'cohort'], how='left'); au.to_csv(f'{TB}/f2x_exposure_crossover_auroc.csv', index=False)
    S, X = summarize(boots)
    S.to_csv(f'{TB}/f2x_exposure_crossover_summary.csv', index=False); X.to_csv(f'{TB}/f2x_exposure_crossover_cross2x2.csv', index=False)
    fd = lambda p, lo, hi: f'{p:+.1f} [{lo:+.1f}, {hi:+.1f}]'
    print('mean Delta over external cohorts (pp, as specified):')
    for tag in TREAT + [NOISE]:
        x = S[(S.det == tag) & (S.variant == 'as_specified')].set_index('thr')
        print(f'  {tag:14s} ' + '  '.join(f'{k} {fd(x.loc[k, "mean_delta"], x.loc[k, "mean_lo"], x.loc[k, "mean_hi"])}' for k in SINGLE))
    xi = X[X.det == INTERACTION[0]].iloc[0]
    print(f'{INTERACTION[1]} = {fd(xi.est, xi.lo, xi.hi)} pp')
    print('F2X_DONE', f'{time.time() - t0:.0f}s', flush=True)


def obm_auc(reg, metas, B, aurows, aub):
    """Paired AI-OpenBMAT window AUROC, music-bearing windows only: the 'auroc' part of agg_window_variants.py (same
    seed_of('auc', 'musicbearing'), batches of 500), for the five detectors."""
    regr = reg.reset_index()
    WC = f'{TB}/openbmat_window_classes.csv'
    assert os.path.exists(WC), 'run code/openbmat_window_classes.py first'
    wc = pd.read_csv(WC, dtype={'item_id': str})
    rows = {n: regr[regr.name == n].iloc[0] for n in ('openbmat_human', 'openbmat_ai')}
    for tag, dn in DETS:
        wins = {r['dump']: C.load_chunks(tag, r['dump'], cols=('ID', 'j', 'n', 'dur', 'short', 'lg')) for r in rows.values()}
        arms = {}
        for arm, cname in (('human', 'openbmat_human'), ('ai', 'openbmat_ai')):
            r = rows[cname]; j = cohort(r, wins, metas); m = metas[r['list']][['_row', 'item_id']].rename(columns={'_row': 'ID'})
            jj = j.merge(m, on='ID').merge(wc[wc.arm == arm][['item_id', 'j', 'pair', 'cls']], on=['item_id', 'j'], how='inner')
            arms[arm] = jj[['pair', 'j', 'lg', 'cluster_key', 'cls']]
        pw = arms['human'].merge(arms['ai'], on=['pair', 'j'], suffixes=('_h', '_a'))
        assert (pw.cls_h == pw.cls_a).all()
        sel = pw[pw.cls_h.isin(['mus', 'fg', 'bg', 'bgvl', 'sim'])]
        s = np.r_[sel.lg_h.values, sel.lg_a.values].astype(float); y = np.r_[np.zeros(len(sel)), np.ones(len(sel))]
        codes, uniq = pd.factorize(pd.concat([sel.cluster_key_h, sel.cluster_key_h]).astype(str)); G = len(uniq); vals = []
        rng = np.random.default_rng(seed_of('auc', 'musicbearing'))
        for a0 in range(0, B, 500):
            m_ = min(B, a0 + 500) - a0; cw = rng.multinomial(G, np.full(G, 1.0 / G), size=m_).astype(np.float64); vals.append(wauc(s, y, cw[:, codes]))
        vals = np.concatenate(vals); pt = float(wauc(s, y, np.ones((1, len(s))))[0])
        aurows.append(dict(det=tag, name=dn, measure='AI-OpenBMAT paired AUROC, music-bearing windows', cohort='openbmat_musicbearing_pairs', B=B,
                           auroc=pt, lo=pc(vals)[0], hi=pc(vals)[1], n_pairs=len(sel), clusters=G))
        aub[(tag, 'obm_paired', 'openbmat_musicbearing_pairs')] = (pt, vals)


def summarize(boots):
    """(T, R) summary: mean Delta (paired interval: cohort mean per replicate), largest Delta (cohort of the largest point
    estimate, with that cell's interval), same-source cells, and the crossover 2x2 on the common cohort set."""
    rows = []
    for tag in TREAT + [NOISE]:
        for k in SINGLE:
            base = [c for c in U if c not in SRC_R[k]]
            for variant, cs in (('as_specified', base), ('strict_no_T_source', [c for c in base if c not in EXP_T[tag]]),
                                ('music_only', [c for c in base if c != 'librispeech_long'])):
                cs = [c for c in cs if (tag, c, k) in boots]
                dp = np.array([boots[(tag, c, k)][0] - boots[(CTRL, c, k)][0] for c in cs])
                dv = np.stack([boots[(tag, c, k)][1] - boots[(CTRL, c, k)][1] for c in cs])
                mv = dv.mean(0); i = int(np.argmax(dp)); lo, hi = pc(dv[i])
                npos = int(sum(1 for j in range(len(cs)) if pc(dv[j])[0] > 0)); nneg = int(sum(1 for j in range(len(cs)) if pc(dv[j])[1] < 0))
                rows.append(dict(det=tag, thr=k, variant=variant, n_cohorts=len(cs), cohorts='+'.join(cs), mean_delta=float(dp.mean()), mean_lo=pc(mv)[0], mean_hi=pc(mv)[1],
                                 max_delta=float(dp[i]), max_cohort=cs[i], max_lo=lo, max_hi=hi, n_ci_above0=npos, n_ci_below0=nneg))
            for c in sorted({c for (c, kk) in SAMESRC if kk == k}):
                if (tag, c, k) not in boots: continue
                dv = boots[(tag, c, k)][1] - boots[(CTRL, c, k)][1]
                rows.append(dict(det=tag, thr=k, variant='same_source_cell', n_cohorts=1, cohorts=c, mean_delta=boots[(tag, c, k)][0] - boots[(CTRL, c, k)][0],
                                 mean_lo=pc(dv)[0], mean_hi=pc(dv)[1]))
    S = pd.DataFrame(rows)
    mv, X = {}, []
    for tag in TREAT + [NOISE]:
        for k in SINGLE:
            if any(c in SRC_R[k] for c in CROSS): continue
            dv = np.stack([boots[(tag, c, k)][1] - boots[(CTRL, c, k)][1] for c in CROSS]).mean(0)
            dp = float(np.mean([boots[(tag, c, k)][0] - boots[(CTRL, c, k)][0] for c in CROSS]))
            mv[(tag, k)] = (dp, dv)
            X.append(dict(det=tag, contrast=f'mean_delta@{k}', est=dp, lo=pc(dv)[0], hi=pc(dv)[1], cohorts='+'.join(CROSS)))
    for tag in TREAT + [NOISE]:
        p = mv[(tag, 'JamA')][0] - mv[(tag, 'AVA-MO')][0]; v = mv[(tag, 'JamA')][1] - mv[(tag, 'AVA-MO')][1]
        X.append(dict(det=tag, contrast='JamA_minus_AVA-MO', est=p, lo=pc(v)[0], hi=pc(v)[1], cohorts='+'.join(CROSS)))
    ib = (mv[('B0+broad', 'JamA')][1] - mv[('B0+broad', 'AVA-MO')][1]) - (mv[('B0+AVA', 'JamA')][1] - mv[('B0+AVA', 'AVA-MO')][1])
    ip = (mv[('B0+broad', 'JamA')][0] - mv[('B0+broad', 'AVA-MO')][0]) - (mv[('B0+AVA', 'JamA')][0] - mv[('B0+AVA', 'AVA-MO')][0])
    X.append(dict(det=INTERACTION[0], contrast=INTERACTION[1], est=ip, lo=pc(ib)[0], hi=pc(ib)[1], cohorts='+'.join(CROSS)))
    return S, pd.DataFrame(X)


if __name__ == '__main__':
    main()
