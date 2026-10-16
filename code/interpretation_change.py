"""How the conclusion "the detector fails" changes from a common naive evaluation to the controlled protocol (post hoc;
calibration source x calibration unit).

The deployment unit is fixed to single windows (k_dep = 1); only the calibration source and the calibration unit change
(alpha = 2%, track-weighted quantile):
  calibration source: FMA-dev (fma_dev800; unseen tracks of the training platform, 755 of 800 by training artists) versus
                      Jamendo-A (platform- and artist-disjoint)
  calibration unit:   track aggregate (trim20 of all windows of the file = unit_provenance FileSet.units('file'); Jamendo tracks
                      mostly have 12 windows, the 30-s FMA-dev clips 5 EAT windows) versus single window
  A = FMA-dev + track, B = FMA-dev + window, C = Jamendo-A + track, D = Jamendo-A + window (the protocol of the paper =
  public_window_rates tau_JamA@2%).
  Sensitivity C12 = Jamendo-A, 12 consecutive windows (ArtifactNet 7) trim20 = the k_cal = 12 definition of unit_matrix (tracks
  with fewer windows left out); it differs from C only for detectors whose 'file' tau differs.
Reported per detector and condition: window FPR of Jamendo-B, AVA-SM, AudioSet-SM and OBM-H with 95% intervals, the number of
  cohorts above 6% (3 alpha) and the window TPR at the same tau on the four unseen generator families (lyria3, heartmula,
  minimax_2.6, mureka_v9; generators and files weighted equally = unit_multiref_tpr).
  Speech cohorts = AVA-SM, AudioSet-SM, OBM-H (any_speech_gt6: at least one above 6%).
  Verdict under D (rule 'orig' of public_window_g0.csv: ratio R = FPR_H / FPR_Jamendo-B >= 3 and 2.5th percentile of the
  same-replicate ratio > 1, at alpha = 1, 2 and 5% separately; named in-the-wild cohorts MUSDB18, GTZAN, OBM-H, AVA-SM, AVA-MO,
  AudioSet-SM, AudioSet-MO without sub-classes or [long]/[short] variants): the rule met at all three alpha in one cohort =
  failure, at one or two = borderline, otherwise none.
Existing tables first: the FPR of D and B come from public_window_rates.csv (B = 10,000), the Jamendo-B and OBM-H cells of C and
  C12 from unit_matrix.csv (k_cal = 12, k_dep = 1), the TPR of D from unit_multiref_tpr.csv (JamA, B = 2,000), each only when
  the point estimate, tau and number of files equal the recomputation below within 1e-9 (src_* and tpr_src name the source).
  Every other cell is computed here (B = 2,000, seed 0 per detector; in every replicate the calibration artists are resampled
  and tau re-estimated, evaluation clusters are resampled independently, the unseen reference within generators).
  Point estimates do not depend on the source; only the intervals do.
Inputs: scores/windows/<detector>/ and, from --src (out/ by default, or results/), public_window_rates.csv, public_window_g0.csv
  (code/agg_window.py), unit_matrix.csv (code/unit_provenance.py) and unit_multiref_tpr.csv (code/unit_multiref.py)
Output (out/): interpretation_change.csv (detector x condition)
usage: python code/interpretation_change.py [--B 2000] [--dets "B0,C0"] [--src out|results]
"""
import os, sys, argparse, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from unit_provenance import FileSet, load_det, deploy, calib, wilson, ref_set, ref_boot_w, unseen_gens, BATCH  # noqa: E402
from policy_compare import DETS, FINETUNED, FOUR  # noqa: E402  (the detectors and the four evaluation cohorts of the policy comparison)

R = C.R; TB = C.TB
SPEECH = ['ava_sm5', 'audioset_sm', 'openbmat_human_musicbearing']
NAMED = ['musdb18', 'gtzan', 'openbmat_human_musicbearing', 'ava_sm5', 'ava_mo', 'audioset_sm', 'audioset_mo']
NAMED_LAB = {'musdb18': 'MUSDB18', 'gtzan': 'GTZAN', 'openbmat_human_musicbearing': 'OBM-H', 'ava_sm5': 'AVA-SM', 'ava_mo': 'AVA-MO',
             'audioset_sm': 'AudioSet-SM', 'audioset_mo': 'AudioSet-MO'}
COND = [('A', 'fma_dev800', 'file', 'FMA-dev + track aggregate (naive evaluation)'), ('B', 'fma_dev800', 1, 'FMA-dev + window'),
        ('C', 'jamendo_A', 'file', 'Jamendo-A + track aggregate'), ('D', 'jamendo_A', 1, 'Jamendo-A + window (protocol of the paper)'),
        ('C12', 'jamendo_A', 'k12', '(sensitivity) Jamendo-A + 12 consecutive windows = unit_matrix k_cal 12')]
LIM = 6.0
TOL = 1e-9


def compute(B, dets):
    reg = C.load_registry().set_index('name')
    metas = {p: C.read_list(f'{R}/{p}').assign(_row=lambda m: np.arange(len(m))) for p in reg.list.unique()}

    def fileset(name, c, with_strata=False):
        r = reg.loc[name]; m = metas[r['list']]; sub = m.query(r['subset']) if r['subset'] else m
        st = sub['group'].astype(str).values if with_strata and 'group' in sub else None
        return FileSet(c, sub._row.values, sub[r['cluster']].astype(str).values, st)

    names = ['fma_dev800', 'jamendo_A'] + [c for c, _, _ in FOUR]
    rows, trows, t0 = [], [], time.time()
    for det, nm in DETS:
        if dets and det not in dets: continue
        need = {reg.loc[n, 'dump'] for n in names + ['newgen', 'music8k']}
        ch = load_det(det, need)
        if not need <= set(ch): print(f'[skip] {det}: missing scores {sorted(need - set(ch))}', flush=True); continue
        rng = np.random.default_rng(0)
        FS = {n: fileset(n, ch[reg.loc[n, 'dump']]) for n in names}
        kmax = 7 if 'ArtifactNet' in det else 12
        Tp, Tb, ncal = [], [], []
        for cond, src, unit, _ in COND:
            k = kmax if unit == 'k12' else unit
            T, Tbk, n = calib(FS[src], k, rng, B); Tp.append(T); Tb.append(Tbk); ncal.append(n)
        Tp = np.array(Tp); Tb = np.stack(Tb)
        for c, cl, key in FOUR:
            Co, nf, nw, ncl = deploy(FS[c], 1)
            pt = Co.rate(Tp); bt = Co.boot(Tb, rng, B)  # the five conditions share the cohort resampling (paired)
            for i, (cond, src, unit, desc) in enumerate(COND):
                lo, hi = wilson(pt[i], ncl)
                rows.append(dict(det=det, name=nm, condition=cond, source=src, cal_unit=str(unit), tau=Tp[i], cal_n=ncal[i], cohort=c, cohort_label=cl,
                                 n_files=nf, n_windows=nw, clusters=ncl, fpr=100 * pt[i], ci_lo=100 * np.percentile(bt[i], 2.5),
                                 ci_hi=100 * np.percentile(bt[i], 97.5), wil_hi_clusters=100 * hi))
        ref = ref_set(ch, lambda n, c_: fileset(n, c_, True), reg, unseen_gens()); rs = ref['s']
        tb = [[] for _ in COND]
        for b0 in range(0, B, BATCH):
            m = min(B, b0 + BATCH) - b0; wr = ref_boot_w(ref, m, rng); den = wr.sum(1)
            for i in range(len(COND)):
                tb[i].append((wr * (rs[None, :] >= Tb[i, b0:b0 + m, None])).sum(1) / den)
        for i, (cond, src, unit, desc) in enumerate(COND):
            v = 100 * np.concatenate(tb[i]); r = dict(det=det, name=nm, condition=cond, tau=Tp[i], tpr=100 * float((ref['base'] * (rs >= Tp[i])).sum()),
                                                      tpr_lo=np.percentile(v, 2.5), tpr_hi=np.percentile(v, 97.5))
            for gi, it in enumerate(ref['info']):
                sel = ref['fgen'][ref['fidx']] == gi
                r[f"tpr_{it['generator']}"] = 100 * float((ref['base'][sel] * (rs[sel] >= Tp[i])).sum() / ref['base'][sel].sum())
            trows.append(r)
        print(f'[{det}] done {time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(rows), pd.DataFrame(trows)


def assemble(comp, ctpr, src, B):
    """Use the existing table value when its point estimate, tau and file count equal the recomputation; otherwise the new value."""
    rd = lambda f: pd.read_csv(f'{src}/{f}', keep_default_na=False, na_values=[''])
    pw = rd('public_window_rates.csv'); um = rd('unit_matrix.csv'); mt = rd('unit_multiref_tpr.csv')
    new = f'new(B={B:,})'
    out, used = [], []
    for _, r in comp.iterrows():
        x = r.to_dict(); x.update(src=new, fpr_new=r.fpr, lo_new=r.ci_lo, hi_new=r.ci_hi)
        cand = None
        if r.condition in ('B', 'D'):
            thr = 'tau_JamA@2%' if r.condition == 'D' else 'tau_FMAov@2%'
            e = pw[(pw.model == r.det) & (pw.thr == thr) & (pw.cohort == r.cohort)]
            if not e.empty: cand = ('public_window_rates(B=10,000)', e.iloc[0], 'rate', 'n')
        elif r.condition in ('C', 'C12') and r.cohort in ('jamendo_B', 'openbmat_human_musicbearing'):
            kc = 7 if 'ArtifactNet' in r.det else 12
            e = um[(um.det == r.det) & (um.k_cal == kc) & (um.k_dep == 1) & (um.cohort == r.cohort)]
            if not e.empty: cand = ('unit_matrix k_cal12→k_dep1(B=10,000)', e.iloc[0], 'rate', 'n_files')
        if cand is not None:
            lab, e, col, ncol = cand
            ok = abs(float(e[col]) - r.fpr) <= TOL and abs(float(e['tau']) - r.tau) <= TOL and int(e[ncol]) == int(r.n_files)
            used.append((lab, ok))
            if ok:
                x.update(src=lab, fpr=float(e[col]), ci_lo=float(e['ci_lo']), ci_hi=float(e['ci_hi']),
                         wil_hi_clusters=float(e['wil_hi_clusters']) if 'wil_hi_clusters' in e else r.wil_hi_clusters)
        out.append(x)
    fpr = pd.DataFrame(out)
    tp = []
    for _, r in ctpr.iterrows():
        x = r.to_dict(); x.update(tpr_src=new, tpr_new=r.tpr)
        if r.condition == 'D':
            e = mt[(mt.det == r.det) & (mt.kind == 'JamA')]
            if not e.empty:
                e = e.iloc[0]; ok = abs(float(e.tpr) - r.tpr) <= TOL and abs(float(e.tau) - r.tau) <= TOL
                used.append(('unit_multiref_tpr(B=2,000)', ok))
                if ok: x.update(tpr_src='unit_multiref_tpr(B=2,000)', tpr=float(e.tpr), tpr_lo=float(e.ci_lo), tpr_hi=float(e.ci_hi))
        tp.append(x)
    u = pd.DataFrame(used, columns=['table', 'used'])
    print('existing-table cells used / matched candidates: ' + ', '.join(f'{t} {int(g.used.sum())}/{len(g)}' for t, g in u.groupby('table', sort=False)), flush=True)
    return fpr, pd.DataFrame(tp)


def verdicts(src, dets):
    g0 = pd.read_csv(f'{src}/public_window_g0.csv', keep_default_na=False, na_values=[''])
    g0['pass_orig'] = g0.pass_orig.astype(str).str.lower().isin(['true', '1'])
    TH = ('tau_JamA@1%', 'tau_JamA@2%', 'tau_JamA@5%')
    out = {}
    for det in dets:
        g = g0[(g0.model == det) & g0.thr.isin(TH) & g0.cohort.isin(NAMED)]
        cnt = g.groupby('cohort').pass_orig.sum().reindex(NAMED).fillna(0).astype(int)
        mx = int(cnt.max()) if len(cnt) else 0
        v = 'failure' if mx == 3 else ('borderline' if mx >= 1 else 'none')
        sp = cnt.reindex(SPEECH); mxs = int(sp.max())
        vs = 'failure' if mxs == 3 else ('borderline' if mxs >= 1 else 'none')
        det_txt = []
        gl = g0[(g0.model == det) & g0.thr.isin(TH)]
        for c in NAMED:
            if cnt[c] > 0:
                al = [t.split('@')[1] for t in TH if bool(g[(g.cohort == c) & (g.thr == t)].pass_orig.any())]
                r2 = g[(g.cohort == c) & (g.thr == 'tau_JamA@2%')]
                rr = '' if r2.empty else f' (2%: R̂ {r2.iloc[0].ratio:.2f} [{r2.iloc[0].ratio_lo:.2f}–{r2.iloc[0].ratio_hi:.2f}])'
                lg = gl[gl.cohort == f'{c}[long]']  # unpadded variant (for reference; not used by the verdict)
                lt = '' if lg.empty else f", unpadded [long] variant meets α={'/'.join(t.split('@')[1] for t in lg[lg.pass_orig].thr) or 'none'}"
                det_txt.append(f"{NAMED_LAB[c]} α={'/'.join(al)}{rr}{lt}")
        g2 = g[g.thr == 'tau_JamA@2%'].sort_values('ratio', ascending=False)  # cohort with the largest ratio at 2% (for reference)
        top = '' if g2.empty else f"{NAMED_LAB[g2.iloc[0].cohort]} R̂ {g2.iloc[0].ratio:.2f} [{g2.iloc[0].ratio_lo:.2f}–{g2.iloc[0].ratio_hi:.2f}]"
        out[det] = dict(verdict_D=v, verdict_D_speech=vs, verdict_detail='; '.join(det_txt) or 'none', max_ratio_2pct=top,
                        n_named=int(g.cohort.nunique()), jamB_2pct=float(g[g.thr == 'tau_JamA@2%'].jamB.iloc[0]) if not g2.empty else np.nan)
    return out


def wide_table(fpr, tpr, ver):
    rows = []
    for (det, cond), g in fpr.groupby(['det', 'condition'], sort=False):
        g = g.set_index('cohort'); x0 = g.iloc[0]
        w = dict(det=det, name=x0['name'], condition=cond, desc=dict((c, d) for c, _, _, d in COND)[cond], source=x0.source, cal_unit=x0.cal_unit,
                 tau=x0.tau, cal_n=int(x0.cal_n))
        for c, cl, key in FOUR:
            y = g.loc[c]
            w.update({f'fpr_{key}': y.fpr, f'lo_{key}': y.ci_lo, f'hi_{key}': y.ci_hi, f'wilhi_{key}': y.wil_hi_clusters, f'src_{key}': y.src,
                      f'fprnew_{key}': y.fpr_new, f'lonew_{key}': y.lo_new, f'hinew_{key}': y.hi_new})
        f4 = {c: g.loc[c].fpr for c, _, _ in FOUR}; lo4 = {c: g.loc[c].ci_lo for c, _, _ in FOUR}
        ws = max(SPEECH, key=lambda c: f4[c])
        w.update(n_gt6=int(sum(v > LIM for v in f4.values())), n_gt6_sig=int(sum(v > LIM for v in lo4.values())),
                 n_gt6_speech=int(sum(f4[c] > LIM for c in SPEECH)), any_speech_gt6=bool(any(f4[c] > LIM for c in SPEECH)),
                 any_speech_gt6_sig=bool(any(lo4[c] > LIM for c in SPEECH)),
                 worst_speech_cohort=dict((c, cl) for c, cl, _ in FOUR)[ws], worst_speech_fpr=f4[ws],
                 worst_speech_lo=g.loc[ws].ci_lo, worst_speech_hi=g.loc[ws].ci_hi)
        t = tpr[(tpr.det == det) & (tpr.condition == cond)].iloc[0]
        w.update(tpr=t.tpr, tpr_lo=t.tpr_lo, tpr_hi=t.tpr_hi, tpr_src=t.tpr_src, **{k: t[k] for k in t.index if k.startswith('tpr_') and k not in ('tpr_lo', 'tpr_hi', 'tpr_src', 'tpr_new')})
        if cond == 'D': w.update(ver[det])
        rows.append(w)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--B', type=int, default=2000); ap.add_argument('--dets', default='', help='paper or directory names, comma-separated')
    ap.add_argument('--src', default='out', choices=['out', 'results'], help='folder of the existing tables (out/ = this run, results/ = frozen)')
    a = ap.parse_args(); dets = set(C.parse_names(a.dets))
    src = TB if a.src == 'out' else str(C.RESULTS)
    miss = [f for f in ('public_window_rates.csv', 'public_window_g0.csv', 'unit_matrix.csv', 'unit_multiref_tpr.csv') if not os.path.exists(f'{src}/{f}')]
    assert not miss, f'{miss} not in {src}: run code/agg_window.py, code/unit_provenance.py and code/unit_multiref.py first, or use --src results'
    comp, ctpr = compute(a.B, dets)
    fpr, tpr = assemble(comp, ctpr, src, a.B)
    W = wide_table(fpr, tpr, verdicts(src, list(dict.fromkeys(fpr.det))))
    W.to_csv(f'{TB}/interpretation_change.csv', index=False)
    print('condition: detectors with a speech cohort above 6% (of', W.det.nunique(), ')')
    for cond, _, _, desc in COND:
        g = W[W.condition == cond]
        print(f'  {cond:3s} {int(g.any_speech_gt6.sum())}  {", ".join(g[g.any_speech_gt6].name) or "none"}  ({desc}); '
              f'unseen TPR median, fine-tuned {g[g.det.isin(FINETUNED)].tpr.median():.1f}')
    d = W[W.condition == 'D']
    print('verdict under D: ' + ', '.join(f'{r.name} {r.verdict_D}' for r in d.itertuples()))
    print('INTERP_DONE', len(W))


if __name__ == '__main__':
    main()
