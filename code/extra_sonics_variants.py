"""Pre-specified replication with two further official SONICS checkpoints, SpecTTTra-alpha and SpecTTTra-beta (5-s
models), next to the paper's SpecTTTra (gamma, 5 s). Builds out/extra_sonics_variants.{md,csv} (frozen copy: results/).

The two detectors were scored after the main analysis, against four expectations recorded in advance:
  E1 unit inflation: Jamendo-B single-window FPR with a track-level tau (k_cal = 12 -> k_dep = 1) >= 2x the matched-unit FPR (1 -> 1).
  E2 small calibration-source effect for detectors not trained on FMA: the FMA-dev window tau gives at most about 2x the Jamendo-A
     window-tau FPR on Jamendo-B (SpecTTTra: 1.8x). Rule: <= 2.0 holds, 2.0-2.5 borderline, >= 2.5 fails.
  E3 coverage failure on new generator families: mean window TPR over the four unseen families at tau_JamA <= 20%.
  E4 no detector-specific in-the-wild excess: AVA-SM, AudioSet-SM and AI-OpenBMAT-H do not meet the failure criterion
     (point ratio to Jamendo-B >= 3 and 2.5th percentile of the ratio > 1) at alpha = 2%.
Inputs (--in, written by code/extra_sonics.sh with the same scripts and settings as the main tables, for the three detectors):
  unit_matrix, unit_prov, unit_gen, unit_rank_unseen          <- code/unit_provenance.py
  public_window_rates, public_window_g0                        <- code/agg_window.py
  unit_refswap(_summary)                                      <- code/unit_refswap.py
  unit_multiref(_summary), unit_multiref_tpr                  <- code/unit_multiref.py
Added here: intervals of the E1 and E2 ratios from a joint bootstrap (same two-stage cluster bootstrap with tau re-estimated in
  every replicate; one Jamendo-A artist draw serves both thresholds (k_cal = 1 and 12), one Jamendo-B artist draw serves numerator
  and denominator, FMA-dev artists are drawn separately). Ratio_b = FPR_num,b / FPR_den,b (denominator 0: inf if the numerator
  is > 0, dropped if both are 0); 2.5th and 97.5th percentiles. Point estimates must equal unit_matrix / unit_prov (asserted).
SpecTTTra check: every SpecTTTra row of the inputs must reproduce the point estimates of the frozen main tables (results/).
Outputs: <out>/extra_sonics_variants.{md,csv}, <in>/joint_ratio_boot.csv, <in>/repro_spectttra.csv
usage: python code/extra_sonics_variants.py --in out/extra_sonics [--out out] [--B 10000]
"""
import os, sys, argparse, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from unit_provenance import FileSet, load_det, deploy, wq_tau, BATCH  # noqa: E402
import compare_tables as CP  # noqa: E402  (key, point and interval columns of every result table)

DETS = [('SpecTTTra', 'SpecTTTra (gamma)'), ('SpecTTTra-alpha', 'SpecTTTra-alpha'), ('SpecTTTra-beta', 'SpecTTTra-beta')]
GAMMA = 'SpecTTTra'
ALPHA = 0.02
E4_COH = [('ava_sm5', 'AVA-SM'), ('audioset_sm', 'AudioSet-SM'), ('openbmat_human_musicbearing', 'AI-OpenBMAT-H')]
SPEECH_ONLY = [('ava_cs', 'AVA-S'), ('librispeech_tc', 'LibriSpeech'), ('openbmat_nomusic', 'AI-OpenBMAT speech only')]  # speech without music (nulls)
MUSIC_ONLY = [('ava_mo', 'AVA-MO'), ('audioset_mo', 'AudioSet-MO'), ('musdb18', 'MUSDB18'), ('gtzan', 'GTZAN')]  # music without speech
NEWFAM = [('newgen', 'lyria3', 'Lyria 3'), ('music8k', 'heartmula', 'HeartMuLa'), ('music8k', 'minimax_2.6', 'MiniMax 2.6'), ('music8k', 'mureka_v9', 'Mureka v9')]
# SpecTTTra values quoted in the paper: (description, table, row, column, value in the paper)
REPRO_TARGETS = [
    ('Jamendo-B window FPR @tau_JamA 2%', 'public_window_rates', dict(model=GAMMA, cohort='jamendo_B', thr='tau_JamA@2%'), 'rate', 2.4),
    ('FMA-dev window tau -> Jamendo-B FPR', 'unit_prov', dict(det=GAMMA, source='fma_dev800', unit='win', cohort='jamendo_B'), 'rate', 4.4),
    ('unit matrix k_cal=12 -> k_dep=1, Jamendo-B', 'unit_matrix', dict(det=GAMMA, k_cal=12, k_dep=1, cohort='jamendo_B'), 'rate', 10.6),
    ('new-family mean TPR @tau_JamA', 'unit_rank_unseen', dict(det=GAMMA, ref='unseen4', cohort='jamendo_B'), 'tpr_tauJamA', 4.4),
]
REPRO_TABLES = ['unit_matrix', 'unit_prov', 'unit_gen', 'unit_rank_unseen', 'public_window_rates', 'public_window_g0',
                'unit_refswap', 'unit_refswap_summary', 'unit_multiref', 'unit_multiref_summary', 'unit_multiref_tpr']


def rd(base, name):
    return pd.read_csv(f'{base}/{name}.csv', keep_default_na=False, na_values=[''])


def pick(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items(): m &= (df[k] == v).values
    x = df[m]
    assert len(x) == 1, (kw, len(x))
    return x.iloc[0]


def repro_check(src):
    """SpecTTTra rows of the inputs versus the frozen main tables: rows matched on the key columns, point estimates compared."""
    spec = {t[0]: t for t in CP.TABLES}
    rows = []
    for name in REPRO_TABLES:
        _, keys, vals, _ = spec[name]
        dc = 'model' if 'model' in keys else 'det'
        old = rd(C.RESULTS, name); new = rd(src, name)
        old = old[old[dc] == GAMMA]; new = new[new[dc] == GAMMA]
        m = old.merge(new, on=keys, suffixes=('_o', '_n'))
        dmax = 0.0
        for v in vals:
            a, b = CP.num(m[v + '_o']), CP.num(m[v + '_n'])
            same = (a.isna() & b.isna()) | (np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b)))
            d = (a - b).abs().where(~same, 0.0)
            dmax = max(dmax, float(d.max()) if len(d) else 0.0)
        rows.append(dict(table=name, rows_frozen=len(old), rows_matched=len(m), max_point_diff=dmax,
                         identical=bool(len(m) == len(old) and dmax <= 1e-9)))
    tg = []
    for lab, fn, kw, col, want in REPRO_TARGETS:
        v = float(pick(rd(src, fn), **kw)[col]); o = float(pick(rd(C.RESULTS, fn), **kw)[col])
        tg.append(dict(check=lab, table=fn, paper=want, frozen=o, reproduced=v, match=bool(abs(v - o) < 1e-9 and round(v, 1) == want)))
    return pd.DataFrame(rows), pd.DataFrame(tg)


def joint_ratio_boot(tag, B, seed, reg, metas, mat, prov):
    """Joint-bootstrap intervals of the E1 and E2 ratios; the point estimates must equal unit_matrix and unit_prov."""
    def fileset(name, c):
        r = reg.loc[name]; m = metas[r['list']]; sub = m.query(r['subset']) if r['subset'] else m
        return FileSet(c, sub._row.values, sub[r['cluster']].astype(str).values, None)
    ch = load_det(tag, {'jamendo2k', 'fma_dev800'})
    FA, FB, FF = fileset('jamendo_A', ch['jamendo2k']), fileset('jamendo_B', ch['jamendo2k']), fileset('fma_dev800', ch['fma_dev800'])

    def prep(fs, k):
        sc, fidx = fs.units(k)
        cnt = np.bincount(fidx, minlength=len(fs.ids)).astype(float); base = 1.0 / cnt[fidx]
        acode = pd.factorize(pd.Series(fs.clus))[0]; o = np.argsort(-sc, kind='stable')
        return dict(ss=sc[o], bw=base[o], cc=acode[fidx][o], T=wq_tau(sc, base, ALPHA)[0], G=int(acode.max()) + 1)

    def wtau(P, cw):
        cum = np.cumsum(cw[:, P['cc']] * P['bw'][None, :], axis=1); cnt = (cum <= ALPHA * cum[:, -1:] + 1e-12).sum(1)
        return np.where(cnt > 0, P['ss'][np.maximum(cnt - 1, 0)], np.inf)
    A1, A12, F1 = prep(FA, 1), prep(FA, 12), prep(FF, 1)
    assert A1['G'] == A12['G']
    Co, nf, nw, ncl = deploy(FB, 1)
    pt = Co.rate([A1['T'], A12['T'], F1['T']])
    m11 = pick(mat, det=tag, k_cal=1, k_dep=1, cohort='jamendo_B'); m121 = pick(mat, det=tag, k_cal=12, k_dep=1, cohort='jamendo_B')
    pf = pick(prov, det=tag, source='fma_dev800', unit='win', cohort='jamendo_B'); pj = pick(prov, det=tag, source='jamendo_A', unit='win', cohort='jamendo_B')
    for a, b in ((100 * pt[0], m11.rate), (100 * pt[1], m121.rate), (100 * pt[2], pf.rate), (100 * pt[0], pj.rate)):
        assert abs(a - b) < 1e-9, (tag, a, b)
    rng = np.random.default_rng(seed); TA1, TA12, TF1 = np.empty(B), np.empty(B), np.empty(B)
    for a in range(0, B, BATCH):
        m = min(B, a + BATCH) - a
        cwA = rng.multinomial(A1['G'], np.full(A1['G'], 1.0 / A1['G']), size=m).astype(np.float64)  # Jamendo-A artists: shared by both thresholds
        TA1[a:a + m] = wtau(A1, cwA); TA12[a:a + m] = wtau(A12, cwA)
        cwF = rng.multinomial(F1['G'], np.full(F1['G'], 1.0 / F1['G']), size=m).astype(np.float64)
        TF1[a:a + m] = wtau(F1, cwF)
    bt = Co.boot(np.stack([TA1, TA12, TF1]), rng, B)  # Jamendo-B artists: shared by the three FPRs

    def ratio(num, den):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(den > 0, num / den, np.where(num > 0, np.inf, np.nan))
        f = r[~np.isnan(r)]
        return np.percentile(f, 2.5), np.percentile(f, 97.5), float((f >= 2).mean())
    e1 = ratio(bt[1], bt[0]); e2 = ratio(bt[2], bt[0])
    pc = lambda v: (100 * np.percentile(v, 2.5), 100 * np.percentile(v, 97.5))
    return dict(det=tag, B=B, seed=seed, n_files_B=nf, n_windows_B=nw, clusters_B=ncl, artists_A=A1['G'], artists_FMA=F1['G'],
                tau_A1=A1['T'], tau_A12=A12['T'], tau_F1=F1['T'],
                fpr_1to1=100 * pt[0], fpr_1to1_lo=pc(bt[0])[0], fpr_1to1_hi=pc(bt[0])[1],
                fpr_12to1=100 * pt[1], fpr_12to1_lo=pc(bt[1])[0], fpr_12to1_hi=pc(bt[1])[1],
                fpr_fmawin=100 * pt[2], fpr_fmawin_lo=pc(bt[2])[0], fpr_fmawin_hi=pc(bt[2])[1],
                e1_ratio=pt[1] / pt[0], e1_lo=e1[0], e1_hi=e1[1], e1_boot_share_ge2=e1[2],
                e2_ratio=pt[2] / pt[0], e2_lo=e2[0], e2_hi=e2[1], e2_boot_share_ge2=e2[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', default=f'{C.TB}/extra_sonics', help='folder with the tables of code/extra_sonics.sh')
    ap.add_argument('--out', default=C.TB, help='folder for extra_sonics_variants.{md,csv}')
    ap.add_argument('--B', type=int, default=10000); ap.add_argument('--seed', type=int, default=20261015)
    a = ap.parse_args(); t0 = time.time(); src = a.src
    # 0) SpecTTTra reproduces the frozen main tables
    rep, tgt = repro_check(src)
    rep.to_csv(f'{src}/repro_spectttra.csv', index=False)
    print(rep.to_string(index=False)); print(tgt.to_string(index=False))
    if not (rep.identical.all() and tgt.match.all()):
        print('SPECTTTRA_REPRO_FAILED'); sys.exit(2)
    # 1) inputs
    mat, prov, gen = rd(src, 'unit_matrix'), rd(src, 'unit_prov'), rd(src, 'unit_gen')
    ru = rd(src, 'unit_rank_unseen'); rates, g0 = rd(src, 'public_window_rates'), rd(src, 'public_window_g0')
    for c in ('pass_strict', 'pass_orig', 'pass_ci_sep'): g0[c] = g0[c].astype(str).str.lower().isin(['true', '1'])
    rs, mrs, mrt = rd(src, 'unit_refswap_summary'), rd(src, 'unit_multiref_summary'), rd(src, 'unit_multiref_tpr')
    rs['all_below_3x'] = rs['all_below_3x'].astype(str).str.lower().isin(['true', '1'])
    mrs['all_below_3x'] = mrs['all_below_3x'].astype(str).str.lower().isin(['true', '1'])
    reg = C.load_registry().set_index('name')
    metas = {p: C.read_list(f'{C.R}/{p}').assign(_row=lambda m: np.arange(len(m))) for p in reg.list.unique()}
    # 2) joint bootstrap of the E1 and E2 ratios
    JB = pd.DataFrame([joint_ratio_boot(t, a.B, a.seed, reg, metas, mat, prov) for t, _ in DETS])
    JB.to_csv(f'{src}/joint_ratio_boot.csv', index=False)
    print(JB[['det', 'fpr_1to1', 'fpr_12to1', 'e1_ratio', 'e1_lo', 'e1_hi', 'fpr_fmawin', 'e2_ratio', 'e2_lo', 'e2_hi']].to_string(), f'{time.time() - t0:.0f}s')
    # 3) values per detector
    V = {}
    for t, lab in DETS:
        v = {}
        x = pick(rates, model=t, cohort='jamendo_B', thr='tau_JamA@2%'); v['jamB_win'] = (x.rate, x.ci_lo, x.ci_hi)
        x = pick(prov, det=t, source='fma_dev800', unit='win', cohort='jamendo_B'); v['jamB_fmawin'] = (x.rate, x.ci_lo, x.ci_hi)
        v['ctx'] = {c: tuple(pick(rates, model=t, cohort=c, thr='tau_JamA@2%')[['rate', 'ci_lo', 'ci_hi']]) for c, _ in SPEECH_ONLY + MUSIC_ONLY}
        x = pick(mat, det=t, k_cal=1, k_dep=1, cohort='jamendo_B'); v['m11'] = (x.rate, x.ci_lo, x.ci_hi)
        x = pick(mat, det=t, k_cal=12, k_dep=1, cohort='jamendo_B'); v['m121'] = (x.rate, x.ci_lo, x.ci_hi)
        j = JB[JB.det == t].iloc[0]; v['e1'] = (j.e1_ratio, j.e1_lo, j.e1_hi); v['e2'] = (j.e2_ratio, j.e2_lo, j.e2_hi)
        d = ru[(ru.det == t) & (ru.ref == 'unseen4')]
        assert len(d) == 7, (t, len(d))
        jb = d[d.cohort == 'jamendo_B'].iloc[0]; v['e3'] = (jb.tpr_tauJamA, jb.tprJ_lo, jb.tprJ_hi)
        v['auroc_rng'] = (d.auroc.min(), d.auroc.max(), d.loc[d.auroc.idxmin(), 'cohort'], d.loc[d.auroc.idxmax(), 'cohort'])
        v['oracle_rng'] = (d.tpr_oracle.min(), d.tpr_oracle.max())
        gv = []
        for c, g, gl in NEWFAM:
            y = pick(gen, det=t, cohort=c, generator=g); gv.append((gl, y.tpr, y.ci_lo, y.ci_hi))
        v['gens'] = gv
        assert abs(np.mean([z[1] for z in gv]) - jb.tpr_tauJamA) < 1e-6, (t, np.mean([z[1] for z in gv]), jb.tpr_tauJamA)  # same definition
        y = pick(gen, det=t, cohort='sonics_fake', generator='all'); v['sonics_tpr'] = (y.tpr, y.ci_lo, y.ci_hi)
        e4 = {}
        for al in ('tau_JamA@1%', 'tau_JamA@2%', 'tau_JamA@5%'):
            for c, cl in E4_COH:
                y = pick(g0, model=t, thr=al, cohort=c)
                e4[(al, c)] = (y.fpr, y.ci_lo, y.ci_hi, y.ratio, y.ratio_lo, y.ratio_hi, bool(y.pass_orig), bool(y.pass_strict), y.jamB)
        v['e4'] = e4
        allg = {}
        for al in ('tau_JamA@1%', 'tau_JamA@2%', 'tau_JamA@5%'):
            d0 = g0[(g0.model == t) & (g0.thr == al)]; real = d0[d0.role == 'real']
            top = real[(real.n >= 30) & (real.clusters >= 10)].sort_values('ratio', ascending=False).head(1).iloc[0]
            allg[al] = (list(real[real.pass_orig].cohort), list(real[real.pass_strict].cohort), top.cohort, top.fpr, top.ratio, top.ratio_lo, top.ratio_hi)
        v['g0_all'] = allg
        x = pick(rs, det=t, ref='jamendo_A'); v['swapJ'] = (bool(x.all_below_3x), x.worst_cohort, x.worst_fpr, x.worst_ci_lo, x.worst_ci_hi, int(x.n_exceed_point), int(x.n_exceed_sig))
        sw_all = rs[rs.det == t]
        v['swap_n_ok'] = (int(sw_all.all_below_3x.sum()), len(sw_all), ', '.join(f'{r.ref_label} {r.worst_fpr:.1f}{"" if r.all_below_3x else "✗"}' for r in sw_all.itertuples()))
        x = pick(mrs, det=t, kind='MAX'); v['max'] = (bool(x.all_below_3x), x.worst_cohort, x.worst_fpr, x.worst_ci_lo, x.worst_ci_hi, int(x.n_exceed), int(x.n_exceed_sig))
        tj = pick(mrt, det=t, kind='JamA'); tm = pick(mrt, det=t, kind='MAX(5)'); v['max_tpr'] = (tm.tpr, tm.ci_lo, tm.ci_hi, tm.tpr - tj.tpr)
        V[t] = v

    # 4) verdicts
    def verdicts(t):
        v = V[t]; out = {}
        r, lo, hi = v['e1']; out['E1'] = ('holds' if r >= 2 else 'fails', f'{r:.2f} [{lo:.2f}–{hi:.2f}]', 'CI lower bound ≥ 2' if lo >= 2 else 'CI includes values < 2')
        r, lo, hi = v['e2']
        out['E2'] = ('holds' if r <= 2.0 else ('borderline (about 2x)' if r < 2.5 else 'fails'), f'{r:.2f} [{lo:.2f}–{hi:.2f}]',
                     'CI upper bound ≤ 2' if hi <= 2 else 'CI reaches above 2')
        p, lo, hi = v['e3']; out['E3'] = ('holds' if p <= 20 else 'fails', f'{p:.1f}% [{lo:.1f}–{hi:.1f}]', 'CI upper bound ≤ 20%' if hi <= 20 else 'CI reaches above 20%')
        e4 = v['e4']; passes = [cl for c, cl in E4_COH if e4[('tau_JamA@2%', c)][6]]
        mx = max(E4_COH, key=lambda cc: e4[('tau_JamA@2%', cc[0])][3])
        mm = e4[('tau_JamA@2%', mx[0])]
        out['E4'] = ('holds' if not passes else 'fails', f'max ratio {mm[3]:.2f} [{mm[4]:.2f}–{mm[5]:.2f}] ({mx[1]})',
                     'G0 passes: ' + (', '.join(passes) if passes else 'none') + ' at α=2%; '
                     + '; '.join(f'α={al.split("@")[1]}: ' + (', '.join(cl for c, cl in E4_COH if e4[(al, c)][6]) or 'none') for al in ('tau_JamA@1%', 'tau_JamA@5%')))
        return out
    VD = {t: verdicts(t) for t, _ in DETS}
    # 5) tables
    f3 = lambda x: f'{x[0]:.1f} [{x[1]:.1f}–{x[2]:.1f}]'
    rows_md, rows_csv = [], []

    def add(quantity, key_fn, csv_fn=None, note=''):
        cells = [key_fn(V[t]) for t, _ in DETS]
        rows_md.append(f'| {quantity} | ' + ' | '.join(cells) + ' |')
        for (t, lab), cell in zip(DETS, cells):
            vals = csv_fn(V[t]) if csv_fn else (np.nan, np.nan, np.nan)
            rows_csv.append(dict(section='compact', quantity=quantity, detector=t, value=vals[0], ci_lo=vals[1], ci_hi=vals[2], text=cell, verdict='', note=note))
    add('Jamendo-B FPR, τ_JamA window (matched unit, k=1)', lambda v: f3(v['jamB_win']), lambda v: v['jamB_win'], 'public_window_rates.csv')
    add('Jamendo-B FPR, FMA-dev window τ', lambda v: f3(v['jamB_fmawin']), lambda v: v['jamB_fmawin'], 'unit_prov.csv (source fma_dev800, unit win)')
    add('E2 ratio: FMA-dev-window / Jamendo-A-window FPR on Jamendo-B', lambda v: f'{v["e2"][0]:.2f} [{v["e2"][1]:.2f}–{v["e2"][2]:.2f}]', lambda v: v['e2'], 'joint_ratio_boot.csv')
    add('Jamendo-B FPR, track-level τ on single windows (k_cal=12 → k_dep=1)', lambda v: f3(v['m121']), lambda v: v['m121'], 'unit_matrix.csv')
    add('E1 ratio: (k_cal=12 → k_dep=1) / matched (1 → 1)', lambda v: f'{v["e1"][0]:.2f} [{v["e1"][1]:.2f}–{v["e1"][2]:.2f}]', lambda v: v['e1'], 'joint_ratio_boot.csv')
    add('New-family mean TPR @τ_JamA (4 generators equal, songs 1/n_f)', lambda v: f3(v['e3']), lambda v: v['e3'], 'unit_rank_unseen.csv (ref unseen4; CI from the jamendo_B row)')
    add('  per generator: Lyria 3 / HeartMuLa / MiniMax 2.6 / Mureka v9', lambda v: ' / '.join(f'{g[1]:.1f}' for g in v['gens']), None, 'unit_gen.csv')
    add('Unseen-family AUROC, range over 7 human cohorts', lambda v: f'{v["auroc_rng"][0]:.3f}–{v["auroc_rng"][1]:.3f}', lambda v: (np.nan, v['auroc_rng'][0], v['auroc_rng'][1]),
        'unit_rank_unseen.csv; cohorts fma_dev800, jamendo_B, ava_sm5, audioset_sm, openbmat_human_musicbearing, musdb18, gtzan; ci_lo/ci_hi hold the range')
    add('  (lowest / highest cohort)', lambda v: f'{v["auroc_rng"][2]} / {v["auroc_rng"][3]}', None)
    add('  unseen-family TPR at each cohort\'s own 2% threshold (oracle), range', lambda v: f'{v["oracle_rng"][0]:.1f}–{v["oracle_rng"][1]:.1f}',
        lambda v: (np.nan, v['oracle_rng'][0], v['oracle_rng'][1]), 'ci_lo/ci_hi hold the range')
    for c, cl in E4_COH:
        add(f'{cl}: window FPR @τ_JamA 2% (ratio to Jamendo-B [95% CI])',
            lambda v, c=c: (lambda e: f'{e[0]:.1f} (×{e[3]:.2f} [{e[4]:.2f}–{e[5]:.2f}])')(v['e4'][('tau_JamA@2%', c)]),
            lambda v, c=c: (lambda e: (e[3], e[4], e[5]))(v['e4'][('tau_JamA@2%', c)]), 'public_window_g0.csv (value = ratio)')
    add('Largest ratio over all human cohorts (n ≥ 30, ≥ 10 clusters), α=2%', lambda v: (lambda g: f'{g[2]} {g[3]:.1f}% (×{g[4]:.2f} [{g[5]:.2f}–{g[6]:.2f}])')(v['g0_all']['tau_JamA@2%']),
        lambda v: (lambda g: (g[4], g[5], g[6]))(v['g0_all']['tau_JamA@2%']), 'public_window_g0.csv')
    add('Context, speech without music (null): ' + ' / '.join(l for _, l in SPEECH_ONLY) + ', window FPR @τ_JamA 2%',
        lambda v: ' / '.join(f'{v["ctx"][c][0]:.1f}' for c, _ in SPEECH_ONLY), None, 'public_window_rates.csv (CIs in the context rows)')
    add('Context, music without speech: ' + ' / '.join(l for _, l in MUSIC_ONLY) + ', window FPR @τ_JamA 2%',
        lambda v: ' / '.join(f'{v["ctx"][c][0]:.1f}' for c, _ in MUSIC_ONLY), None, 'public_window_rates.csv (CIs in the context rows)')
    for t, lab in DETS:
        for c, cl in SPEECH_ONLY + MUSIC_ONLY:
            x = V[t]['ctx'][c]
            rows_csv.append(dict(section='context', quantity=f'window FPR @tau_JamA 2%: {c} ({cl})', detector=t, value=x[0], ci_lo=x[1], ci_hi=x[2],
                                 text=f'{x[0]:.1f} [{x[1]:.1f}–{x[2]:.1f}]', verdict='', note='public_window_rates.csv'))
    add('G0 (orig) passes on any human cohort, α = 1 / 2 / 5%', lambda v: ' / '.join((', '.join(v['g0_all'][al][0]) or 'none') for al in ('tau_JamA@1%', 'tau_JamA@2%', 'tau_JamA@5%')), None, 'public_window_g0.csv')
    add('Ref swap, single Jamendo-A τ: worst of 8 cohorts; all < 3α? (cohorts ≥ 6%, of which CI lower bound > 6%)', lambda v: (lambda s: f'{s[2]:.1f} [{s[3]:.1f}–{s[4]:.1f}] {s[1]}; {"yes" if s[0] else "NO"} ({s[5]}, {s[6]})')(v['swapJ']),
        lambda v: (lambda s: (s[2], s[3], s[4]))(v['swapJ']), 'unit_refswap_summary.csv (ref jamendo_A)')
    add('  ref swap: references with all cohorts < 3α (of 6) — worst FPR per reference', lambda v: f'{v["swap_n_ok"][0]}/{v["swap_n_ok"][1]}: {v["swap_n_ok"][2]}', None, 'unit_refswap_summary.csv')
    add('Multi-ref MAX (leave-one-cohort-out): worst of 8 cohorts; all < 3α? (cohorts ≥ 6%, of which CI lower bound > 6%)', lambda v: (lambda s: f'{s[2]:.1f} [{s[3]:.1f}–{s[4]:.1f}] {s[1]}; {"yes" if s[0] else "NO"} ({s[5]}, {s[6]})')(v['max']),
        lambda v: (lambda s: (s[2], s[3], s[4]))(v['max']), 'unit_multiref_summary.csv (kind MAX)')
    add('  new-family TPR @MAX(5) (change vs τ_JamA, points)', lambda v: f'{v["max_tpr"][0]:.1f} [{v["max_tpr"][1]:.1f}–{v["max_tpr"][2]:.1f}] ({v["max_tpr"][3]:+.1f})',
        lambda v: v['max_tpr'][:3], 'unit_multiref_tpr.csv')
    add('Sanity: SONICS-fake window TPR @τ_JamA (training distribution)', lambda v: f3(v['sonics_tpr']), lambda v: v['sonics_tpr'], 'unit_gen.csv (sonics_fake, all)')
    CRIT = {'E1': 'Unit effect: Jamendo-B single-window FPR with a track-level τ (k_cal=12 → k_dep=1) ≥ 2× the matched-unit FPR (1 → 1)',
            'E2': 'Small calibration-source effect: FMA-dev-window τ gives ≤ about 2× the Jamendo-A-window τ FPR on Jamendo-B (SpecTTTra: 1.8×); rule: ≤ 2.0 holds, 2.0–2.5 borderline, ≥ 2.5 fails',
            'E3': 'Coverage failure: mean TPR over the 4 new families at τ_JamA ≤ 20%',
            'E4': 'No in-the-wild excess: AVA-SM, AudioSet-SM, AI-OpenBMAT-H do not pass G0 (point ratio ≥ 3 and 2.5th pct > 1) at α = 2%'}
    vrows = []
    for e in ('E1', 'E2', 'E3', 'E4'):
        cells = [f'{VD[t][e][1]} — **{VD[t][e][0]}** ({VD[t][e][2]})' if t != GAMMA else f'{VD[t][e][1]} ({VD[t][e][0]})' for t, _ in DETS]
        vrows.append(f'| {e} | {CRIT[e]} | ' + ' | '.join(cells) + ' |')
        for t, lab in DETS:
            rows_csv.append(dict(section='verdict', quantity=e, detector=t, value=np.nan, ci_lo=np.nan, ci_hi=np.nan, text=VD[t][e][1], verdict=VD[t][e][0],
                                 note=f'{CRIT[e]} | {VD[t][e][2]}'))
    hdr = ' | '.join(lab for _, lab in DETS)
    L = ['# Pre-specified replication: SpecTTTra-alpha and SpecTTTra-beta next to SpecTTTra (gamma)', '',
         'Window unit (5 s, at most 12 windows per file, reflect padding), α = 2%, τ_JamA = track-weighted (1/n_f) 98th percentile of the Jamendo-A '
         'window scores; same scripts and settings as the main tables. FPR/TPR in % [two-stage cluster bootstrap 95% CI, τ re-estimated in every '
         'replicate]. E1/E2 ratio CIs: joint bootstrap '
         f'(B = {a.B:,}) sharing one Jamendo-A artist draw between the two thresholds and one Jamendo-B artist draw between numerator and denominator.', '',
         '## 1. SpecTTTra reproduces the main tables', '',
         '| check | paper | reproduced | match |', '|---|---|---|---|']
    for r in tgt.itertuples():
        L.append(f'| {r.check} | {r.paper:.1f} | {r.reproduced:.4f} | {"yes" if r.match else "NO"} |')
    L += ['', f'Point estimates of all SpecTTTra rows of {len(rep)} input tables equal the frozen main tables (rows matched '
          f'{int(rep.rows_matched.sum())}/{int(rep.rows_frozen.sum())}, largest absolute difference {rep.max_point_diff.max():.1e}).', '',
          '## 2. Side by side', '', f'| quantity | {hdr} |', '|---|' + '---|' * len(DETS)] + rows_md
    L += ['', '## 3. Expectations recorded in advance', '', f'| | criterion | {hdr} |', '|---|---|' + '---|' * len(DETS)] + vrows
    L += ['', '## How to read', '',
          '- E1 and E3 are judged on the point estimate; the parenthesis says whether the 95% CI also clears the bound. E2 uses the stated band. '
          'E4 is the failure criterion of the paper (G0) on the three named cohorts at α = 2%; α = 1% and 5% are shown as robustness checks.',
          '- The SpecTTTra column reproduces the paper\'s SpecTTTra numbers, so differences between columns come from the checkpoints, not the pipeline.',
          '- The two context rows split human audio into speech without music and music without speech (same τ_JamA, window unit).',
          '- The new-family mean TPR equals the equal-weight mean of the four per-generator TPRs (checked); its CI comes from the generator-stratified bootstrap '
          'with τ re-estimated (unit_rank_unseen, Jamendo-B row).', '']
    os.makedirs(a.out, exist_ok=True)
    open(f'{a.out}/extra_sonics_variants.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    pd.DataFrame(rows_csv).to_csv(f'{a.out}/extra_sonics_variants.csv', index=False)
    print('\n'.join(L)); print('EXTRA_SONICS_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
