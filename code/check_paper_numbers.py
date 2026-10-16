"""Compare the regenerated tables (out/) with the frozen tables behind the paper (results/) and print key paper numbers.

Part 1: for every result table, rows are matched on their key columns and the point estimates are compared (they do not depend
        on the bootstrap). Interval columns are compared separately; they match only when the same number of replicates is used
        and the random stream is the paper's (see README, "What reproduces exactly").
Part 2: selected numbers quoted in the paper, recomputed from out/.
usage: python code/check_paper_numbers.py [--out out] [--tol 1e-9]
"""
import os, sys, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402


def V(pre, parts):
    """Tables of one agg_window_variants.py run (--prefix pre --parts parts): (name, key columns, point columns, interval columns)."""
    spec = [('rates', 'rates', ['model', 'cohort', 'subset', 'thr'], ['rate', 'n_files', 'clusters'], ['lo', 'hi']),
            ('auroc', 'auroc', ['model', 'unit', 'cls'], ['auroc'], ['lo', 'hi']),
            ('rates', 'ratios', ['model', 'cohort', 'subset'], ['fpr', 'jamB', 'ratio'], ['lo', 'hi']),
            ('paired', 'paired', ['model', 'cohort', 'subset'], ['delta', 'ratio'], ['d_lo', 'd_hi'])]
    return [(f'{pre}_{t}', k, v, c) for part, t, k, v, c in spec if part in parts]


TABLES = [
    ('public_window_rates', ['model', 'cohort', 'thr'], ['rate', 'n', 'clusters'], ['ci_lo', 'ci_hi']),
    ('public_window_g0', ['model', 'thr', 'cohort'], ['fpr', 'jamB', 'ratio'], ['ratio_lo', 'ratio_hi']),
    ('unit_matrix', ['det', 'k_cal', 'k_dep', 'cohort'], ['rate', 'tau'], ['ci_lo', 'ci_hi']),
    ('unit_prov', ['det', 'source', 'unit', 'cohort'], ['rate', 'tau'], ['ci_lo', 'ci_hi']),
    ('unit_rank_unseen', ['det', 'ref', 'cohort'], ['auroc', 'fpr_tauJamA', 'tpr_tauJamA', 'tpr_oracle'], ['au_lo', 'au_hi', 'fpr_lo', 'fpr_hi']),
    ('unit_gen', ['det', 'cohort', 'generator'], ['tpr'], ['ci_lo', 'ci_hi']),
    ('unit_refswap', ['det', 'ref', 'cohort'], ['fpr', 'tau'], ['ci_lo', 'ci_hi']),
    ('unit_refswap_summary', ['det', 'ref'], ['worst_fpr', 'all_below_3x', 'n_exceed_point'], ['worst_ci_lo', 'worst_ci_hi']),
    ('unit_multiref', ['det', 'cohort', 'sensitivity', 'kind'], ['fpr', 'tau'], ['ci_lo', 'ci_hi']),
    ('unit_multiref_summary', ['det', 'kind'], ['worst_fpr', 'all_below_3x', 'n_exceed'], ['worst_ci_lo', 'worst_ci_hi']),
    ('unit_multiref_tpr', ['det', 'kind'], ['tpr', 'tau'], ['ci_lo', 'ci_hi']),
    ('unit_refsize', ['det', 'size', 'draw'], ['tau', 'worst_all8', 'worst_as_MUSDB', 'worst_as_AudioSetMO'], []),
    ('unit_refsize_summary', ['det', 'size'], ['med', 'p05', 'p95', 'medM', 'p05M', 'p95M', 'medA', 'p05A', 'p95A'], []),
    ('unit_multiref_vs_strict', ['det'], ['W_max', 'tau_all5', 'tpr_max', 'alpha_prime', 'tau_strict', 'tpr_strict', 'dTPR'],
     ['W_max_lo', 'W_max_hi', 'tpr_max_lo', 'tpr_max_hi', 'alpha_prime_lo', 'alpha_prime_hi', 'tpr_strict_lo', 'tpr_strict_hi', 'dTPR_lo', 'dTPR_hi']),
    ('unit_alpha_dependence', ['det', 'alpha'], ['jamB_fpr_FMA', 'jamB_fpr_JamA', 'jamB_ratio_vsJamA', 'third_fpr_FMA'], ['jamB_ci_lo', 'jamB_ci_hi']),
    ('f2_platform_overlap', ['det', 'cohort', 'thr'], ['fpr', 'tau'], ['ci_lo', 'ci_hi']),
    ('f2_platform_overlap_delta', ['cohort', 'thr', 'comp'], ['delta'], ['ci_lo', 'ci_hi']),
    ('f2_platform_overlap_tpr', ['det', 'thr'], ['tpr', 'tau'], ['ci_lo', 'ci_hi']),
] + V('variants_seednoise', ('rates', 'paired', 'auroc')) + V('variants_m0', ('auroc',)) + V('variants_m0su_pair', ('rates',)) + [
    ('track_tau_all_rates', ['model', 'cohort', 'alpha'], ['rate'], ['lo', 'hi']),
    ('matched_tpr_variants', ['treat', 'matched_on', 'cohort', 'subset'], ['tau_ref_star', 'fpr_treat', 'fpr_ref_star', 'delta'], ['d_lo', 'd_hi']),
    ('openbmat_window_auroc', ['model', 'cls'], ['auroc', 'n_chunks_human'], ['ci_lo', 'ci_hi']),
    ('heads_rates', ['bb', 'head', 'alpha', 'cohort', 'subset'], ['rate', 'T'], ['ci_lo', 'ci_hi']),
    ('heads_provenance', ['bb', 'head', 'calib'], ['T', 'dT_sd', 'jamendo_B', 'rw_worst'], ['jamendo_B_lo', 'jamendo_B_hi']),
    ('heads_fa_types', ['bb', 'head', 'cohort'], ['n_fp_windows', 'el_5', 'ep_5', 'mx_5'], []),
    ('heads_null', ['bb', 'head', 'cohort'], ['med', 'fpr', 'margin_sd'], []),
    ('heads_contrast_all', ['bb', 'base', 'method'], ['worst_base', 'worst_method', 'd_worst', 'oot_base', 'oot_method', 'd_oot'], ['d_worst_lo', 'd_worst_hi', 'p_worst']),
    ('heads_matched_tpr', ['bb'], ['T_uc', 'tpr_uc', 'uc_worst', 'T_orig_matched', 'tpr_orig_matched', 'orig_worst', 'd_worst'], ['d_worst_lo', 'd_worst_hi']),
    # replication with SpecTTTra-alpha and SpecTTTra-beta (code/extra_sonics.sh); 'value' holds the point estimate of a row
    ('extra_sonics_variants', ['section', 'quantity', 'detector'], ['value'], ['ci_lo', 'ci_hi']),
]


def num(col):
    """Column -> float (booleans as 0/1)."""
    return pd.to_numeric(col.map(lambda v: {'True': 1, 'False': 0}.get(v, v) if isinstance(v, str) else (int(v) if isinstance(v, (bool, np.bool_)) else v)),
                         errors='coerce').astype(float)


def compare(out, tol):
    rows = []
    for name, keys, vals, cis in TABLES:
        fo, fr = f'{out}/{name}.csv', f'{C.RESULTS}/{name}.csv'
        if not os.path.exists(fo):
            rows.append(dict(table=name, status='not regenerated')); continue
        o = pd.read_csv(fo, keep_default_na=False, na_values=['']); r = pd.read_csv(fr, keep_default_na=False, na_values=[''])
        m = r.merge(o, on=keys, how='inner', suffixes=('_ref', '_new'))
        miss = len(r) - len(m)
        diffs = {}
        for v in vals:
            a = num(m[f'{v}_ref']); b = num(m[f'{v}_new'])
            both_nan = a.isna() & b.isna(); inf_eq = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
            d = (a - b).abs().where(~(both_nan | inf_eq), 0.0)
            diffs[v] = float(d.max()) if len(d) else 0.0
        cimax = 0.0
        for v in cis:
            if f'{v}_ref' in m and f'{v}_new' in m:
                a = pd.to_numeric(m[f'{v}_ref'], errors='coerce'); b = pd.to_numeric(m[f'{v}_new'], errors='coerce')
                d = (a - b).abs().dropna(); cimax = max(cimax, float(d.max()) if len(d) else 0.0)
        worst = max(diffs.values()) if diffs else 0.0
        rows.append(dict(table=name, rows_frozen=len(r), rows_matched=len(m), rows_unmatched=miss, max_point_diff=worst,
                         point_estimates='identical' if (worst <= tol and miss == 0) else ('identical (matched rows)' if worst <= tol else 'DIFFERENT'),
                         max_interval_diff=cimax))
    return pd.DataFrame(rows)


def key_numbers(out):
    rd = lambda f: pd.read_csv(f'{out}/{f}.csv', keep_default_na=False, na_values=[''])
    L = []

    def add(desc, paper, value, note=''):
        L.append(dict(number=desc, paper=paper, repo=value, note=note))
    try:
        r = rd('public_window_rates'); q = lambda m, c, t: float(r[(r.model == m) & (r.cohort == c) & (r.thr == t)].rate.iloc[0])
        add('B0 Jamendo-B window FPR at tau_JamA 2% (%)', '3.0', f"{q('B0', 'jamendo_B', 'tau_JamA@2%'):.1f}")
        add('B0 Jamendo-B window FPR at the FMA-dev threshold (%)', '20.3', f"{q('B0', 'jamendo_B', 'tau_FMAov@2%'):.1f}")
        add('M0 Jamendo-B FPR, FMA-dev vs Jamendo-A threshold (%)', '6.1 vs 2.5', f"{q('M0', 'jamendo_B', 'tau_FMAov@2%'):.1f} vs {q('M0', 'jamendo_B', 'tau_JamA@2%'):.1f}")
        add('Fakeprint LR / SpecTTTra / ArtifactNet Jamendo-B FPR at the FMA-dev threshold (%)', '9.6 / 4.4 / 7.4',
            f"{q('Fakeprint LR', 'jamendo_B', 'tau_FMAov@2%'):.1f} / {q('SpecTTTra', 'jamendo_B', 'tau_FMAov@2%'):.1f} / {q('ArtifactNet', 'jamendo_B', 'tau_FMAov@2%'):.1f}")
        add('B0 ESC-50 (padded) / M0 ESC-50 window FPR (%)', '13.6 / 76.9', f"{q('B0', 'esc50', 'tau_JamA@2%'):.1f} / {q('M0', 'esc50', 'tau_JamA@2%'):.1f}")
        add('M0 AVA-SM all vs unpadded window FPR (%)', '5.4 -> 1.1', f"{q('M0', 'ava_sm5', 'tau_JamA@2%'):.1f} -> {q('M0', 'ava_sm5[long]', 'tau_JamA@2%'):.1f}")
    except Exception as e: add('public_window_rates', '', f'not available ({e})')
    try:
        u = rd('unit_prov'); g = lambda d, s, un, c: float(u[(u.det == d) & (u.source == s) & (u.unit == un) & (u.cohort == c)].rate.iloc[0])
        add('B0 Jamendo-B FPR, FMA-dev window threshold (unit_prov, %)', '20.3', f"{g('B0', 'fma_dev800', 'win', 'jamendo_B'):.1f}")
        add('B0 AVA-SM / AudioSet-SM with track-level Jamendo-A thresholds vs matched units (%)', '19.5 / 23.7 vs 4.6 / 6.4',
            f"{g('B0', 'jamendo_A', 'file', 'ava_sm5'):.1f} / {g('B0', 'jamendo_A', 'file', 'audioset_sm'):.1f} vs {g('B0', 'jamendo_A', 'win', 'ava_sm5'):.1f} / {g('B0', 'jamendo_A', 'win', 'audioset_sm'):.1f}")
        ft = ['B0', "B0'", 'B0+0', 'C0', 'C0+0']
        jb = u[(u.cohort == 'jamendo_B') & (u.unit == 'win') & u.det.isin(ft)]
        add('B0 and C0 families, Jamendo-B FPR at FMA-dev vs Jamendo-A window thresholds (%)', '19.2-39.7 vs 2.6-3.6 (incl. C1)',
            f"{jb[jb.source == 'fma_dev800'].rate.min():.1f}-{jb[jb.source == 'fma_dev800'].rate.max():.1f} vs {jb[jb.source == 'jamendo_A'].rate.min():.1f}-{jb[jb.source == 'jamendo_A'].rate.max():.1f}",
            'C1 not released')
    except Exception as e: add('unit_prov', '', f'not available ({e})')
    try:
        m = rd('unit_matrix'); b = m[(m.det == 'B0') & (m.cohort == 'jamendo_B')]
        diag = b[b.k_cal == b.k_dep].rate
        k121 = b[(b.k_cal == 12) & (b.k_dep == 1)].iloc[0]; k112 = b[(b.k_cal == 1) & (b.k_dep == 12)].iloc[0]
        add('B0 Jamendo-B matched-unit FPR, all units (%)', '3.0-3.8', f'{diag.min():.1f}-{diag.max():.1f}')
        add('B0 Jamendo-B, 12-window threshold applied to single windows (%)', '10.8 [7.6, 14.2]', f'{k121.rate:.1f} [{k121.ci_lo:.1f}, {k121.ci_hi:.1f}]', 'interval depends on B')
        add('B0 Jamendo-B, single-window threshold applied to 12-window units (%)', '0.4 [0.0, 1.0]', f'{k112.rate:.1f} [{k112.ci_lo:.1f}, {k112.ci_hi:.1f}]', 'interval depends on B')
    except Exception as e: add('unit_matrix', '', f'not available ({e})')
    try:
        s = rd('unit_refswap_summary'); cnt = lambda ref: f"{int(s[s.ref == ref].all_below_3x.sum())}/{len(s[s.ref == ref])}"
        add('Reference swap: detectors below 6% on all other cohorts, FMA-dev / Jamendo-A / AudioSet-MO', '1 / 5 / 9 of 10 (incl. C1)',
            f"{cnt('fma_dev800')} / {cnt('jamendo_A')} / {cnt('audioset_mo')}", 'C1 not released')
        add('Reference swap: same, MUSDB18 / GTZAN / AVA-MO', '2 or 3 of 10 (incl. C1)', f"{cnt('musdb18')} / {cnt('gtzan')} / {cnt('ava_mo')}", 'C1 not released')
        rs = rd('unit_refswap'); b = rs[(rs.det == 'B0') & (rs.cohort == 'ava_sm5') & (rs.ref != 'fma_dev800')]
        add('B0 AVA-SM FPR across platform-disjoint references (%)', '1.7-18.5', f'{b.fpr.min():.1f}-{b.fpr.max():.1f}')
    except Exception as e: add('unit_refswap', '', f'not available ({e})')
    try:
        s = rd('unit_multiref_summary'); x = s[s.kind == 'MAX']; p = s[s.kind == 'POOL']
        add('Multi-reference MAX: detectors below 6% on all eight cohorts', '7/10 (incl. C1)', f'{int(x.all_below_3x.sum())}/{len(x)}',
            'C1 (above 6% under MAX in the paper) not released; with C1 the count is 7/10 as reported')
        add('Multi-reference MAX: significant excesses (CI lower bound > 6%)', '0', f'{int((x.n_exceed_sig > 0).sum())}', 'interval depends on B')
        add('Pooled reference: detectors below 6% on all eight cohorts', '4/10 (incl. C1)', f'{int(p.all_below_3x.sum())}/{len(p)}', 'C1 not released')
        t = rd('unit_multiref_tpr').pivot_table(index='det', columns='kind', values='tpr')
        cost = (t['JamA'] - t['MAX(5)'])
        ftc = cost.reindex(['B0', "B0'", 'B0+0', 'C0', 'C0+0', 'M0']).dropna()
        add('TPR cost of MAX on unseen families, range / median of fine-tuned (points)', '0-13 / 6.5 (incl. C1)', f'{cost.min():.1f}-{cost.max():.1f} / {ftc.median():.1f}', 'C1 not released')
    except Exception as e: add('unit_multiref', '', f'not available ({e})')
    try:
        s = rd('unit_refsize_summary'); b = s[(s.det == 'B0') & (s['size'] == 150)].iloc[0]
        add('500 draws of 150 Jamendo-A tracks: B0 worst FPR, median [5th, 95th percentile] (%)', '6.4 [4.4, 9.6]', f'{b.med:.1f} [{b.p05:.1f}, {b.p95:.1f}]')
        d = rd('unit_refsize'); n = d.assign(ok=d.worst_all8 < 6).groupby(['size', 'draw']).ok.sum()
        add('Detectors below 6% on all eight cohorts across 150- / 322-track draws (5th-95th percentile)', '4-9 / 5-8 (of 10, incl. C1)',
            ' / '.join(f'{np.percentile(n.loc[k], 5):.0f}-{np.percentile(n.loc[k], 95):.0f}' for k in (150, 322)) + f' (of {d.det.nunique()})', 'C1 not released')
        v = rd('unit_multiref_vs_strict').set_index('det'); gain = -v.loc[['B0', "B0'", 'B0+0'], 'dTPR']
        add("Oracle alpha' range (%); B0 family TPR gain of the single threshold over MAX (points)", '0.89-2.11; 5.9-7.6',
            f'{v.alpha_prime.min():.2f}-{v.alpha_prime.max():.2f}; {gain.min():.1f}-{gain.max():.1f}')
    except Exception as e: add('unit_refsize / unit_multiref_vs_strict', '', f'not available ({e})')
    try:
        a = rd('unit_alpha_dependence').set_index(['det', 'alpha']).jamB_ratio_vsJamA
        add('FMA-dev/Jamendo-A FPR ratio on Jamendo-B, alpha 5% -> 1%: B0, C0', '4.5 -> 9.1, 7.0 -> 20.8', f"{a[('B0', 5)]:.1f} -> {a[('B0', 1)]:.1f}, {a[('C0', 5)]:.1f} -> {a[('C0', 1)]:.1f}")
    except Exception as e: add('unit_alpha_dependence', '', f'not available ({e})')
    try:
        f = rd('f2_platform_overlap'); g = lambda d, c, t: f[(f.det == d) & (f.cohort == c) & (f.thr == t)].iloc[0]
        add('B0+broad: Jamendo-A threshold, B0+0 -> B0+broad (logits)', '-0.11 -> -0.90', f"{g('B0+0', 'jamendo_B', 'JamA').tau:.2f} -> {g('B0+broad', 'jamendo_B', 'JamA').tau:.2f}")
        add('B0+broad at tau_JamA: AVA-SM / AI-OpenBMAT-H / long LibriSpeech (%)', '4.5->8.8 / 3.0->9.0 / 1.8->30.9',
            ' / '.join(f"{g('B0+0', c, 'JamA').fpr:.1f}->{g('B0+broad', c, 'JamA').fpr:.1f}" for c in ('ava_sm5', 'openbmat_human_musicbearing', 'librispeech_long')))
    except Exception as e: add('f2_platform_overlap', '', f'not available ({e})')
    try:
        g0 = rd('public_window_g0'); x = g0[(g0.model == 'B0') & (g0.thr == 'tau_JamA@2%') & (g0.cohort == 'audioset_sm')].iloc[0]
        add('B0 AudioSet-SM excess ratio over Jamendo-B', '2.18 [1.28, 3.83], 6.4%', f'{x.ratio:.2f} [{x.ratio_lo:.2f}, {x.ratio_hi:.2f}], {x.fpr:.1f}%', 'interval depends on B')
        x = g0[(g0.model == 'M0') & (g0.thr == 'tau_JamA@2%') & (g0.cohort == 'openbmat_human_musicbearing')].iloc[0]
        add('M0 AI-OpenBMAT-H vs Jamendo-B, ratio', '8.1% vs 2.5%, 3.29 [2.38, 4.60]', f'{x.fpr:.1f}% vs {x.jamB:.1f}%, {x.ratio:.2f} [{x.ratio_lo:.2f}, {x.ratio_hi:.2f}]', 'interval depends on B')
    except Exception as e: add('public_window_g0', '', f'not available ({e})')
    try:
        au = rd('variants_m0_auroc'); w = au[(au.unit == 'window') & (au.cls == 'musicbearing')].set_index('model').auroc
        add('Paired AUROC on music-bearing AI-OpenBMAT windows, M0 vs B0+0', '0.863 vs 0.963', f"{w['M0']:.3f} vs {w['B0+0']:.3f}")
        sn = rd('variants_seednoise_rates'); q = lambda m, c, sb: float(sn[(sn.model == m) & (sn.cohort == c) & (sn.subset == sb) & (sn.thr == 'tau_JamA@2%')].rate.iloc[0])
        add('B0+SU: unpadded AVA-SM / AudioSet-SM (%)', '2.2 / 5.6', f"{q('B0+SU', 'ava_sm5', 'long'):.1f} / {q('B0+SU', 'audioset_sm', 'all'):.1f}")
        add('B0+0 -> B0+SU: AI-OpenBMAT-H / GTZAN (%)', '3.0->5.0 / 0.4->3.8', f"{q('B0+0', 'openbmat_human_musicbearing', 'long'):.1f}->{q('B0+SU', 'openbmat_human_musicbearing', 'long'):.1f} / {q('B0+0', 'gtzan', 'long'):.1f}->{q('B0+SU', 'gtzan', 'long'):.1f}")
        add('B0+0 -> B0+AVA: unpadded AVA-SM / AudioSet-SM (%)', '4.7->1.8 / 7.0->4.1', f"{q('B0+0', 'ava_sm5', 'long'):.1f}->{q('B0+AVA', 'ava_sm5', 'long'):.1f} / {q('B0+0', 'audioset_sm', 'all'):.1f}->{q('B0+AVA', 'audioset_sm', 'all'):.1f}")
        mp = rd('variants_m0su_pair_rates'); qm = lambda m: float(mp[(mp.model == m) & (mp.cohort == 'gtzan') & (mp.subset == 'long') & (mp.thr == 'tau_JamA@2%')].rate.iloc[0])
        add('M0 -> M0+SU: GTZAN (%)', '2.9 -> 10.1', f"{qm('M0'):.1f} -> {qm('M0+SU'):.1f}")
    except Exception as e: add('variants', '', f'not available ({e})')
    try:
        mt = rd('matched_tpr_variants'); x = mt[(mt.treat == 'B0+AVA')].set_index(['cohort', 'subset'])
        add('B0+AVA vs B0+0 at matched AI-OpenBMAT-AI TPR: AVA-SM (long) / AudioSet-SM (%)', '1.8 vs 0.7 / 4.1 vs 3.2',
            f"{x.loc[('ava_sm5', 'long')].fpr_treat:.1f} vs {x.loc[('ava_sm5', 'long')].fpr_ref_star:.1f} / {x.loc[('audioset_sm', 'all')].fpr_treat:.1f} vs {x.loc[('audioset_sm', 'all')].fpr_ref_star:.1f}")
    except Exception as e: add('matched_tpr_variants', '', f'not available ({e})')
    try:
        ob = rd('openbmat_window_auroc').pivot_table(index='model', columns='cls', values='auroc')
        add('Paired AUROC B0: similar / bg / bgvl', '0.949 / 0.935 / 0.900', f"{ob.loc['B0', 'sim']:.3f} / {ob.loc['B0', 'bg']:.3f} / {ob.loc['B0', 'bgvl']:.3f}")
        add('Paired AUROC bgvl: C0 / SpecTTTra (SpecTTTra music)', '0.818 / 0.591 (0.978)', f"{ob.loc['C0', 'bgvl']:.3f} / {ob.loc['SpecTTTra', 'bgvl']:.3f} ({ob.loc['SpecTTTra', 'mus']:.3f})")
    except Exception as e: add('openbmat_window_auroc', '', f'not available ({e})')
    try:
        ge = rd('unit_gen'); nf = ['lyria3', 'heartmula', 'minimax_2.6', 'mureka_v9']
        mean = lambda d: np.mean([float(ge[(ge.det == d) & (ge.generator == g)].tpr.iloc[0]) for g in nf])
        add('Table 4 mean TPR on four new families: B0 / C0 / M0', '84.1 / 82.7 / 89.1', f"{mean('B0'):.1f} / {mean('C0'):.1f} / {mean('M0'):.1f}")
    except Exception as e: add('unit_gen', '', f'not available ({e})')
    try:
        hr = rd('heads_rates')
        def worst(bb, h):
            x = hr[(hr.bb == bb) & (hr['head'] == h) & (hr.alpha == 0.02) & (hr.subset == 'all') & hr.cohort.isin(['openbmat_human_musicbearing', 'musdb18', 'gtzan', 'ava_sm5'])]
            return x.rate.max()
        add('Table 3 worst FPR orig / UC / PE: B0', '4.6 / 0.9 / 0.8', f"{worst('B0', 'orig'):.1f} / {worst('B0', 'free_split'):.1f} / {worst('B0', 'pos_split'):.1f}")
        add('Table 3 worst FPR orig / UC / PE: M0', '7.4 / 2.6 / 4.4', f"{worst('M0', 'orig'):.1f} / {worst('M0', 'free_split'):.1f} / {worst('M0', 'pos_split'):.1f}")
        hm = rd('heads_matched_tpr').set_index('bb')
        add('Original head at matched TPR vs UC, worst FPR: B0 / C0 / M0 (%)', '1.3 vs 0.9 / 0.8 vs 1.4 / 5.8 vs 2.6',
            ' / '.join(f"{hm.loc[b, 'orig_worst']:.1f} vs {hm.loc[b, 'uc_worst']:.1f}" for b in ('B0', 'C0', 'M0')))
    except Exception as e: add('heads', '', f'not available ({e})')
    try:
        x = rd('extra_sonics_variants')
        row = lambda m, d: x[(x.section == 'compact') & x.quantity.str.startswith(m) & (x.detector == d)].iloc[0]
        add('Replication: SpecTTTra-alpha Jamendo-B / AudioSet-SM window FPR at tau_JamA 2% (%)', '2.6 / 10.1',
            f"{float(row('Jamendo-B FPR, τ_JamA window', 'SpecTTTra-alpha').value):.1f} / {float(row('AudioSet-SM:', 'SpecTTTra-alpha').text.split()[0]):.1f}")
        add('Replication: SpecTTTra-beta mean TPR on the four unseen families at tau_JamA (%)', '5.7',
            f"{float(row('New-family mean TPR', 'SpecTTTra-beta').value):.1f}")
    except Exception as e: add('extra_sonics_variants', '', f'not available ({e})')
    return pd.DataFrame(L)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=C.TB); ap.add_argument('--tol', type=float, default=1e-9); a = ap.parse_args()
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90); pd.set_option('display.max_rows', 200)
    cmp = compare(a.out, a.tol); cmp.to_csv(f'{a.out}/check_tables.csv', index=False)
    print('== Part 1: regenerated tables (out/) versus frozen tables (results/), point estimates ==')
    print(cmp.to_string(index=False))
    kn = key_numbers(a.out); kn.to_csv(f'{a.out}/check_paper_numbers.csv', index=False)
    print('\n== Part 2: numbers quoted in the paper, recomputed from out/ ==')
    print(kn.to_string(index=False))


if __name__ == '__main__':
    main()
