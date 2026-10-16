"""Compare the regenerated tables (out/) with the frozen tables (results/).

Rows are matched on their key columns and the point estimates are compared (they do not depend on the bootstrap). Interval
columns match only when the same number of replicates and the same random stream are used.
usage: python code/compare_tables.py [--out out] [--tol 1e-9]
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
] + V('variants_seednoise', ('rates', 'paired', 'auroc')) + V('variants_m0', ('auroc',)) + [
    ('track_tau_all_rates', ['model', 'cohort', 'alpha'], ['rate'], ['lo', 'hi']),
    ('matched_tpr_variants', ['treat', 'matched_on', 'cohort', 'subset'], ['tau_ref_star', 'fpr_treat', 'fpr_ref_star', 'delta'], ['d_lo', 'd_hi']),
    ('openbmat_window_auroc', ['model', 'cls'], ['auroc', 'n_chunks_human'], ['ci_lo', 'ci_hi']),
    ('heads_rates', ['bb', 'head', 'alpha', 'cohort', 'subset'], ['rate', 'T'], ['ci_lo', 'ci_hi']),
    ('heads_provenance', ['bb', 'head', 'calib'], ['T', 'dT_sd', 'jamendo_B', 'rw_worst'], ['jamendo_B_lo', 'jamendo_B_hi']),
    ('heads_fa_types', ['bb', 'head', 'cohort'], ['n_fp_windows', 'el_5', 'ep_5', 'mx_5'], []),
    ('heads_null', ['bb', 'head', 'cohort'], ['med', 'fpr', 'margin_sd'], []),
    ('heads_contrast_all', ['bb', 'base', 'method'], ['worst_base', 'worst_method', 'd_worst', 'oot_base', 'oot_method', 'd_oot'], ['d_worst_lo', 'd_worst_hi', 'p_worst']),
    ('heads_matched_tpr', ['bb'], ['T_uc', 'tpr_uc', 'uc_worst', 'T_orig_matched', 'tpr_orig_matched', 'orig_worst', 'd_worst'], ['d_worst_lo', 'd_worst_hi']),
    # post hoc analyses: exposure crossover, TPR-matched baseline of Table 3, threshold policies, interpretation change
    ('f2x_exposure_crossover', ['det', 'cohort', 'thr'], ['fpr', 'tau', 'n_files', 'clusters', 'delta'], ['ci_lo', 'ci_hi', 'd_lo', 'd_hi']),
    ('f2x_exposure_crossover_summary', ['det', 'thr', 'variant', 'cohorts'], ['n_cohorts', 'mean_delta', 'max_delta'], ['mean_lo', 'mean_hi', 'max_lo', 'max_hi']),
    ('f2x_exposure_crossover_cross2x2', ['det', 'contrast'], ['est'], ['lo', 'hi']),
    ('f2x_exposure_crossover_auroc', ['det', 'set', 'cohort'], ['auroc', 'delta'], ['lo', 'hi', 'd_lo', 'd_hi']),
    ('table3_matched_baseline', ['bb', 'refit'], ['orig_tau', 'orig_tpr', 'orig_worst', 'matched_tau', 'matched_tpr', 'matched_worst', 'refit_tau', 'refit_tpr',
                                                  'refit_worst', 'd_refit_minus_matched', 'd_refit_minus_orig'],
     ['matched_worst_lo', 'matched_worst_hi', 'refit_worst_lo', 'refit_worst_hi', 'd_rm_lo', 'd_rm_hi', 'd_ro_lo', 'd_ro_hi']),
    ('policy_compare', ['det', 'policy'], ['tau_deployed', 'fpr_JamB', 'fpr_AVASM', 'fpr_ASSM', 'fpr_OBMH', 'worst_fpr', 'below6_point', 'tpr', 'tpr_cost_vs_P1'],
     ['worst_sim_lo', 'worst_sim_hi', 'worst_coh_lo', 'worst_coh_hi', 'tpr_lo', 'tpr_hi', 'cost_lo', 'cost_hi']),
    ('interpretation_change', ['det', 'condition'], ['tau', 'fpr_JamB', 'fpr_AVASM', 'fpr_ASSM', 'fpr_OBMH', 'n_gt6', 'any_speech_gt6', 'worst_speech_fpr', 'tpr'],
     ['lo_JamB', 'hi_JamB', 'lo_AVASM', 'hi_AVASM', 'lo_ASSM', 'hi_ASSM', 'lo_OBMH', 'hi_OBMH', 'tpr_lo', 'tpr_hi']),
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=C.TB); ap.add_argument('--tol', type=float, default=1e-9); a = ap.parse_args()
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90); pd.set_option('display.max_rows', 200)
    cmp = compare(a.out, a.tol); cmp.to_csv(f'{a.out}/compare_tables.csv', index=False)
    print('== regenerated tables (out/) versus frozen tables (results/), point estimates ==')
    print(cmp.to_string(index=False))


if __name__ == '__main__':
    main()
