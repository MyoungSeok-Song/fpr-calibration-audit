"""Alpha dependence of the training-platform distortion (Section 5.1). Reads public_window_rates.csv only (no new estimation).

Window unit, alpha in {1, 2, 5}%:
  (a) Jamendo-B window FPR at the FMA-dev threshold (tau_FMAov@alpha) / alpha
  (b) Jamendo-B window FPR at the FMA-dev threshold / window FPR at the Jamendo-A threshold (tau_JamA@alpha); the ratio quoted
      in the paper ("rises from 4.5 (B0) ... at alpha = 5% to 9.1 ... at alpha = 1%")
  The same two values for the worst third-platform cohort (AVA-SM, AudioSet-SM, AI-OpenBMAT-H music-bearing).
Output (out/): unit_alpha_dependence.csv
usage: python code/unit_alpha_dependence.py [--src out|results]
"""
import os, sys, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

TB = C.TB
DETS = [('B0', 'B0'), ("B0'", "B0'"), ('B0+0', 'B0+0'), ('C0', 'C0'), ('C0+0', 'C0+0'), ('M0', 'M0')]  # B0 and C0 families, M0 (Section 5.1)
THIRD = [('ava_sm5', 'AVA-SM'), ('audioset_sm', 'AudioSet-SM'), ('openbmat_human_musicbearing', 'OBM-H')]
AL = [1, 2, 5]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--src', default='out', choices=['out', 'results']); a = ap.parse_args()
    src = TB if a.src == 'out' else str(C.RESULTS)
    r = pd.read_csv(f'{src}/public_window_rates.csv', keep_default_na=False, na_values=[''])
    get = lambda t, c, k: r[(r.model == t) & (r.cohort == c) & (r.thr == k)]
    rows = []
    for t, nm in DETS:
        for a_ in AL:
            jf, jj = get(t, 'jamendo_B', f'tau_FMAov@{a_}%'), get(t, 'jamendo_B', f'tau_JamA@{a_}%')
            if jf.empty: continue
            jf, jj = jf.iloc[0], jj.iloc[0]
            row = dict(det=t, name=nm, alpha=a_, jamB_fpr_FMA=jf.rate, jamB_ci_lo=jf.ci_lo, jamB_ci_hi=jf.ci_hi, jamB_fpr_JamA=jj.rate,
                       jamB_ratio_alpha=round(jf.rate / a_, 2), jamB_ratio_vsJamA=round(jf.rate / jj.rate, 2) if jj.rate > 0 else np.inf)
            best = None
            for c, cl in THIRD:
                xf, xj = get(t, c, f'tau_FMAov@{a_}%'), get(t, c, f'tau_JamA@{a_}%')
                if xf.empty: continue
                xf, xj = xf.iloc[0], xj.iloc[0]
                if best is None or xf.rate > best[1].rate: best = (cl, xf, xj)
            if best:
                cl, xf, xj = best
                row.update(third_worst=cl, third_fpr_FMA=xf.rate, third_ci_lo=xf.ci_lo, third_ci_hi=xf.ci_hi, third_fpr_JamA=xj.rate,
                           third_ratio_alpha=round(xf.rate / a_, 2), third_ratio_vsJamA=round(xf.rate / xj.rate, 2) if xj.rate > 0 else np.inf)
            rows.append(row)
    d = pd.DataFrame(rows); d.to_csv(f'{TB}/unit_alpha_dependence.csv', index=False)
    print('FMA-dev / Jamendo-A threshold FPR ratio on Jamendo-B (definition b), alpha 5% -> 2% -> 1%:')
    for t, nm in DETS:
        g = d[d.det == t].set_index('alpha')
        if len(g): print(f'  {nm:14s} {g.loc[5, "jamB_ratio_vsJamA"]:.2f} -> {g.loc[2, "jamB_ratio_vsJamA"]:.2f} -> {g.loc[1, "jamB_ratio_vsJamA"]:.2f}')
    print('ALPHA_DONE', len(d))


if __name__ == '__main__':
    main()
