"""Holm adjustment of the confirmatory head contrast (PE - UC, worst in-the-wild FPR) across families (Section 5.4, Table 3).

Reads out/<tag>_contrast.csv files written by heads_window.py (one or several runs), recomputes the Holm adjustment over the
confirmatory contrasts of all families present and writes out/heads_contrast_all.csv.
Note: the paper adjusts over nine families including C1; without C1 the adjustment runs over eight families, so p_holm differs
from the paper (the raw p-values do not).
usage: python code/heads_report.py [--tags heads]
"""
import os, sys, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

TB = C.TB
GA = list(C.FAMILIES)


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tags', default='heads'); a = ap.parse_args()
    ct = pd.concat([pd.read_csv(f'{TB}/{t}_contrast.csv') for t in a.tags.split(',')], ignore_index=True)
    conf = ct[(ct.kind == 'confirmatory') & ct.bb.isin(GA)].copy()
    conf['p_holm'] = holm(conf.p_worst.values) if len(conf) else []
    ct = ct.drop(columns=[c for c in ('p_holm',) if c in ct.columns]).merge(conf[['bb', 'base', 'method', 'p_holm']], on=['bb', 'base', 'method'], how='left')
    ct.to_csv(f'{TB}/heads_contrast_all.csv', index=False)
    print(f'Confirmatory PE - UC, worst in-the-wild FPR (Holm over {len(conf)} families):')
    for r in ct[ct.kind == 'confirmatory'].itertuples():
        print(f'  {r.bb:18s} {r.worst_base:5.1f} -> {r.worst_method:5.1f}  d {r.d_worst:+.1f} [{r.d_worst_lo:+.1f}, {r.d_worst_hi:+.1f}]  p {r.p_worst:.3f}  Holm {r.p_holm:.3f}')


if __name__ == '__main__':
    main()
