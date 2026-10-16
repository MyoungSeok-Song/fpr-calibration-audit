"""Track-level FPR with thresholds matched to the window count, tau_n (Section 5.2, last paragraph).

Track score = trim20u of the window logits (mean after dropping the floor(0.2 n) largest; for n = 1..4 nothing is dropped).
For a target file with n_f windows (at most 12), tau_n(alpha) = the k-th largest trim20u score of the first n windows of every
Jamendo-A track (k = floor(alpha * N), calibration FPR <= alpha), so calibration and evaluation use the same number of windows.
Rate = share of files with s >= tau_{n_f}. Intervals: two stages, the Jamendo-A artists are resampled to re-estimate every tau_n
and the evaluation clusters are resampled (B replicates); loT/hiT keep tau fixed.
Outputs (out/): track_tau_<prefix>_{thresholds,rates}.csv
usage: python code/agg_track_tau.py --models "B0;M0" [--B 2000] [--prefix all]
"""
import os, sys, argparse, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import agg_window_variants as W  # noqa: E402  (load_windows, cohort, load_metas, seed_of, wilson_hi)

TB = C.TB; LEVELS = [0.01, 0.02, 0.05]; NMAX = 12


def trim20u(lg):
    lg = np.sort(np.asarray(lg, float)); n = len(lg); k = int(np.floor(0.2 * n))
    return float(lg[:n - k].mean()) if n - k >= 1 else float(lg.mean())


def file_scores(j):
    """Window table -> file table (ID, cluster_key, n_w, short, score = trim20u)."""
    g = j.sort_values(['ID', 'j']).groupby('ID', sort=False)
    f = g.agg(cluster_key=('cluster_key', 'first'), n_w=('lg', 'size'), short=('short', 'first')).reset_index()
    f['score'] = g.lg.apply(trim20u).values
    return f


def calib_scores_by_n(j):
    """Jamendo-A: trim20u of the first n windows of every track, n = 1..NMAX -> DataFrame(ID, cluster_key, n, score)."""
    rows = []
    for ID, g in j.sort_values(['ID', 'j']).groupby('ID', sort=False):
        lg = g.lg.values; ck = g.cluster_key.iloc[0]
        for n in range(1, min(NMAX, len(lg)) + 1): rows.append((ID, ck, n, trim20u(lg[:n])))
    return pd.DataFrame(rows, columns=['ID', 'cluster_key', 'n', 'score'])


def kth(s, alpha):
    s = np.sort(np.asarray(s, float))[::-1]; k = int(np.floor(alpha * len(s)))
    return (s[k - 1], k) if k >= 1 else (np.inf, 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--models', required=True); ap.add_argument('--B', type=int, default=2000); ap.add_argument('--prefix', default='all')
    a = ap.parse_args(); B = a.B; t0 = time.time()
    reg = C.load_registry(); metas = W.load_metas(reg)
    th_rows, rate_rows = [], []
    for tag in C.parse_names(a.models):
        mname = tag; wins = W.load_windows(tag, reg)
        if not wins: print(f'[skip] {tag}', flush=True); continue
        coh = {}
        for _, r in reg.iterrows():
            j = W.cohort(r, wins, metas)
            if j is not None: coh[r['name']] = (r, j)
        if 'jamendo_A' not in coh: print(f'[skip] {tag}: no jamendo_A'); continue
        cal = calib_scores_by_n(coh['jamendo_A'][1])
        # point thresholds tau_n(alpha) and bootstrap thresholds (artists resampled, fixed seed)
        TAU = {}; TAUB = {}
        codes_all, uniq = pd.factorize(cal[cal.n == 1].cluster_key.astype(str)); G = len(uniq)
        rng = np.random.default_rng(W.seed_of('filetau', 'jamA'))
        CW = np.vstack([rng.multinomial(G, np.full(G, 1.0 / G), size=min(500, B - a0)) for a0 in range(0, B, 500)]).astype(np.float64)  # (B, G)
        for n in range(1, NMAX + 1):
            c = cal[cal.n == n]
            if len(c) == 0: continue
            codes = pd.Series(c.cluster_key.astype(str)).map({u: i for i, u in enumerate(uniq)}).values
            s = c.score.values.astype(float); o = np.argsort(-s, kind='stable'); ss = s[o]; co = codes[o]
            for al in LEVELS:
                T, k = kth(s, al); TAU[(n, al)] = T
                Wm = CW[:, co]; cum = np.cumsum(Wm, axis=1); kb = np.floor(al * Wm.sum(1)); ok = kb >= 1
                idx = np.argmax(cum >= np.maximum(kb, 1)[:, None], axis=1); TAUB[(n, al)] = np.where(ok, ss[idx], np.inf)
                th_rows.append(dict(model=tag, name=mname, n=n, alpha=al, n_tracks=len(c), k=k, tau=T, tau_lo=np.percentile(TAUB[(n, al)], 2.5), tau_hi=np.percentile(TAUB[(n, al)], 97.5)))
        for name, (r, j) in coh.items():
            f = file_scores(j); f['n_cap'] = f.n_w.clip(upper=NMAX)
            codes_f, uf = pd.factorize(f.cluster_key.astype(str)); Gf = len(uf)
            rng = np.random.default_rng(W.seed_of('coh', name, 'all'))
            for al in LEVELS:
                tau_f = np.array([TAU.get((n, al), np.inf) for n in f.n_cap]); flag = (f.score.values >= tau_f).astype(float)
                pt = 100 * flag.mean()
                # two stages: tau_n re-estimated in every replicate + evaluation clusters resampled
                TB_ = np.stack([TAUB[(n, al)] if (n, al) in TAUB else np.full(B, np.inf) for n in range(1, NMAX + 1)])  # (NMAX, B)
                Fb = (f.score.values[None, :] >= TB_[f.n_cap.values - 1, :].T)  # (B, nf)
                vals = np.empty(B); valsT = np.empty(B)
                for a0 in range(0, B, 500):
                    b = min(B, a0 + 500); cw = rng.multinomial(Gf, np.full(Gf, 1.0 / Gf), size=b - a0).astype(np.float64); Wf = cw[:, codes_f]; den = np.maximum(Wf.sum(1), 1e-9)
                    vals[a0:b] = (Wf * Fb[a0:b]).sum(1) / den; valsT[a0:b] = (Wf * flag[None, :]).sum(1) / den
                rate_rows.append(dict(model=tag, name=mname, cohort=name, role=r['role'], alpha=al, n_files=len(f), clusters=Gf, n_w_median=float(f.n_w.median()), padded_files=int(f['short'].sum()),
                                      rate=pt, lo=100 * np.percentile(vals, 2.5), hi=100 * np.percentile(vals, 97.5), loT=100 * np.percentile(valsT, 2.5), hiT=100 * np.percentile(valsT, 97.5),
                                      wilson_hi_G=W.wilson_hi(0, Gf) if pt == 0 else np.nan))
        print(f'  {tag} done {time.time() - t0:.0f}s', flush=True)
    th = pd.DataFrame(th_rows); rt = pd.DataFrame(rate_rows); pre = f'{TB}/track_tau_{a.prefix}'
    th.to_csv(pre + '_thresholds.csv', index=False); rt.to_csv(pre + '_rates.csv', index=False)
    print('TRACK_TAU_DONE', pre, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
