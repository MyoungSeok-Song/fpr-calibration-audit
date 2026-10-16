"""Window-level aggregation: thresholds, FPR/TPR per cohort, excess ratios over Jamendo-B (Sections 5.1-5.3, native rate).

Decision unit = the detector's native window (EAT/MERT detectors 10.24 s, SpecTTTra 5 s, ArtifactNet 4 s with at most 7
windows, Fakeprint LR on the 10.24-s window grid).
Definitions
  tau_win(src, alpha) = weighted (1 - alpha) quantile of the window scores of the calibration cohort, every window of a
                        file weighted 1/(windows of that file), so each track counts once. Implementation: the score of the
                        last window whose descending cumulative weight does not exceed alpha * total weight (for equal
                        weights this is the floor(alpha * n)-th largest score). A window is flagged when s >= tau.
  window FPR (TPR)    = per file, the fraction of its windows at or above tau, averaged over files (each file weighs the same).
  short inputs        = files shorter than 10.24 s (one reflect-padded window for the EAT detectors) and longer files are
                        also reported separately (suffixes [short] and [long]).
Intervals: cluster bootstrap with B replicates (default 10,000, seed 0, restarted for every detector).
  - Every replicate re-estimates tau: multinomial weights over the calibration artists (clusters), window weight =
    cluster count / windows of the file, then the weighted quantile.
  - Evaluation cohorts are resampled independently by their registry cluster (artist, film, clip, source recording,
    speaker or file); all cohorts share the replicate's tau.
  - AI cohorts are stratified by generator: files are resampled within each generator; per-generator rows are added.
  - Wilson 95% intervals use the number of clusters as n (conservative).
Excess ratio (window level): r_b = FPR_X,b / FPR_JamB,b with the same tau_b (X and Jamendo-B clusters resampled independently).
  - strict: 2.5th percentile of r >= 3
  - orig  : point ratio >= 3 and 2.5th percentile of r > 1 (the failure criterion of the paper)
  - ci_sep: FPR_X >= 3 * FPR_JamB and CI_lo(X) > CI_hi(JamB) (reference only)
Outputs (out/): public_window_{thresholds,rates,g0}.csv
usage: python code/agg_window.py [--B 10000] [--models "B0,C0"] [--part] [--merge]
  The replication detectors SpecTTTra-alpha and SpecTTTra-beta run only when named in --models (code/extra_sonics.sh).
"""
import os, sys, argparse, time, glob
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

R = C.R; TB = C.TB; REG = C.REG
MODELS = [('B0', 'EAT 10.24 s'), ("B0'", 'EAT 10.24 s'), ('B0+0', 'EAT 10.24 s'), ('C0', 'EAT 10.24 s'), ('C0+0', 'EAT 10.24 s'),
          ('M0', 'MERT-330M 10.24 s'), ('ArtifactNet', '4 s x 7'), ('SpecTTTra', '5 s'), ('ArtifactNet (native rate)', '4 s x 7'),
          ('Fakeprint LR', 'EAT 10.24 s grid')]
EXTRA = [(n, '5 s') for n in C.REPLICATION]  # replication detectors, only when named in --models
SOURCES = {'JamA': 'jamendo_A', 'FMAov': 'fma_dev800'}
ALPHAS = [0.01, 0.02, 0.05]
G0_REF = 'jamendo_B'
SHORT_S = 10.24
BATCH = 250


def wq_tau(s, w, alpha):
    """Weighted (1 - alpha) quantile: the last score whose descending cumulative weight <= alpha * total. Returns (tau, realised weight share)."""
    o = np.argsort(-s, kind='stable'); ss = s[o]; cw = np.cumsum(w[o]); W = cw[-1]
    cnt = int((cw <= alpha * W + 1e-12).sum())
    if cnt == 0: return np.inf, 0.0
    T = ss[cnt - 1]
    return T, float(w[s >= T].sum() / W)


def boot_tau(s, base_w, ccode, G, alpha, rng, B):
    o = np.argsort(-s, kind='stable'); ss = s[o]; bw = base_w[o]; cc = ccode[o]; out = np.empty(B)
    for a in range(0, B, BATCH):
        m = min(B, a + BATCH) - a
        cw = rng.multinomial(G, np.full(G, 1.0 / G), size=m).astype(np.float64)
        cum = np.cumsum(cw[:, cc] * bw[None, :], axis=1)
        cnt = (cum <= alpha * cum[:, -1:] + 1e-12).sum(1)
        out[a:a + m] = np.where(cnt > 0, ss[np.maximum(cnt - 1, 0)], np.inf)
    return out


class Cohort:
    """Window scores, files, clusters and strata of one cohort.
    Window rate = sum_c w_f(c)/n_f(c) * 1[s_c >= tau] / sum_f w_f (one descending cumulative sum serves every tau)."""

    def __init__(self, sc, fidx, fclust, fstrata=None):
        self.nf = int(fidx.max()) + 1
        self.nchk = np.bincount(fidx, minlength=self.nf).astype(np.float64)
        o = np.argsort(-sc, kind='stable')
        self.ss = sc[o]; self.fi = fidx[o]; self.inv_n = 1.0 / self.nchk[self.fi]
        self.fclust = pd.factorize(pd.Series(fclust))[0]; self.G = int(self.fclust.max()) + 1
        self.fstrata = None if fstrata is None else pd.factorize(pd.Series(fstrata))[0]

    def rate(self, T):
        pos = np.searchsorted(-self.ss, -np.asarray(T, float), side='right')
        cum = np.r_[0.0, np.cumsum(self.inv_n)]
        return cum[pos] / self.nf

    def boot(self, Tmat, rng, B):
        out = np.empty((Tmat.shape[0], B))
        pos = np.searchsorted(-self.ss, -Tmat.ravel(), side='right').reshape(Tmat.shape)
        for a in range(0, B, BATCH):
            m = min(B, a + BATCH) - a
            if self.fstrata is None:
                cw = rng.multinomial(self.G, np.full(self.G, 1.0 / self.G), size=m).astype(np.float64)
                Wf = cw[:, self.fclust]
            else:  # generator strata: resample files within each generator
                Wf = np.zeros((m, self.nf))
                for st in np.unique(self.fstrata):
                    idx = np.where(self.fstrata == st)[0]
                    Wf[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=m)
            cum = np.concatenate([np.zeros((m, 1)), np.cumsum(Wf[:, self.fi] * self.inv_n[None, :], axis=1)], axis=1)
            den = Wf.sum(1)
            for t in range(Tmat.shape[0]):
                out[t, a:a + m] = cum[np.arange(m), pos[t, a:a + m]] / np.maximum(den, 1e-12)
        return out


def wilson(p, n, z=1.959964):
    if n <= 0 or not np.isfinite(p): return (np.nan, np.nan)
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_chunks(det, dump):
    return C.load_chunks(det, dump, cols=('ID', 'j', 'lg', 'dur', 'mp', 'short'))


def merge_parts():
    out = {}
    order = {t: i for i, (t, _) in enumerate(MODELS + EXTRA)}
    for kind in ('thresholds', 'rates', 'g0'):
        fs = sorted(glob.glob(f'{TB}/window_parts/*_{kind}.csv'))
        dd = [pd.read_csv(f, keep_default_na=False, na_values=['']) for f in fs if os.path.getsize(f) > 1]
        out[kind] = pd.concat(dd, ignore_index=True) if dd else pd.DataFrame()
        if not out[kind].empty: out[kind] = out[kind].sort_values('model', key=lambda c: c.map(order), kind='stable')
        out[kind].to_csv(f'{TB}/public_window_{kind}.csv', index=False)
    for k in ('pass_strict', 'pass_orig', 'pass_ci_sep'):
        if k in out['g0']: out['g0'][k] = out['g0'][k].astype(str).str.lower().isin(['true', '1'])
    if not out['g0'].empty:  # a ratio is undefined when the Jamendo-B point FPR is 0, so strict/orig do not pass
        g = out['g0']; jb0 = g.jamB.astype(float) <= 0
        g.loc[jb0, 'pass_strict'] = False; g.loc[jb0, 'pass_orig'] = False
        g.to_csv(f'{TB}/public_window_g0.csv', index=False)
    print('MERGE_DONE', {k: len(v) for k, v in out.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=10000); ap.add_argument('--models', default='', help='paper or directory names, comma-separated')
    ap.add_argument('--part', action='store_true', help='per-detector run: write out/window_parts/<detector>_*.csv')
    ap.add_argument('--merge', action='store_true', help='merge out/window_parts into out/public_window_*.csv')
    a = ap.parse_args()
    if a.merge: return merge_parts()
    t0 = time.time(); B = a.B
    reg = C.load_registry()
    metas = {}
    for p in reg.list.unique():
        m = C.read_list(f'{R}/{p}'); m['_row'] = np.arange(len(m)); metas[p] = m
    models = C.select(MODELS, EXTRA, a.models)
    TH, RT, G0 = [], [], []
    for tag, unit in models:
        mname = tag
        rng = np.random.default_rng(0)
        chunks = {d: load_chunks(tag, d) for d in reg.dump.unique()}
        chunks = {d: c for d, c in chunks.items() if c is not None}
        if not chunks: print(f'[skip] {tag}: no scores', flush=True); continue
        if tag not in C.PUBLIC:
            need = set(reg[~reg.role.isin(['calib_probe'])].dump) - {'suno_ext_dev', 'udio_hf_dev'}
            miss = sorted(need - set(chunks))
            if miss: print(f'[skip] {tag}: missing scores {miss}', flush=True); continue
        # cohorts (+ length variants; AI cohorts also per generator)
        coh = {}
        for _, r in reg.iterrows():
            if r['dump'] not in chunks or r['role'] == 'calib_probe': continue
            m = metas[r['list']]; sub = m.query(r['subset']) if r['subset'] else m
            c = chunks[r['dump']]; c = c[c.ID.isin(set(sub._row))]
            if c.empty: continue
            fm = sub.set_index('_row')
            ids = np.sort(c.ID.unique()); fpos = {v: i for i, v in enumerate(ids)}
            fidx = c.ID.map(fpos).values
            fclust = fm.loc[ids, r['cluster']].astype(str).values
            dur = c.groupby('ID').dur.first().reindex(ids).values
            grp = fm.loc[ids, 'group'].astype(str).values if 'group' in fm else np.array(['all'] * len(ids))
            variants = [('', np.ones(len(ids), bool))]
            if (dur < SHORT_S).any() and (dur >= SHORT_S).any():
                variants += [('[long]', dur >= SHORT_S), ('[short]', dur < SHORT_S)]
            elif (dur < SHORT_S).all():
                variants[0] = ('', np.ones(len(ids), bool))
            for suf, keep in variants:
                kf = np.where(keep)[0]; kset = set(kf)
                ck = np.array([fi in kset for fi in fidx])
                remap = {f: i for i, f in enumerate(kf)}
                fi2 = np.array([remap[f] for f in fidx[ck]])
                strata = grp[kf] if r['role'] == 'fake' else None
                coh[r['name'] + suf] = (r, Cohort(c.lg.values[ck].astype(float), fi2, fclust[kf], strata),
                                        dict(n=len(kf), chunks=int(ck.sum()), clusters=len(set(fclust[kf])), short=int((dur[kf] < SHORT_S).sum())))
                if r['role'] == 'fake' and suf == '' and len(set(grp)) > 1:
                    for g in sorted(set(grp)):
                        kg = np.where(grp == g)[0]; gs = set(kg); cg = np.array([fi in gs for fi in fidx])
                        rm = {f: i for i, f in enumerate(kg)}
                        coh[f"{r['name']}::{g}"] = (r, Cohort(c.lg.values[cg].astype(float), np.array([rm[f] for f in fidx[cg]]), fclust[kg]),
                                                    dict(n=len(kg), chunks=int(cg.sum()), clusters=len(kg), short=int((dur[kg] < SHORT_S).sum())))
        # thresholds
        Tb_all, Tpt, keys = [], [], []
        for src, cname in SOURCES.items():
            if cname not in coh: continue
            r, Co, info = coh[cname]
            base_w = Co.inv_n; ccode = Co.fclust[Co.fi]
            for al in ALPHAS:
                T, real = wq_tau(Co.ss, base_w, al)
                Tb = boot_tau(Co.ss, base_w, ccode, Co.G, al, rng, B)
                fin = Tb[np.isfinite(Tb)]
                key = f'tau_{src}@{int(al * 100)}%'
                TH.append(dict(model=tag, name=mname, unit=unit, thr=key, source=cname, alpha=al, n_files=info['n'], n_chunks=info['chunks'],
                               clusters=info['clusters'], tau=T, cal_rate_chunkweighted=100 * real, cal_rate_window=100 * float(Co.rate([T])[0]),
                               tau_lo=np.percentile(fin, 2.5) if len(fin) else np.nan, tau_hi=np.percentile(fin, 97.5) if len(fin) else np.nan,
                               n_inf=int((~np.isfinite(Tb)).sum())))
                Tb_all.append(Tb); Tpt.append(T); keys.append(key)
        Tmat = np.stack(Tb_all); Tpt = np.array(Tpt)
        boots = {}
        for name, (r, Co, info) in coh.items():
            pt = Co.rate(Tpt); bt = Co.boot(Tmat, rng, B); boots[name] = bt
            for ti, key in enumerate(keys):
                lo, hi = wilson(pt[ti], info['clusters'])
                RT.append(dict(model=tag, name=mname, unit=unit, cohort=name, role=r['role'], thr=key, tau=Tpt[ti], n=info['n'], chunks=info['chunks'],
                               clusters=info['clusters'], n_short=info['short'], rate=100 * pt[ti], ci_lo=100 * np.percentile(bt[ti], 2.5),
                               ci_hi=100 * np.percentile(bt[ti], 97.5), wil_lo_clusters=100 * lo, wil_hi_clusters=100 * hi))
        # excess ratios over Jamendo-B (tau_JamA, every alpha)
        if G0_REF in coh:
            for ti, key in enumerate(keys):
                if not key.startswith('tau_JamA'): continue
                rb = boots[G0_REF][ti]; rp = coh[G0_REF][1].rate([Tpt[ti]])[0]
                for name, (r, Co, info) in coh.items():
                    if r['role'] not in ('real', 'null', 'real_composer_exposed') or name.startswith(G0_REF) or '::' in name: continue
                    xb = boots[name][ti]; xp = Co.rate([Tpt[ti]])[0]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio_b = np.where(rb > 0, xb / rb, np.where(xb > 0, np.inf, np.nan))
                    rr = xp / rp if rp > 0 else (np.inf if xp > 0 else np.nan)
                    fin = ratio_b[~np.isnan(ratio_b)]
                    rlo = np.percentile(fin, 2.5) if len(fin) else np.nan; rhi = np.percentile(fin, 97.5) if len(fin) else np.nan
                    xlo, xhi = np.percentile(xb, [2.5, 97.5]); jlo, jhi = np.percentile(rb, [2.5, 97.5])
                    G0.append(dict(model=tag, name=mname, thr=key, cohort=name, role=r['role'], n=info['n'], clusters=info['clusters'],
                                   fpr=100 * xp, ci_lo=100 * xlo, ci_hi=100 * xhi, jamB=100 * rp, jamB_lo=100 * jlo, jamB_hi=100 * jhi,
                                   ratio=rr, ratio_lo=rlo, ratio_hi=rhi, pass_strict=bool(rp > 0 and rlo >= 3), pass_orig=bool(rp > 0 and rr >= 3 and rlo > 1),
                                   pass_ci_sep=bool(xp >= 3 * rp and xlo > jhi)))
        print(f'[{tag}] cohorts {len(coh)} thr {keys} {time.time() - t0:.0f}s', flush=True)
    th = pd.DataFrame(TH); rt = pd.DataFrame(RT); g0 = pd.DataFrame(G0)
    if a.part:
        os.makedirs(f'{TB}/window_parts', exist_ok=True); tg = '_'.join(C.DETECTORS[x[0]][0] for x in models)
        for kind, df in (('thresholds', th), ('rates', rt), ('g0', g0)):
            df.to_csv(f'{TB}/window_parts/{tg}_{kind}.csv', index=False)
        print(f'AGG_WINDOW_PART_DONE {tg} {time.time() - t0:.0f}s', flush=True); return
    th.to_csv(f'{TB}/public_window_thresholds.csv', index=False); rt.to_csv(f'{TB}/public_window_rates.csv', index=False)
    g0.to_csv(f'{TB}/public_window_g0.csv', index=False)
    print(f'AGG_WINDOW_DONE {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
