"""Window-level aggregation with paired comparisons between detector variants (Sections 5.1, 5.3, 5.4, 7).

Used for the one-epoch variants (B0+0, B0+0 (seed 2), B0+SU, B0+AVA, B0+broad) and for the paired AI-OpenBMAT AUROC of M0. Only the public cohort registry is read.
Decision unit = one window (one row of the window scores). Padded windows (inputs shorter than 10.24 s, reflect-padded) are
reported separately.
tau_{d,alpha} = track-weighted (1/n_f) empirical (1 - alpha) quantile of the Jamendo-A window scores: the last score whose
         descending cumulative weight does not exceed alpha * total weight (calibration FPR <= alpha), alpha in {1, 2, 5}%
         (main 2%). Secondary source FMAov = fma_dev800 (same rule).
FPR / TPR = per file f the share of flagged windows (1/n_f) sum 1[s >= tau], averaged over files. Subsets: all / long (short == 0)
         / padded (short == 1).
Intervals: two-stage cluster bootstrap with B replicates: every replicate resamples the calibration clusters (artists) and
         re-estimates tau, and resamples the evaluation clusters (registry cluster column) independently; percentile 95%.
         Seeds are fixed per cohort and threshold, so different detectors get the same cluster draws (paired comparisons).
Zero counts: when the point estimate is 0, the Wilson 95% upper bound with the number of clusters G as n is also given.
Paired AUROC (AI-OpenBMAT): window level, human and AI windows at the same position of the same mix (pair, j), a window is
         assigned to a class when >= 80% of it carries that class (openbmat_window_classes.csv); clusters = cluster_ob (pairs
         resampled together).
Parts (--parts): rates   = thresholds, window rates per cohort, ratios to Jamendo-B (+ the replicates used by matched_tpr_variants.py)
                 paired  = paired differences with the reference detector (--ref)
                 auroc   = paired window AUROC on AI-OpenBMAT
Inputs: scores/windows/<detector>/<dataset>.csv.gz, out/openbmat_window_classes.csv (code/openbmat_window_classes.py)
Outputs (out/): variants_<prefix>_{thresholds,rates,ratios}.csv and variants_<prefix>_boot.npz (rates), variants_<prefix>_paired.csv
         (paired), variants_<prefix>_auroc.csv (auroc)
usage: python code/agg_window_variants.py --models "B0+0;B0+0 (seed 2);B0+SU;B0+AVA;B0+broad" --prefix seednoise [--B 10000] [--ref B0+0]
       [--parts rates,paired,auroc]
"""
import os, sys, argparse, time, hashlib
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from auroc import wauc  # noqa: E402

R = C.R; TB = C.TB; REG = C.REG
SOURCES = {'JamA': 'jamendo_A', 'FMAov': 'fma_dev800'}
LEVELS = [0.01, 0.02, 0.05]
SUBSETS = {'all': None, 'long': 0, 'padded': 1}
MAIN = 'tau_JamA@2%'
G0_COH = ['ava_sm5', 'ava_sm_long', 'audioset_sm', 'openbmat_human_musicbearing', 'ava_mo', 'audioset_mo', 'gtzan', 'musdb18', 'librispeech_tc', 'esc50']


def seed_of(*parts):
    """Fixed seed per cohort, threshold or AUROC class -> the same cluster draws for every detector (paired comparisons)."""
    return int(hashlib.md5('|'.join(str(x) for x in parts).encode()).hexdigest()[:8], 16)


def wilson_hi(k, n, z=1.96):
    if n <= 0: return np.nan
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c + h) / d


def tau_weighted(s, w, alpha):
    """The last score whose descending cumulative weight does not exceed alpha * total weight (inf if none)."""
    o = np.argsort(-s, kind='stable'); ss = s[o]; cum = np.cumsum(w[o]); k = alpha * w.sum()
    idx = int(np.searchsorted(cum, k, side='right')) - 1
    return ss[idx] if idx >= 0 else np.inf


def boot_tau_w(s, w, cl, alpha, rng, B):
    """B replicates of tau with the calibration clusters resampled; window weight w (1/n_f) times cluster multiplicity."""
    o = np.argsort(-s, kind='stable'); ss = s[o]; w = w[o]
    codes, uniq = pd.factorize(pd.Series(cl).astype(str)); codes = codes[o]; G = len(uniq)
    out = np.empty(B)
    for a in range(0, B, 500):
        b = min(B, a + 500); m = b - a
        cw = rng.multinomial(G, np.full(G, 1.0 / G), size=m).astype(np.float64)
        W = cw[:, codes] * w[None, :]; cum = np.cumsum(W, axis=1); k = alpha * W.sum(1)
        cnt = (cum <= k[:, None] + 1e-12).sum(1)  # number of windows not exceeding the target weight
        out[a:b] = np.where(cnt >= 1, ss[np.maximum(cnt - 1, 0)], np.inf)
    return out


def boot_rates(s, wf, codes_w, codes_f, G, Tmat, rng, B):
    """Evaluation cohort: window scores s, window weights wf = 1/n_f, cluster code per window codes_w and per file codes_f.
    Tmat (n_thr x B) -> rates (n_thr x B)."""
    out = np.empty((Tmat.shape[0], B))
    for a in range(0, B, 500):
        b = min(B, a + 500); m = b - a
        cw = rng.multinomial(G, np.full(G, 1.0 / G), size=m).astype(np.float32)
        Ww = cw[:, codes_w] * wf[None, :].astype(np.float32); den = np.maximum(cw[:, codes_f].sum(1), 1e-9)
        for t in range(Tmat.shape[0]):
            F = (s[None, :] >= Tmat[t, a:b, None]).astype(np.float32)
            out[t, a:b] = (F * Ww).sum(1) / den
    return out


def load_windows(tag, reg):
    out = {}
    for d in reg.dump.unique():
        c = C.load_chunks(tag, d, cols=('ID', 'j', 'n', 'dur', 'short', 'lg'))
        if c is not None: out[d] = c
    return out


def cohort(r, wins, metas):
    if r['dump'] not in wins: return None
    m = metas[r['list']]; sub = m.query(r['subset']) if r['subset'] else m
    j = sub[['_row', r['cluster']]].rename(columns={'_row': 'ID', r['cluster']: 'cluster_key'}).merge(wins[r['dump']], on='ID', how='inner')
    if len(j) == 0: return None
    j['cluster_key'] = j['cluster_key'].astype(str)
    return j.sort_values(['ID', 'j']).reset_index(drop=True)


def file_table(j):
    """Window table -> file table (ID, cluster_key, n_w, short)."""
    f = j.groupby('ID', sort=False).agg(cluster_key=('cluster_key', 'first'), n_w=('lg', 'size'), short=('short', 'first')).reset_index()
    return f


def load_metas(reg):
    metas = {}
    for p in reg.list.unique():
        m = C.read_list(f'{R}/{p}'); m['_row'] = np.arange(len(m)); metas[p] = m
    return metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True, help='paper or directory names separated by ";" (or "," if no name contains a comma)')
    ap.add_argument('--B', type=int, default=10000); ap.add_argument('--prefix', default='all')
    ap.add_argument('--tb', default=TB); ap.add_argument('--ref', default='B0+0', help='reference detector of the paired comparisons')
    ap.add_argument('--parts', default='rates,paired,auroc', help='comma-separated subset of rates, paired, auroc')
    a = ap.parse_args(); B = a.B; os.makedirs(a.tb, exist_ok=True); t0 = time.time()
    parts = {x.strip() for x in a.parts.split(',') if x.strip()}
    assert parts <= {'rates', 'paired', 'auroc'} and parts, a.parts
    if 'paired' in parts: parts.add('rates')  # the paired differences reuse the replicates of the rates
    ref_det = C.resolve(a.ref)
    reg = C.load_registry()
    metas = load_metas(reg)
    thr_rows, rate_rows, auc_rows = [], [], []; boot = {}
    for tag in C.parse_names(a.models):
        mname = tag
        wins = load_windows(tag, reg)
        if not wins: print(f'[skip] {tag}: no scores', flush=True); continue
        coh = {}
        for _, r in reg.iterrows():
            j = cohort(r, wins, metas)
            if j is not None: coh[r['name']] = (r, j)
        # ---- thresholds ----
        TH = {}
        for src, cname in (SOURCES.items() if 'rates' in parts else []):
            if cname not in coh: continue
            r, j = coh[cname]; f = file_table(j)
            nf = j.groupby('ID').lg.transform('size').values.astype(float); w = 1.0 / nf; s = j.lg.values.astype(float)
            for al in LEVELS:
                T = tau_weighted(s, w, al); Tb = boot_tau_w(s, w, j.cluster_key.values, al, np.random.default_rng(seed_of('tau', src, al)), B); key = f'tau_{src}@{int(al * 100)}%'
                TH[key] = (T, Tb, cname); fin = Tb[np.isfinite(Tb)]
                cal = 100 * float(((s >= T) * w).sum() / w.sum())
                thr_rows.append(dict(model=tag, name=mname, thr=key, source=cname, alpha=al, n_files=len(f), n_windows=len(j), clusters=f.cluster_key.nunique(),
                                     tau=T, cal_fpr_win=cal, tau_lo=np.percentile(fin, 2.5), tau_hi=np.percentile(fin, 97.5)))
        # ---- window rates per cohort ----
        if 'rates' in parts:
            keys = list(TH); Tmat = np.stack([TH[k][1] for k in keys]); Tpt = np.array([TH[k][0] for k in keys])
        for name, (r, j) in (coh.items() if 'rates' in parts else []):
            for sub_name, sv in SUBSETS.items():
                jj = j if sv is None else j[j['short'] == sv]
                if len(jj) == 0: continue
                f = file_table(jj); codes_f, uniq = pd.factorize(f.cluster_key); G = len(uniq)
                fid = pd.factorize(jj.ID)[0]; codes_w = codes_f[fid]
                nf = f.n_w.values[fid].astype(float); wf = 1.0 / nf; s = jj.lg.values.astype(float)
                br = boot_rates(s, wf, codes_w, codes_f, G, Tmat, np.random.default_rng(seed_of('coh', name, sub_name)), B)
                bf = boot_rates(s, wf, codes_w, codes_f, G, np.repeat(Tpt[:, None], B, axis=1), np.random.default_rng(seed_of('coh', name, sub_name)), B)
                if MAIN in keys: boot[(tag, name, sub_name)] = br[keys.index(MAIN)].astype(np.float32)
                for t, k in enumerate(keys):
                    Fw = (s >= Tpt[t]).astype(float); v = np.bincount(fid, weights=Fw * wf, minlength=len(f))  # share of flagged windows per file
                    pt = 100 * v.mean(); kpos = int((v > 0).sum())
                    rate_rows.append(dict(model=tag, name=mname, cohort=name, role=r['role'], subset=sub_name, thr=k, tau=Tpt[t], n_files=len(f), n_windows=len(jj),
                                          clusters=G, rate=pt, lo=100 * np.percentile(br[t], 2.5), hi=100 * np.percentile(br[t], 97.5),
                                          loT=100 * np.percentile(bf[t], 2.5), hiT=100 * np.percentile(bf[t], 97.5),
                                          files_any=kpos, wilson_hi_G=wilson_hi(0, G) if pt == 0 else np.nan))
            print(f'  {tag} {name} done {time.time() - t0:.0f}s', flush=True)
        # ---- paired AUROC (AI-OpenBMAT) ----
        # window classes from out/openbmat_window_classes.csv (>= 80% overlap, otherwise 'mixed'); human and AI windows of the same
        # (pair, j) are paired; clusters cluster_ob resampled (pairs together). music-bearing = class in {mus, fg, bg, bgvl, sim}.
        WC = f'{a.tb}/openbmat_window_classes.csv'
        if 'auroc' in parts:
            assert os.path.exists(WC), 'run code/openbmat_window_classes.py first'
        if 'auroc' in parts and 'openbmat_human' in coh and 'openbmat_ai' in coh:
            wc = pd.read_csv(WC, dtype={'item_id': str})
            arms = {}
            for arm, cname in (('human', 'openbmat_human'), ('ai', 'openbmat_ai')):
                r, j = coh[cname]; m = metas[r['list']][['_row', 'item_id']].rename(columns={'_row': 'ID'})
                jj = j.merge(m, on='ID').merge(wc[wc.arm == arm][['item_id', 'j', 'pair', 'cls']], on=['item_id', 'j'], how='inner')
                arms[arm] = jj[['pair', 'j', 'lg', 'cluster_key', 'cls']]
            pw = arms['human'].merge(arms['ai'], on=['pair', 'j'], suffixes=('_h', '_a'))
            assert (pw.cls_h == pw.cls_a).all()
            for cls in ['musicbearing', 'mus', 'fg', 'bg', 'bgvl', 'sim', 'nomus', 'mixed']:
                sel = pw[pw.cls_h.isin(['mus', 'fg', 'bg', 'bgvl', 'sim'])] if cls == 'musicbearing' else pw[pw.cls_h == cls]
                if len(sel) < 20: continue
                s = np.r_[sel.lg_h.values, sel.lg_a.values].astype(float); y = np.r_[np.zeros(len(sel)), np.ones(len(sel))]
                codes, uniq = pd.factorize(pd.concat([sel.cluster_key_h, sel.cluster_key_h]).astype(str)); G = len(uniq); vals = []; rng = np.random.default_rng(seed_of('auc', cls))
                for a0 in range(0, B, 500):
                    m_ = min(B, a0 + 500) - a0; cw = rng.multinomial(G, np.full(G, 1.0 / G), size=m_).astype(np.float64); vals.append(wauc(s, y, cw[:, codes]))
                vals = np.concatenate(vals); pt = float(wauc(s, y, np.ones((1, len(s))))[0])
                auc_rows.append(dict(model=tag, name=mname, unit='window', cls=cls, n_human=len(sel), n_ai=len(sel), clusters=G, auroc=pt,
                                     lo=np.percentile(vals, 2.5), hi=np.percentile(vals, 97.5), score='paired windows, class = >=80% overlap (openbmat_window_classes.csv)'))
    th = pd.DataFrame(thr_rows); rt = pd.DataFrame(rate_rows); au = pd.DataFrame(auc_rows)
    pre = f'{a.tb}/variants_{a.prefix}'
    if 'rates' in parts:
        th.to_csv(pre + '_thresholds.csv', index=False); rt.to_csv(pre + '_rates.csv', index=False)
        np.savez_compressed(pre + '_boot.npz', **{f'{t}|{c}|{sb}': v for (t, c, sb), v in boot.items()})
        write_ratios(rt, boot, pre, ref_det, paired='paired' in parts)
    if 'auroc' in parts: au.to_csv(pre + '_auroc.csv', index=False)
    print('AGG_WINDOW_VARIANTS_DONE', pre, f'{time.time() - t0:.0f}s', flush=True)


def write_ratios(rt, boot, pre, ref, paired=True):
    """(1) ratio = cohort FPR / Jamendo-B FPR (same detector, same replicate) with interval and the window failure rule
       (ratio >= 3 and 2.5th percentile > 1). (2) Paired comparison with the reference detector: difference in FPR (pp) and ratio of
       ratios (same seeds -> same cluster draws)."""
    pt = {(r.model, r.cohort, r.subset): r.rate for r in rt[rt.thr == MAIN].itertuples()}
    models = list(dict.fromkeys(rt.model)); rows = []; prow = []
    for m in models:
        for sb in ('all', 'long'):
            jb = boot.get((m, 'jamendo_B', sb))
            if jb is None: continue
            for c in G0_COH:
                x = boot.get((m, c, sb))
                if x is None: continue
                q = x / np.maximum(jb, 1e-9); r0 = pt[(m, c, sb)] / max(pt[(m, 'jamendo_B', sb)], 1e-9)
                lo, hi = np.percentile(q, 2.5), np.percentile(q, 97.5)
                rows.append(dict(model=m, cohort=c, subset=sb, fpr=pt[(m, c, sb)], jamB=pt[(m, 'jamendo_B', sb)], ratio=r0, lo=lo, hi=hi, g0=bool(r0 >= 3 and lo > 1)))
                if paired and ref in models and m != ref and (ref, c, sb) in boot and (ref, 'jamendo_B', sb) in boot:
                    xr = boot[(ref, c, sb)]; jr = boot[(ref, 'jamendo_B', sb)]
                    d = x - xr  # rates in [0, 1]; x100 below
                    rr = q / np.maximum(xr / np.maximum(jr, 1e-9), 1e-9)
                    prow.append(dict(model=m, ref=ref, cohort=c, subset=sb, fpr=pt[(m, c, sb)], fpr_ref=pt[(ref, c, sb)], delta=pt[(m, c, sb)] - pt[(ref, c, sb)],
                                     d_lo=100 * np.percentile(d, 2.5), d_hi=100 * np.percentile(d, 97.5), ratio=r0, ratio_ref=pt[(ref, c, sb)] / max(pt[(ref, 'jamendo_B', sb)], 1e-9),
                                     rr=float(np.median(rr)), rr_lo=np.percentile(rr, 2.5), rr_hi=np.percentile(rr, 97.5)))
    pd.DataFrame(rows).to_csv(pre + '_ratios.csv', index=False)
    if paired: pd.DataFrame(prow).to_csv(pre + '_paired.csv', index=False)


if __name__ == '__main__':
    main()
