"""Last-layer heads at the window level (,, probes in, null inputs).

Input: per-window outputs of the heads (scores/heads/<family>/<dataset>.npz): logit z, positive evidence E+ (ep) and negative
evidence E- (en) for every head: orig (the detector's own last layer; fine-tuned detectors and Fakeprint LR), free_split (UC,
re-trained, unconstrained), pos_split (PE, positive evidence, non-negative weights on sign-split features), free_mlp / pos_mlp
(unconstrained / non-negative last layer on a fixed ReLU hidden layer).
Decision unit = one window (10.24 s); no track aggregation.
  tau_win(alpha): per family and head, the weighted (1 - alpha) quantile of the Jamendo-A window scores (838 tracks, 163 artists),
            window weight 1/n_f (one track, one vote); flagged when s >= tau. Same definition as agg_window.wq_tau.
            alpha in {1, 2, 5}% (main 2%).
  AI cohorts are resampled within generators (as in agg_window). Calibration sources at the window level (alpha = 2%) use the
  same rule (FMA cohorts clustered by artist); FMA-ov and FMA-test calibrate the frozen-encoder probes only.
  Cohort FPR/TPR = per file the share of windows at or above tau, averaged over files.
  Subsets: all, nonpad (>= 10.24 s, no padding, main), pad (reflect-padded inputs, reported separately).
  Intervals: two-stage cluster bootstrap with B replicates (default 10,000): every replicate draws multinomial weights over the
      Jamendo-A artists (tau_b re-estimated) and independent multinomial weights over the clusters of the evaluation cohort.
      Seeds are derived from the cohort names, so every head and family shares the same draws (paired comparisons). When the
      point estimate is 0, the Wilson 95% upper bound with the number of clusters G as n is given.
  Head contrasts (pp, paired bootstrap): confirmatory = PE (pos_split) - UC (free_split) in the worst in-the-wild cohort FPR
      (RW4: AI-OpenBMAT-H music-bearing, MUSDB18, GTZAN, AVA-SM; max over cohorts per replicate), two-sided bootstrap p, Holm over
      families (code/heads_report.py). Secondary: per-cohort differences, orig -> PE, MLP-UC -> MLP-PE, TPR difference on OOT3
      (MUSIC8K, AIME, AI-OpenBMAT-AI music-bearing).
  Null inputs: unpadded null windows (AVA clean dialogue and ambience >= 10.24 s, long LibriSpeech, synthetic 30-s noise generated at
      16 kHz): median score, its margin above tau in units of the Jamendo-A window SD, and the FPR.
  Evidence types (window level): quantiles of E+ and E- over the calibration true negatives (Jamendo-A, s < tau).
      evidence-loss = E- < P5 and P5 <= E+ <= P95; E+-type = E+ > P95; otherwise mixed. Sensitivity cuts (P1, P99) and (P10, P90).
Outputs (out/): heads_{rates,contrast,null,fa_types,provenance}.csv, heads_meta.json
usage: python code/heads_window.py [--B 10000] [--families "B0;C0;M0;EAT (frozen);EAT-AS2M (frozen);MERT (frozen);CLAP (frozen);Fakeprint LR"]
"""
import os, sys, json, argparse, time, hashlib
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

TB = C.TB
RW4 = ['openbmat_human_musicbearing', 'musdb18', 'gtzan', 'ava_sm5']
OOT3 = ['music8k', 'aime_sa', 'openbmat_ai_musicbearing']
NULL_LONG = ['ava_cs_long', 'ava_amb_long', 'librispeech_tc', 'null_pink30', 'null_white30', 'null_dither30']  # LibriSpeech: nonpad subset
ALPHAS = (0.01, 0.02, 0.05)
MAIN_HEADS = ('orig', 'free_split', 'pos_split')
ALL_HEADS = ('orig', 'free_split', 'pos_split', 'free_mlp', 'pos_mlp')
PROBE_ONLY_COHORTS = ('fma_ov_probe800', 'fma_test_disjoint800')  # calibration sources of the frozen-encoder probes only
PROBES = ('EAT (frozen)', 'EAT-AS2M (frozen)', 'MERT (frozen)', 'CLAP (frozen)')
SUP_NULL = ['null_pink10', 'null_white10', 'null_dither10', 'null_pink30', 'null_white30', 'null_dither30']  # synthetic nulls generated at 16 kHz


def seed_of(name):
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


def load_registry():
    """Cohort registry rows (+ the synthetic 16-kHz nulls, one cluster per file)."""
    reg = pd.read_csv(C.REG, keep_default_na=False, na_values=[''])
    rows = []
    for _, r in reg.iterrows():
        rows.append(dict(name=r['name'], dump=r['dump'], list=f"{C.R}/{r['list']}", subset='' if pd.isna(r['subset']) else str(r['subset']),
                         role=str(r['role']), cluster=str(r['cluster'])))
    for nm in SUP_NULL:
        rows.append(dict(name=nm, dump=nm, list=str(C.ITEMS / f'{nm}.csv'), subset='', role='null_sup', cluster='item_id'))
    return rows


def cohort_meta(row):
    m = C.read_list(row['list'])
    if row['subset']: m = m.query(row['subset'], engine='python')
    cl = row['cluster']
    key = m[cl].astype(str).values if cl in m.columns else m.item_id.astype(str).values
    return m.assign(_cluster=key).reset_index(drop=True)


def wilson_hi(k, n, z=1.959964):
    if n <= 0: return np.nan
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return min(1.0, c + h)


def wq_desc(s, w, alpha):
    """Same definition as agg_window.wq_tau: score of the last window whose descending cumulative weight <= alpha * total
    (for equal weights the floor(alpha * n)-th largest score)."""
    o = np.argsort(-s, kind='stable'); ss = s[o]; cw = np.cumsum(w[o])
    cnt = int((cw <= alpha * cw[-1] + 1e-12).sum())
    return ss[cnt - 1] if cnt > 0 else np.inf


class Win:
    """One cohort (window table): file ids, window scores -> file-mean share of flagged windows and its bootstrap.
    Human/null cohorts: multinomial resampling of registry clusters. AI cohorts: stratified by generator (files resampled
    within each generator, as in agg_window)."""
    def __init__(self, fid, nfile_w, clusters_file, name, B, strata=None):
        self.fid = fid; self.nf = nfile_w; self.G = len(self.nf)
        rng = np.random.default_rng(seed_of(name))
        if strata is None:
            codes, uniq = pd.factorize(pd.Series(clusters_file).astype(str))
            M = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=B).astype(np.float32)
            self.W = M[:, codes]  # (B, n_files)
            self.ncl = len(uniq)
        else:
            st = pd.factorize(pd.Series(strata).astype(str))[0]
            self.W = np.zeros((B, self.G), np.float32)
            for k in np.unique(st):
                idx = np.where(st == k)[0]
                self.W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=B)
            self.ncl = self.G

    def ratio(self, s, T):
        return np.bincount(self.fid, (s >= T).astype(float), self.G) / self.nf

    def point(self, s, T):
        return float(self.ratio(s, T).mean())

    def boot(self, s, Tb):
        ut, inv = np.unique(Tb, return_inverse=True)
        Rm = np.stack([self.ratio(s, t) for t in ut]).astype(np.float32)  # (n_unique, n_files)
        out = np.empty(len(Tb), np.float32)
        W = self.W[:len(Tb)]  # secondary analyses use the first B_aux replicates (a subset of the same draws)
        for i in range(len(ut)):
            rows = np.where(inv == i)[0]
            if len(rows) == 0: continue
            Wr = W[rows]
            out[rows] = (Wr @ Rm[i]) / np.maximum(Wr.sum(1), 1e-9)
        return out


def load_scores(reg, fams, dtype=np.float32, heads_wanted=ALL_HEADS):
    """(dataset, family, head) -> (z, E+, E-, z0) and the window index (item_id, j, short) per dataset."""
    S, IDX = {}, {}
    for dump in sorted({r['dump'] for r in reg}):
        for fam in fams:
            if fam not in PROBES and dump in PROBE_ONLY_COHORTS: continue
            z = C.load_head_scores(fam, dump)
            if z is None: continue
            ix = pd.DataFrame(dict(item_id=z['item_id'].astype(str), j=z['j'].astype(int), short=z['short'].astype(int)))
            if dump not in IDX: IDX[dump] = ix
            else: assert (IDX[dump].item_id.values == ix.item_id.values).all() and (IDX[dump].j.values == ix.j.values).all(), (dump, fam)
            for h in heads_wanted:
                if f'{h}__z' in z:
                    S[(dump, fam, h)] = (z[f'{h}__z'].astype(dtype), z[f'{h}__ep'].astype(np.float32), z[f'{h}__en'].astype(np.float32), float(z[f'{h}__z0'][0]))
    return S, IDX


def build_cohorts(reg, IDX):
    coh = {}
    for r in reg:
        dump = r['dump']
        if dump not in IDX: continue
        m = cohort_meta(r)
        idx = IDX[dump]
        sel = idx.item_id.isin(set(m.item_id))
        sub = idx[sel]
        fp_codes, fp_uniq = pd.factorize(sub.item_id.values)
        clmap = dict(zip(m.item_id, m._cluster))
        short = sub.groupby(fp_codes)['short'].first().values if len(sub) else np.zeros(0)
        coh[r['name']] = dict(row=r, rows=np.where(sel.values)[0], fid=fp_codes, files=fp_uniq,
                              clusters=np.array([clmap[f] for f in fp_uniq]), short=short, n_meta=len(m),
                              group=dict(zip(m.item_id, m['group'].astype(str))) if 'group' in m else {})
    return coh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=10000)
    ap.add_argument('--B_aux', type=int, default=2000, help='replicates for alpha = 1 and 5% and for the secondary heads (MLP)')
    ap.add_argument('--families', default=';'.join(C.FAMILIES), help='head families separated by ";"')
    ap.add_argument('--tag', default='heads')
    a = ap.parse_args()
    t0 = time.time(); B = a.B
    fams = [C.resolve(f, C.FAMILIES) for f in a.families.split(';') if f.strip()]
    reg = load_registry()
    S, IDX = load_scores(reg, fams)
    print(f'[scores] {len(S)} (dataset, family, head) arrays {time.time() - t0:.0f}s', flush=True)
    coh = build_cohorts(reg, IDX)
    # subset Win objects (all/nonpad/pad)
    WIN = {}

    def get_win(name, subset):
        key = (name, subset)
        if key in WIN: return WIN[key]
        c = coh[name]
        if subset == 'all': keepf = np.ones(len(c['files']), bool)
        elif subset == 'nonpad': keepf = c['short'] == 0
        else: keepf = c['short'] == 1
        if keepf.sum() == 0: WIN[key] = None; return None
        newid = -np.ones(len(c['files']), int); newid[keepf] = np.arange(keepf.sum())
        rows_mask = keepf[c['fid']]
        fid = newid[c['fid'][rows_mask]]
        nf = np.bincount(fid).astype(float)
        strata = np.array([c['group'].get(f, '') for f in c['files'][keepf]]) if c['row']['role'] == 'fake' else None
        w = Win(fid, nf, c['clusters'][keepf], f'{name}|{subset}', B, strata)
        w.rows = c['rows'][rows_mask]; w.files = c['files'][keepf]
        WIN[key] = w
        return w
    # calibration cohorts: multinomial over artists (registry clusters), window weight = cluster count x 1/n_f (one track, one vote)
    CAL = {}

    def make_cal(name):
        if name in CAL: return CAL[name]
        if name not in coh: CAL[name] = None; return None
        c = coh[name]; fid = c['fid']; nf = np.bincount(fid).astype(float)
        codes, uniq = pd.factorize(pd.Series(c['clusters']).astype(str))
        rng = np.random.default_rng(seed_of(f'{name}|win'))
        M = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=B).astype(np.float32)
        CAL[name] = dict(c=c, w=1.0 / nf[fid], art=codes[fid], M=M)
        return CAL[name]

    def tau_boot(cal, s_cal, alpha, Bn):
        o = np.argsort(-s_cal, kind='stable'); so = s_cal[o]; wo = cal['w'][o]; ao = cal['art'][o]
        out = np.empty(Bn, np.float32)
        for b0 in range(0, Bn, 500):
            Wb = cal['M'][b0:min(Bn, b0 + 500)][:, ao] * wo[None, :]
            cs = np.cumsum(Wb, 1); cnt = (cs <= alpha * cs[:, -1:] + 1e-12).sum(1)
            out[b0:b0 + len(Wb)] = np.where(cnt > 0, so[np.maximum(cnt - 1, 0)], np.inf)
        return out

    calA = make_cal('jamendo_A'); cA = calA['c']; wA = calA['w']

    rates, contrasts, nulls, fats = [], [], [], []
    boot_store = {}
    for bb in fams:
        heads = [h for h in ALL_HEADS if (('jamendo2k', bb, h) in S)]
        for h in heads:
            zA, epA, enA, z0 = S[('jamendo2k', bb, h)]
            sA = zA[cA['rows']]
            sdA = float(np.sqrt(np.cov(sA, aweights=wA)))
            for alpha in (ALPHAS if h in MAIN_HEADS else (0.02,)):
                Bn = B if (h in MAIN_HEADS and alpha == 0.02) else min(B, a.B_aux)  # main contrasts (alpha = 2%, orig/UC/PE) use all B
                Tp = float(wq_desc(sA, wA, alpha)); Tb = tau_boot(calA, sA, alpha, Bn)
                for nm, c in coh.items():
                    dump = c['row']['dump']
                    if (dump, bb, h) not in S: continue
                    z = S[(dump, bb, h)][0]
                    for subset in ('all', 'nonpad', 'pad'):
                        w = get_win(nm, subset)
                        if w is None: continue
                        if subset != 'all' and (c['short'] == (1 if subset == 'nonpad' else 0)).sum() == 0: continue  # all files on one side: same as 'all'
                        s = z[w.rows]
                        pt = w.point(s, Tp); br = w.boot(s, Tb)
                        kf = int((w.ratio(s, Tp) > 0).sum())
                        rates.append(dict(bb=bb, head=h, alpha=alpha, cohort=nm, role=c['row']['role'], subset=subset, n_files=len(w.files),
                                          n_windows=len(s), clusters=w.ncl, rate=100 * pt, ci_lo=100 * np.percentile(br, 2.5), ci_hi=100 * np.percentile(br, 97.5),
                                          files_any_flag=kf, wilson_hi_G=100 * wilson_hi(0, w.ncl) if pt == 0 else np.nan, T=Tp, sd_cal=sdA, B=len(Tb)))
                        if alpha == 0.02 and subset in ('all', 'nonpad'):
                            boot_store[f'{bb}|{h}|{nm}|{subset}'] = br
                # evidence types (alpha = 2%, window level)
                if alpha == 0.02:
                    tnA = sA < Tp
                    eA, nA = epA[cA['rows']][tnA], enA[cA['rows']][tnA]
                    qs = {cut: (np.percentile(eA, cut[0]), np.percentile(eA, cut[1]), np.percentile(nA, cut[0])) for cut in ((5, 95), (1, 99), (10, 90))}
                    for nm in RW4 + ['jamendo_B', 'ava_sm_long', 'audioset_sm', 'ava_mo'] + NULL_LONG:
                        if nm not in coh or (coh[nm]['row']['dump'], bb, h) not in S: continue
                        w = get_win(nm, 'all'); zz, ee, nn, _ = S[(coh[nm]['row']['dump'], bb, h)]
                        s = zz[w.rows]; fp = s >= Tp; e1, n1 = ee[w.rows][fp], nn[w.rows][fp]
                        row = dict(bb=bb, head=h, cohort=nm, n_windows=len(s), n_fp_windows=int(fp.sum()))
                        for cut, (pl, ph, pn) in qs.items():
                            el = (n1 < pn) & (e1 >= pl) & (e1 <= ph); et = e1 > ph
                            row.update({f'el_{cut[0]}': int(el.sum()), f'ep_{cut[0]}': int(et.sum()), f'mx_{cut[0]}': int((~el & ~et).sum())})
                        fats.append(row)
                    # null inputs: median score of unpadded null windows and its margin
                    for nm in NULL_LONG:
                        if nm not in coh or (coh[nm]['row']['dump'], bb, h) not in S: continue
                        w = get_win(nm, 'nonpad')
                        if w is None: continue
                        s = S[(coh[nm]['row']['dump'], bb, h)][0][w.rows]
                        nulls.append(dict(bb=bb, head=h, cohort=nm, n_files=len(w.files), n_windows=len(s), med=float(np.median(s)), T=Tp, sd_cal=sdA,
                                          margin_sd=float((np.median(s) - Tp) / sdA), fpr=100 * w.point(s, Tp)))
            print(f'[win] {bb}:{h} {time.time() - t0:.0f}s', flush=True)
    # ---- calibration source (window level, alpha = 2%): FMA-dev / (probes only) FMA-ov / FMA-test / Jamendo-A ----
    prov = []
    PROV_EVAL = ['jamendo_B'] + RW4 + OOT3 + ['newgen', 'ava_sm_long', 'audioset_sm']
    for bb in fams:
        srcs = ['fma_dev800', 'fma_ov_probe800', 'fma_test_disjoint800', 'jamendo_A'] if bb in PROBES else ['fma_dev800', 'jamendo_A']
        for h in [x for x in ALL_HEADS if ('jamendo2k', bb, x) in S]:
            zJ = S[('jamendo2k', bb, h)][0][cA['rows']]; TJ = float(wq_desc(zJ, wA, 0.02)); sdJ = float(np.sqrt(np.cov(zJ, aweights=wA)))
            for src in srcs:
                cal = make_cal(src)
                if cal is None or (cal['c']['row']['dump'], bb, h) not in S: continue
                sc = S[(cal['c']['row']['dump'], bb, h)][0][cal['c']['rows']]
                Bn = B if h in MAIN_HEADS else min(B, a.B_aux)
                Tp = float(wq_desc(sc, cal['w'], 0.02)); Tb = tau_boot(cal, sc, 0.02, Bn)
                row = dict(bb=bb, head=h, calib=src, T=Tp, dT=Tp - TJ, dT_sd=(Tp - TJ) / sdJ, n_cal_files=int(len(cal['c']['files'])),
                           n_cal_clusters=int(cal['M'].shape[1]))
                bs = {}
                for nm in PROV_EVAL:
                    if nm not in coh or (coh[nm]['row']['dump'], bb, h) not in S: continue
                    w = get_win(nm, 'all'); x = S[(coh[nm]['row']['dump'], bb, h)][0][w.rows]
                    row[nm] = 100 * w.point(x, Tp); br = w.boot(x, Tb); bs[nm] = br
                    row[nm + '_lo'] = 100 * np.percentile(br, 2.5); row[nm + '_hi'] = 100 * np.percentile(br, 97.5)
                if all(c_ in bs for c_ in RW4):
                    wst = 100 * np.max(np.stack([bs[c_] for c_ in RW4]), 0); mac = 100 * np.mean(np.stack([bs[c_] for c_ in RW4]), 0)
                    row['rw_worst'] = max(row[c_] for c_ in RW4); row['rw_worst_cohort'] = max(RW4, key=lambda c_: row[c_])
                    row['rw_worst_lo'], row['rw_worst_hi'] = np.percentile(wst, 2.5), np.percentile(wst, 97.5)
                    row['rw_macro'] = float(np.mean([row[c_] for c_ in RW4])); row['rw_macro_lo'], row['rw_macro_hi'] = np.percentile(mac, 2.5), np.percentile(mac, 97.5)
                if all(c_ in bs for c_ in OOT3):
                    oo = 100 * np.mean(np.stack([bs[c_] for c_ in OOT3]), 0)
                    row['oot'] = float(np.mean([row[c_] for c_ in OOT3])); row['oot_lo'], row['oot_hi'] = np.percentile(oo, 2.5), np.percentile(oo, 97.5)
                prov.append(row)
        print(f'[prov] {bb} {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(prov).to_csv(f'{TB}/{a.tag}_provenance.csv', index=False)
    rt = pd.DataFrame(rates); rt.to_csv(f'{TB}/{a.tag}_rates.csv', index=False)
    pd.DataFrame(fats).to_csv(f'{TB}/{a.tag}_fa_types.csv', index=False)
    pd.DataFrame(nulls).to_csv(f'{TB}/{a.tag}_null.csv', index=False)
    # ---- head contrasts (paired bootstrap, alpha = 2%) ----
    def bkey(bb, h, nm, subset='all'): return f'{bb}|{h}|{nm}|{subset}'
    pairs = (('free_split', 'pos_split', 'confirmatory'), ('orig', 'pos_split', 'aux'), ('orig', 'free_split', 'aux'), ('free_mlp', 'pos_mlp', 'aux'))
    for bb in fams:
        for hb, hm, kind in pairs:
            if not all(bkey(bb, x, c) in boot_store for x in (hb, hm) for c in RW4): continue
            row = dict(bb=bb, base=hb, method=hm, kind=kind)
            pt = lambda h, c: float(rt[(rt.bb == bb) & (rt['head'] == h) & (rt.alpha == 0.02) & (rt.cohort == c) & (rt.subset == 'all')].rate.iloc[0])
            for c in RW4 + OOT3 + ['jamendo_B', 'newgen']:
                if bkey(bb, hb, c) not in boot_store or bkey(bb, hm, c) not in boot_store: continue
                dlt = 100 * (boot_store[bkey(bb, hm, c)] - boot_store[bkey(bb, hb, c)])
                row[f'd_{c}'] = pt(hm, c) - pt(hb, c); row[f'd_{c}_lo'] = np.percentile(dlt, 2.5); row[f'd_{c}_hi'] = np.percentile(dlt, 97.5)
            wb = 100 * np.max(np.stack([boot_store[bkey(bb, hb, c)] for c in RW4]), 0)
            wm = 100 * np.max(np.stack([boot_store[bkey(bb, hm, c)] for c in RW4]), 0)
            row['worst_base'] = max(pt(hb, c) for c in RW4); row['worst_method'] = max(pt(hm, c) for c in RW4)
            row['worst_base_cohort'] = max(RW4, key=lambda c: pt(hb, c)); row['worst_method_cohort'] = max(RW4, key=lambda c: pt(hm, c))
            dd = wm - wb
            row['d_worst'] = row['worst_method'] - row['worst_base']; row['d_worst_lo'] = np.percentile(dd, 2.5); row['d_worst_hi'] = np.percentile(dd, 97.5)
            row['p_worst'] = float(min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean())))
            if all(bkey(bb, x, c) in boot_store for x in (hb, hm) for c in OOT3):
                tb = 100 * np.mean(np.stack([boot_store[bkey(bb, hb, c)] for c in OOT3]), 0); tm = 100 * np.mean(np.stack([boot_store[bkey(bb, hm, c)] for c in OOT3]), 0)
                row['oot_base'] = np.mean([pt(hb, c) for c in OOT3]); row['oot_method'] = np.mean([pt(hm, c) for c in OOT3])
                row['d_oot'] = row['oot_method'] - row['oot_base']; row['d_oot_lo'] = np.percentile(tm - tb, 2.5); row['d_oot_hi'] = np.percentile(tm - tb, 97.5)
            row['macro_base'] = np.mean([pt(hb, c) for c in RW4]); row['macro_method'] = np.mean([pt(hm, c) for c in RW4])
            mb = 100 * np.mean(np.stack([boot_store[bkey(bb, hb, c)] for c in RW4]), 0); mm_ = 100 * np.mean(np.stack([boot_store[bkey(bb, hm, c)] for c in RW4]), 0)
            row['d_macro'] = row['macro_method'] - row['macro_base']; row['d_macro_lo'] = np.percentile(mm_ - mb, 2.5); row['d_macro_hi'] = np.percentile(mm_ - mb, 97.5)
            contrasts.append(row)
    ct = pd.DataFrame(contrasts)
    if len(ct):  # Holm over the confirmatory contrasts (PE - UC) of this run; heads_report.py recomputes it over all families
        conf = ct.kind == 'confirmatory'
        p = ct.loc[conf, 'p_worst'].values; order = np.argsort(p); m = len(p); adj = np.empty(m)
        run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (m - rank) * p[i])); adj[i] = run
        ct.loc[conf, 'p_worst_holm'] = adj
    ct.to_csv(f'{TB}/{a.tag}_contrast.csv', index=False)
    json.dump(dict(B=B, unit='10.24 s window', tau='Jamendo-A window weighted (1-alpha) upper quantile, weight 1/n_f', alphas=list(ALPHAS),
                   families=fams, registry='lists/cohorts.csv', cohorts={k: dict(role=v['row']['role'], n_files=int(len(v['files'])), n_meta=v['n_meta'],
                                                                                pad_share=float(v['short'].mean()) if len(v['short']) else np.nan) for k, v in coh.items()},
                   finished=time.strftime('%Y-%m-%d %H:%M:%S')), open(f'{TB}/{a.tag}_meta.json', 'w'), indent=1)
    print(f'HEADS_WINDOW_DONE {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
