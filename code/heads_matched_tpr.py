"""Original head versus re-trained (UC) head at the same unseen-AI TPR (, "Raising the original threshold ...").

Same definitions as heads_window.py (window unit, track-weighted Jamendo-A tau, file mean of window rates, two-stage cluster
bootstrap, AI cohorts stratified by generator), plus a matched-TPR threshold.
Procedure (per family with an original head whose result the paper reports: B0, M0, C0; C1 is not released):
  1) UC head: tau_UC = tau_win(2%) on Jamendo-A. Unseen-AI TPR = macro mean of the window TPR over OOT3 (MUSIC8K, AIME,
     AI-OpenBMAT-AI music-bearing).
  2) Original head: the threshold is raised to the lowest tau* whose OOT3 macro TPR does not exceed the UC value (step function,
     so the TPR at tau* is the nearest value at or below the target).
  3) Window FPR of the original head at tau* (worst in-the-wild cohort = max over RW4: AI-OpenBMAT-H music-bearing, MUSDB18,
     GTZAN, AVA-SM; and per cohort) next to the UC head's FPR at tau_UC.
  Intervals: paired two-stage cluster bootstrap, B = 2,000. Every replicate resamples Jamendo-A artists (tau_UC,b) and the OOT3 files
     within generators (target TPR_b); the original head's tau*_b is the first grid value (4,001 quantiles of its OOT3 window scores)
     whose macro TPR under the same replicate weights is at or below the target; evaluation cohorts use the same cluster weights.
  Reading rule: if the 95% lower bound of (worst FPR of the original head at matched TPR - worst FPR of UC) > 0 -> "re-training
  improves the operating curve", otherwise "re-training mainly moves the operating point".
The natural-zero positive-evidence variant (MLP-PE versus MLP-UC, all families) is summarised from out/heads_contrast_all.csv.
Output (out/): heads_matched_tpr.csv
usage: python code/heads_matched_tpr.py [--B 2000]
"""
import os, sys, argparse, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import heads_window as HW  # noqa: E402

TB = C.TB
RW4 = HW.RW4; OOT3 = HW.OOT3
FAMS = ['B0', 'M0', 'C0']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=2000)
    ap.add_argument('--grid', type=int, default=4001)
    a = ap.parse_args()
    t0 = time.time(); B = a.B
    HUMAN = ['jamendo_B', 'ava_sm5', 'audioset_sm', 'openbmat_human_musicbearing', 'musdb18', 'gtzan']
    NEED = set(HUMAN + OOT3 + ['jamendo_A'])
    reg = [r for r in HW.load_registry() if r['name'] in NEED]
    S, IDX = HW.load_scores(reg, FAMS, dtype=np.float64, heads_wanted=('orig', 'free_split'))
    S = {k: v[0] for k, v in S.items()}
    print(f'[scores] {time.time() - t0:.0f}s', flush=True)
    coh = {}
    for r in reg:
        m = HW.cohort_meta(r); idx = IDX[r['dump']]
        sel = idx.item_id.isin(set(m.item_id)); sub = idx[sel]
        fp_codes, fp_uniq = pd.factorize(sub.item_id.values); clmap = dict(zip(m.item_id, m._cluster))
        grp = dict(zip(m.item_id, m['group'].astype(str))) if 'group' in m else {}
        rows = np.where(sel.values)[0]; nf = np.bincount(fp_codes).astype(float)
        strata = np.array([grp.get(f, '') for f in fp_uniq]) if r['role'] == 'fake' else None
        w = HW.Win(fp_codes, nf, np.array([clmap[f] for f in fp_uniq]), f'{r["name"]}|all', B, strata)  # same seed names as heads_window
        w.rows = rows; w.files = fp_uniq
        coh[r['name']] = dict(row=r, w=w, fid=fp_codes, nf=nf)
    cA = coh['jamendo_A']; fidA = cA['fid']; wA = 1.0 / cA['nf'][fidA]
    mA = HW.cohort_meta(next(r for r in reg if r['name'] == 'jamendo_A'))
    codesA, uniqA = pd.factorize(pd.Series(cA['w'].files).map(dict(zip(mA.item_id, mA._cluster))).astype(str))
    rngA = np.random.default_rng(HW.seed_of('jamendo_A|win'))
    MA = rngA.multinomial(len(uniqA), np.full(len(uniqA), 1.0 / len(uniqA)), size=B).astype(np.float32)
    artw = codesA[fidA]

    def tau_boot(sA, alpha=0.02):
        o = np.argsort(-sA, kind='stable'); so = sA[o]; wo = wA[o]; ao = artw[o]; out = np.empty(B)
        for b0 in range(0, B, 500):
            Wb = MA[b0:b0 + 500][:, ao] * wo[None, :]
            cs = np.cumsum(Wb, 1); cnt = (cs <= alpha * cs[:, -1:] + 1e-12).sum(1)
            out[b0:b0 + len(Wb)] = np.where(cnt > 0, so[np.maximum(cnt - 1, 0)], np.inf)
        return out

    def ratio_mat(c, s, grid):
        """(G, n_files): per grid tau, the share of flagged windows per file."""
        fid = c['fid']; nf = c['nf']; G = len(grid)
        o = np.argsort(s); so = s[o]; fo = fid[o]
        out = np.zeros((G, len(nf)), np.float32)
        pos = np.searchsorted(so, grid, side='left')  # windows with s >= tau start at position pos
        for g in range(G):
            out[g] = np.bincount(fo[pos[g]:], minlength=len(nf)) / nf
        return out

    def macro_curve_exact(sc_by_c):
        """Exact step function: OOT3 macro TPR at every candidate tau (unique OOT window scores, ascending)."""
        cand = np.unique(np.concatenate(list(sc_by_c.values())))
        tot = np.zeros(len(cand))
        for c, s in sc_by_c.items():
            fid = coh[c]['fid']; nf = coh[c]['nf']; wv = 1.0 / nf[fid] / len(nf)
            o = np.argsort(-s, kind='stable'); so = s[o]; cum = np.r_[0.0, np.cumsum(wv[o])]
            tot += cum[np.searchsorted(-so, -cand, side='right')]
        return cand, tot / len(sc_by_c)

    rows = []
    for bb in FAMS:
        if ('jamendo2k', bb, 'free_split') not in S: continue
        has_orig = all((coh[c]['row']['dump'], bb, 'orig') in S for c in NEED)
        sc = lambda c, h: S[(coh[c]['row']['dump'], bb, h)][coh[c]['w'].rows]
        sA_uc = sc('jamendo_A', 'free_split'); T_uc = HW.wq_desc(sA_uc, wA, 0.02); Tb_uc = tau_boot(sA_uc)
        tpr_uc = np.mean([coh[c]['w'].point(sc(c, 'free_split'), T_uc) for c in OOT3])
        tpr_uc_b = np.mean(np.stack([coh[c]['w'].boot(sc(c, 'free_split'), Tb_uc) for c in OOT3]), 0)
        uc_pt = {h: coh[h]['w'].point(sc(h, 'free_split'), T_uc) for h in HUMAN}
        uc_b = {h: coh[h]['w'].boot(sc(h, 'free_split'), Tb_uc) for h in HUMAN}
        row = dict(bb=bb, family=bb, T_uc=T_uc, tpr_uc=100 * tpr_uc, tpr_uc_lo=100 * np.percentile(tpr_uc_b, 2.5), tpr_uc_hi=100 * np.percentile(tpr_uc_b, 97.5))
        for h in HUMAN:
            row[f'uc_{h}'] = 100 * uc_pt[h]; row[f'uc_{h}_lo'] = 100 * np.percentile(uc_b[h], 2.5); row[f'uc_{h}_hi'] = 100 * np.percentile(uc_b[h], 97.5)
        wuc = 100 * np.max(np.stack([uc_b[h] for h in RW4]), 0)
        row['uc_worst'] = 100 * max(uc_pt[h] for h in RW4); row['uc_worst_cohort'] = max(RW4, key=lambda h: uc_pt[h])
        row['uc_worst_lo'], row['uc_worst_hi'] = np.percentile(wuc, 2.5), np.percentile(wuc, 97.5)
        if not has_orig:
            row['note'] = 'no original head (frozen encoder + new heads only)'; rows.append(row); print(f'[no original head] {bb}', flush=True); continue
        # original head: its own tau_win(2%) for reference
        sA_o = sc('jamendo_A', 'orig'); T_o2 = HW.wq_desc(sA_o, wA, 0.02)
        row['T_orig_2pct'] = T_o2; row['tpr_orig_2pct'] = 100 * np.mean([coh[c]['w'].point(sc(c, 'orig'), T_o2) for c in OOT3])
        row['orig2_worst'] = 100 * max(coh[h]['w'].point(sc(h, 'orig'), T_o2) for h in RW4)
        # point estimate: exact matched-TPR threshold
        oot_o = {c: sc(c, 'orig') for c in OOT3}
        cand, curve = macro_curve_exact(oot_o)
        ok = np.where(curve <= tpr_uc + 1e-12)[0]
        i = ok[0] if len(ok) else len(cand) - 1
        T_star = cand[i]; row['T_orig_matched'] = T_star; row['tpr_orig_matched'] = 100 * curve[i]
        row['tpr_orig_matched_next_lower_tau'] = 100 * curve[i - 1] if i > 0 else np.nan
        row['jamA_fpr_orig_matched'] = 100 * coh['jamendo_A']['w'].point(sA_o, T_star)
        o_pt = {h: coh[h]['w'].point(sc(h, 'orig'), T_star) for h in HUMAN}
        # bootstrap: matched-TPR threshold on a grid
        allo = np.concatenate(list(oot_o.values()))
        grid = np.unique(np.quantile(allo, np.linspace(0, 1, a.grid)))
        tp = np.zeros((B, len(grid)))
        for c in OOT3:
            Rm = ratio_mat(coh[c], oot_o[c], grid); W = coh[c]['w'].W
            tp += (W @ Rm.T) / np.maximum(W.sum(1, keepdims=True), 1e-9)
        tp /= len(OOT3)
        hit = tp <= tpr_uc_b[:, None] + 1e-12
        g_star = np.where(hit.any(1), hit.argmax(1), len(grid) - 1)
        row['grid_tau_err'] = float(np.abs(grid[np.searchsorted(grid, T_star).clip(0, len(grid) - 1)] - T_star))
        o_b = {}
        for h in HUMAN:
            Rh = ratio_mat(coh[h], sc(h, 'orig'), grid); W = coh[h]['w'].W
            o_b[h] = (W * Rh[g_star]).sum(1) / np.maximum(W.sum(1), 1e-9)
            row[f'orig_{h}'] = 100 * o_pt[h]; row[f'orig_{h}_lo'] = 100 * np.percentile(o_b[h], 2.5); row[f'orig_{h}_hi'] = 100 * np.percentile(o_b[h], 97.5)
            dd = 100 * (o_b[h] - uc_b[h])
            row[f'd_{h}'] = 100 * (o_pt[h] - uc_pt[h]); row[f'd_{h}_lo'] = np.percentile(dd, 2.5); row[f'd_{h}_hi'] = np.percentile(dd, 97.5)
        wo = 100 * np.max(np.stack([o_b[h] for h in RW4]), 0)
        row['orig_worst'] = 100 * max(o_pt[h] for h in RW4); row['orig_worst_cohort'] = max(RW4, key=lambda h: o_pt[h])
        row['orig_worst_lo'], row['orig_worst_hi'] = np.percentile(wo, 2.5), np.percentile(wo, 97.5)
        dw = wo - wuc
        row['d_worst'] = row['orig_worst'] - row['uc_worst']; row['d_worst_lo'] = np.percentile(dw, 2.5); row['d_worst_hi'] = np.percentile(dw, 97.5)
        row['p_worst'] = float(min(1.0, 2 * min((dw <= 0).mean(), (dw >= 0).mean())))
        row['tpr_orig_matched_boot_mean'] = 100 * float(np.mean(tp[np.arange(B), g_star]))
        row['tpr_uc_boot_mean'] = 100 * float(np.mean(tpr_uc_b))
        row['verdict'] = ('re-training improves the operating curve' if row['d_worst_lo'] > 0 else
                          're-training mainly moves the operating point (original head lower at matched TPR)' if row['d_worst_hi'] < 0 else
                          're-training mainly moves the operating point')
        rows.append(row)
        print(f"[{bb}] TPR_uc {row['tpr_uc']:.1f} -> orig tau {T_o2:.3f}->{T_star:.3f} (TPR {row['tpr_orig_2pct']:.1f}->{row['tpr_orig_matched']:.1f}); "
              f"worst orig@matched {row['orig_worst']:.2f} vs UC {row['uc_worst']:.2f} d {row['d_worst']:+.2f} [{row['d_worst_lo']:+.2f},{row['d_worst_hi']:+.2f}] {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f'{TB}/heads_matched_tpr.csv', index=False)
    # natural-zero positive-evidence variant (MLP-PE versus MLP-UC), from the paired head contrasts
    p = f'{TB}/heads_contrast_all.csv'
    if os.path.exists(p):
        ct = pd.read_csv(p)
        mlp = ct[(ct.base == 'free_mlp') & (ct.method == 'pos_mlp') & ct.bb.isin(list(C.FAMILIES))]
        if len(mlp):
            print(f'MLP-PE - MLP-UC over {len(mlp)} families: worst in-the-wild FPR {mlp.d_worst.min():+.2f} to {mlp.d_worst.max():+.2f} pp, '
                  f'OOT3 TPR {mlp.d_oot.min():+.2f} to {mlp.d_oot.max():+.2f} pp')
    print(f'HEADS_MATCHED_DONE {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
