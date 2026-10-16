"""Original head at its deployment threshold -> original head at a TPR-matched threshold -> re-fitted heads (Table 3; post hoc
diagnostic).

Same definitions and code path as heads_matched_tpr.py (window unit, Win, wq_desc, RW4, OOT3, the same seed names, a grid of
4,001 quantiles, B = 2,000); the target head is extended from UC (free_split) to UC and PE (pos_split).
Families: B0, C0, M0 (C1 is not released).
  - Re-fitted head h in {UC, PE}: tau_h = Jamendo-A window 2% (track-weighted); TPR_h = macro mean of the window TPR over OOT3
    (MUSIC8K, AIME, AI-OpenBMAT-AI music-bearing).
  - Original head tau*_h: the lowest tau whose OOT3 macro TPR does not exceed TPR_h (exact step function for the point estimate;
    in every bootstrap replicate the first grid value at or below that replicate's target TPR_h,b). The match uses the TPR of the
    evaluation set, so tau*_h is a post hoc diagnostic, not a deployable threshold: it separates "re-fitting changes the
    operating curve" from "re-fitting moves the operating point".
  - Worst FPR = max over RW4 (AI-OpenBMAT-H music-bearing, MUSDB18, GTZAN, AVA-SM evaluation films). Paired difference =
    re-fitted h - original head at tau*_h (same cluster draws, tau re-estimated in every replicate).
Columns (FPR and TPR in %): orig_* = original head at its deployment tau, matched_* = original head at tau*_h, refit_* =
  re-fitted head h; per-cohort FPR for the six human cohorts; d_<cohort> = refit - matched with interval; d_refit_minus_matched
  (worst FPR, pp) with interval and two-sided bootstrap p (p_rm, unadjusted); d_refit_minus_orig with interval.
  The UC rows reproduce out/heads_matched_tpr.csv (d_refit_minus_matched = -d_worst).
Output (out/): table3_matched_baseline.csv
usage: python code/table3_matched_baseline.py [--B 2000]
"""
import os, sys, argparse, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import heads_window as HW  # noqa: E402

TB = C.TB
RW4 = HW.RW4; OOT3 = HW.OOT3
FAMS = ['B0', 'C0', 'M0']
TARGETS = {'free_split': 'UC', 'pos_split': 'PE'}
HUMAN = ['jamendo_B', 'ava_sm5', 'audioset_sm', 'openbmat_human_musicbearing', 'musdb18', 'gtzan']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--B', type=int, default=2000); ap.add_argument('--grid', type=int, default=4001)
    a = ap.parse_args(); t0 = time.time(); B = a.B
    NEED = set(HUMAN + OOT3 + ['jamendo_A'])
    reg = [r for r in HW.load_registry() if r['name'] in NEED]
    S, IDX = HW.load_scores(reg, FAMS, dtype=np.float64, heads_wanted=('orig', 'free_split', 'pos_split'))
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

    pc = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    rows = []
    for bb in FAMS:
        sc = lambda c, h: S[(coh[c]['row']['dump'], bb, h)][coh[c]['w'].rows]
        # original head at its deployment tau (own Jamendo-A 2%): point and interval (tau re-estimated)
        sA_o = sc('jamendo_A', 'orig'); T_o2 = HW.wq_desc(sA_o, wA, 0.02); Tb_o2 = tau_boot(sA_o)
        o2_pt = {h: coh[h]['w'].point(sc(h, 'orig'), T_o2) for h in HUMAN}
        o2_b = {h: coh[h]['w'].boot(sc(h, 'orig'), Tb_o2) for h in HUMAN}
        tpr_o2 = np.mean([coh[c]['w'].point(sc(c, 'orig'), T_o2) for c in OOT3])
        tpr_o2_b = np.mean(np.stack([coh[c]['w'].boot(sc(c, 'orig'), Tb_o2) for c in OOT3]), 0)
        w_o2 = 100 * np.max(np.stack([o2_b[h] for h in RW4]), 0)
        oot_o = {c: sc(c, 'orig') for c in OOT3}
        cand, curve = macro_curve_exact(oot_o)
        allo = np.concatenate(list(oot_o.values()))
        grid = np.unique(np.quantile(allo, np.linspace(0, 1, a.grid)))
        tp = np.zeros((B, len(grid)))
        for c in OOT3:
            Rm = ratio_mat(coh[c], oot_o[c], grid); W = coh[c]['w'].W
            tp += (W @ Rm.T) / np.maximum(W.sum(1, keepdims=True), 1e-9)
        tp /= len(OOT3)
        Rh = {h: ratio_mat(coh[h], sc(h, 'orig'), grid) for h in HUMAN}
        for th, tl in TARGETS.items():
            sA_t = sc('jamendo_A', th); T_t = HW.wq_desc(sA_t, wA, 0.02); Tb_t = tau_boot(sA_t)
            tpr_t = np.mean([coh[c]['w'].point(sc(c, th), T_t) for c in OOT3])
            tpr_t_b = np.mean(np.stack([coh[c]['w'].boot(sc(c, th), Tb_t) for c in OOT3]), 0)
            t_pt = {h: coh[h]['w'].point(sc(h, th), T_t) for h in HUMAN}
            t_b = {h: coh[h]['w'].boot(sc(h, th), Tb_t) for h in HUMAN}
            w_t = 100 * np.max(np.stack([t_b[h] for h in RW4]), 0)
            # matched threshold: exact step function (point), grid search per replicate (bootstrap)
            ok = np.where(curve <= tpr_t + 1e-12)[0]; i = ok[0] if len(ok) else len(cand) - 1
            T_star = cand[i]
            m_pt = {h: coh[h]['w'].point(sc(h, 'orig'), T_star) for h in HUMAN}
            hit = tp <= tpr_t_b[:, None] + 1e-12
            g_star = np.where(hit.any(1), hit.argmax(1), len(grid) - 1)
            m_b = {}
            for h in HUMAN:
                W = coh[h]['w'].W; m_b[h] = (W * Rh[h][g_star]).sum(1) / np.maximum(W.sum(1), 1e-9)
            w_m = 100 * np.max(np.stack([m_b[h] for h in RW4]), 0)
            d_tm = w_t - w_m; d_to = w_t - w_o2
            row = dict(bb=bb, family=bb, refit=tl, refit_head=th,
                       orig_tau=T_o2, orig_tpr=100 * tpr_o2, orig_tpr_lo=100 * pc(tpr_o2_b)[0], orig_tpr_hi=100 * pc(tpr_o2_b)[1],
                       orig_worst=100 * max(o2_pt[h] for h in RW4), orig_worst_cohort=max(RW4, key=lambda h: o2_pt[h]), orig_worst_lo=pc(w_o2)[0], orig_worst_hi=pc(w_o2)[1],
                       match_target='OOT3 macro window TPR of the re-fitted head at its own deployment tau (MUSIC8K, AIME, AI-OpenBMAT-AI music-bearing)',
                       matched_tau=T_star, matched_tpr=100 * curve[i], matched_tpr_boot_mean=100 * float(np.mean(tp[np.arange(B), g_star])),
                       matched_jamA_fpr=100 * coh['jamendo_A']['w'].point(sA_o, T_star),
                       matched_worst=100 * max(m_pt[h] for h in RW4), matched_worst_cohort=max(RW4, key=lambda h: m_pt[h]), matched_worst_lo=pc(w_m)[0], matched_worst_hi=pc(w_m)[1],
                       refit_tau=T_t, refit_tpr=100 * tpr_t, refit_tpr_lo=100 * pc(tpr_t_b)[0], refit_tpr_hi=100 * pc(tpr_t_b)[1], refit_tpr_boot_mean=100 * float(np.mean(tpr_t_b)),
                       refit_worst=100 * max(t_pt[h] for h in RW4), refit_worst_cohort=max(RW4, key=lambda h: t_pt[h]), refit_worst_lo=pc(w_t)[0], refit_worst_hi=pc(w_t)[1],
                       d_refit_minus_matched=100 * (max(t_pt[h] for h in RW4) - max(m_pt[h] for h in RW4)), d_rm_lo=pc(d_tm)[0], d_rm_hi=pc(d_tm)[1],
                       p_rm=float(min(1.0, 2 * min((d_tm <= 0).mean(), (d_tm >= 0).mean()))),
                       d_refit_minus_orig=100 * (max(t_pt[h] for h in RW4) - max(o2_pt[h] for h in RW4)), d_ro_lo=pc(d_to)[0], d_ro_hi=pc(d_to)[1])
            for h in HUMAN:
                row[f'orig_{h}'] = 100 * o2_pt[h]; row[f'matched_{h}'] = 100 * m_pt[h]; row[f'refit_{h}'] = 100 * t_pt[h]
                dd = 100 * (t_b[h] - m_b[h]); row[f'd_{h}'] = 100 * (t_pt[h] - m_pt[h]); row[f'd_{h}_lo'], row[f'd_{h}_hi'] = pc(dd)
            rows.append(row)
            print(f"[{bb}/{tl}] orig {row['orig_worst']:.2f}/{row['orig_tpr']:.1f} -> matched {row['matched_worst']:.2f} (TPR {row['matched_tpr']:.1f}) -> "
                  f"{tl} {row['refit_worst']:.2f}/{row['refit_tpr']:.1f}; refit - matched {row['d_refit_minus_matched']:+.2f} "
                  f"[{row['d_rm_lo']:+.2f}, {row['d_rm_hi']:+.2f}] {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(f'{TB}/table3_matched_baseline.csv', index=False)
    print(f'TABLE3_MATCHED_DONE {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
