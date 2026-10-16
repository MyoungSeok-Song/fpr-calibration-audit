"""Matched-TPR comparison of a one-epoch variant with its control (, B0+AVA "at matched AI-OpenBMAT-AI TPR").

The threshold of the control detector (B0+0) is moved until its reference TPR (long windows, file mean) equals the reference TPR
of the treated detector at its own tau_JamA@2%; at that operating point the human-cohort FPR of the control is compared with
the treated detector's FPR at tau_JamA@2% with a paired bootstrap.
 - Pairing: fixed seeds per cohort and subset (agg_window_variants.seed_of) -> the treated detector's stored replicates
   (variants_<prefix>_boot.npz, first B) and the control's replicates use the same cluster draws.
 - The control's tau* is a fixed point estimate (bisection on the TPR curve); its interval reflects only the evaluation clusters.
 - Reading rule: lower treated FPR at the same TPR = better operating curve; similar FPR = moved operating point.
Inputs: out/variants_<prefix>_{thresholds.csv,boot.npz} from agg_window_variants.py
Output (out/): matched_tpr_variants.csv
usage: python code/matched_tpr_variants.py --pairs "B0+AVA:openbmat_ai_musicbearing" [--ref B0+0] [--prefix seednoise] [--B 2000]
"""
import os, sys, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import agg_window_variants as W  # noqa: E402

TB = C.TB
FPR_COH = [('ava_sm5', 'long'), ('ava_sm5', 'all'), ('audioset_sm', 'all'), ('openbmat_human_musicbearing', 'long'), ('gtzan', 'long'), ('jamendo_B', 'long'), ('ava_mo', 'long')]


def file_mean_rate(j, tau):
    """Window table -> share of flagged windows per file, averaged over files (0..1)."""
    f = (j.lg.values >= tau).astype(float); fid = pd.factorize(j.ID)[0]
    return float(np.bincount(fid, weights=f).__truediv__(np.bincount(fid)).mean())


def solve_tau(j, target):
    """tau with TPR(tau) = target (non-increasing curve, bisection)."""
    lo, hi = float(j.lg.min()) - 1, float(j.lg.max()) + 1
    for _ in range(60):
        mid = (lo + hi) / 2
        if file_mean_rate(j, mid) >= target: lo = mid
        else: hi = mid
    return lo


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pairs', required=True, help='"detector:cohort" items separated by ";"')
    ap.add_argument('--ref', default='B0+0'); ap.add_argument('--prefix', default='seednoise'); ap.add_argument('--B', type=int, default=2000)
    a = ap.parse_args(); B = a.B; ref = C.resolve(a.ref)
    reg = C.load_registry(); metas = W.load_metas(reg)
    boot = np.load(f'{TB}/variants_{a.prefix}_boot.npz')
    th = pd.read_csv(f'{TB}/variants_{a.prefix}_thresholds.csv')
    rt_rows = []

    def cohorts(tag):
        wins = W.load_windows(tag, reg); out = {}
        for _, r in reg.iterrows():
            j = W.cohort(r, wins, metas)
            if j is not None: out[r['name']] = j
        return out
    C_ref = cohorts(ref)
    for pair in a.pairs.split(';'):
        tname, tcoh = pair.rsplit(':', 1); tag = C.resolve(tname.strip()); tcoh = tcoh.strip(); C_t = cohorts(tag)
        tau_t = float(th[(th.model == tag) & (th.thr == 'tau_JamA@2%')].tau.iloc[0]); tau_r0 = float(th[(th.model == ref) & (th.thr == 'tau_JamA@2%')].tau.iloc[0])
        jt = C_t[tcoh]; jt = jt[jt['short'] == 0]; target = file_mean_rate(jt, tau_t)
        jr = C_ref[tcoh]; jr = jr[jr['short'] == 0]; tau_star = solve_tau(jr, target); tpr_ref_at_star = file_mean_rate(jr, tau_star); tpr_ref0 = file_mean_rate(jr, tau_r0)
        print(f'{tag}: {tcoh} TPR(tau_JamA2={tau_t:.3f}) = {100*target:.1f}% -> {ref} tau* = {tau_star:.3f} (tau_JamA2 {tau_r0:.3f}, TPR {100*tpr_ref0:.1f} -> {100*tpr_ref_at_star:.1f}%)', flush=True)
        for coh, sb in FPR_COH:
            if coh not in C_ref or coh not in C_t: continue
            jr_ = C_ref[coh]; jt_ = C_t[coh]
            if sb == 'long': jr_ = jr_[jr_['short'] == 0]; jt_ = jt_[jt_['short'] == 0]
            if len(jr_) == 0 or len(jt_) == 0: continue
            key = f'{tag}|{coh}|{sb}'
            if key not in boot.files: print('  no replicates for', key); continue
            bt = boot[key][:B].astype(float)
            # control at fixed tau* (same seed and batches -> paired)
            f = W.file_table(jr_); codes_f, uniq = pd.factorize(f.cluster_key); G = len(uniq); fid = pd.factorize(jr_.ID)[0]; codes_w = codes_f[fid]
            wf = 1.0 / f.n_w.values[fid].astype(float); s = jr_.lg.values.astype(float)
            Tm = np.full((1, B), tau_star); br = W.boot_rates(s, wf, codes_w, codes_f, G, Tm, np.random.default_rng(W.seed_of('coh', coh, sb)), B)[0]
            pr = 100 * file_mean_rate(jr_, tau_star); pr0 = 100 * file_mean_rate(jr_, tau_r0); pt = 100 * file_mean_rate(jt_, tau_t)
            d = 100 * (bt - br)
            rt_rows.append(dict(treat=tag, matched_on=tcoh, target_tpr=100 * target, ref=ref, tau_treat=tau_t, tau_ref_star=tau_star, tau_ref_jamA2=tau_r0, cohort=coh, subset=sb,
                                fpr_treat=pt, fpr_ref_star=pr, fpr_ref_jamA2=pr0, delta=pt - pr, d_lo=np.percentile(d, 2.5), d_hi=np.percentile(d, 97.5),
                                ref_star_lo=100 * np.percentile(br, 2.5), ref_star_hi=100 * np.percentile(br, 97.5), n_files=len(f), clusters=G))
    df = pd.DataFrame(rt_rows); df.to_csv(f'{TB}/matched_tpr_variants.csv', index=False)
    print('MATCHED_TPR_VARIANTS_DONE', len(df))


if __name__ == '__main__':
    main()
