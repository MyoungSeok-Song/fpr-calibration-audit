"""Decision unit x calibration source (Figs. 1-3, Table 4). Re-aggregates the released window scores.

(1) Unit matrix: for every detector, tau at alpha = 2% is calibrated on Jamendo-A with calibration unit
    k_cal in {1, 2, 4, 8, 12} windows (trim20 of k consecutive windows, track-weighted quantile) and applied with deployment
    unit k_dep in {1, 2, 4, 12} (same aggregation, files with at least k_dep windows) -> realised FPR (target 2%) and the
    TPR of AI cohorts at the same tau (Fig. 2).
    - k consecutive windows: runs of consecutive j on the file's window grid (EAT: hop 5 s up to 65 s, otherwise 12 windows
      spread evenly; SpecTTTra: 5-s windows, at most 12; ArtifactNet: 7 windows spread evenly; Fakeprint LR: the EAT grid).
      Every start position (n - k + 1 per file) is used and the windows of a file sum to weight 1 (one track, one vote).
    - trim20 drops the floor(0.2k) largest logits and averages the rest; the fine-tuned detectors weight windows by the
      music-presence probability of a pretrained tagger, the public detectors weight them equally.
    - ArtifactNet has at most 7 windows, so k = 8 and 12 become k = 7.
(2) Calibration source x unit: tau from {FMA-dev (training platform), Jamendo-A (platform-disjoint)} x unit {track (trim20
    of all windows of the file), window (k = 1)}, applied at the same unit to Jamendo-B, AVA-SM, AudioSet-SM and
    AI-OpenBMAT-H (Fig. 1).
(3) Exposure status per generator and detector family; per-generator window TPR at tau_JamA (Table 4).
(4) Rank versus calibration against the four generator families that no compared detector has seen (Fig. 3): AUROC of human
    windows versus AI windows, FPR at tau_JamA, and the reference TPR at tau_JamA and at the cohort's own 2% threshold (oracle).
Intervals: cluster bootstrap with B replicates (default 10,000, seed 0; Jamendo-A artists resampled in every replicate to
re-estimate tau, evaluation clusters resampled independently, AI cohorts stratified by generator). Wilson n = clusters.
Outputs (out/): unit_{matrix,prov,gen,rank_unseen,exposure}.csv
usage: python code/unit_provenance.py [--B 10000] [--dets "B0,C0"] [--part] [--merge]
  The replication detectors SpecTTTra-alpha and SpecTTTra-beta run only when named in --dets (code/extra_sonics.sh).
"""
import os, sys, argparse, time, glob
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from agg_window import wq_tau, boot_tau, Cohort, wilson  # noqa: E402
from auroc import wauc  # noqa: E402

R = C.R; TB = C.TB; REG = C.REG
DETS = [('B0', 'B0'), ("B0'", "B0'"), ('B0+0', 'B0+0'), ('C0', 'C0'), ('C0+0', 'C0+0'), ('M0', 'M0'),
        ('ArtifactNet', 'ArtifactNet'), ('SpecTTTra', 'SpecTTTra'), ('Fakeprint LR', 'Fakeprint LR')]
EXTRA = [(n, n) for n in C.REPLICATION]  # replication detectors, only when named in --dets
KCAL = [1, 2, 4, 8, 12]; KDEP = [1, 2, 4, 12]; ALPHA = 0.02
T_REAL = ['jamendo_B', 'musdb18', 'gtzan', 'openbmat_human_musicbearing']
T_FAKE = ['sonics_fake', 'music8k', 'aime_sa', 'openbmat_ai_musicbearing']
PROV_T = ['jamendo_B', 'ava_sm5', 'audioset_sm', 'openbmat_human', 'openbmat_human_musicbearing']
FAKE_COH = ['sonics_fake', 'newgen', 'music8k', 'aime_sa', 'openbmat_ai_musicbearing', 'suno_ext_dev', 'udio_hf_dev']  # AI-OpenBMAT: the 1,231 music-bearing files
BATCH = 250

# (3) Exposure status. Families: B0 family (B0, B0', B0+0; M0 shares B0's training data), C0 family (C0, C0+0), SONICS
#     (SpecTTTra, trained on all SONICS generators). Training generators of the fine-tuned detectors: SONICS chirp-v3.5 and
#     udio-120s only; the twelve Echoes generators (acestep, audioldm, brev, diffrhythm, elevenlabs, mubert, musicgen, producer,
#     songgen, stableaudio, suno, udio); FakeMusicCaps (MusicGen-medium, AudioLDM2, MusicLDM, Mustango, Stable Audio Open).
#     The training data of ArtifactNet and Fakeprint LR are not fully known, so no status is given for them.
#     same = the same generator version was in training / new_version = another version of a trained family /
#     new_family = a family absent from training / new_condition = the trained version under a new condition (speech) /
#     '?' = version information could not be verified (reason in the note).
EXPO = [
    # cohort, generator, family, B0 family (and M0), C0 family, SONICS, note
    ('sonics_fake', 'chirp-v2-xxl-alpha', 'Suno', 'new_version', 'new_version', 'same', 'only chirp-v3.5 of SONICS was used for fine-tuning'),
    ('sonics_fake', 'chirp-v3', 'Suno', 'new_version', 'new_version', 'same', ''),
    ('sonics_fake', 'chirp-v3.5', 'Suno', 'same', 'same', 'same', ''),
    ('sonics_fake', 'udio-120s', 'Udio', 'same', 'same', 'same', ''),
    ('sonics_fake', 'udio-30s', 'Udio', 'new_version', 'new_version', 'same', 'Udio 30-s model (udio-32)'),
    ('newgen', 'lyria3', 'Lyria', 'new_family', 'new_family', 'new_family', ''),
    ('newgen', 'suno-musicality', 'Suno', 'new_version', 'new_version', 'new_version', ''),
    ('newgen', 'suno-v5.5', 'Suno', 'new_version', 'new_version', 'new_version', 'same version as MUSIC8K suno_v5.5'),
    ('music8k', 'acestep1.5', 'ACE-Step', 'new_version', 'new_version', 'new_family', 'Echoes acestep (version unverified) in training; B0 partly exposed through the ACE-Step vocal overlay'),
    ('music8k', 'heartmula', 'HeartMuLa', 'new_family', 'new_family', 'new_family', ''),
    ('music8k', 'minimax_2.6', 'MiniMax', 'new_family', 'new_family', 'new_family', ''),
    ('music8k', 'mureka_v9', 'Mureka', 'new_family', 'new_family', 'new_family', ''),
    ('music8k', 'suno_v5.5', 'Suno', 'new_version', 'new_version', 'new_version', 'same version as Recent suno-v5.5'),
    ('music8k', 'sunov5_new', 'Suno', 'new_version', 'new_version', 'new_version', 'Suno v5'),
    ('aime_sa', 'suno_v3_aime', 'Suno', 'new_version', 'new_version', 'same', 'Suno v3 of AIME'),
    ('aime_sa', 'riffusion_aime', 'Riffusion', 'new_family?', 'new_family?', 'new_family', 'another model of the company behind the Echoes producer generator; effectively a new family (unverified)'),
    ('aime_sa', 'stableaudio_v1', 'Stable Audio', 'new_version?', 'new_version?', 'new_family', 'Echoes stableaudio and FakeMusicCaps Stable Audio Open in training (version unverified)'),
    ('aime_sa', 'stableaudio_v2', 'Stable Audio', 'new_version?', 'new_version?', 'new_family', 'as above'),
    ('openbmat_ai_musicbearing', 'suno_v3.5_openbmat', 'Suno', 'new_condition', 'new_condition', 'new_condition', 'Suno v3.5 (trained version) under broadcast-style speech; 1,231 music-bearing pairs'),
    ('suno_ext_dev', 'suno_ext', 'Suno', 'new_version?', 'new_version?', 'new_version?', 'platform compilation, mixed versions (unverified)'),
    ('udio_hf_dev', 'udio_hf', 'Udio', 'new_version?', 'new_version?', 'new_version?', 'as above'),
]


# (4) Rank versus calibration against unseen AI music. Reference = generators whose status is new_family for every family
#     with known training data. Every generator weighs the same, every file within a generator weighs the same (1/n_f), every
#     window within a file weighs the same.
RU_COH = ['fma_dev800', 'jamendo_B', 'ava_sm5', 'audioset_sm', 'openbmat_human_musicbearing', 'musdb18', 'gtzan']


def unseen_gens():
    """(cohort, generator) pairs whose status is a verified new_family for the B0 family, the C0 family and SONICS."""
    return [(c, g) for (c, g, fam, b, cc, so, note) in EXPO if all(x == 'new_family' for x in (b, cc, so))]


def ref_set(ch, fileset, reg, pairs):
    S, FI, GI, info, off = [], [], [], [], 0
    for gi, (cn, g) in enumerate(pairs):
        d = reg.loc[cn, 'dump']
        if d not in ch: continue
        fs = fileset(cn, ch[d])
        sel = fs.ids[fs.strata == g] if fs.strata is not None else fs.ids
        sub = FileSet(ch[d], sel, [str(x) for x in sel]); sc, fi = sub.units(1)
        S.append(sc); FI.append(fi + off); GI.append(np.full(len(sub.ids), len(info))); off += len(sub.ids)
        info.append(dict(cohort=cn, generator=g, n_files=len(sub.ids), n_windows=len(sc)))
    if not info: return None
    s = np.concatenate(S); fidx = np.concatenate(FI); fgen = np.concatenate(GI)
    nwin = np.bincount(fidx).astype(float); ngf = np.bincount(fgen).astype(float); G = len(info)
    base = 1.0 / (G * ngf[fgen[fidx]] * nwin[fidx])  # sums to 1
    return dict(s=s, fidx=fidx, fgen=fgen, nwin=nwin, ngf=ngf, G=G, base=base, info=info)


def ref_boot_w(ref, m, rng):
    Cw = np.zeros((m, len(ref['nwin'])))
    for g in range(ref['G']):
        idx = np.where(ref['fgen'] == g)[0]
        Cw[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=m)
    return (Cw / ref['ngf'][ref['fgen']][None, :] / ref['G'])[:, ref['fidx']] / ref['nwin'][ref['fidx']][None, :]


def rank_unseen(tag, name, FS, ch, fileset, reg, B, seed=5):
    """One detector: reference unseen4 (the four unseen generator families) x human cohort -> AUROC, FPR and reference TPR
    at tau_JamA, reference TPR at the cohort's own 2% threshold (oracle). All with cluster-bootstrap intervals."""
    rng = np.random.default_rng(seed); rows = []
    TJ, TJb, _ = calib(FS['jamendo_A'], 1, rng, B)
    for rname, pairs in (('unseen4', unseen_gens()),):
        ref = ref_set(ch, fileset, reg, pairs)
        if ref is None: continue
        rs = ref['s']; tprJ = float((ref['base'] * (rs >= TJ)).sum())
        for cn in RU_COH:
            if cn not in FS: continue
            hs = FS[cn]; hsc, hfi = hs.units(1); hw = 1.0 / np.bincount(hfi)[hfi]
            s = np.r_[hsc, rs]; y = np.r_[np.zeros(len(hsc)), np.ones(len(rs))]
            au = float(wauc(s, y, np.r_[hw, ref['base']][None, :])[0])
            Ch = Cohort(hsc, hfi, hs.clus, None); fprJ = float(Ch.rate([TJ])[0]); fprJb = Ch.boot(TJb[None, :], rng, B)[0]
            ccode = pd.factorize(pd.Series(hs.clus[hfi]))[0]; Gh = int(ccode.max()) + 1
            Tx, _ = wq_tau(hsc, hw, ALPHA); Txb = boot_tau(hsc, hw, ccode, Gh, ALPHA, rng, B)
            tprX = float((ref['base'] * (rs >= Tx)).sum())
            aub, tJb, tXb = [], [], []
            for b0 in range(0, B, BATCH):
                m = min(B, b0 + BATCH) - b0
                wh = rng.multinomial(Gh, np.full(Gh, 1.0 / Gh), size=m)[:, ccode] * hw[None, :]
                wr = ref_boot_w(ref, m, rng); den = wr.sum(1)
                aub.append(wauc(s, y, np.concatenate([wh, wr], axis=1)))
                tJb.append((wr * (rs[None, :] >= TJb[b0:b0 + m, None])).sum(1) / den)
                tXb.append((wr * (rs[None, :] >= Txb[b0:b0 + m, None])).sum(1) / den)
            aub, tJb, tXb = np.concatenate(aub), np.concatenate(tJb), np.concatenate(tXb)
            pc = lambda v: (100 * np.percentile(v, 2.5), 100 * np.percentile(v, 97.5))
            rows.append(dict(det=tag, name=name, ref=rname, cohort=cn, n_files=len(hs.ids), n_windows=len(hsc), clusters=Gh,
                             auroc=au, au_lo=np.percentile(aub, 2.5), au_hi=np.percentile(aub, 97.5),
                             tau_JamA=TJ, fpr_tauJamA=100 * fprJ, fpr_lo=pc(fprJb)[0], fpr_hi=pc(fprJb)[1],
                             tpr_tauJamA=100 * tprJ, tprJ_lo=pc(tJb)[0], tprJ_hi=pc(tJb)[1],
                             tau_oracle=Tx, tpr_oracle=100 * tprX, tprX_lo=pc(tXb)[0], tprX_hi=pc(tXb)[1]))
    return rows


def load_det(tag, dumps):
    out = {}
    for d in dumps:
        c = C.load_chunks(tag, d, cols=('ID', 'j', 'lg', 'mp', 'dur'))
        if c is None: continue
        c['w'] = (np.nan_to_num(c.mp.values.astype(float), nan=0.0) + 1e-3) if 'mp' in c else 1.0
        out[d] = c.sort_values(['ID', 'j']).reset_index(drop=True)
    return out


class FileSet:
    """Per-file windows (logits, weights), clusters and strata of one cohort."""

    def __init__(self, c, ids, clus, strata=None):
        c = c[c.ID.isin(set(ids))]
        self.ids = np.array(sorted(c.ID.unique()))
        pos = {v: i for i, v in enumerate(self.ids)}
        self.fi = c.ID.map(pos).values; self.lg = c.lg.values.astype(float); self.w = c.w.values.astype(float)
        self.n = np.bincount(self.fi, minlength=len(self.ids)); self.off = np.r_[0, np.cumsum(self.n)[:-1]]
        cm = dict(zip(ids, clus)); self.clus = np.array([cm[i] for i in self.ids])
        self.strata = None if strata is None else np.array([dict(zip(ids, strata))[i] for i in self.ids])

    def units(self, k):
        """k='file' -> one unit per file (all windows); integer k -> every run of k consecutive windows (files with n >= k).
        Returns (scores, file index)."""
        if k == 'file':
            sc = np.empty(len(self.ids))
            for f in range(len(self.ids)):
                a, n = self.off[f], self.n[f]
                sc[f] = trim20_w(self.lg[a:a + n][None, :], self.w[a:a + n][None, :])[0]
            return sc, np.arange(len(self.ids))
        ok = np.where(self.n >= k)[0]
        if len(ok) == 0: return np.array([]), np.array([], int)
        nw = self.n[ok] - k + 1
        fidx = np.repeat(ok, nw)
        start = np.concatenate([np.arange(m) for m in nw]) + np.repeat(self.off[ok], nw)
        idx = start[:, None] + np.arange(k)[None, :]
        return trim20_w(self.lg[idx], self.w[idx]), fidx


def trim20_w(lg, w):
    N, k = lg.shape
    if k == 1: return lg[:, 0]
    keep = max(1, k - int(np.floor(0.2 * k)))
    o = np.argsort(lg, axis=1, kind='stable')[:, :keep]
    l2 = np.take_along_axis(lg, o, 1); w2 = np.take_along_axis(w, o, 1)
    return (l2 * w2).sum(1) / w2.sum(1)


def calib(fs, k, rng, B):
    """tau at calibration unit k (track-weighted quantile) and tau_b (artists resampled)."""
    sc, fidx = fs.units(k)
    if len(sc) == 0: return np.nan, np.full(B, np.nan), 0
    cnt = np.bincount(fidx, minlength=len(fs.ids)).astype(float); base = 1.0 / cnt[fidx]
    ccode = pd.factorize(pd.Series(fs.clus))[0][fidx]; G = int(ccode.max()) + 1
    T, _ = wq_tau(sc, base, ALPHA)
    return T, boot_tau(sc, base, ccode, G, ALPHA, rng, B), int((cnt > 0).sum())


def deploy(fs, k):
    sc, fidx = fs.units(k)
    if len(sc) == 0: return None
    keep = np.unique(fidx); remap = -np.ones(len(fs.ids), int); remap[keep] = np.arange(len(keep))
    st = None if fs.strata is None else fs.strata[keep]
    return Cohort(sc, remap[fidx], fs.clus[keep], st), len(keep), len(sc), len(set(fs.clus[keep]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=10000); ap.add_argument('--dets', default='', help='paper or directory names, comma-separated')
    ap.add_argument('--part', action='store_true'); ap.add_argument('--merge', action='store_true')
    ap.add_argument('--only_ru', action='store_true', help='only the rank analysis against unseen families (4)')
    a = ap.parse_args(); B = a.B
    if a.merge: return merge()
    reg = C.load_registry().set_index('name')
    metas = {p: C.read_list(f'{R}/{p}').assign(_row=lambda m: np.arange(len(m))) for p in reg.list.unique()}

    def fileset(name, c):
        r = reg.loc[name]; m = metas[r['list']]; sub = m.query(r['subset']) if r['subset'] else m
        strata = sub['group'].astype(str).values if r['role'] == 'fake' and 'group' in sub else None
        return FileSet(c, sub._row.values, sub[r['cluster']].astype(str).values, strata)
    dets = C.select(DETS, EXTRA, a.dets)
    t0 = time.time(); M, P, G, RU = [], [], [], []
    for tag, name in dets:
        need = {reg.loc[n, 'dump'] for n in set(T_REAL + T_FAKE + PROV_T + FAKE_COH + RU_COH + ['jamendo_A', 'fma_dev800', 'newgen', 'music8k'])}
        ch = load_det(tag, need)
        core = {reg.loc[n, 'dump'] for n in ['jamendo_A', 'fma_dev800', 'jamendo_B']}
        if not core <= set(ch): print(f'[skip] {tag}: calibration or reference scores missing', flush=True); continue
        rng = np.random.default_rng(0)
        kmax = 7 if 'ArtifactNet' in tag else 12
        kc = sorted({min(k, kmax) for k in KCAL}); kd = sorted({min(k, kmax) for k in KDEP})
        FS = {n: fileset(n, ch[reg.loc[n, 'dump']]) for n in set(T_REAL + T_FAKE + PROV_T + RU_COH + ['jamendo_A', 'fma_dev800'])
              if reg.loc[n, 'dump'] in ch}
        # (4) rank versus calibration against unseen families (own random seed, independent of the other parts)
        RU += rank_unseen(tag, name, FS, ch, fileset, reg, B)
        if a.only_ru:
            print(f'[{tag}] rank_unseen done {time.time() - t0:.0f}s', flush=True); continue
        # (1) unit matrix
        cal = {k: calib(FS['jamendo_A'], k, rng, B) for k in kc}
        Tpt = np.array([cal[k][0] for k in kc]); Tmat = np.stack([cal[k][1] for k in kc])
        for cn in T_REAL + T_FAKE:
            if cn not in FS: continue
            for k2 in kd:
                dp = deploy(FS[cn], k2)
                if dp is None: continue
                Co, nf, nw, ncl = dp; pt = Co.rate(Tpt); bt = Co.boot(Tmat, rng, B)
                for i, k1 in enumerate(kc):
                    lo, hi = wilson(pt[i], ncl)
                    M.append(dict(det=tag, name=name, k_cal=k1, k_dep=k2, cohort=cn, role=reg.loc[cn, 'role'], tau=Tpt[i], n_files=nf, n_units=nw,
                                  clusters=ncl, rate=100 * pt[i], ci_lo=100 * np.percentile(bt[i], 2.5), ci_hi=100 * np.percentile(bt[i], 97.5),
                                  wil_hi_clusters=100 * hi, cal_files=cal[k1][2]))
        # (2) source x unit
        units = [('file', 'file'), ('win', 1)]
        Tp, Tb, lab = [], [], []
        for src in ('fma_dev800', 'jamendo_A'):
            for ul, k in units:
                T, Tbk, ncal = calib(FS[src], k, rng, B); Tp.append(T); Tb.append(Tbk); lab.append((src, ul, ncal))
        Tp = np.array(Tp); Tb = np.stack(Tb)
        for cn in PROV_T:
            if cn not in FS: continue
            for ul, k in units:
                dp = deploy(FS[cn], k)
                if dp is None: continue
                Co, nf, nw, ncl = dp; idx = [i for i, l in enumerate(lab) if l[1] == ul]
                pt = Co.rate(Tp[idx]); bt = Co.boot(Tb[idx], rng, B)
                for jj, i in enumerate(idx):
                    P.append(dict(det=tag, name=name, source=lab[i][0], unit=ul, cohort=cn, tau=Tp[i], cal_n=lab[i][2], n_files=nf, n_units=nw, clusters=ncl,
                                  rate=100 * pt[jj], ci_lo=100 * np.percentile(bt[jj], 2.5), ci_hi=100 * np.percentile(bt[jj], 97.5)))
        # (3) per-generator TPR (window, tau_JamA, k = 1)
        TJ1 = cal[1][0]; TJb = cal[1][1][None, :]
        for cn in FAKE_COH:
            dname = reg.loc[cn, 'dump'] if cn in reg.index else None
            if dname not in ch: continue
            fs = fileset(cn, ch[dname]); gens = ['all'] + (sorted(set(fs.strata)) if fs.strata is not None and len(set(fs.strata)) > 1 else [])
            for g in gens:
                if g == 'all':
                    sub = fs  # whole cohort: generator-stratified bootstrap
                else:
                    sel_ = fs.ids[fs.strata == g]
                    sub = FileSet(ch[dname], sel_, [str(x) for x in sel_])
                dp = deploy(sub, 1)
                if dp is None: continue
                Co, nf, nw, ncl = dp; pt = Co.rate([TJ1])[0]; bt = Co.boot(TJb, rng, B)[0]
                G.append(dict(det=tag, name=name, cohort=cn, generator=g, n_files=nf, tpr=100 * pt, ci_lo=100 * np.percentile(bt, 2.5), ci_hi=100 * np.percentile(bt, 97.5)))
        print(f'[{tag}] done {time.time() - t0:.0f}s', flush=True)
    dfs = dict(matrix=pd.DataFrame(M), prov=pd.DataFrame(P), gen=pd.DataFrame(G), rank_unseen=pd.DataFrame(RU))
    if a.only_ru: dfs = {k: dfs[k] for k in ('rank_unseen',)}
    if a.part:
        os.makedirs(f'{TB}/unit_parts', exist_ok=True); tg = '_'.join(C.DETECTORS[t][0] for t, _ in dets) or 'all'
        for k, d in dfs.items(): d.to_csv(f'{TB}/unit_parts/{tg}_{k}.csv', index=False)
        print('UNIT_PART_DONE', tg, f'{time.time() - t0:.0f}s'); return
    write(dfs)


def write(dfs):
    for k, d in dfs.items():
        d.to_csv(f'{TB}/unit_{k}.csv', index=False)
    pd.DataFrame(EXPO, columns=['cohort', 'generator', 'family', 'B0fam', 'C0fam', 'SONICS', 'note']).to_csv(f'{TB}/unit_exposure.csv', index=False)
    print('UNIT_DONE', {k: len(v) for k, v in dfs.items()})


def merge():
    order = {t: i for i, (t, _) in enumerate(DETS + EXTRA)}
    dfs = {}
    for k in ('matrix', 'prov', 'gen', 'rank_unseen'):
        fs = sorted(glob.glob(f'{TB}/unit_parts/*_{k}.csv'))
        dd = [pd.read_csv(f) for f in fs if os.path.getsize(f) > 1]
        d = pd.concat(dd, ignore_index=True) if dd else pd.DataFrame()
        if not d.empty: d = d.sort_values('det', key=lambda c: c.map(order), kind='stable')
        dfs[k] = d
    write(dfs)


if __name__ == '__main__':
    main()
