"""Paired window AUROC on AI-OpenBMAT by music class (; B0 family, C0 family and SpecTTTra).

Each window is assigned the class that covers at least 80% of it (otherwise 'mixed'); classes come from the dataset's per-file
segment annotations (lists/openbmat_segments.csv, human arm; the AI arm has the same structure).
  EAT detectors: window start from the released scores, window length 10.24 s.  SpecTTTra: 5-s windows, start = 5 * j.
Human file gen_x and AI file gen_x share the structure (same window -> same class). Clusters = cluster_ob (first music source /
speech source); pairs are resampled together, B replicates.
Output (out/): openbmat_window_auroc.csv (same columns as the frozen results/openbmat_window_auroc.csv)
usage: python code/openbmat_window_auroc.py [--B 10000]
"""
import os, sys, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from auroc import wauc  # noqa: E402

TB = C.TB
SHORT = {'music': 'mus', 'fg-music': 'fg', 'bg-music': 'bg', 'bgvl-music': 'bgvl', 'similar': 'sim', 'no-music': 'nomus'}
MODELS = [('B0', 'B0', 10.24), ("B0'", "B0'", 10.24), ('B0+0', 'B0+0', 10.24), ('C0', 'C0', 10.24), ('C0+0', 'C0+0', 10.24),
          ('SpecTTTra', 'SpecTTTra', 5.0)]


def seg_tables():
    """Human-arm segments -> {row of the item in lists/items/openbmat_human.csv: [(start, end, class)]} (segments are consecutive)."""
    seg = pd.read_csv(C.LISTS / 'openbmat_segments.csv', dtype={'item_id': str}, keep_default_na=False)
    seg = seg[seg.arm == 'human']
    pos = C.item_index('openbmat_human')
    out = {}
    for iid, g in seg.groupby('item_id', sort=False):
        t = 0.0; lst = []
        for s in g.sort_values('seg').itertuples():
            d = float(s.duration); c = SHORT.get(s.event_type, 'other'); lst.append((t, t + d, c)); t += d
        out[int(pos[iid])] = lst
    return out


def chunk_class(segs, st, win, pure=0.8):
    ov = {}
    for a, b, c in segs:
        o = max(0.0, min(b, st + win) - max(a, st))
        if o > 0: ov[c] = ov.get(c, 0.0) + o
    if not ov: return 'none'
    c = max(ov, key=ov.get)
    return c if ov[c] >= pure * win else 'mixed'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--B', type=int, default=10000); a = ap.parse_args()
    H = C.read_list(C.ITEMS / 'openbmat_human.csv'); AI = C.read_list(C.ITEMS / 'openbmat_ai.csv')
    segs = seg_tables()
    rows = []
    for tag, name, win in MODELS:
        ch = C.load_chunks(tag, 'openbmat_human'); ca = C.load_chunks(tag, 'openbmat_ai')
        if ch is None or ca is None: continue
        for c in (ch, ca):
            if c['start'].isna().all():  # SpecTTTra: 5-s windows, start = 5 * j
                c['start'] = c['j'] * win
        ch['cls'] = [chunk_class(segs[i], st, win) for i, st in zip(ch.ID, ch.start)]
        ca['cls'] = [chunk_class(segs[i], st, win) for i, st in zip(ca.ID, ca.start)]
        ch['cl'] = H.cluster_ob.values[ch.ID.values]; ca['cl'] = AI.cluster_ob.values[ca.ID.values]
        rng = np.random.default_rng(0)
        for cls in ['fg', 'bg', 'bgvl', 'sim', 'mus', 'nomus', 'mixed']:
            h = ch[ch.cls == cls]; x = ca[ca.cls == cls]
            if len(h) < 10 or len(x) < 10: continue
            s = np.r_[h.lg.values, x.lg.values]; y = np.r_[np.zeros(len(h)), np.ones(len(x))]
            codes, uniq = pd.factorize(pd.Series(np.r_[h.cl.values, x.cl.values])); G = len(uniq)
            au = float(wauc(s, y, np.ones((1, len(s))))[0]); bs = []
            for b0 in range(0, a.B, 500):
                m = min(a.B, b0 + 500) - b0
                bs.append(wauc(s, y, rng.multinomial(G, np.full(G, 1.0 / G), size=m).astype(np.float64)[:, codes]))
            bs = np.concatenate(bs)
            rows.append(dict(model=tag, name=name, cls=cls, n_chunks_human=len(h), n_chunks_ai=len(x), n_files=h.ID.nunique(), clusters=G,
                             auroc=au, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
        print(f'[{tag}] done', flush=True)
    d = pd.DataFrame(rows); d.to_csv(f'{TB}/openbmat_window_auroc.csv', index=False)
    print(d.pivot_table(index='name', columns='cls', values='auroc', sort=False).round(3).to_string())


if __name__ == '__main__':
    main()
