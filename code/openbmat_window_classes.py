"""Class label of every 10.24-s AI-OpenBMAT window (input of the paired window AUROC in agg_window_variants.py).

- Segment times: the per-file segment list of AI-OpenBMAT (lists/openbmat_segments.csv, taken from the dataset's per-file JSON
  annotations) is sequential (start = cumulative duration of the preceding segments) and is cut at the file duration (60 s).
  Class = event_type (segment_type for the 4 'skipped' events) in {music, fg-music, bg-music, bgvl-music, similar, no-music}
  -> mus / fg / bg / bgvl / sim / nomus.
- For the window [start, min(start + 10.24, dur)] the overlap in seconds with every class is computed; the class covering
  at least 80% of the window is assigned, otherwise 'mixed'.
- Window positions are those of the released B0 window scores (all EAT/MERT detectors share this window grid).
- Both arms are labelled from their own annotations (the structure of a human/AI pair is identical; agreement is printed).
Output: out/openbmat_window_classes.csv (arm, pair, item_id, j, start, end, cls, frac, mus_s, fg_s, bg_s, bgvl_s, sim_s, nomus_s)
usage: python code/openbmat_window_classes.py
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

OUT = f'{C.TB}/openbmat_window_classes.csv'
CROP = 10.24; SHORT = {'music': 'mus', 'fg-music': 'fg', 'bg-music': 'bg', 'bgvl-music': 'bgvl', 'similar': 'sim', 'no-music': 'nomus'}
CL = list(SHORT.values())


def main():
    seg = pd.read_csv(C.LISTS / 'openbmat_segments.csv', dtype={'item_id': str}, keep_default_na=False)
    rows = []
    for arm, dump in (('human', 'openbmat_human'), ('ai', 'openbmat_ai')):
        c = C.load_chunks('B0', dump, cols=('ID', 'item_id', 'j', 'start', 'dur'))
        sa = seg[seg.arm == arm]
        segs_by_item = {k: g.sort_values('seg') for k, g in sa.groupby('item_id', sort=False)}
        for iid, g in c.groupby('item_id', sort=False):
            sg = segs_by_item[iid]; dur = float(sg.file_duration.iloc[0]); t = 0.0; segs = []
            for s in sg.itertuples():
                e = s.event_type; e = s.segment_type if e not in SHORT else e
                a, b = t, min(t + float(s.duration), dur); t += float(s.duration)
                if b > a: segs.append((a, b, SHORT[e]))
            pair = os.path.basename(iid).replace('_ai.wav', '').replace('.wav', '')
            for r in g.itertuples():
                ws, we = float(r.start), min(float(r.start) + CROP, min(float(r.dur), dur)); L = we - ws
                ov = {k: 0.0 for k in CL}
                for a, b, k in segs:
                    ov[k] += max(0.0, min(b, we) - max(a, ws))
                best = max(ov, key=ov.get); frac = ov[best] / L if L > 0 else 0.0
                rows.append(dict(arm=arm, pair=pair, item_id=iid, j=int(r.j), start=round(ws, 3), end=round(we, 3), cls=best if frac >= 0.8 else 'mixed', frac=round(frac, 3),
                                 **{f'{k}_s': round(v, 2) for k, v in ov.items()}))
    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    h = df[df.arm == 'human'].set_index(['pair', 'j']).cls; a = df[df.arm == 'ai'].set_index(['pair', 'j']).cls
    print('windows', len(df), 'per arm', len(h), len(a), 'human==ai class agreement', float((h == a.reindex(h.index)).mean()))
    print('class counts (human):', df[df.arm == 'human'].cls.value_counts().to_dict())
    print('->', OUT)


if __name__ == '__main__':
    main()
