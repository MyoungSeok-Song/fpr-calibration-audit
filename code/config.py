"""Paths, detector names and data loaders shared by every script.

All paths are relative to the repository root:
  lists/cohorts.csv                      cohort registry (one row per cohort: item list, subset query, role, cluster column)
  lists/items/<dataset>.csv              item lists; the row number of an item is its index ID
  scores/windows/<detector>/<dataset>.csv.gz   per-window detector logits (item_id, window index j, ...)
  scores/heads/<family>/<dataset>.npz          per-window outputs of the last-layer heads (Table 3)
  results/                               frozen result tables behind the paper
  out/                                   default output directory of every script (override with OUT_DIR)
"""
import os
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = str(ROOT)
LISTS = ROOT / 'lists'
ITEMS = LISTS / 'items'
SCORES = ROOT / 'scores'
RESULTS = ROOT / 'results'
OUT = pathlib.Path(os.environ.get('OUT_DIR', str(ROOT / 'out')))
TB = str(OUT)
REG = str(LISTS / 'cohorts.csv')
OUT.mkdir(parents=True, exist_ok=True)

# Detectors with released window scores: paper name -> (directory under scores/windows, scored window).
# C1 is not released.
DETECTORS = {
    'B0': ('B0', 'EAT 10.24 s'),
    "B0'": ('B0prime', 'EAT 10.24 s'),
    'B0+0': ('B0plus0', 'EAT 10.24 s'),
    'B0+0 (seed 2)': ('B0plus0_seed2', 'EAT 10.24 s'),
    'C0': ('C0', 'EAT 10.24 s'),
    'C0+0': ('C0plus0', 'EAT 10.24 s'),
    'M0': ('M0', 'MERT-330M 10.24 s'),
    'B0+SU': ('B0plusSU', 'EAT 10.24 s'),
    'B0+AVA': ('B0plusAVA', 'EAT 10.24 s'),
    'B0+broad': ('B0plusbroad', 'EAT 10.24 s'),
    'M0+SU': ('M0plusSU', 'MERT-330M 10.24 s'),
    'SpecTTTra': ('SpecTTTra', '5 s'),
    'ArtifactNet': ('ArtifactNet', '4 s x 7'),
    'ArtifactNet (native rate)': ('ArtifactNet_native', '4 s x 7'),
    'Fakeprint LR': ('FakeprintLR', 'EAT 10.24 s grid'),
    'SpecTTTra-alpha': ('SpecTTTra-alpha', '5 s'),
    'SpecTTTra-beta': ('SpecTTTra-beta', '5 s'),
}
PUBLIC = ('SpecTTTra', 'ArtifactNet', 'ArtifactNet (native rate)', 'Fakeprint LR', 'SpecTTTra-alpha', 'SpecTTTra-beta')
# The ten detectors of the main analyses in the paper (C1 not released).
MAIN = ['B0', "B0'", 'B0+0', 'C0', 'C0+0', 'M0', 'ArtifactNet', 'SpecTTTra', 'Fakeprint LR']
# Two further official SONICS checkpoints, scored after the main analysis for the pre-specified replication (Section 5.6). They
# are never part of a default run: name them with --models / --dets (code/extra_sonics.sh does this).
REPLICATION = ('SpecTTTra-alpha', 'SpecTTTra-beta')


# Head families of Table 3: paper name -> directory under scores/heads (C1 not released).
FAMILIES = {
    'B0': 'B0', 'C0': 'C0', 'M0': 'M0',
    'EAT (frozen)': 'EAT_frozen', 'EAT-AS2M (frozen)': 'EAT-AS2M_frozen', 'MERT (frozen)': 'MERT_frozen',
    'CLAP (frozen)': 'CLAP_frozen', 'Fakeprint LR': 'FakeprintLR',
}
FINETUNED_FAMILIES = ('B0', 'C0', 'M0')


def resolve(name, table=None):
    """Paper name or directory name (case-insensitive) -> paper name."""
    table = DETECTORS if table is None else table
    for k, v in table.items():
        slug = v[0] if isinstance(v, tuple) else v
        if name == k or name.lower() == k.lower() or name.lower() == slug.lower():
            return k
    raise KeyError(f'unknown detector {name!r}; known: {list(table)}')


def parse_names(arg, table=None):
    """Comma- or semicolon-separated list of paper or directory names -> paper names."""
    if not arg:
        return []
    sep = ';' if ';' in arg else ','
    return [resolve(x.strip(), table) for x in arg.split(sep) if x.strip()]


def select(default, extra, arg):
    """Detectors of one script run. `default` and `extra` are lists of tuples whose first element is the paper name.
    No names in `arg` -> `default`; otherwise the named detectors among `default` + `extra`, in that order."""
    names = set(parse_names(arg))
    if not names:
        return list(default)
    return [x for x in list(default) + list(extra) if x[0] in names]


def read_list(path):
    """Read an item list (lists/items/<dataset>.csv). item_id stays a string."""
    return pd.read_csv(path, dtype={'item_id': str}, keep_default_na=False, na_values=[''], low_memory=False)


def load_registry():
    return pd.read_csv(REG, keep_default_na=False)


_ITEM_INDEX = {}


def item_index(dump):
    """item_id -> row number in lists/items/<dump>.csv."""
    if dump not in _ITEM_INDEX:
        ids = read_list(ITEMS / f'{dump}.csv').item_id
        _ITEM_INDEX[dump] = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids.values)
    return _ITEM_INDEX[dump]


def chunk_path(det, dump):
    return SCORES / 'windows' / DETECTORS[det][0] / f'{dump}.csv.gz'


def load_chunks(det, dump, cols=None):
    """Window scores of detector `det` on dataset `dump`, one row per scored window.

    Columns: ID (row of the item in lists/items/<dump>.csv), item_id, j (window index within the item), n (windows
    scored for the item), start (window start in s, empty when not recorded), dur (item duration in s), short (1 if the
    item is shorter than one window and was reflect-padded), lg (detector logit), mp (music-presence probability used
    as window weight by the fine-tuned detectors; empty for the public detectors). Returns None if not released.
    """
    p = chunk_path(det, dump)
    if not p.exists():
        return None
    c = pd.read_csv(p, dtype={'item_id': str}, keep_default_na=False, na_values=[''], low_memory=False)
    c.insert(0, 'ID', item_index(dump).loc[c.item_id.values].values)
    if cols is not None:
        c = c[[x for x in cols if x in c.columns]]
    return c


def head_path(family, dump):
    return SCORES / 'heads' / FAMILIES[family] / f'{dump}.npz'


def load_head_scores(family, dump):
    """Per-window head outputs: dict with item_id, j, short and, per head h, h__z (logit), h__ep (E+), h__en (E-), h__z0."""
    p = head_path(family, dump)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}
