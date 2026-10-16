#!/usr/bin/env python3
"""Anonymity scan run before release.

Every file name and every file content of the repository is checked, including the members of .gz, .zip and .npz files
(string arrays inside .npz files are decoded; numeric arrays carry no text). Two kinds of checks:
  1. e-mail addresses (built in);
  2. a list of identifying strings. The list is private, because it would itself identify the authors, and is kept outside
     the repository; pass it with --patterns or the ANON_PATTERNS environment variable. One pattern per line, matched
     case-insensitively as a substring; a pattern made only of digits is matched as a whole number (not inside a longer
     number or a decimal fraction); a line starting with 're:' is a regular expression; '#' starts a comment line.
The scan fails closed: a missing pattern list, an unreadable file or any hit gives exit code 1.
usage: python scan_anon.py --patterns PATH [--root .] [--max-show 5]
"""
import argparse
import gzip
import io
import os
import re
import sys
import zipfile

EMAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}')


def load_patterns(path):
    lit, rx = [], []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            s = line.rstrip('\n')
            if not s.strip() or s.lstrip().startswith('#'):
                continue
            if s.startswith('re:'):
                rx.append((s, re.compile(s[3:], re.IGNORECASE)))
            elif s.strip().isdigit():
                d = s.strip()
                rx.append((d, re.compile(r'(?<![0-9.])' + d + r'(?![0-9.])')))
            else:
                lit.append(s.lower())
    if not lit and not rx:
        raise ValueError('empty pattern list')
    return lit, rx


def find_hits(text, lit, rx):
    """-> list of (pattern, position)."""
    out = []
    low = text.lower()
    for p in lit:
        i = low.find(p)
        while i >= 0:
            out.append((p, i)); i = low.find(p, i + 1)
    for name, r in rx:
        out += [(name, m.start()) for m in r.finditer(text)]
    out += [('e-mail address', m.start()) for m in EMAIL.finditer(text)]
    return out


def where(text, pos):
    line = text.count('\n', 0, pos) + 1
    a = text.rfind('\n', 0, pos) + 1; b = text.find('\n', pos); b = len(text) if b < 0 else b
    return line, text[a:b][:160]


def decode(data):
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return data.decode('latin-1')  # binary content: ASCII runs are still visible to the scan


def scan_npz_member(name, data):
    import numpy as np
    arr = np.load(io.BytesIO(data), allow_pickle=False)
    if arr.dtype.kind in 'US':
        return '\n'.join(str(x) for x in arr.ravel().tolist())
    if arr.dtype.kind == 'O':
        raise ValueError(f'object array in {name}')
    return ''  # numeric array: no text


def contents(path):
    """Yield (location, text) for a file and its members."""
    with open(path, 'rb') as fh:
        data = fh.read()
    low = path.lower()
    if low.endswith('.gz'):
        yield path + ' [gzip]', decode(gzip.decompress(data))
    elif low.endswith('.npz') or low.endswith('.zip'):
        zf = zipfile.ZipFile(io.BytesIO(data))
        yield path + ' [member names]', '\n'.join(zf.namelist()) + '\n' + decode(zf.comment or b'')
        for n in zf.namelist():
            raw = zf.read(n)
            if low.endswith('.npz') and n.endswith('.npy'):
                yield f'{path} [{n}]', scan_npz_member(n, raw)
            else:
                yield f'{path} [{n}]', decode(raw)
    else:
        yield path, decode(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patterns', default=os.environ.get('ANON_PATTERNS', ''))
    ap.add_argument('--root', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--max-show', type=int, default=5)
    a = ap.parse_args()
    if not a.patterns or not os.path.exists(a.patterns):
        print('FAIL: pattern list not found (pass --patterns PATH or set ANON_PATTERNS)'); sys.exit(1)
    lit, rx = load_patterns(a.patterns)
    root = os.path.abspath(a.root)
    files, hits, errors, nbytes = [], [], [], 0
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for fn in sorted(fns):
            files.append(os.path.relpath(os.path.join(dp, fn), root))
        for dn in dns:
            rel = os.path.relpath(os.path.join(dp, dn), root) + '/'
            for p, pos in find_hits(rel, lit, rx):
                hits.append(('directory name', rel, p, 0, rel))
    for rel in files:
        for p, pos in find_hits(rel, lit, rx):
            hits.append(('file name', rel, p, 0, rel))
        try:
            for loc, text in contents(os.path.join(root, rel)):
                nbytes += len(text)
                for p, pos in find_hits(text, lit, rx):
                    line, ctx = where(text, pos)
                    hits.append(('content', loc, p, line, ctx))
        except Exception as e:  # unreadable content fails the scan
            errors.append((rel, repr(e)))
    print(f'scanned {len(files)} files, {nbytes / 1e6:.1f} MB of text, {len(lit)} literal and {len(rx)} pattern rules + e-mail')
    if errors:
        print(f'FAIL: {len(errors)} unreadable files'); [print('  ', *e) for e in errors]
    if hits:
        print(f'FAIL: {len(hits)} hits')
        shown = {}
        for kind, loc, p, line, ctx in hits:
            k = (loc, p)
            shown[k] = shown.get(k, 0) + 1
            if shown[k] <= a.max_show:
                print(f'  [{kind}] {loc}:{line}  pattern={p!r}  ...{ctx}...')
        summary = {}
        for kind, loc, p, line, ctx in hits:
            summary[p] = summary.get(p, 0) + 1
        print('hits per pattern:', summary)
    if hits or errors:
        sys.exit(1)
    print('PASS: no hits')


if __name__ == '__main__':
    main()
