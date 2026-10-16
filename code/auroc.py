"""Weighted AUROC used by the rank analyses."""
import numpy as np


def wauc(s, y, W):
    """Weighted AUROC (Mann-Whitney, ties count 0.5). s, y: (n,), W: (B, n) weights -> (B,)."""
    order = np.argsort(s, kind='stable'); s = s[order]; y = y[order]; W = W[:, order]
    starts = np.r_[0, np.where(np.diff(s) != 0)[0] + 1]
    Wp = W * (y == 1); Wn = W * (y == 0)
    P = np.add.reduceat(Wp, starts, axis=1); N = np.add.reduceat(Wn, starts, axis=1)
    Cn = np.cumsum(N, axis=1) - N
    num = (P * (Cn + 0.5 * N)).sum(1); den = P.sum(1) * N.sum(1)
    return num / np.maximum(den, 1e-12)
