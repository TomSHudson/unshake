"""Misfit / similarity measures, vectorised over samples.

All functions take data (n_rec, n_t) and synth (n, n_rec, n_t) (or (n_rec, n_t))
and return one value per sample, shape (n,) (or a scalar).

mode='stacked'   : all traces flattened and compared together.
mode='per_trace' : each trace compared separately, then (weighted) averaged.
"""

import numpy as np


def _as_batch(synth):
    synth = np.asarray(synth, dtype=float)
    return (synth[None], True) if synth.ndim == 2 else (synth, False)


def _out(x, single):
    return x[0] if single else x


def _combine(per_trace, weights):
    w = np.ones(per_trace.shape[-1]) if weights is None else np.asarray(weights, dtype=float)
    return per_trace @ w / w.sum()


def normalise_traces(x):
    """Divide each trace by its max absolute value."""
    return x / (np.max(np.abs(x), axis=-1, keepdims=True) + 1e-30)


def vr(data, synth, mode='stacked', weights=None):
    """Variance reduction 1 - sum((d-s)^2) / sum(d^2) (Templeton & Dreger 2006).
    Not clipped at zero (SeisSrcInv clipped it)."""
    synth, single = _as_batch(synth)
    res2 = np.sum((data - synth) ** 2, axis=-1)          # (n, n_rec)
    d2 = np.sum(data ** 2, axis=-1)                      # (n_rec,)
    if mode == 'stacked':
        w = 1.0 if weights is None else np.asarray(weights)
        out = 1.0 - np.sum(w * res2, axis=-1) / np.sum(w * d2)
    else:
        out = _combine(1.0 - res2 / d2, weights)
    return _out(out, single)


def pcc(data, synth, mode='stacked', weights=None):
    """Pearson correlation coefficient (-1 to 1)."""
    synth, single = _as_batch(synth)
    if mode == 'stacked':
        d = data.ravel() - data.mean()
        s = synth.reshape(len(synth), -1)
        s = s - s.mean(axis=1, keepdims=True)
        out = s @ d / (np.linalg.norm(s, axis=1) * np.linalg.norm(d))
    else:
        d = data - data.mean(axis=-1, keepdims=True)
        s = synth - synth.mean(axis=-1, keepdims=True)
        r = np.sum(d * s, axis=-1) / (np.linalg.norm(d, axis=-1) * np.linalg.norm(s, axis=-1) + 1e-30)
        out = _combine(r, weights)
    return _out(out, single)


def ncc(data, synth, max_lag=0, weights=None):
    """Per-trace normalised cross-correlation, maximised over integer lags in
    [-max_lag, max_lag] (circular shift), then (weighted) averaged over traces."""
    synth, single = _as_batch(synth)
    d = data - data.mean(axis=-1, keepdims=True)
    s = synth - synth.mean(axis=-1, keepdims=True)
    denom = np.linalg.norm(d, axis=-1) * np.linalg.norm(s, axis=-1) + 1e-30
    best = np.full(s.shape[:2], -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        best = np.maximum(best, np.sum(d * np.roll(s, lag, axis=-1), axis=-1) / denom)
    return _out(_combine(best, weights), single)


def gaussian_loglike(data, synth, sigma, weights=None):
    """Gaussian log-likelihood (up to a constant): -0.5 sum(w (d - s)^2 / sigma^2).

    sigma : scalar or (n_rec,) noise standard deviation.
    """
    synth, single = _as_batch(synth)
    sigma = np.asarray(sigma, dtype=float)
    w = 1.0 if weights is None else np.asarray(weights, dtype=float)
    sig2 = sigma ** 2 if sigma.ndim == 0 else (sigma ** 2)[:, None]
    chi2 = np.sum(((data - synth) ** 2 / sig2).sum(axis=-1) * w, axis=-1)
    return _out(-0.5 * chi2, single)


SIMILARITY = {'vr': vr, 'pcc': pcc, 'ncc': ncc}
