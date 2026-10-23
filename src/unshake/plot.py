"""Minimal plotting: lune, beachball, waveform fits, corner frequency vs azimuth."""

import matplotlib.pyplot as plt
import numpy as np

from . import mt
from .misfit import normalise_traces


def _hammer(gamma, delta):
    """Hammer projection of lune longitude gamma / latitude delta (radians)."""
    c = np.sqrt(1.0 + np.cos(delta) * np.cos(gamma / 2.0))
    return (2.0 * np.sqrt(2.0) * np.cos(delta) * np.sin(gamma / 2.0) / c,
            np.sqrt(2.0) * np.sin(delta) / c)


_LUNE_REFS = {'DC': [1, 0, -1], '+ISO': [1, 1, 1], '-ISO': [-1, -1, -1],
              '+CLVD': [2, -1, -1], '-CLVD': [1, 1, -2], '+crack': [3, 1, 1], '-crack': [-1, -1, -3]}


def lune(m6=None, weights=None, result=None, best=None, values=None, stat='max', vmin=None,
         colorbar=False, bins=(30, 60), ax=None, cmap='viridis'):
    """Lune plot (Hammer projection) of MT samples.

    Pass either `result` (a `Result`) or `m6` (n, 6) with optional `weights`
    (plots a weighted histogram). Alternatively pass per-sample `values`
    (e.g. log-likelihoods) to plot their per-bin `stat` ('max' gives a profile over
    orientation; any `scipy.stats.binned_statistic_2d` statistic works), shown
    relative to the overall maximum and clipped at `vmin`.
    `best` (6,) is marked with a star (defaults to the max-likelihood result sample).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 6))
    if result is not None:
        m6, weights = result.mts, result.weights
        best = result.best[:6] if best is None else best
    if m6 is not None:
        g, d = mt.lune(np.atleast_2d(m6))
        lune_range = [[-np.pi / 6, np.pi / 6], [-np.pi / 2, np.pi / 2]]
        if values is None:
            H, ge, de = np.histogram2d(g, d, bins=bins, weights=weights, range=lune_range)
            H = np.ma.masked_equal(H, 0)
        else:
            from scipy.stats import binned_statistic_2d
            H, ge, de, _ = binned_statistic_2d(g, d, values, statistic=stat, bins=bins, range=lune_range)
            H = np.ma.masked_invalid(H - np.nanmax(H))
        GE, DE = np.meshgrid(ge, de, indexing='ij')
        X, Y = _hammer(GE, DE)
        mesh = ax.pcolormesh(X, Y, H, cmap=cmap, shading='flat', vmin=vmin)
        if colorbar:
            plt.colorbar(mesh, ax=ax, shrink=0.6, label='weight' if values is None else f'{stat} - max')
    # Boundary, grid lines and reference sources:
    dd = np.linspace(-np.pi / 2, np.pi / 2, 181)
    for gg in np.radians([-30, -20, -10, 0, 10, 20, 30]):
        ax.plot(*_hammer(np.full_like(dd, gg), dd), color='k', lw=1.2 if abs(gg) > 0.5 else 0.3)
    gl = np.linspace(-np.pi / 6, np.pi / 6, 61)
    for dl in np.radians([-60, -30, 0, 30, 60]):
        ax.plot(*_hammer(gl, np.full_like(gl, dl)), color='k', lw=0.3)
    for name, lam in _LUNE_REFS.items():
        g, d = mt.lune(np.array([*lam, 0.0, 0.0, 0.0]))
        x, y = _hammer(g, d)
        ax.plot(x, y, 'ko', ms=3)
        ax.annotate(name, (x, y), fontsize=7, xytext=(3, 3), textcoords='offset points')
    if best is not None:
        ax.plot(*_hammer(*mt.lune(best)), marker='*', color='r', ms=12, ls='')
    ax.set_aspect('equal')
    ax.axis('off')
    return ax


def beachball(m6, ax=None, facecolor='k', width=2.0):
    """Beachball (lower hemisphere) of an NED moment tensor, via obspy."""
    from obspy.imaging.beachball import beach
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
    ax.add_collection(beach(mt.to_use(np.ravel(m6)), xy=(0, 0), width=width, facecolor=facecolor))
    lim = 0.55 * width
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axis('off')
    return ax


def wavefield(dataset, synth, normalise=True, x=None, xlabel='Receiver index'):
    """Observed / modelled / residual record sections (e.g. along a DAS fibre)."""
    d, s = dataset.data, np.asarray(synth)
    if normalise:
        d, s = normalise_traces(d), normalise_traces(s)
    x = np.arange(dataset.n_rec) if x is None else x
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    vmax = np.max(np.abs(d))
    for ax, arr, title in zip(axes, (d, s, d - s), ('Observed', 'Model', 'Residual')):
        im = ax.pcolormesh(x, dataset.times, arr.T, cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel('Time (s)')
    fig.colorbar(im, ax=axes, shrink=0.8)
    return fig


def traces(dataset, synth, idx, normalise=True):
    """Data vs model for selected receivers (indices or labels)."""
    idx = np.atleast_1d(idx)
    if idx.dtype.kind in 'USO':
        idx = np.array([np.where(dataset.labels == lab)[0][0] for lab in idx])
    fig, axes = plt.subplots(len(idx), 1, figsize=(6, 1.2 * len(idx) + 1), sharex=True, squeeze=False)
    for ax, i in zip(axes[:, 0], idx):
        d, s = dataset.data[i], np.asarray(synth)[i]
        if normalise:
            d, s = normalise_traces(d), normalise_traces(s)
        ax.plot(dataset.times, d, 'k', lw=1, label='data')
        ax.plot(dataset.times, s, 'r', lw=1, label='model')
        ax.set_ylabel(str(dataset.labels[i]), rotation=0, ha='right')
    axes[0, 0].legend(loc='upper right', fontsize=8)
    axes[-1, 0].set_xlabel('Time (s)')
    return fig


def fc_azimuth(model, params, dataset, ax=None):
    """Apparent corner frequency at each receiver vs azimuth (for `MTSpectrumModel`)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    fc = model.corner_frequencies(params, dataset)
    for row in np.atleast_2d(fc):
        ax.plot(dataset.azimuth, row, '.', ms=3)
    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Apparent fc (Hz)')
    return ax
