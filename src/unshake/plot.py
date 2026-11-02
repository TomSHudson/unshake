"""Minimal plotting: lune, beachball, waveform fits, corner frequency vs azimuth."""

import matplotlib.pyplot as plt
import numpy as np

from . import mt, spectra
from .forward import stf_spectrum, synthetics
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


def spectra_check(dataset, m, stf=None, mesh_fmax=None, band=None, fc=(50, 100, 200),
                  noise_window=(None, 0.0), fmax_plot=300.0, gf_key='mt'):
    """Diagnostic spectra for deciding how to treat the GF source time function.

    Panels: (1) data, synthetic (GFs x m) and pre-pick noise amplitude spectra (median and
    10-90 % over channels); (2) data/synthetic ratio, which estimates S_true / STF, with
    Brune(fc) / STF overlays scaled to the ratio over 20-80 Hz; (3) the per-channel ratio
    (normalised per channel) sorted by azimuth (receiver index if no coordinates).

    stf : moment-rate STF still in the GFs (callable or (t, s), e.g. Salvus
        ``wavelet.get_stf_rate()``, see `forward.stf_spectrum`); None if formal.
    mesh_fmax : GF mesh limit (e.g. 2 * f0) and band : (f0, f1, f2, f3), drawn as lines.
    noise_window : (t1, t2) for the noise spectrum (None = first sample); scaled to the
        signal window length.
    """
    d, s = dataset.data, synthetics(dataset.gf[gf_key], m)
    f, D = spectra.amplitude_spectrum(d, dataset.dt)
    _, S = spectra.amplitude_spectrum(s, dataset.dt)
    t = dataset.times
    t1 = t[0] if noise_window[0] is None else noise_window[0]
    sel = (t >= t1) & (t < noise_window[1])
    _, N = spectra.amplitude_spectrum(d[:, sel], dataset.dt, nfft=2 * (len(f) - 1))
    N = N * np.sqrt(dataset.n_t / sel.sum())                 # stationary noise scales as sqrt(duration)
    R = np.ones_like(f) if stf is None else np.abs(stf_spectrum(stf, f))
    ratio = D / (S + 1e-30)
    keep = f <= fmax_plot

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    for X, col, lab in ((D, 'k', 'data'), (S, 'r', 'synthetic'), (N, '0.5', 'noise')):
        lo, med, hi = np.percentile(X[:, keep], [10, 50, 90], axis=0)
        ax.fill_between(f[keep], lo, hi, color=col, alpha=0.2, lw=0)
        ax.plot(f[keep], med, color=col, label=lab)
    ax.set_yscale('log')
    ax.set_ylabel('Amplitude')
    ax.set_title('Spectra (median, 10-90 %)')

    ax = axes[1]
    lo, med, hi = np.percentile(ratio[:, keep], [10, 50, 90], axis=0)
    ax.fill_between(f[keep], lo, hi, color='k', alpha=0.2, lw=0)
    ax.plot(f[keep], med, 'k', label='data / synthetic')
    ref = (f >= 20) & (f <= 80)
    for fc_i in (*fc, None):
        curve = (1.0 if fc_i is None else spectra.brune(f, fc_i)) / np.maximum(R, 1e-6)
        curve = curve * np.exp(np.median(np.log(med[ref[keep]]) - np.log(curve[ref])))
        ax.plot(f[keep], curve[keep], '--', label='1 / STF' if fc_i is None else f'Brune {fc_i} Hz / STF')
    ax.set_yscale('log')
    ax.set_ylim(lo[ref[keep]].min() / 30, hi[ref[keep]].max() * 30)
    ax.set_title('Ratio (estimates S_true / STF)')

    ax = axes[2]
    try:
        order, y, ylab = np.argsort(dataset.azimuth), np.sort(dataset.azimuth), 'Azimuth (deg)'
    except ValueError:
        order, y, ylab = np.arange(dataset.n_rec), np.arange(dataset.n_rec), 'Receiver index'
    lr = np.log10(ratio[order][:, keep] + 1e-30)
    lr -= np.median(lr[:, ref[keep]], axis=1, keepdims=True)
    im = ax.pcolormesh(f[keep], y, lr, cmap='RdBu_r', vmin=-1.5, vmax=1.5, shading='nearest')
    fig.colorbar(im, ax=ax, label='log10 ratio (per-channel, rel. 20-80 Hz)')
    ax.set_ylabel(ylab)
    ax.set_title('Ratio per channel')

    for ax in axes:
        ax.set_xlabel('Frequency (Hz)')
        ax.set_xlim(0, fmax_plot)
        if mesh_fmax is not None:
            ax.axvline(mesh_fmax, color='m', ls=':', label='mesh limit')
        if band is not None:
            ax.axvspan(band[1], band[2], color='g', alpha=0.08, label='band B (flat)')
            for fb in (band[0], band[3]):
                ax.axvline(fb, color='g', ls=':', lw=0.8)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    return fig
