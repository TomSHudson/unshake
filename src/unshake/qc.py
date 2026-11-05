"""Post-inversion quality control.

Cross-correlation lags between the data and the best-fitting synthetics, per receiver.
A good source model fitted with good Green's functions leaves lags near zero at every
receiver. Systematic patterns point to problems the source model cannot absorb:

    constant offset              -> origin-time error (or a GF / data timing offset);
    ~cos(azimuth - phi) pattern  -> horizontal source mislocation (towards phi);
    trend with take-off angle    -> depth error or velocity-model error;
    scattered large lags         -> poor fits, noisy channels or cycle skipping.

Usage, after any inversion:

    from unshake import qc
    lags, cc = qc.lag_qc(ds, res, model=model)            # Result from monte_carlo / mcmc
    lags, cc = qc.lag_qc(ds, p, model=model)              # params from nonlinear_least_squares
    lags, cc = qc.lag_qc(ds, m)                           # m from least_squares (linear)
    qc.lag_qc(ds, synth, return_lags=False)               # synthetics (n_rec, n_t), plot only

Sign convention: a positive lag means the data arrive *later* than the synthetic,
i.e. d(t) ~ s(t - lag).

Lags are per trace (channel), not per station: the components of a 3-component station
are kept separate because P and S dominate on different components, so their lags
measure different phases.
"""

import warnings

import numpy as np

from .forward import synthetics
from .results import Result


def best_synthetics(dataset, best, model=None, gf_key='mt'):
    """Synthetics (n_rec, n_t) for the outcome of an inversion.

    best : one of
        `Result`                      -> model.forward(result.best) (needs `model`);
        (n_params,) parameter vector  -> model.forward(best) if `model` is given, else
                                         linear synthetics of dataset.gf[gf_key] (e.g. the
                                         m from `least_squares`);
        (n_rec, n_t) array            -> used as the synthetics directly.
    """
    if isinstance(best, Result):
        if model is None:
            raise ValueError("A Result needs the model that produced it: pass model=")
        best = best.best
    best = np.asarray(best, dtype=float)
    if best.shape == dataset.data.shape:
        return best
    if best.ndim != 1:
        raise ValueError(f"best has shape {best.shape}: expected (n_params,) or {dataset.data.shape}")
    if model is not None:
        return model.forward(np.atleast_2d(best), dataset)[0]
    n_comp = dataset.gf[gf_key].shape[1]
    if len(best) != n_comp:
        raise ValueError(f"{len(best)} parameters but gf['{gf_key}'] has {n_comp} components: "
                         "pass the model= that the parameters belong to")
    return synthetics(dataset.gf[gf_key], best)


def xcorr_lags(data, synth, dt, max_lag=None, interpolate=True):
    """Lag of the maximum normalised cross-correlation between each data and synthetic trace.

    data, synth : (n_rec, n_t). Each trace is demeaned. The cross-correlation is linear
        (zero-padded), not circular, so the lags do not wrap around.
    dt : sample interval (s).
    max_lag : only search lags with |lag| <= max_lag (s); None searches all lags.
    interpolate : refine the peak to sub-sample precision with a parabola through the
        peak and its two neighbours.

    Returns (lags, cc), each (n_rec,): the lag in seconds (positive: data later than
    synthetic, d(t) ~ s(t - lag)) and the normalised cross-correlation at that lag.
    Both are NaN for traces that are all zero.
    """
    d = np.atleast_2d(np.asarray(data, dtype=float))
    s = np.atleast_2d(np.asarray(synth, dtype=float))
    if d.shape != s.shape:
        raise ValueError(f"data {d.shape} and synth {s.shape} have different shapes")
    d = d - d.mean(axis=-1, keepdims=True)
    s = s - s.mean(axis=-1, keepdims=True)
    n_t = d.shape[-1]
    nfft = int(2 ** np.ceil(np.log2(2 * n_t)))
    c = np.fft.irfft(np.fft.rfft(d, nfft) * np.conj(np.fft.rfft(s, nfft)), nfft)   # c[k] = sum d[t] s[t-k]
    c = np.concatenate([c[:, nfft - (n_t - 1):], c[:, :n_t]], axis=-1)            # lags -(n_t-1) .. n_t-1
    k = np.arange(-(n_t - 1), n_t)
    with np.errstate(invalid='ignore', divide='ignore'):
        c = c / (np.linalg.norm(d, axis=-1) * np.linalg.norm(s, axis=-1))[:, None]

    if max_lag is not None:
        c[:, np.abs(k) > max_lag / dt] = -np.inf
    rows = np.arange(len(c))
    i = np.argmax(np.where(np.isnan(c), -np.inf, c), axis=-1)
    lag, cc = k[i].astype(float), c[rows, i]

    if interpolate:
        inner = (i > 0) & (i < c.shape[-1] - 1)
        y0 = cc[inner]
        ym, yp = c[rows[inner], i[inner] - 1], c[rows[inner], i[inner] + 1]
        curv = ym - 2.0 * y0 + yp
        ok = np.isfinite(ym) & np.isfinite(yp) & (curv < 0)
        delta, peak = np.zeros_like(y0), y0.copy()
        delta[ok] = np.clip(0.5 * (ym[ok] - yp[ok]) / curv[ok], -0.5, 0.5)
        peak[ok] = y0[ok] - 0.25 * (ym[ok] - yp[ok]) * delta[ok]
        lag[inner] += delta
        cc[inner] = peak

    bad = ~np.isfinite(cc)
    lag[bad], cc[bad] = np.nan, np.nan
    return lag * dt, cc


_UNITS = {'s': 1.0, 'ms': 1e3}


def _x_values(dataset, x):
    """(values, label) for the x axis of `plot_lags`."""
    if not isinstance(x, str):
        return np.asarray(x, dtype=float), None
    labels = {'azimuth': 'Source-receiver azimuth (deg)', 'takeoff': 'Take-off angle (deg from down)',
              'distance': 'Source-receiver distance', 'index': 'Receiver index'}
    if x not in labels:
        raise ValueError(f"x must be one of {list(labels)} or an array, not '{x}'")
    if x == 'index':
        return np.arange(dataset.n_rec, dtype=float), labels[x]
    try:
        return np.asarray(getattr(dataset, x), dtype=float), labels[x]
    except ValueError:
        warnings.warn(f"No source/receiver coordinates for '{x}': plotting against receiver index")
        return np.arange(dataset.n_rec, dtype=float), labels['index']


def plot_lags(dataset, lags, cc=None, x='azimuth', unit='ms', ax=None, xlabel=None, cmap='Blues'):
    """Cross-correlation lag at each receiver against take-off direction.

    lags, cc : (n_rec,) from `xcorr_lags` (lags in seconds).
    x : 'azimuth' (source-receiver, degrees from north; default), 'takeoff', 'distance',
        'index', or an (n_rec,) array of your own (then set `xlabel`).
    unit : 'ms' or 's' for the lag axis.
    cc : if given, points are coloured by the peak correlation (darker = better fit), so
        lags from poorly fitting traces can be discounted.
    Returns the axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    xv, default_label = _x_values(dataset, x)
    y = np.asarray(lags, dtype=float) * _UNITS[unit]

    ax.axhline(0.0, color='0.6', lw=0.8, zorder=0)
    if cc is None:
        ax.plot(xv, y, 'o', ms=4, color='C0')
    else:
        # Truncate the colour map so that low-cc points stay visible on a white background.
        cm = plt.get_cmap(cmap)
        from matplotlib.colors import ListedColormap
        cm = ListedColormap(cm(np.linspace(0.3, 1.0, 256)))
        sc = ax.scatter(xv, y, c=cc, cmap=cm, vmin=max(0.0, np.nanmin(cc)), vmax=1.0, s=18,
                        edgecolors='none')
        ax.figure.colorbar(sc, ax=ax, label='Peak normalised CC')
    if isinstance(x, str) and x == 'azimuth':
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 90))
    med = np.nanmedian(y)
    mad = 1.4826 * np.nanmedian(np.abs(y - med))
    ax.set_title(f"Data-synthetic lag: median {med:.3g} {unit}, robust std {mad:.3g} {unit}", fontsize=9)
    ax.set_xlabel(xlabel or default_label or 'x')
    ax.set_ylabel(f'Lag ({unit}), + = data later')
    return ax


def lag_qc(dataset, best, model=None, gf_key='mt', max_lag=None, x='azimuth', unit='ms',
           plot=True, ax=None, return_lags=True, interpolate=True):
    """Post-inversion QC: lag of maximum cross-correlation between the data and the best
    synthetics at every receiver, plotted against take-off direction.

    dataset : the `Dataset` that was inverted.
    best, model, gf_key : the inversion outcome, see `best_synthetics` (a `Result`, a
        parameter vector, a linear source vector, or (n_rec, n_t) synthetics).
    max_lag : maximum |lag| searched (s); None searches all lags. Set it to under half the
        dominant period to avoid cycle skipping.
    x, unit, ax : passed to `plot_lags`.
    plot : make the plot.
    return_lags : return (lags, cc), each (n_rec,): lags in seconds (positive: data later
        than synthetic) and the peak normalised cross-correlation. Otherwise return None.
    """
    synth = best_synthetics(dataset, best, model=model, gf_key=gf_key)
    lags, cc = xcorr_lags(dataset.data, synth, dataset.dt, max_lag=max_lag, interpolate=interpolate)
    if plot:
        plot_lags(dataset, lags, cc, x=x, unit=unit, ax=ax)
    return (lags, cc) if return_lags else None
