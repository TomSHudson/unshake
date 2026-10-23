"""Inversion engines.

least_squares : linear inversion of the Green's functions (no sampling).
monte_carlo   : independent random samples from the model prior, weighted by likelihood.

A sampler takes (model, dataset, ...) and returns a `Result`; an MCMC sampler
(e.g. emcee) can be added alongside `monte_carlo` using the same `loglike` builder.
"""

import time

import numpy as np

from . import misfit as mf
from .forward import synthetics
from .results import Result


def _design_matrix(dataset, gf_key):
    gf = dataset.gf[gf_key]                                   # (n_rec, n_comp, n_t)
    return gf.transpose(0, 2, 1).reshape(-1, gf.shape[1])     # (n_rec * n_t, n_comp), receiver-major


def least_squares(dataset, gf_key='mt', sigma=None, cov_m=None, m_prior=None):
    """Linear least-squares source inversion d = G m.

    sigma : optional scalar or (n_rec,) data noise std (diagonal data covariance).
        Falls back to `dataset.noise_std`. If neither is set, the data variance is
        estimated from the residual (as SeisSrcInv.inversion.perform_inversion).
    cov_m, m_prior : optional Gaussian prior on m (Tarantola / Fichtner eq. 6.8-6.9).

    Dataset weights (if set) scale each receiver's rows by sqrt(weight).
    Returns (m, cov_m_post), shapes (n_comp,) and (n_comp, n_comp).
    """
    G = _design_matrix(dataset, gf_key)
    d = dataset.data.ravel()
    n_t = dataset.n_t
    if dataset.weights is not None:
        sw = np.repeat(np.sqrt(dataset.weights), n_t)
        G, d = G * sw[:, None], d * sw
    sigma = dataset.noise_std if sigma is None else sigma

    if sigma is None and cov_m is None:
        m, res, _, _ = np.linalg.lstsq(G, d, rcond=None)
        var = (res[0] if len(res) else np.sum((d - G @ m) ** 2)) / (len(d) - len(m))
        return m, var * np.linalg.inv(G.T @ G)

    if sigma is None:
        raise ValueError("A prior needs a data noise level: pass sigma= or set dataset.noise_std")
    sig2 = np.broadcast_to(np.asarray(sigma, dtype=float) ** 2, (dataset.n_rec,))
    cd_inv = np.repeat(1.0 / sig2, n_t)                       # diagonal of C_d^-1
    A = G.T @ (G * cd_inv[:, None])
    b = G.T @ (d * cd_inv)
    if cov_m is not None:
        cm_inv = np.linalg.inv(cov_m)
        m_prior = np.zeros(G.shape[1]) if m_prior is None else np.ravel(m_prior)
        A = A + cm_inv
        b = b + cm_inv @ m_prior
    cov_post = np.linalg.inv(A)
    return cov_post @ b, cov_post


def _default_sigma(dataset, gf_key, normalise):
    """Noise level for the Gaussian likelihood when none is given: from
    dataset.noise_std, else the std of the least-squares residual."""
    if dataset.noise_std is not None:
        sigma = np.asarray(dataset.noise_std, dtype=float)
        if normalise:   # convert to per-trace normalised units
            sigma = sigma / np.max(np.abs(dataset.data), axis=1)
        return sigma
    m, _ = least_squares(dataset, gf_key=gf_key)
    d, s = dataset.data, synthetics(dataset.gf[gf_key], m)
    if normalise:
        d, s = mf.normalise_traces(d), mf.normalise_traces(s)
    sigma = float(np.std(d - s))
    print(f"monte_carlo: no noise level given, using least-squares residual std = {sigma:.3g}")
    return sigma


def make_loglike(dataset, misfit='gaussian', sigma=None, mode='stacked', normalise=False,
                 temperature=1.0, max_lag=0, gf_key='mt'):
    """Build loglike(synth) -> (n,) for synthetics of shape (n, n_rec, n_t).

    misfit : 'gaussian' -> -0.5 chi^2 with noise level sigma.
             'vr' | 'pcc' | 'ncc' -> -(1 - similarity) / (2 temperature)
             (temperature=1 reproduces SeisSrcInv's exp(-(1 - sim) / 2)).
    normalise : normalise each data and synthetic trace by its max (amplitude-free fit).
    """
    data = mf.normalise_traces(dataset.data) if normalise else dataset.data
    w = dataset.weights
    if misfit == 'gaussian':
        sigma = _default_sigma(dataset, gf_key, normalise) if sigma is None else sigma
    sim_fn = mf.SIMILARITY.get(misfit)

    def loglike(synth):
        if normalise:
            synth = mf.normalise_traces(synth)
        if misfit == 'gaussian':
            return mf.gaussian_loglike(data, synth, sigma, weights=w)
        if misfit == 'ncc':
            sim = sim_fn(data, synth, max_lag=max_lag, weights=w)
        else:
            sim = sim_fn(data, synth, mode=mode, weights=w)
        return -(1.0 - sim) / (2.0 * temperature)

    return loglike


def monte_carlo(model, dataset, n, misfit='gaussian', sigma=None, mode='stacked',
                normalise=False, temperature=1.0, max_lag=0, batch=1000, rng=None):
    """Monte Carlo inversion: draw n samples from the model prior and weight by likelihood.

    See `make_loglike` for misfit options. Vectorised in batches of `batch` samples
    (numpy/BLAS threading does the parallel work). Returns a `Result`.
    """
    rng = np.random.default_rng(rng)
    if hasattr(model, 'prepare'):
        model.prepare(dataset)
    gf_key = getattr(model, 'gf_key', 'mt')
    loglike = make_loglike(dataset, misfit, sigma, mode, normalise, temperature, max_lag, gf_key)
    data_vr = mf.normalise_traces(dataset.data) if normalise else dataset.data

    params, ll, vr = [], [], []
    tic = time.time()
    for i0 in range(0, int(n), batch):
        p = model.sample_prior(min(batch, int(n) - i0), rng)
        synth = model.forward(p, dataset)
        params.append(p)
        ll.append(loglike(synth))
        vr.append(mf.vr(data_vr, mf.normalise_traces(synth) if normalise else synth,
                        mode=mode, weights=dataset.weights))
    print(f"monte_carlo: {int(n)} samples in {time.time() - tic:.1f} s")

    return Result(params=np.concatenate(params), param_names=list(model.param_names),
                  loglike=np.concatenate(ll), vr=np.concatenate(vr),
                  meta={'model': type(model).__name__, 'kind': getattr(model, 'kind', None),
                        'scale': getattr(model, 'scale', None), 'misfit': misfit,
                        'mode': mode, 'normalise': normalise, 'temperature': temperature,
                        'uid': dataset.source.uid, 'n': int(n)})
