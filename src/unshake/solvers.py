"""Inversion engines.

least_squares           : linear inversion of the Green's functions (no sampling).
nonlinear_least_squares : non-linear (Gauss-Newton type) inversion of any model, with
                          multi-start and a Gauss-Newton covariance.
monte_carlo             : independent random samples from the model prior, weighted by likelihood.
mcmc                    : adaptive Metropolis MCMC (several chains), e.g. started from the
                          `nonlinear_least_squares` optimum and covariance.

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


def least_squares(dataset, gf_key='mt', sigma=None, cov_m=None, m_prior=None, check_band=True):
    """Linear least-squares source inversion d = G m.

    sigma : optional scalar or (n_rec,) data noise std (diagonal data covariance).
        Falls back to `dataset.noise_std`. If neither is set, the data variance is
        estimated from the residual (as SeisSrcInv.inversion.perform_inversion).
    cov_m, m_prior : optional Gaussian prior on m (Tarantola / Fichtner eq. 6.8-6.9).

    Dataset weights (if set) scale each receiver's rows by sqrt(weight).
    check_band : print the GF/data band-filter note (`Dataset.check_band`).
    Returns (m, cov_m_post), shapes (n_comp,) and (n_comp, n_comp).
    """
    if check_band:
        dataset.check_band()
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


def lsq_loglike(m_dirs, m_lsq, cov):
    """Log-density of the Gaussian LSQ posterior N(m_lsq, cov) for MT directions,
    maximised (profiled) over a non-negative amplitude a in m = a * m_dir.

    Lets the LSQ uncertainty be evaluated anywhere, e.g. over the whole lune with
    m_dirs from `mt.sample_lune`. Returns (n,), relative to the peak (0 at m_lsq).
    """
    P = np.linalg.inv(cov)
    u = np.atleast_2d(m_dirs)
    uPm = u @ (P @ m_lsq)
    uPu = np.einsum('ni,ij,nj->n', u, P, u)
    return -0.5 * np.where(uPm > 0, -uPm ** 2 / uPu, 0.0) - 0.5 * (m_lsq @ P @ m_lsq)


def lsq_lune_profile(m_lsq, cov, bins=(30, 60), n_random_starts=2, rng=None):
    """Profile of the LSQ posterior over the whole lune.

    At every lune bin centre, the source type (gamma, delta) is fixed and the
    orientation (and amplitude, analytically) is optimised to maximise
    `lsq_loglike`. Starts from the LSQ orientation plus `n_random_starts` random ones.

    Returns (m_dirs (n_bins, 6), loglike (n_bins,)); plot with
    ``plot.lune(m6=m_dirs, values=loglike, bins=bins)``.
    """
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    from . import mt
    rng = np.random.default_rng(rng)
    ng, nd = bins
    ge = np.linspace(-np.pi / 6, np.pi / 6, ng + 1)
    de = np.linspace(-np.pi / 2, np.pi / 2, nd + 1)
    G, D = np.meshgrid(0.5 * (ge[1:] + ge[:-1]), 0.5 * (de[1:] + de[:-1]), indexing='ij')
    lams = mt.lune_to_eigenvalues(G.ravel(), D.ravel())
    _, V = mt.eig(m_lsq)                                     # LSQ principal axes (descending)
    starts = [np.zeros(3)] + [Rotation.random(random_state=rng).as_rotvec()
                              for _ in range(n_random_starts)]

    def m_dir(rotvec, lam):
        R = V @ Rotation.from_rotvec(rotvec).as_matrix()
        return mt.from_33(R @ np.diag(lam) @ R.T)

    m_out, ll_out = np.zeros((len(lams), 6)), np.zeros(len(lams))
    for i, lam in enumerate(lams):
        fits = [minimize(lambda r: -lsq_loglike(m_dir(r, lam), m_lsq, cov)[0], x0, method='BFGS')
                for x0 in starts]
        best = min(fits, key=lambda f: f.fun)
        m_out[i], ll_out[i] = m_dir(best.x, lam), -best.fun
    return m_out, ll_out


def _default_sigma(dataset, gf_key, normalise):
    """Noise level for the Gaussian likelihood when none is given: from
    dataset.noise_std, else the std of the least-squares residual."""
    if dataset.noise_std is not None:
        sigma = np.asarray(dataset.noise_std, dtype=float)
        if normalise:   # convert to per-trace normalised units
            sigma = sigma / np.max(np.abs(dataset.data), axis=1)
        return sigma
    m, _ = least_squares(dataset, gf_key=gf_key, check_band=False)
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
    dataset.check_band()
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
    p0 = getattr(model, 'p0', None)
    if misfit == 'gaussian' and sigma is None and dataset.noise_std is None and p0 is not None:
        # The model has a starting fit (e.g. MTDCDirectivityModel): use its residual, since the
        # default (LSQ on the GFs alone) ignores the source time function. Inflated for
        # correlated noise, as in `nonlinear_least_squares`.
        s = model.forward(p0, dataset)[0]
        d = dataset.data
        if normalise:
            d, s = mf.normalise_traces(d), mf.normalise_traces(s)
        factor = mf.noise_correlation_factor(d - s)
        sigma = float(np.std(d - s) * np.sqrt(factor))
        print(f"monte_carlo: no noise level given, using the starting-model residual std x "
              f"sqrt(noise correlation factor {factor:.2f}) = {sigma:.3g}")
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


def nonlinear_least_squares(model, dataset, p0=None, sigma=None, n_starts=4, correlated_noise=True,
                            rng=None, **lsq_kw):
    """Non-linear least-squares inversion with `scipy.optimize.least_squares` (trust-region
    reflective, bounded, finite-difference Jacobian).

    Minimises 0.5 |r|^2 with
        r = [ sqrt(w_i) (d_i - u_i(p)) / sigma_i  (all traces, flattened),
              model.prior_residuals(p)            (if the model has it, e.g. damping) ].

    model : needs `forward`, `bounds()` and `param_scales(p)`; optionally `prepare`,
        `prior_residuals`, `starts` and `check_params` (see `models.MTDCDirectivityModel`).
    p0 : starting model; default `model.p0` from `model.prepare(dataset)`.
    sigma : scalar or (n_rec,) data noise std. Falls back to `dataset.noise_std`. If neither
        is set, it starts from the std of the starting residual, is re-estimated as
        sigma^2 = chi^2 / (N - n_params) at the optimum, and the fit is repeated once.
    n_starts : extra starts from `model.starts(p0, n_starts, rng)`; the lowest cost wins.
    correlated_noise : if True, sigma is inflated by sqrt(`misfit.noise_correlation_factor`)
        of the residual at the optimum, and the fit is repeated once. The time samples of
        band-limited noise are not independent, so without this the covariance is too small
        (by about 1.6x for the band in examples/directional_inversion_demo.ipynb).
    lsq_kw : passed to `scipy.optimize.least_squares`.

    Returns (p, cov, info): the optimum (n_params,), the Gauss-Newton covariance
    (J^T J)^-1 (n_params, n_params) and a dict with
        sigma        : the effective noise std used (inflated if correlated_noise); pass it
                       to `mcmc` or `make_loglike` so that they use the same likelihood,
        noise_factor : the variance inflation factor (1 if correlated_noise is False),
        cost, vr, nfev, start_costs, result (the scipy result).
    """
    from scipy.optimize import least_squares as scipy_lsq

    dataset.check_band()
    if p0 is None or getattr(model, 'scale', 0.0) is None:
        model.prepare(dataset)
    p0 = np.asarray(model.p0 if p0 is None else p0, dtype=float)
    lo, hi = model.bounds()
    x_scale = model.param_scales(p0)
    prior = getattr(model, 'prior_residuals', None)
    w = np.ones(dataset.n_rec) if dataset.weights is None else dataset.weights
    data = dataset.data

    sigma = dataset.noise_std if sigma is None else sigma
    estimate_sigma = sigma is None
    if estimate_sigma:
        sigma = float(np.std(data - model.forward(p0, dataset)[0]))

    def make_resid(sigma):
        rw = np.sqrt(w) / np.broadcast_to(np.asarray(sigma, dtype=float), (dataset.n_rec,))

        def resid(p):
            r = ((data - model.forward(p, dataset)[0]) * rw[:, None]).ravel()
            return r if prior is None else np.r_[r, prior(p)]
        return resid

    def fit(x0, sigma):
        return scipy_lsq(make_resid(sigma), np.clip(x0, lo, hi), bounds=(lo, hi),
                         x_scale=x_scale, method='trf', **lsq_kw)

    starts = [p0]
    if n_starts and hasattr(model, 'starts'):
        starts += list(model.starts(p0, n_starts, rng))
    tic = time.time()
    fits = [fit(x0, sigma) for x0 in starts]
    best = min(fits, key=lambda f: f.cost)
    nfev = sum(f.nfev for f in fits)
    if estimate_sigma:                     # sigma^2 = chi^2 / (N - n_params) at the optimum
        chi2 = 2.0 * best.cost - (0.0 if prior is None else np.sum(prior(best.x) ** 2))
        sigma = sigma * np.sqrt(chi2 / (data.size - len(p0)))
    noise_factor = 1.0
    if correlated_noise:                   # fewer independent samples than n_t
        noise_factor = mf.noise_correlation_factor(data - model.forward(best.x, dataset)[0])
        sigma = sigma * np.sqrt(noise_factor)
    if estimate_sigma or correlated_noise:  # refit with the final sigma (it sets the prior's weight)
        best = fit(best.x, sigma)
        nfev += best.nfev

    p = best.x
    J = best.jac * x_scale                                      # conditioned (scaled) Jacobian
    cov = x_scale[:, None] * np.linalg.pinv(J.T @ J, hermitian=True) * x_scale[None, :]
    synth = model.forward(p, dataset)[0]
    vr = float(mf.vr(data, synth, weights=dataset.weights))
    print(f"nonlinear_least_squares: {len(starts)} start(s), {nfev} forward evaluations in "
          f"{time.time() - tic:.1f} s; cost {best.cost:.4g}, VR {vr:.4f}, "
          f"sigma {np.mean(sigma):.3g}{' (estimated)' if estimate_sigma else ''}, "
          f"noise correlation factor {noise_factor:.2f}")
    if hasattr(model, 'check_params'):
        model.check_params(p)
    return p, cov, {'sigma': sigma, 'noise_factor': noise_factor, 'cost': best.cost, 'vr': vr, 'nfev': nfev,
                    'start_costs': [f.cost for f in fits], 'result': best}


def mcmc(model, dataset, p0, cov, n_steps=4000, n_chains=8, burn=None, sigma=None, adapt=True,
         rng=None, **loglike_kw):
    """Adaptive Metropolis MCMC (Haario et al. 2001), with several chains run in step.

    Target: log posterior = loglike(model.forward(p)) + model.log_prior(p).

    The chains start at draws from N(p0, cov), e.g. the `nonlinear_least_squares` optimum and
    covariance. At every step, each chain proposes p' = p + N(0, 2.38^2 / n_params * C) and
    accepts it with probability min(1, posterior ratio); all chains together cost one batched
    `forward` call. C starts as cov. During burn-in, it is updated every 100 steps from the
    samples so far (all chains, latest half). After burn-in it is fixed, so the kept samples
    come from a standard Metropolis sampler.

    sigma : noise std for the Gaussian likelihood. Pass info['sigma'] from
        `nonlinear_least_squares` (already inflated for correlated noise).
    burn : number of steps discarded (default n_steps // 2).
    adapt : update C during burn-in.
    loglike_kw : passed to `make_loglike` (misfit, mode, ...).

    Returns a `Result` of the n_chains * (n_steps - burn) kept samples, with equal weights.
    meta holds: acceptance (fraction of accepted proposals after burn-in), rhat (Gelman-Rubin,
    per parameter; < 1.05 is good), tau (autocorrelation time in steps, per parameter), ess
    (the smallest effective sample size over the parameters) and chain (all steps, including
    burn-in, (n_steps, n_chains, n_params), for trace plots).
    """
    rng = np.random.default_rng(rng)
    p0, cov = np.asarray(p0, dtype=float), np.asarray(cov, dtype=float)
    n_par = len(p0)
    burn = n_steps // 2 if burn is None else burn
    loglike = make_loglike(dataset, sigma=sigma, gf_key=getattr(model, 'gf_key', 'mt'), **loglike_kw)

    def log_post(p):
        """Log posterior, log-likelihood and VR of a batch of parameter vectors, (n,) each."""
        lp = model.log_prior(p)
        ll, vr = np.full(len(p), -np.inf), np.full(len(p), np.nan)
        ok = np.isfinite(lp)
        if ok.any():
            synth = model.forward(p[ok], dataset)
            ll[ok] = loglike(synth)
            vr[ok] = mf.vr(dataset.data, synth, weights=dataset.weights)
        return lp + ll, ll, vr

    # Starting points: draws around p0, replaced by p0 where they fall outside the prior.
    x = rng.multivariate_normal(p0, cov, n_chains)
    x[~np.isfinite(model.log_prior(x))] = p0
    lp_x, ll_x, vr_x = log_post(x)

    step = _sqrtm(2.38 ** 2 / n_par * cov)        # proposal: x + N(0, step @ step.T)
    chain = np.zeros((n_steps, n_chains, n_par))
    ll_chain, vr_chain = np.zeros((n_steps, n_chains)), np.zeros((n_steps, n_chains))
    n_accepted = 0
    tic = time.time()
    for i in range(n_steps):
        y = x + rng.standard_normal((n_chains, n_par)) @ step.T
        lp_y, ll_y, vr_y = log_post(y)
        accept = np.log(rng.uniform(size=n_chains)) < lp_y - lp_x
        x[accept], lp_x[accept], ll_x[accept], vr_x[accept] = (
            y[accept], lp_y[accept], ll_y[accept], vr_y[accept])
        chain[i], ll_chain[i], vr_chain[i] = x, ll_x, vr_x
        if i >= burn:
            n_accepted += accept.sum()
        elif adapt and i >= 199 and (i + 1) % 100 == 0:
            recent = chain[(i + 1) // 2:i + 1].reshape(-1, n_par)
            step = _sqrtm(2.38 ** 2 / n_par * np.cov(recent.T))

    kept = chain[burn:]
    acceptance = n_accepted / kept.shape[0] / n_chains
    rhat = _rhat(kept)
    tau = _autocorr_time(kept)
    ess = kept.shape[0] * n_chains / tau
    print(f"mcmc: {n_chains} chains x {n_steps} steps (burn-in {burn}) in {time.time() - tic:.1f} s; "
          f"acceptance {acceptance:.2f}, max R-hat {np.nanmax(rhat):.3f}, min ESS {np.nanmin(ess):.0f}")
    return Result(params=kept.reshape(-1, n_par), param_names=list(model.param_names),
                  loglike=ll_chain[burn:].ravel(), vr=vr_chain[burn:].ravel(), equal_weights=True,
                  meta={'model': type(model).__name__, 'sampler': 'mcmc', 'sigma': sigma,
                        'n_steps': n_steps, 'n_chains': n_chains, 'burn': burn,
                        'acceptance': acceptance, 'rhat': rhat, 'tau': tau,
                        'ess': np.nanmin(ess), 'chain': chain, 'uid': dataset.source.uid})


def _sqrtm(c):
    """Matrix square root L (L @ L.T = c) of a symmetric positive semi-definite matrix."""
    w, V = np.linalg.eigh(c)
    return V * np.sqrt(np.clip(w, 0.0, None))


def _rhat(chains):
    """Gelman-Rubin R-hat per parameter, for chains (n_steps, n_chains, n_params)."""
    n = chains.shape[0]
    within = chains.var(axis=0, ddof=1).mean(axis=0)            # mean within-chain variance
    between = chains.mean(axis=0).var(axis=0, ddof=1)          # variance of the chain means
    return np.sqrt(((n - 1) / n * within + between) / within)


def _autocorr_time(chains):
    """Integrated autocorrelation time per parameter, for chains (n_steps, n_chains, n_params):
    1 + 2 * sum of the chain-averaged autocorrelation, up to its first negative value."""
    n = chains.shape[0]
    x = chains - chains.mean(axis=0)
    f = np.fft.rfft(x, n=2 * n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), axis=0)[:n].mean(axis=1)   # (n, n_params)
    acf = acf / acf[0]
    tau = np.ones(acf.shape[1])
    for j in range(acf.shape[1]):
        negative = np.flatnonzero(acf[1:, j] < 0)
        k = negative[0] + 1 if len(negative) else n
        tau[j] = 1.0 + 2.0 * acf[1:k, j].sum()
    return tau
