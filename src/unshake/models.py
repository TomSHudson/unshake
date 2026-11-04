"""Source models: map a parameter vector to synthetics, and own the prior.

To add new physics, write a new class with this interface (no base class needed):

    param_names : list of str
    prepare(dataset)            -> None   (optional; set defaults from the data)
    sample_prior(n, rng)        -> (n, n_params)
    forward(params, dataset)    -> (n, n_rec, n_t)

The solvers in `unshake.solvers` only use this interface. `solvers.nonlinear_least_squares`
also uses these optional methods (see `MTDCDirectivityModel`):

    bounds()                -> (lo, hi), each (n_params,), finite
    param_scales(p)         -> (n_params,) typical size of each parameter near p
    prior_residuals(params) -> (n_prior,) r with 0.5 |r|^2 = -log prior (+ const)
    find_non_linear_starting_model_params(dataset)
                            -> (n_params,) starting model
    starts(p0, n, rng)      -> (n, n_params) extra starting models (multi-start)
    check_params(params)    -> None   (warn about e.g. constraints not set by the bounds)
"""

import warnings
from dataclasses import replace

import numpy as np

from . import mt, spectra, stfs
from .forward import apply_source_spectrum, convolve_stf, synthetics
from .solvers import least_squares

MT_NAMES = ['Mnn', 'Mee', 'Mdd', 'Mne', 'Mnd', 'Med']
FORCE_NAMES = ['Fn', 'Fe', 'Fd']


class _Scaled:
    """Shared handling of source amplitude.

    scale : source norm (Frobenius norm for MTs, vector norm for forces). If None,
        it is set by `prepare` from the least-squares solution (the SeisSrcInv behaviour).
    scale_range : optional (lo, hi) factors. If given, the amplitude is sampled
        log-uniformly in [lo * scale, hi * scale] instead of being fixed.
    """
    gf_key = None

    def __init__(self, scale=None, scale_range=None):
        self.scale = scale
        self.scale_range = scale_range

    def prepare(self, dataset):
        if self.scale is None:
            m_lsq, _ = least_squares(dataset, gf_key=self.gf_key, check_band=False)
            self.scale = float(mt.norm(m_lsq) if self.gf_key == 'mt' else np.linalg.norm(m_lsq))

    def _amplitudes(self, n, rng):
        if self.scale is None:
            raise ValueError("Model scale not set: call model.prepare(dataset) or pass scale=")
        if self.scale_range is None:
            return np.full(n, self.scale)
        lo, hi = np.log10(self.scale * np.asarray(self.scale_range, dtype=float))
        return 10.0 ** rng.uniform(lo, hi, n)


class MTModel(_Scaled):
    """Moment tensor point source. kind: 'full', 'lune' or 'dc' (see `unshake.mt`)."""
    gf_key = 'mt'

    def __init__(self, kind='full', scale=None, scale_range=None):
        super().__init__(scale, scale_range)
        self.kind = kind
        self.param_names = list(MT_NAMES)

    def sample_prior(self, n, rng=None):
        rng = np.random.default_rng(rng)
        return mt.SAMPLERS[self.kind](n, rng) * self._amplitudes(n, rng)[:, None]

    def forward(self, params, dataset):
        return synthetics(dataset.gf['mt'], params[..., :6])


class ForceModel(_Scaled):
    """Single force point source."""
    gf_key = 'force'

    def __init__(self, scale=None, scale_range=None):
        super().__init__(scale, scale_range)
        self.param_names = list(FORCE_NAMES)

    def sample_prior(self, n, rng=None):
        rng = np.random.default_rng(rng)
        return mt.sample_force(n, rng) * self._amplitudes(n, rng)[:, None]

    def forward(self, params, dataset):
        return synthetics(dataset.gf['force'], params[..., :3])


class MTSpectrumModel(MTModel):
    """STUB: joint moment tensor + directional (Doppler) corner frequency model.

    Parameters: 6 MT components, fc0, vr, rupture azimuth and plunge (radians).
    Forward: MT synthetics filtered per receiver by a source spectrum with corner
    frequency `spectra.apparent_fc(fc0, vr, c, dataset.ray_dirs, rupture_dir)`,
    where c is `dataset.receivers.velocity`. Needs formal GFs (`Dataset.remove_stf`).

    The physics here is deliberately minimal (milestone 2 fleshes it out).
    """

    def __init__(self, kind='full', fc0_range=(10.0, 200.0), vr_range=(0.0, 1500.0),
                 mode='unilateral', n=2.0, gamma=1.0,
                 scale=None, scale_range=None):
        super().__init__(kind, scale, scale_range)
        self.fc0_range, self.vr_range = fc0_range, vr_range
        self.mode, self.n, self.gamma = mode, n, gamma
        self.param_names = list(MT_NAMES) + ['fc0', 'vr', 'rupt_az', 'rupt_plunge']

    def sample_prior(self, n, rng=None):
        rng = np.random.default_rng(rng)
        m = super().sample_prior(n, rng)
        fc0 = 10.0 ** rng.uniform(*np.log10(self.fc0_range), n)
        vr = rng.uniform(*self.vr_range, n)
        az = rng.uniform(0.0, 2.0 * np.pi, n)
        plunge = np.arcsin(rng.uniform(-1.0, 1.0, n))   # uniform directions on the sphere
        return np.column_stack([m, fc0, vr, az, plunge])

    def corner_frequencies(self, params, dataset):
        """Apparent corner frequency at each receiver, (n, n_rec)."""
        params = np.atleast_2d(params)
        if dataset.receivers.velocity is None:
            raise ValueError("MTSpectrumModel needs dataset.receivers.velocity (phase velocity)")
        rdir = spectra.rupture_direction(params[:, 8], params[:, 9])
        return spectra.apparent_fc(params[:, 6], params[:, 7], dataset.receivers.velocity,
                                   dataset.ray_dirs, rdir, mode=self.mode)

    def forward(self, params, dataset):
        params = np.atleast_2d(params)
        synth = super().forward(params, dataset)
        fc = self.corner_frequencies(params, dataset)
        return apply_source_spectrum(synth, dataset.dt, fc, n=self.n, gamma=self.gamma)


class MTTwoBruneModel(MTModel):
    """Moment tensor + the sum of two causal Brune STFs that share one corner frequency
    fc0, one of them directional.

    Parameters: 6 MT components, fc0, w2, vr, rupture azimuth and plunge (radians).
    Forward: MT synthetics convolved per receiver with
    `stfs.two_brune_spectrum(f, fc0, fc2, w2)`, i.e.
        S_i(f) = (1 - w2) S(f; fc0) + w2 S(f; fc2_i),
    where
        fc0 : corner frequency of the symmetric STF, the same at every receiver, and the
              corner frequency of the directional STF perpendicular to the rupture;
        w2  : moment fraction of the directional STF, the same at every receiver;
        fc2 : `spectra.apparent_fc(fc0, vr, c, dataset.ray_dirs, rupture_dir)` (Doppler),
              with c = `dataset.receivers.velocity`.
    w2 = 0 or vr = 0 both give a single, azimuth-independent Brune with corner fc0, so
    the rupture parameters are then undetermined. For small vr / c, only the product
    w2 * vr / c is constrained to first order. Needs formal GFs (`Dataset.remove_stf`).
    """

    def __init__(self, kind='full', fc0_range=(10.0, 200.0), w2_range=(0.0, 1.0),
                 vr_range=(0.0, 1500.0), mode='unilateral', scale=None, scale_range=None):
        super().__init__(kind, scale, scale_range)
        self.fc0_range, self.w2_range, self.vr_range, self.mode = fc0_range, w2_range, vr_range, mode
        self.param_names = list(MT_NAMES) + ['fc0', 'w2', 'vr', 'rupt_az', 'rupt_plunge']

    def sample_prior(self, n, rng=None):
        rng = np.random.default_rng(rng)
        m = super().sample_prior(n, rng)
        fc0 = 10.0 ** rng.uniform(*np.log10(self.fc0_range), n)
        w2 = rng.uniform(*self.w2_range, n)
        vr = rng.uniform(*self.vr_range, n)
        az = rng.uniform(0.0, 2.0 * np.pi, n)
        plunge = np.arcsin(rng.uniform(-1.0, 1.0, n))   # uniform directions on the sphere
        return np.column_stack([m, fc0, w2, vr, az, plunge])

    def corner_frequencies(self, params, dataset):
        """Apparent corner frequency of the directional STF at each receiver, (n, n_rec)."""
        params = np.atleast_2d(params)
        if dataset.receivers.velocity is None:
            raise ValueError("MTTwoBruneModel needs dataset.receivers.velocity (phase velocity)")
        rdir = spectra.rupture_direction(params[:, 9], params[:, 10])
        return spectra.apparent_fc(params[:, 6], params[:, 8], dataset.receivers.velocity,
                                   dataset.ray_dirs, rdir, mode=self.mode)

    def forward(self, params, dataset):
        params = np.atleast_2d(params)
        synth = super().forward(params, dataset)
        fc0, w2 = params[:, 6, None, None], params[:, 7, None, None]
        fc2 = self.corner_frequencies(params, dataset)[..., None]
        return convolve_stf(synth, dataset.dt, lambda f: stfs.two_brune_spectrum(f, fc0, fc2, w2))


def _single_brune_lsq(dataset, fc0):
    """Linear MT least squares with one causal Brune STF at fc0 for the whole MT.

    Returns (m (6,), rss), with rss the (weighted) residual sum of squares.
    """
    gf = convolve_stf(dataset.gf['mt'], dataset.dt, lambda f: stfs.brune_stf_spectrum(f, fc0))
    m, _ = least_squares(replace(dataset, gf={'mt': gf}), gf_key='mt', check_band=False)
    r2 = np.sum((dataset.data - synthetics(gf, m)) ** 2, axis=-1)
    return m, float(r2.sum() if dataset.weights is None else r2 @ dataset.weights)


class MTDCDirectivityModel(MTModel):
    """Moment tensor whose double-couple part ruptures with directivity.

    Parameters: 6 MT components, fc0, and the rupture velocity vector (v_e, v_n, v_u)
    (ENU, m/s). The MT is split with `mt.decompose` (Vavrycuk 2015) into m_dc and
    m_ndc = m - m_dc (ISO + CLVD). Forward, per receiver i:
        u_i = S(f; fc2_i) * [G_i . m_dc]  +  S(f; fc0) * [G_i . m_ndc],
    with S the causal Brune spectrum (`stfs.brune_stf_spectrum`) and
        fc2_i = fc0 / (1 - v . r_i / c_i)   (`spectra.apparent_fc_vector`),
    where c_i = `dataset.receivers.velocity`. A pure DC is fully directional; ISO,
    CLVD and tensile sources (no DC part) have no directivity. v = 0 gives a single
    Brune with corner fc0. Needs formal GFs (`Dataset.remove_stf`).

    Prior on v: isotropic zero-mean Gaussian (sigma_v), truncated at |v| <= v_max.

    For `solvers.nonlinear_least_squares`: `find_non_linear_starting_model_params` (variable projection over fc0),
    `bounds` (finite: |m_k| <= a_max * scale, fc0 in fc0_range, |v_j| <= v_max, with
    a_max = scale_range[1] or 3), `param_scales` and `prior_residuals` (v / sigma_v).
    """

    def __init__(self, kind='full', fc0_range=(10.0, 200.0), sigma_v=900.0, v_max=1600.0,
                 scale=None, scale_range=None):
        super().__init__(kind, scale, scale_range)
        self.fc0_range, self.sigma_v, self.v_max = fc0_range, sigma_v, v_max
        self.param_names = list(MT_NAMES) + ['fc0', 'v_e', 'v_n', 'v_u']
        self.p0 = None

    def prepare(self, dataset):
        """Set the starting model `p0` (`find_non_linear_starting_model_params`) and, if not given, `scale` = ||m0||.

        (The base class would take the scale from LSQ on the formal GFs, which have no STF,
        so it is biased for data low-passed by the Brune STF.)
        """
        self.p0 = self.find_non_linear_starting_model_params(dataset)
        if self.scale is None:
            self.scale = float(mt.norm(self.p0[:6]))

    def find_non_linear_starting_model_params(self, dataset, fc0=None, fc0_grid=15):
        """Starting model by variable projection: one Brune STF at fc0 for the whole MT,
        the MT solved linearly (`least_squares`) at each fc0, v = 0.

        fc0 : use this corner frequency (e.g. measured from the data) and skip the search.
        fc0_grid : number of log-spaced fc0 values over fc0_range for the coarse scan; the
            best one is refined by a bounded 1-D minimisation in log fc0 between its neighbours.
        Returns p0 = [m0, fc0, 0, 0, 0].
        """
        if fc0 is None:
            from scipy.optimize import minimize_scalar
            x = np.linspace(*np.log(self.fc0_range), fc0_grid)
            rss = [_single_brune_lsq(dataset, np.exp(xi))[1] for xi in x]
            i = int(np.argmin(rss))
            lo, hi = x[max(i - 1, 0)], x[min(i + 1, len(x) - 1)]
            opt = minimize_scalar(lambda xi: _single_brune_lsq(dataset, np.exp(xi))[1],
                                  bounds=(lo, hi), method='bounded', options={'xatol': 1e-3})
            fc0 = float(np.exp(opt.x))
        m0, _ = _single_brune_lsq(dataset, fc0)
        return np.r_[m0, fc0, 0.0, 0.0, 0.0]

    def bounds(self):
        """Finite (lo, hi) bounds on the parameters."""
        if self.scale is None:
            raise ValueError("Model scale not set: call model.prepare(dataset) or pass scale=")
        a = self.scale * (self.scale_range[1] if self.scale_range is not None else 3.0)
        hi = np.r_[np.full(6, a), self.fc0_range[1], np.full(3, self.v_max)]
        lo = np.r_[np.full(6, -a), self.fc0_range[0], np.full(3, -self.v_max)]
        return lo, hi

    def param_scales(self, p):
        """Typical size of each parameter near p (for conditioning the optimiser)."""
        return np.r_[np.full(6, self.scale), p[6], np.full(3, self.sigma_v)]

    def prior_residuals(self, params):
        """v / sigma_v: the Gaussian prior on v as least-squares residuals (damping)."""
        return np.asarray(params)[7:10] / self.sigma_v

    def starts(self, p0, n, rng=None):
        """n extra starting models for multi-start: p0 with v of size sigma_v / 2 in
        uniformly random directions (MT and fc0 unchanged). Returns (n, n_params)."""
        out = np.tile(np.asarray(p0, dtype=float), (n, 1))
        out[:, 7:10] = 0.5 * self.sigma_v * mt.sample_force(n, rng)   # unit vectors on the sphere
        return out

    def check_params(self, params):
        """Warn if the rupture speed |v| exceeds v_max (the bounds are per component)."""
        vr = np.linalg.norm(np.asarray(params)[7:10])
        if vr > self.v_max:
            warnings.warn(f"|v| = {vr:.0f} m/s exceeds v_max = {self.v_max:.0f} m/s")

    def sample_prior(self, n, rng=None):
        rng = np.random.default_rng(rng)
        m = super().sample_prior(n, rng)
        fc0 = 10.0 ** rng.uniform(*np.log10(self.fc0_range), n)
        v = rng.normal(0.0, self.sigma_v, (n, 3))
        bad = np.linalg.norm(v, axis=1) > self.v_max
        while bad.any():                                  # truncate by resampling
            v[bad] = rng.normal(0.0, self.sigma_v, (bad.sum(), 3))
            bad = np.linalg.norm(v, axis=1) > self.v_max
        return np.column_stack([m, fc0, v])

    def corner_frequencies(self, params, dataset):
        """Apparent corner frequency of the directional (DC) STF at each receiver, (n, n_rec)."""
        params = np.atleast_2d(params)
        if dataset.receivers.velocity is None:
            raise ValueError("MTDCDirectivityModel needs dataset.receivers.velocity (phase velocity)")
        return spectra.apparent_fc_vector(params[:, 6], params[:, 7:10], dataset.receivers.velocity,
                                          dataset.ray_dirs)

    @staticmethod
    def rupture(params):
        """Rupture speed (m/s), azimuth and plunge (degrees) from the velocity-vector parameters."""
        vr, az, plunge = spectra.rupture_from_vector(np.atleast_2d(params)[:, 7:10])
        return vr, np.degrees(az), np.degrees(plunge)

    def forward(self, params, dataset):
        params = np.atleast_2d(params)
        _, _, _, m_dc, m_ndc = mt.decompose(params[:, :6])
        gf = dataset.gf['mt']
        fc0 = params[:, 6, None, None]
        fc2 = self.corner_frequencies(params, dataset)[..., None]
        return (convolve_stf(synthetics(gf, m_dc), dataset.dt, lambda f: stfs.brune_stf_spectrum(f, fc2))
                + convolve_stf(synthetics(gf, m_ndc), dataset.dt, lambda f: stfs.brune_stf_spectrum(f, fc0)))
