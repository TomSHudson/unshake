"""Source models: map a parameter vector to synthetics, and own the prior.

To add new physics, write a new class with this interface (no base class needed):

    param_names : list of str
    prepare(dataset)            -> None   (optional; set defaults from the data)
    sample_prior(n, rng)        -> (n, n_params)
    forward(params, dataset)    -> (n, n_rec, n_t)

The solvers in `unshake.solvers` only use this interface.
"""

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
    """

    def __init__(self, kind='full', fc0_range=(10.0, 200.0), sigma_v=900.0, v_max=1600.0,
                 scale=None, scale_range=None):
        super().__init__(kind, scale, scale_range)
        self.fc0_range, self.sigma_v, self.v_max = fc0_range, sigma_v, v_max
        self.param_names = list(MT_NAMES) + ['fc0', 'v_e', 'v_n', 'v_u']

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
