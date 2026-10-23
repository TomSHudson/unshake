"""Source models: map a parameter vector to synthetics, and own the prior.

To add new physics, write a new class with this interface (no base class needed):

    param_names : list of str
    prepare(dataset)            -> None   (optional; set defaults from the data)
    sample_prior(n, rng)        -> (n, n_params)
    forward(params, dataset)    -> (n, n_rec, n_t)

The solvers in `unshake.solvers` only use this interface.
"""

import numpy as np

from . import mt, spectra
from .forward import apply_source_spectrum, synthetics
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
            m_lsq, _ = least_squares(dataset, gf_key=self.gf_key)
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
    where c is `dataset.receivers.velocity`.

    The physics here is deliberately minimal (milestone 2 fleshes it out).
    """

    def __init__(self, kind='full', fc0_range=(10.0, 200.0), vr_range=(0.0, 1500.0),
                 mode='unilateral', n=2.0, gamma=1.0, reference=None,
                 scale=None, scale_range=None):
        super().__init__(kind, scale, scale_range)
        self.fc0_range, self.vr_range = fc0_range, vr_range
        self.mode, self.n, self.gamma, self.reference = mode, n, gamma, reference
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
        return apply_source_spectrum(synth, dataset.dt, fc, n=self.n, gamma=self.gamma,
                                     reference=self.reference)
