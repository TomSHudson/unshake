"""Source time functions (moment rate, unit moment, causal), in time and frequency.

Convolve them with formal band-limited GFs (see `Dataset.remove_stf`) using
`forward.convolve_stf`, e.g.

    forward.convolve_stf(synth, dt, lambda f: stfs.brune_stf_spectrum(f, fc[:, None]))

Spectra use the np.fft sign convention (X(f) = sum x(t) exp(-2 pi i f t) dt), so they
can multiply `np.fft.rfft` output directly. `spectra.py` keeps the zero-phase amplitude
spectra (`spectra.brune`, `spectra.boatwright`) and the Doppler `spectra.apparent_fc`.
"""

import numpy as np


def brune_stf(t, fc):
    """Brune (1970) moment rate with unit moment: (2 pi fc)^2 t exp(-2 pi fc t) for t >= 0."""
    t = np.asarray(t, dtype=float)
    wc = 2.0 * np.pi * np.asarray(fc, dtype=float)
    return np.where(t >= 0, wc ** 2 * t * np.exp(-wc * np.clip(t, 0, None)), 0.0)


def brune_stf_spectrum(f, fc):
    """Complex spectrum of `brune_stf`: 1 / (1 + i f/fc)^2.

    Its modulus is the zero-phase `spectra.brune(f, fc)` (omega-squared); the phase makes
    the pulse causal. fc broadcasts against f, e.g. fc[..., None] gives (..., n_f).
    """
    return 1.0 / (1.0 + 1j * np.asarray(f) / np.asarray(fc, dtype=float)) ** 2


def two_brune_spectrum(f, fc1, fc2, w2):
    """Linear sum of two unit-moment Brune STFs: (1 - w2) S(f; fc1) + w2 S(f; fc2).

    w2 is the moment fraction of the second STF, so the total moment is unchanged.
    fc1, fc2 and w2 broadcast against f; typically fc1 and w2 are the same at every
    receiver and fc2 varies with direction (e.g. from `spectra.apparent_fc`).
    """
    w2 = np.asarray(w2, dtype=float)
    return (1.0 - w2) * brune_stf_spectrum(f, fc1) + w2 * brune_stf_spectrum(f, fc2)
