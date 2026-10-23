"""Source spectra and apparent (Doppler-shifted) corner frequencies.

Plain functions, so that the theory can be swapped while it is being developed.

TODO (open issue): the Salvus GFs are generated with a Ricker source-time
function. Before fitting corner frequencies on real data, either regenerate the
GFs with a broadband/impulsive STF, or divide the Ricker spectrum out (see
`ricker_spectrum` and `forward.apply_source_spectrum(..., reference=...)`).
"""

import numpy as np


def source_spectrum(f, fc, n=2.0, gamma=1.0):
    """Normalised displacement source spectrum (Boatwright 1980 general form).

    |S(f)| = 1 / (1 + (f/fc)^(gamma n))^(1/gamma)

    gamma=1 gives Brune (1970) (n=2); gamma=2 gives the sharper Boatwright shape.
    Zero-phase: unshake treats the source spectrum as a real amplitude filter.
    """
    return 1.0 / (1.0 + (np.abs(f) / fc) ** (gamma * n)) ** (1.0 / gamma)


def brune(f, fc, n=2.0):
    return source_spectrum(f, fc, n=n, gamma=1.0)


def boatwright(f, fc, n=2.0):
    return source_spectrum(f, fc, n=n, gamma=2.0)


def ricker_spectrum(f, f0):
    """Amplitude spectrum of a Ricker wavelet with peak frequency f0 (peak normalised to 1)."""
    r = (np.asarray(f) / f0) ** 2
    return r * np.exp(1.0 - r)


def rupture_direction(azimuth, plunge):
    """Unit rupture direction(s) in ENU from azimuth (clockwise from north) and
    plunge (positive downwards), both in radians. Returns shape (..., 3)."""
    azimuth, plunge = np.asarray(azimuth), np.asarray(plunge)
    return np.stack([np.cos(plunge) * np.sin(azimuth),
                     np.cos(plunge) * np.cos(azimuth),
                     -np.sin(plunge)], -1)


def apparent_fc(fc0, vr, c, ray_dirs, rupture_dir, mode='unilateral'):
    """Apparent corner frequency at each receiver for a line rupture (Doppler effect).

    Parameters
    ----------
    fc0 : (n,) corner frequency seen perpendicular to the rupture (theta = 90 deg).
    vr : (n,) rupture velocity.
    c : scalar or (n_rec,) phase velocity at the source (vp or vs).
    ray_dirs : (n_rec, 3) unit take-off vectors (ENU), e.g. `Dataset.ray_dirs`.
    rupture_dir : (n, 3) unit rupture directions (ENU).
    mode : 'unilateral' -> fc0 / (1 - (vr/c) cos theta)   (Haskell-type)
           'bilateral'  -> fc0 / (1 + (vr/c) |cos theta|) (symmetric; the longer half dominates)

    Returns (n, n_rec).
    """
    fc0 = np.atleast_1d(np.asarray(fc0, dtype=float))[:, None]
    vr = np.atleast_1d(np.asarray(vr, dtype=float))[:, None]
    cos_theta = np.atleast_2d(rupture_dir) @ np.asarray(ray_dirs).T   # (n, n_rec)
    mach = vr / np.asarray(c, dtype=float)
    if mode == 'unilateral':
        return fc0 / (1.0 - mach * cos_theta)
    if mode == 'bilateral':
        return fc0 / (1.0 + mach * np.abs(cos_theta))
    raise ValueError(f"Unknown mode '{mode}'")
