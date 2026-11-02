"""Source spectra and apparent (Doppler-shifted) corner frequencies.

Plain functions, so that the theory can be swapped while it is being developed.

Green's functions from a simulation are always band-limited and include the
simulation's source time function (e.g. the Salvus Ricker). unshake works with
formal band-limited GFs, G * B, where B = `flat_band(f, *band)`. Get them with
`Dataset.remove_stf` (or `forward.remove_stf`); the data must be band-limited by
the same B (`forward.bandlimit`). The source spectrum is then the only
source-time term in the forward model.
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


def flat_band(f, f0, f1, f2, f3):
    """Band filter B(f): zero below f0, cosine ramp up over f0-f1, flat (1) over f1-f2,
    cosine ramp down over f2-f3, zero above f3. Same shape as Salvus's FlatSpectrumWavelet."""
    f = np.abs(np.asarray(f, dtype=float))
    B = np.zeros_like(f)
    up, flat, down = (f >= f0) & (f < f1), (f >= f1) & (f < f2), (f >= f2) & (f < f3)
    B[up] = 0.5 * (1.0 - np.cos(np.pi * (f[up] - f0) / (f1 - f0)))
    B[flat] = 1.0
    B[down] = 0.5 * (1.0 + np.cos(np.pi * (f[down] - f2) / (f3 - f2)))
    return B


def amplitude_spectrum(x, dt, nfft=None):
    """Hann-tapered amplitude spectrum along the last axis. Returns (f, |X| * dt)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    nfft = nfft or int(2 ** np.ceil(np.log2(4 * n)))
    X = np.fft.rfft(x * np.hanning(n), n=nfft, axis=-1)
    return np.fft.rfftfreq(nfft, dt), np.abs(X) * dt


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
