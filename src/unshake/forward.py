"""Forward modelling: Green's functions x source -> synthetics."""

import numpy as np

from .spectra import source_spectrum


def synthetics(gf, m):
    """Linear synthetics for source vector(s) m.

    gf : (n_rec, n_comp, n_t)
    m : (n_comp,) or (n, n_comp)
    Returns (n_rec, n_t) or (n, n_rec, n_t).
    """
    m = np.asarray(m, dtype=float)
    n_rec, n_comp, n_t = gf.shape
    G = np.moveaxis(gf, 1, 0).reshape(n_comp, n_rec * n_t)   # matmul uses BLAS (much faster than einsum)
    return (m @ G).reshape(m.shape[:-1] + (n_rec, n_t))


def apply_source_spectrum(synth, dt, fc, n=2.0, gamma=1.0, reference=None):
    """Filter synthetics by a zero-phase source spectrum with per-receiver corner frequencies.

    synth : (..., n_rec, n_t)
    fc : broadcastable to (..., n_rec), e.g. (n, n_rec) from `spectra.apparent_fc`.
    reference : optional callable f -> amplitude spectrum already in the GFs
        (e.g. ``lambda f: ricker_spectrum(f, f0)``). It is divided out, water-levelled.
    """
    n_t = synth.shape[-1]
    f = np.fft.rfftfreq(n_t, dt)
    spec = source_spectrum(f, np.asarray(fc, dtype=float)[..., None], n=n, gamma=gamma)
    if reference is not None:
        ref = np.asarray(reference(f), dtype=float)
        spec = spec / np.maximum(ref, 1e-3 * ref.max())
    return np.fft.irfft(np.fft.rfft(synth, axis=-1) * spec, n=n_t, axis=-1)
