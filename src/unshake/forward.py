"""Forward modelling: Green's functions x source -> synthetics."""

import numpy as np

from .spectra import flat_band, source_spectrum


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


def apply_source_spectrum(synth, dt, fc, n=2.0, gamma=1.0):
    """Filter synthetics by a zero-phase source spectrum with per-receiver corner frequencies.

    The GFs must be formal (STF removed, see `remove_stf`), so that this spectrum
    is the only source-time term.

    synth : (..., n_rec, n_t)
    fc : broadcastable to (..., n_rec), e.g. (n, n_rec) from `spectra.apparent_fc`.
    """
    n_t = synth.shape[-1]
    f = np.fft.rfftfreq(n_t, dt)
    spec = source_spectrum(f, np.asarray(fc, dtype=float)[..., None], n=n, gamma=gamma)
    return np.fft.irfft(np.fft.rfft(synth, axis=-1) * spec, n=n_t, axis=-1)


def convolve_stf(synth, dt, spec):
    """Convolve synthetics with a (complex, e.g. causal) source time function spectrum.

    The GFs must be formal (STF removed, see `remove_stf`), so that this STF is the only
    source-time term. Done in the frequency domain (rfft, multiply, irfft), zero-padded
    so that the convolution is linear, not circular; the result is cut back to n_t.

    synth : (..., n_rec, n_t)
    spec : callable f -> spectrum broadcastable to (..., n_rec, n_f), e.g.
        ``lambda f: stfs.brune_stf_spectrum(f, fc[..., None])``.
    """
    return _filter(np.asarray(synth, dtype=float), dt, spec)


def _filter(x, dt, spec_fn):
    """Multiply x (..., n_t) by spec_fn(f) in the frequency domain, zero-padded to avoid wrap-around."""
    n_t = x.shape[-1]
    nfft = int(2 ** np.ceil(np.log2(2 * n_t)))
    f = np.fft.rfftfreq(nfft, dt)
    return np.fft.irfft(np.fft.rfft(x, n=nfft, axis=-1) * spec_fn(f), n=nfft, axis=-1)[..., :n_t]


def bandlimit(x, dt, band):
    """Apply the zero-phase band filter B = `spectra.flat_band(f, *band)` along the last axis.

    band : (f0, f1, f2, f3) in Hz. Use the same band for the data and the GFs.
    """
    return _filter(x, dt, lambda f: flat_band(f, *band))


def stf_spectrum(stf, f):
    """Complex spectrum of a source time function at frequencies f.

    stf : callable f -> spectrum, or a sampled trace (t, s). A sampled trace keeps its
        timing: phase is referenced to t = 0.

    For GFs, pass the simulation's *moment-rate* function: Salvus applies its wavelet
    s(t) as the moment M(t) (verified against Aki & Richards in a full-space test), so use
    ``wavelet.get_stf_rate()``, or ``lambda f: 2j * np.pi * f * spectra.ricker_spectrum(f, f0)``
    for a Ricker centred on t = 0. The formal GFs are then the response to a unit-impulse
    moment rate (a step in moment), matching the moment-rate source spectra in `spectra`.
    """
    if callable(stf):
        return np.asarray(stf(f))
    t, s = (np.asarray(a, dtype=float) for a in stf)
    return np.exp(-2j * np.pi * np.outer(f, t)) @ s * (t[1] - t[0])


def remove_stf(x, dt, stf, band):
    """Formal band-limited Green's functions: x * B / STF along the last axis.

    Divides out the STF used in the simulation (see `stf_spectrum`) and applies the band
    filter B = `spectra.flat_band(f, *band)`. Zero wherever B = 0, so no water level is
    needed; choose `band` inside the band where the STF has energy (and where the mesh
    is accurate). The data must be band-limited by the same B (`bandlimit`).
    """
    def spec(f):
        B = flat_band(f, *band)
        S = stf_spectrum(stf, f)
        out = np.zeros_like(S, dtype=complex)
        out[B > 0] = B[B > 0] / S[B > 0]
        return out
    return _filter(x, dt, spec)
