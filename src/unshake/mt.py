"""Moment tensor conventions, conversions and random samplers.

Convention (used everywhere in unshake)
---------------------------------------
A moment tensor is a 6-vector of *plain tensor components* in NED coordinates::

    m6 = [M_nn, M_ee, M_dd, M_ne, M_nd, M_ed]

This matches the column order of the Green's function array ``gf['mt']`` with
shape (n_rec, 6, n_t), where column ``ne`` is the response to the symmetric pair
M_ne = M_en = 1 (as Salvus ``MomentTensorPoint3D`` defines it). There is no
sqrt(2) scaling of the off-diagonals, unlike SeisSrcInv.

Single forces are 3-vectors [F_n, F_e, F_d].

All functions accept a single vector, shape (6,), or a batch, shape (n, 6).
"""

import numpy as np
from scipy.spatial.transform import Rotation

_SQRT2 = np.sqrt(2.0)
# Tape & Tape (2012) rotation from eigenvalues to lune (u, v, w) space:
_U = (1.0 / np.sqrt(6.0)) * np.array([[np.sqrt(3.0), 0.0, -np.sqrt(3.0)],
                                       [-1.0, 2.0, -1.0],
                                       [_SQRT2, _SQRT2, _SQRT2]])


# --- Conversions ---

def to_33(m6):
    """6-vector(s) -> symmetric 3x3 tensor(s), shape (..., 3, 3)."""
    m6 = np.asarray(m6, dtype=float)
    nn, ee, dd, ne, nd, ed = np.moveaxis(m6, -1, 0)
    return np.stack([np.stack([nn, ne, nd], -1),
                     np.stack([ne, ee, ed], -1),
                     np.stack([nd, ed, dd], -1)], -2)


def from_33(M):
    """Symmetric 3x3 tensor(s) -> 6-vector(s), shape (..., 6)."""
    M = np.asarray(M, dtype=float)
    return np.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2],
                     M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]], -1)


def norm(m6):
    """Frobenius norm sqrt(sum_ij M_ij^2) of 6-vector(s)."""
    m6 = np.asarray(m6, dtype=float)
    return np.sqrt(np.sum(m6[..., :3] ** 2, -1) + 2.0 * np.sum(m6[..., 3:] ** 2, -1))


def normalise(m6):
    """Scale 6-vector(s) to unit Frobenius norm."""
    m6 = np.asarray(m6, dtype=float)
    return m6 / norm(m6)[..., None]


def moment(m6):
    """Scalar seismic moment M0 = ||M||_F / sqrt(2) (Silver & Jordan 1982)."""
    return norm(m6) / _SQRT2


def mw(m6):
    """Moment magnitude (Hanks & Kanamori), assuming m6 is in N m."""
    return (2.0 / 3.0) * (np.log10(moment(m6)) - 9.1)


def to_use(m6):
    """NED 6-vector(s) -> USE (r, t, p) order [rr, tt, pp, rt, rp, tp], as used by
    obspy's beachball and GCMT."""
    m6 = np.asarray(m6, dtype=float)
    nn, ee, dd, ne, nd, ed = np.moveaxis(m6, -1, 0)
    return np.stack([dd, nn, ee, nd, -ed, -ne], -1)


# --- Eigen-decomposition, lune and fault planes ---

def eig(m6):
    """Eigenvalues (descending) and eigenvectors (columns, NED) of 6-vector(s)."""
    w, v = np.linalg.eigh(to_33(m6))
    return w[..., ::-1], v[..., ::-1]


def lune(m6):
    """Lune longitude gamma and latitude delta (radians) of 6-vector(s)
    (Tape & Tape 2012). gamma in [-pi/6, pi/6], delta in [-pi/2, pi/2]."""
    lam, _ = eig(m6)
    l1, l2, l3 = lam[..., 0], lam[..., 1], lam[..., 2]
    gamma = np.arctan2(-l1 + 2.0 * l2 - l3, np.sqrt(3.0) * (l1 - l3))
    lam_norm = np.linalg.norm(lam, axis=-1)
    beta = np.arccos(np.clip((l1 + l2 + l3) / (np.sqrt(3.0) * lam_norm), -1.0, 1.0))
    return gamma, np.pi / 2.0 - beta


def lune_to_eigenvalues(gamma, delta):
    """Unit-norm eigenvalues (descending), shape (..., 3), for lune (gamma, delta)."""
    gamma, delta = np.asarray(gamma, dtype=float), np.asarray(delta, dtype=float)
    beta = np.pi / 2.0 - delta
    uvw = np.stack([np.cos(gamma) * np.sin(beta), np.sin(gamma) * np.sin(beta), np.cos(beta)], -1)
    return uvw @ _U   # equivalent to U^T uvw (Tape & Tape 2012, eq. 20)


def sdr(m6):
    """Strike, dip, rake (degrees) of both nodal planes of the DC part of m6.

    Returns two arrays of shape (..., 3): (plane 1, plane 2). Uses the
    Aki & Richards convention with NED axes.
    """
    _, v = eig(m6)
    t_ax, p_ax = v[..., :, 0], v[..., :, 2]
    planes = []
    for n, s in (((t_ax + p_ax) / _SQRT2, (t_ax - p_ax) / _SQRT2),
                 ((t_ax - p_ax) / _SQRT2, (t_ax + p_ax) / _SQRT2)):
        # Normal must point upwards (hanging wall side), i.e. n_d <= 0:
        flip = np.where(n[..., 2] > 0, -1.0, 1.0)[..., None]
        n, s = n * flip, s * flip
        dip = np.arccos(np.clip(-n[..., 2], -1.0, 1.0))
        strike = np.arctan2(-n[..., 0], n[..., 1])
        sin_dip = np.where(np.abs(np.sin(dip)) < 1e-12, 1e-12, np.sin(dip))
        rake = np.arctan2(-s[..., 2] / sin_dip,
                          s[..., 0] * np.cos(strike) + s[..., 1] * np.sin(strike))
        out = np.degrees(np.stack([strike % (2 * np.pi), dip, rake], -1))
        planes.append(out)
    return planes[0], planes[1]


# --- Random samplers ---
# All take (n, rng) and return unit-norm samples, shape (n, 6) or (n, 3).

def random_rotations(n, rng=None):
    """n uniformly distributed (Haar) random rotation matrices, shape (n, 3, 3).

    (SeisSrcInv rotated by two angles only, which does not cover SO(3) uniformly.)
    """
    rng = np.random.default_rng(rng)
    return Rotation.random(n, random_state=rng).as_matrix()


def from_eigenvalues(lam, R):
    """Build 6-vectors from eigenvalues (n, 3) and rotations (n, 3, 3)."""
    M = np.einsum('nij,nj,nkj->nik', R, lam, R)
    return from_33(M)


def sample_full_mt(n, rng=None):
    """Full MTs uniformly distributed on the unit sphere in tensor space
    (6 Gaussian deviates, Pugh 2015 / Muller 1959)."""
    rng = np.random.default_rng(rng)
    x = rng.standard_normal((n, 6))
    x[:, 3:] /= _SQRT2   # orthonormal coordinates -> plain tensor components
    return normalise(x)


def sample_lune(n, rng=None, equal_area=False):
    """Full MTs sampled uniformly in lune coordinates, then randomly oriented.

    equal_area=False samples gamma and delta uniformly, as in SeisSrcInv.
    equal_area=True samples uniformly by area on the lune (sin(delta) uniform).
    """
    rng = np.random.default_rng(rng)
    gamma = rng.uniform(-np.pi / 6.0, np.pi / 6.0, n)
    if equal_area:
        delta = np.arcsin(rng.uniform(-1.0, 1.0, n))
    else:
        delta = rng.uniform(-np.pi / 2.0, np.pi / 2.0, n)
    return from_eigenvalues(lune_to_eigenvalues(gamma, delta), random_rotations(n, rng))


def sample_dc(n, rng=None):
    """Randomly oriented, unit-norm double-couples."""
    rng = np.random.default_rng(rng)
    lam = np.tile(np.array([1.0, 0.0, -1.0]) / _SQRT2, (n, 1))
    return from_eigenvalues(lam, random_rotations(n, rng))


def sample_force(n, rng=None):
    """Unit single-force vectors [F_n, F_e, F_d], uniform on the sphere."""
    rng = np.random.default_rng(rng)
    x = rng.standard_normal((n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


SAMPLERS = {'full': sample_full_mt, 'lune': sample_lune, 'dc': sample_dc}
