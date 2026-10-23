"""Synthetic check: recover a known moment tensor with least squares and Monte Carlo.

Uses analytic far-field P + S Green's functions in a homogeneous whole space,
3-component receivers on the surface. Run:  python examples/synthetic_mt_recovery.py
"""

import matplotlib.pyplot as plt
import numpy as np

import unshake
from unshake import mt, plot

rng = np.random.default_rng(42)

# --- Geometry and medium ---
vp, vs, dt, n_t = 3500.0, 1800.0, 1e-3, 200
src = np.array([0.0, 0.0, -50.0])                                # ENU (m)
n_sta = 12
az = rng.uniform(0, 2 * np.pi, n_sta)
dist = rng.uniform(40, 150, n_sta)
sta = np.column_stack([dist * np.sin(az), dist * np.cos(az), np.zeros(n_sta)])


def ricker(t, f0):
    a = (np.pi * f0 * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


# --- Analytic Green's functions (NED source components, NED receiver components) ---
pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]   # nn, ee, dd, ne, nd, ed
t = np.arange(n_t) * dt
labels, coords, chans, gfs = [], [], [], []
for s in range(n_sta):
    r_enu = sta[s] - src
    r = np.linalg.norm(r_enu)
    g = np.array([r_enu[1], r_enu[0], -r_enu[2]]) / r          # ray direction in NED
    p_wav = ricker(t - r / vp - 0.02, 60.0) / r
    s_wav = ricker(t - r / vs - 0.02, 60.0) / r
    for i, comp in enumerate('NED'):
        gf = np.zeros((6, n_t))
        for c, (p, q) in enumerate(pairs):
            for (a, b) in {(p, q), (q, p)}:                     # symmetric pair: M_pq = M_qp = 1
                gf[c] += g[i] * g[a] * g[b] * p_wav + ((i == a) - g[i] * g[a]) * g[b] * s_wav
        labels.append(f"S{s:02d}{comp}")
        coords.append(sta[s])
        chans.append({'N': [0, 1, 0], 'E': [1, 0, 0], 'D': [0, 0, -1]}[comp])
        gfs.append(gf)
gf = np.array(gfs)

# --- True source and noisy data ---
m_true = 1e3 * mt.from_eigenvalues(np.array([[0.9, 0.2, -0.8]]), mt.random_rotations(1, rng))[0]
data = unshake.forward.synthetics(gf, m_true)
noise = 0.05 * np.max(np.abs(data))
data = data + noise * rng.standard_normal(data.shape)

ds = unshake.Dataset(data=data, gf={'mt': gf}, dt=dt,
                     receivers=unshake.Receivers(labels=labels, coords=coords, channel_dirs=chans,
                                                 phase='P', velocity=vp),
                     source=unshake.Source(uid='synthetic', coords=src), noise_std=noise)
print(ds)

# --- Least squares ---
m_lsq, cov = unshake.least_squares(ds)
print("true:", np.round(mt.normalise(m_true), 3))
print("lsq: ", np.round(mt.normalise(m_lsq), 3), " +/-", np.round(np.sqrt(np.diag(cov)) / mt.norm(m_lsq), 3))

# --- Monte Carlo (lune sampling, Gaussian likelihood, amplitude free) ---
model = unshake.MTModel(kind='lune', scale_range=(0.5, 2.0))
res = unshake.monte_carlo(model, ds, n=50_000, rng=1)
print(res)
print("best:", np.round(mt.normalise(res.best), 3), " VR =", round(res.vr[res.best_idx], 3))
g_true, d_true = mt.lune(m_true)
g_best, d_best = mt.lune(res.best)
print(f"lune true (g, d) = ({np.degrees(g_true):.1f}, {np.degrees(d_true):.1f}),"
      f" best = ({np.degrees(g_best):.1f}, {np.degrees(d_best):.1f})")

# --- Plots ---
fig, ax = plt.subplots(1, 3, figsize=(9, 4), gridspec_kw={'width_ratios': [1, 1, 1]})
plot.lune(result=res, ax=ax[0])
plot.beachball(m_true, ax=ax[1])
ax[1].set_title('true')
plot.beachball(res.best, ax=ax[2])
ax[2].set_title('MC best')
plot.wavefield(ds, unshake.forward.synthetics(gf, res.best))
plt.show()
