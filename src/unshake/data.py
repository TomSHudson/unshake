"""The Dataset: everything about one event that an inversion needs.

Coordinates
-----------
Receiver and source coordinates are in a local Cartesian ENU frame
(x = east, y = north, z = up), in any consistent length unit. Only directions
and relative distances are derived from them. Moment tensors and forces use NED
(see `unshake.mt`).
"""

import copy
import pickle
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class Receivers:
    """Per-receiver information. Every array has n_rec rows."""
    labels: np.ndarray
    coords: np.ndarray | None = None          # (n_rec, 3), ENU
    channel_dirs: np.ndarray | None = None    # (n_rec, 3), ENU unit vectors (e.g. DAS fibre axis)
    phase: np.ndarray | None = None           # (n_rec,), e.g. 'P' or 'S'
    velocity: np.ndarray | None = None        # (n_rec,), phase velocity used for Doppler terms

    def __post_init__(self):
        self.labels = np.asarray(self.labels)
        n = len(self.labels)
        for name in ('coords', 'channel_dirs', 'phase', 'velocity'):
            val = getattr(self, name)
            if val is None:
                continue
            val = np.asarray(val)
            if name == 'velocity' and val.ndim == 0:
                val = np.full(n, float(val))
            if name == 'phase' and val.ndim == 0:
                val = np.full(n, val)
            if len(val) != n:
                raise ValueError(f"Receivers.{name} has {len(val)} rows, expected {n}")
            setattr(self, name, val)

    def __len__(self):
        return len(self.labels)

    def take(self, idx):
        return Receivers(**{k: (None if v is None else np.asarray(v)[idx])
                            for k, v in vars(self).items()})


@dataclass
class Source:
    """Event information."""
    uid: str = ''
    coords: np.ndarray | None = None   # (3,), ENU
    origin_time: object = None         # e.g. obspy.UTCDateTime

    def __post_init__(self):
        if self.coords is not None:
            self.coords = np.asarray(self.coords, dtype=float).reshape(3)


@dataclass
class Dataset:
    """Observed waveforms, Green's functions, geometry and metadata for one event.

    Parameters
    ----------
    data : (n_rec, n_t) observed waveforms.
    gf : dict of Green's function arrays keyed by source type, each (n_rec, n_comp, n_t),
        e.g. {'mt': (n_rec, 6, n_t), 'force': (n_rec, 3, n_t)}. A bare array is taken as 'mt'.
    dt : sample interval (s).
    receivers : `Receivers` (or a sequence of labels).
    source : `Source`.
    t0 : time of the first sample, relative to the origin time (s).
    weights : optional (n_rec,) per-receiver weights.
    noise_std : optional scalar or (n_rec,) noise standard deviation.
    meta : free-form provenance (GF generator, filter band, STF used for GFs, ...).
    """
    data: np.ndarray
    gf: dict
    dt: float
    receivers: Receivers
    source: Source = field(default_factory=Source)
    t0: float = 0.0
    weights: np.ndarray | None = None
    noise_std: np.ndarray | float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        if not isinstance(self.gf, dict):
            self.gf = {'mt': self.gf}
        self.gf = {k: np.asarray(v, dtype=float) for k, v in self.gf.items()}
        if not isinstance(self.receivers, Receivers):
            self.receivers = Receivers(labels=self.receivers)
        n_rec, n_t = self.data.shape
        for k, g in self.gf.items():
            if g.ndim != 3 or g.shape[0] != n_rec or g.shape[2] != n_t:
                raise ValueError(f"gf['{k}'] has shape {g.shape}, expected ({n_rec}, n_comp, {n_t})")
        if len(self.receivers) != n_rec:
            raise ValueError(f"{len(self.receivers)} receivers but data has {n_rec} traces")
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float)

    def __repr__(self):
        gfs = ', '.join(f"{k}{v.shape}" for k, v in self.gf.items())
        return (f"Dataset(uid='{self.source.uid}', n_rec={self.n_rec}, n_t={self.n_t}, "
                f"dt={self.dt}, gf=[{gfs}])")

    # --- Basic properties ---

    @property
    def n_rec(self):
        return self.data.shape[0]

    @property
    def n_t(self):
        return self.data.shape[1]

    @property
    def times(self):
        return self.t0 + self.dt * np.arange(self.n_t)

    @property
    def labels(self):
        return self.receivers.labels

    # --- Geometry (derived from coordinates, straight rays) ---

    def _src_to_rec(self):
        if self.receivers.coords is None or self.source.coords is None:
            raise ValueError("Receiver and source coordinates are needed for geometry")
        return self.receivers.coords - self.source.coords

    @property
    def distance(self):
        return np.linalg.norm(self._src_to_rec(), axis=1)

    @property
    def ray_dirs(self):
        """Unit take-off vectors from source to receivers along straight rays, (n_rec, 3) ENU."""
        r = self._src_to_rec()
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    @property
    def azimuth(self):
        """Source-to-receiver azimuth, degrees clockwise from north."""
        r = self._src_to_rec()
        return np.degrees(np.arctan2(r[:, 0], r[:, 1])) % 360.0

    @property
    def takeoff(self):
        """Take-off angle, degrees from vertically down."""
        return np.degrees(np.arccos(np.clip(-self.ray_dirs[:, 2], -1.0, 1.0)))

    # --- Methods returning new Datasets ---

    def copy(self):
        return copy.deepcopy(self)

    def subset(self, idx):
        """New Dataset with only the selected receivers (indices, boolean mask or labels)."""
        idx = np.asarray(idx)
        if idx.dtype.kind in 'USO':
            lookup = {lab: i for i, lab in enumerate(self.labels)}
            idx = np.array([lookup[lab] for lab in idx])
        noise = self.noise_std
        if noise is not None and np.ndim(noise) > 0:
            noise = np.asarray(noise)[idx]
        return replace(self, data=self.data[idx], gf={k: v[idx] for k, v in self.gf.items()},
                       receivers=self.receivers.take(idx),
                       weights=None if self.weights is None else self.weights[idx],
                       noise_std=noise, meta=dict(self.meta))

    def window(self, t1, t2):
        """New Dataset cut to times t1 <= t < t2 (same time base as `times`)."""
        sel = (self.times >= t1) & (self.times < t2)
        i0 = np.argmax(sel)
        return replace(self, data=self.data[:, sel], gf={k: v[..., sel] for k, v in self.gf.items()},
                       t0=self.t0 + i0 * self.dt, meta=dict(self.meta))

    def shift(self, samples, target='gf'):
        """New Dataset with per-receiver integer shifts applied (zero fill, no wrap-around).

        samples : (n_rec,) shifts in samples; positive delays the trace.
        target : 'gf' (shift all Green's functions) or 'data'.
        """
        samples = np.broadcast_to(np.asarray(samples, dtype=int), (self.n_rec,))
        new = self.copy()
        arrs = list(new.gf.values()) if target == 'gf' else [new.data]
        for a in arrs:
            for i, s in enumerate(samples):
                a[i] = np.roll(a[i], s, axis=-1)
                if s > 0:
                    a[i, ..., :s] = 0.0
                elif s < 0:
                    a[i, ..., s:] = 0.0
        new.meta.setdefault('shifts', []).append({'target': target, 'samples': samples.copy()})
        return new

    def estimate_noise(self, t1, t2):
        """Set and return per-receiver noise_std from the data in a (pre-event) window."""
        sel = (self.times >= t1) & (self.times < t2)
        self.noise_std = np.std(self.data[:, sel], axis=1)
        return self.noise_std

    # --- IO ---

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    # --- Quick look ---

    def plot(self, gf_key='mt', normalise=True):
        """Quick look at the data next to each Green's function component."""
        import matplotlib.pyplot as plt
        g = self.gf[gf_key]
        panels = [('data', self.data)] + [(f"{gf_key}[{i}]", g[:, i]) for i in range(g.shape[1])]
        fig, axes = plt.subplots(1, len(panels), figsize=(2.2 * len(panels), 4), sharey=True)
        for ax, (title, arr) in zip(axes, panels):
            if normalise:
                arr = arr / (np.max(np.abs(arr), axis=1, keepdims=True) + 1e-30)
            vmax = np.max(np.abs(arr))
            ax.pcolormesh(self.times, np.arange(self.n_rec), arr, cmap='RdBu', vmin=-vmax, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel('Time (s)')
        axes[0].set_ylabel('Receiver index')
        fig.tight_layout()
        return fig
