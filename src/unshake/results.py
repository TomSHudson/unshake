"""Inversion results."""

import pickle
from dataclasses import dataclass, field

import numpy as np

from . import mt


@dataclass
class Result:
    """Samples and their log-likelihoods from a sampling inversion.

    params : (n, n_params) samples; columns named by param_names.
    loglike : (n,) log-likelihood of each sample.
    vr : (n,) variance reduction of each sample (for interpretation).
    """
    params: np.ndarray
    param_names: list
    loglike: np.ndarray
    vr: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"Result({self.meta.get('model', '')}, n={len(self)}, "
                f"max VR={np.max(self.vr) if self.vr is not None else None:.3f}, "
                f"ESS={self.ess:.1f})")

    def __len__(self):
        return len(self.loglike)

    def __getitem__(self, name):
        """Column of params by name, e.g. result['Mnn'] or result['fc0']."""
        return self.params[:, self.param_names.index(name)]

    # --- Posterior ---

    @property
    def weights(self):
        """Normalised posterior weights (flat prior over the samples drawn)."""
        w = np.exp(self.loglike - np.max(self.loglike))
        return w / w.sum()

    @property
    def ess(self):
        """Effective sample size, 1 / sum(w^2)."""
        return 1.0 / np.sum(self.weights ** 2)

    @property
    def best_idx(self):
        return int(np.argmax(self.loglike))

    @property
    def best(self):
        """Maximum likelihood sample."""
        return self.params[self.best_idx]

    @property
    def mean(self):
        return self.weights @ self.params

    def top(self, k=100):
        """Indices of the k highest-likelihood samples (best first)."""
        return np.argsort(self.loglike)[::-1][:k]

    # --- MT helpers (when the first 6 params are an MT) ---

    @property
    def mts(self):
        if self.param_names[:6] != ['Mnn', 'Mee', 'Mdd', 'Mne', 'Mnd', 'Med']:
            raise AttributeError("Result params do not start with a moment tensor")
        return self.params[:, :6]

    def lune(self):
        """(gamma, delta) in radians for every sample."""
        return mt.lune(self.mts)

    # --- IO ---

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
