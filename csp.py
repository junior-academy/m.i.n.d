from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin


def _as_3d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (n_trials, n_channels, n_times); got {X.shape}")
    return X


def _trial_cov(trial: np.ndarray, reg: Optional[float]) -> np.ndarray:
    # trial: (n_channels, n_times)
    cov = trial @ trial.T
    cov = (cov + cov.T) / 2.0
    trace = float(np.trace(cov))
    if trace > 0:
        cov = cov / trace
    if reg is not None and reg != 0:
        cov = cov + (float(reg) * np.eye(cov.shape[0], dtype=cov.dtype))
    return cov


def _mean_cov(X: np.ndarray, reg: Optional[float]) -> np.ndarray:
    covs = [_trial_cov(trial, reg=None) for trial in X]
    cov = np.mean(covs, axis=0)
    cov = (cov + cov.T) / 2.0
    if reg is not None and reg != 0:
        cov = cov + (float(reg) * np.eye(cov.shape[0], dtype=cov.dtype))
    return cov


@dataclass
class CSP(BaseEstimator, TransformerMixin):
    """
    Minimal CSP transformer with a 1-vs-rest multi-class strategy.

    - Fits one set of spatial filters per class vs the composite covariance
    - Uses the top eigenvector from each class (works well when n_components == n_classes)
    - Transforms epochs into log-variance features like typical CSP pipelines
    """

    n_components: int = 4
    reg: Optional[float] = 1e-6
    log: bool = True
    # Accepted for compatibility with `mne.decoding.CSP`; ignored here.
    norm_trace: bool = False

    filters_: Optional[np.ndarray] = None  # shape: (n_components, n_channels)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = _as_3d(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("y must be 1D and match X.shape[0]")

        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("Need at least 2 classes for CSP")

        # Composite covariance across all trials (trace-normalized per-trial then averaged)
        cov_composite = _mean_cov(X, reg=self.reg)

        # One spatial filter per class by default
        n_filters = min(self.n_components, len(classes))
        filters = []
        for cls in classes[:n_filters]:
            Xc = X[y == cls]
            if Xc.shape[0] == 0:
                continue
            cov_c = _mean_cov(Xc, reg=self.reg)
            # Solve cov_c v = lambda cov_composite v
            evals, evecs = linalg.eigh(cov_c, cov_composite)
            # Take the eigenvector with the largest eigenvalue (class-discriminative)
            v = evecs[:, np.argmax(evals)]
            v = v / (np.linalg.norm(v) + 1e-12)
            filters.append(v)

        if not filters:
            raise RuntimeError("Failed to compute CSP filters")

        W = np.stack(filters, axis=0)
        # If asked for more components than classes, pad with additional eigenvectors from composite PCA.
        if W.shape[0] < self.n_components:
            # PCA on composite covariance to get extra orthonormal directions
            evals, evecs = linalg.eigh(cov_composite)
            order = np.argsort(evals)[::-1]
            for idx in order:
                if W.shape[0] >= self.n_components:
                    break
                v = evecs[:, idx]
                # avoid duplicates
                if np.max(np.abs(W @ v)) > 0.98:
                    continue
                W = np.vstack([W, v[None, :]])

        self.filters_ = W[: self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.filters_ is None:
            raise RuntimeError("CSP must be fit before transform")
        X = _as_3d(X)

        W = self.filters_
        feats = np.empty((X.shape[0], W.shape[0]), dtype=float)
        for i, trial in enumerate(X):
            projected = W @ trial  # (n_components, n_times)
            var = np.var(projected, axis=1, ddof=0)
            if self.log:
                var = np.log(var + 1e-12)
            feats[i] = var
        return feats
