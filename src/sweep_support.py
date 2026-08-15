"""Small, importable preprocessing components used by sweep.py."""
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

SEED = 42

class VarianceGate(TransformerMixin, BaseEstimator):
    def __init__(self, threshold: float = 1e-12):
        self.threshold = threshold

    def fit(self, X: Any, y: Any = None):
        variance = np.nanvar(np.asarray(X), axis=0)
        self.keep_ = variance > self.threshold
        if not self.keep_.any():
            self.keep_ = np.ones(len(variance), dtype=bool)
        self.n_dropped_ = int((~self.keep_).sum())
        return self

    def transform(self, X: Any):
        return np.asarray(X)[:, self.keep_]

class SafePCA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components: int = 17, random_state: int = SEED):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None):
        array = np.asarray(X)
        count = max(1, min(self.n_components, array.shape[0] - 1, array.shape[1]))
        self.n_components_used_ = int(count)
        self.pca_ = PCA(n_components=count, svd_solver="randomized",
                        random_state=self.random_state).fit(array)
        return self

    def transform(self, X: Any):
        return self.pca_.transform(np.asarray(X))
