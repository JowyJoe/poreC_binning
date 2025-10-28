from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans


def kmeans_labels(U: np.ndarray, k: int, n_init: int = 20, seed: int = 42) -> np.ndarray:
    # Row-normalize U to unit norm to stabilize k-means
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = U / norms
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed, verbose=0)
    labels = km.fit_predict(X)
    return labels
