"""Stability selection with sparsevb."""
import numpy as np
from sparsevb import svb_fit_linear


def select_stability(X, y, B=50, subsample_frac=0.5, freq_threshold=0.9,
                     gamma_threshold=0.5, rng=None):
    """Stability selection: bootstrap subsamples + sparsevb.

    Parameters
    ----------
    X : ndarray (n, p)
    y : ndarray (n,)
    B : int — number of bootstrap subsamples
    subsample_frac : float — fraction of samples per subsample
    freq_threshold : float — minimum selection frequency to keep a variable
    gamma_threshold : float — gamma threshold for per-subsample selection
    rng : numpy Generator (optional)

    Returns
    -------
    selected : ndarray of selected variable indices
    info : dict with selection frequencies
    """
    if rng is None:
        rng = np.random.default_rng(2)

    n, p = X.shape
    m = int(n * subsample_frac)
    selection_counts = np.zeros(p)

    for b in range(B):
        idx = rng.choice(n, size=m, replace=False)
        X_b, y_b = X[idx], y[idx]
        res_b = svb_fit_linear(X_b, y_b)
        selection_counts += (res_b["gamma"] > gamma_threshold).astype(float)

    frequencies = selection_counts / B
    selected = np.sort(np.where(frequencies > freq_threshold)[0])

    return selected, {"frequencies": frequencies}
