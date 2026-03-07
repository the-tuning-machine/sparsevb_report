"""Knockoff-based variable selection (Barber-Candes)."""
import numpy as np
from sparsevb import svb_fit_linear


def select_knockoffs(X, y, gamma, target_fdr=0.1, gamma_prefilter=0.2, rng=None):
    """Knockoff filter using sparsevb inclusion probabilities as statistics.

    Prefilters candidates (gamma > gamma_prefilter) to keep the augmented
    matrix manageable, then generates knockoff variables by permuting each
    candidate column independently, fits sparsevb on [X_cand, X_tilde_cand],
    and applies the knockoff filter.

    Parameters
    ----------
    X : ndarray (n, p)
    y : ndarray (n,)
    gamma : ndarray (p,) — original inclusion probabilities from sparsevb
    target_fdr : float — target FDR level
    gamma_prefilter : float — minimum gamma to be considered as candidate
    rng : numpy Generator (optional)

    Returns
    -------
    selected : ndarray of selected variable indices (in original indexing)
    info : dict with W statistics and threshold
    """
    if rng is None:
        rng = np.random.default_rng(1)

    n, p = X.shape

    # Prefilter: only consider variables with gamma above threshold
    candidates = np.where(gamma > gamma_prefilter)[0]
    if len(candidates) == 0:
        return np.array([], dtype=int), {"W": np.array([]), "threshold": np.inf,
                                          "candidates": candidates,
                                          "target_fdr": target_fdr}

    X_cand = X[:, candidates]
    p_cand = len(candidates)

    # Generate knockoff matrix: permute each candidate column independently
    X_tilde = np.empty_like(X_cand)
    for j in range(p_cand):
        X_tilde[:, j] = rng.permutation(X_cand[:, j])

    # Fit sparsevb on augmented candidate matrix [X_cand, X_tilde]
    X_aug = np.hstack([X_cand, X_tilde])
    res_aug = svb_fit_linear(X_aug, y)
    gamma_aug = res_aug["gamma"]
    mu_aug = res_aug["mu"]

    gamma_orig = gamma_aug[:p_cand]
    gamma_knock = gamma_aug[p_cand:]
    mu_orig = mu_aug[:p_cand]
    mu_knock = mu_aug[p_cand:]

    # Knockoff statistic: use |mu * gamma| for better discrimination
    stat_orig = np.abs(mu_orig * gamma_orig)
    stat_knock = np.abs(mu_knock * gamma_knock)
    W = stat_orig - stat_knock

    # Knockoff filter: find threshold T
    # T = min{t > 0 : (1 + #{j: W_j <= -t}) / max(1, #{j: W_j >= t}) <= q}
    # Using knockoff+ (with +1 in numerator) for finite-sample FDR control
    candidate_thresholds = np.sort(np.abs(W[W != 0]))[::-1]

    threshold = np.inf
    for t in candidate_thresholds:
        if t <= 0:
            continue
        numerator = 1 + np.sum(W <= -t)  # knockoff+ correction
        denominator = max(1, np.sum(W >= t))
        if numerator / denominator <= target_fdr:
            threshold = t

    if threshold == np.inf:
        selected = np.array([], dtype=int)
    else:
        selected = np.sort(candidates[W >= threshold])

    return selected, {
        "W": W,
        "threshold": threshold,
        "candidates": candidates,
        "gamma_orig": gamma_orig,
        "gamma_knock": gamma_knock,
        "stat_orig": stat_orig,
        "stat_knock": stat_knock,
        "target_fdr": target_fdr,
    }
