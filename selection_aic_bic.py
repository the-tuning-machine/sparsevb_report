"""AIC / BIC forward selection on candidates sorted by gamma descending."""
import numpy as np


def select_aic_bic(X, y, gamma):
    """Forward selection using AIC and BIC on top-k variables sorted by gamma.

    Parameters
    ----------
    X : ndarray (n, p)
    y : ndarray (n,)
    gamma : ndarray (p,) — inclusion probabilities from sparsevb

    Returns
    -------
    selected_aic : ndarray of selected variable indices (AIC)
    selected_bic : ndarray of selected variable indices (BIC)
    info : dict with AIC/BIC curves
    """
    n = X.shape[0]
    order = np.argsort(-gamma)  # descending gamma

    aic_values = []
    bic_values = []

    k_max = min(len(order), n // 2)  # cap to avoid OLS overfitting near k~n
    for k in range(1, k_max + 1):
        idx = order[:k]
        X_k = X[:, idx]
        # OLS via least squares
        coef, residuals, _, _ = np.linalg.lstsq(X_k, y, rcond=None)
        rss = np.sum((y - X_k @ coef) ** 2)
        log_rss_n = np.log(rss / n)
        aic = n * log_rss_n + 2 * k
        bic = n * log_rss_n + k * np.log(n)
        aic_values.append(aic)
        bic_values.append(bic)

    aic_values = np.array(aic_values)
    bic_values = np.array(bic_values)

    k_aic = np.argmin(aic_values) + 1
    k_bic = np.argmin(bic_values) + 1

    selected_aic = np.sort(order[:k_aic])
    selected_bic = np.sort(order[:k_bic])

    return selected_aic, selected_bic, {
        "aic_values": aic_values,
        "bic_values": bic_values,
        "k_aic": k_aic,
        "k_bic": k_bic,
    }
