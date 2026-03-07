"""Cross-validation forward selection on candidates sorted by gamma descending."""
import numpy as np


def select_cv(X, y, gamma, n_folds=5, rng=None):
    """5-fold CV forward selection on top-k variables sorted by gamma.

    Parameters
    ----------
    X : ndarray (n, p)
    y : ndarray (n,)
    gamma : ndarray (p,) — inclusion probabilities from sparsevb
    n_folds : int
    rng : numpy Generator (optional)

    Returns
    -------
    selected : ndarray of selected variable indices
    info : dict with CV curves
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = X.shape[0]
    order = np.argsort(-gamma)

    # Build fold indices
    indices = rng.permutation(n)
    folds = np.array_split(indices, n_folds)

    # Cap k to avoid OLS overfitting when k approaches train set size
    n_train = n - max(len(f) for f in folds)
    k_max = min(len(order), n_train // 2)
    cv_mse = []
    for k in range(1, k_max + 1):
        idx = order[:k]
        fold_errors = []
        for f in range(n_folds):
            val_idx = folds[f]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
            X_tr, y_tr = X[train_idx][:, idx], y[train_idx]
            X_val, y_val = X[val_idx][:, idx], y[val_idx]
            coef, _, _, _ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
            fold_errors.append(np.mean((y_val - X_val @ coef) ** 2))
        cv_mse.append(np.mean(fold_errors))

    cv_mse = np.array(cv_mse)
    k_cv = np.argmin(cv_mse) + 1
    selected = np.sort(order[:k_cv])

    return selected, {"cv_mse": cv_mse, "k_cv": k_cv}
