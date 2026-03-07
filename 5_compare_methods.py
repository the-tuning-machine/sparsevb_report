"""
Compare variable selection methods
===================================
Compare sparsevb (gamma > 0.5 threshold) against classical post-hoc
selection methods: AIC, BIC, Cross-Validation, Knockoffs, and
Stability Selection on a synthetic sparse linear problem.
"""

import numpy as np
from sparsevb import svb_fit_linear
from selection_aic_bic import select_aic_bic
from selection_cv import select_cv
from selection_knockoffs import select_knockoffs
from selection_stability import select_stability

rng = np.random.default_rng(42)
n, p, s = 100, 200, 5

X = rng.standard_normal((n, p))
beta_true = np.zeros(p)
beta_true[:s] = [3.0, -2.0, 1.5, -1.0, 0.5]

noise_sd = 1.0
y = X @ beta_true + noise_sd * rng.standard_normal(n)
true_support = set(np.where(beta_true != 0)[0])

res = svb_fit_linear(X, y, noise_sd=noise_sd)
gamma = res["gamma"]
selected_baseline = np.where(gamma > 0.5)[0]

print("Running AIC/BIC forward selection...")
sel_aic, sel_bic, info_ab = select_aic_bic(X, y, gamma)

print("Running cross-validation...")
sel_cv, info_cv = select_cv(X, y, gamma, rng=np.random.default_rng(0))

print("Running knockoff filter...")
sel_knock, info_knock = select_knockoffs(X, y, gamma, target_fdr=0.1,
                                         rng=np.random.default_rng(1))

print("Running stability selection (B=50)...")
sel_stab, info_stab = select_stability(X, y, B=50, rng=np.random.default_rng(2))

methods = {
    "gamma>0.5": set(selected_baseline),
    "AIC":       set(sel_aic),
    "BIC":       set(sel_bic),
    "CV":        set(sel_cv),
    "Knockoffs": set(sel_knock),
    "Stability": set(sel_stab),
}

print("\n" + "=" * 72)
print(f"{'Method':<12} | {'Selected':>8} | {'TP':>3} | {'FP':>3} | {'FN':>3} | Variables")
print("-" * 72)
for name, sel in methods.items():
    tp = len(sel & true_support)
    fp = len(sel - true_support)
    fn = len(true_support - sel)
    vars_str = str(sorted(int(v) for v in sel)) if sel else "[]"
    print(f"{name:<12} | {len(sel):>8} | {tp:>3} | {fp:>3} | {fn:>3} | {vars_str}")
print("=" * 72)
print(f"True support: {sorted(int(v) for v in true_support)}")
