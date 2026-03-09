"""
Signal strength: effect of SNR on variable selection
=====================================================
We place coefficients of varying magnitudes in a single model and measure
per-coefficient detection rate, estimation bias, and shrinkage.
"""

import numpy as np
from sparsevb import svb_fit_linear
import matplotlib.pyplot as plt
from collections import defaultdict


COEFF_VALUES = np.array([5.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01])


def run_experiment(n, p, coeff_values, noise_sd=1.0, n_rep=50):
    s = len(coeff_values)
    detection_rates = defaultdict(list)
    bias_by_coeff = defaultdict(list)
    gamma_by_coeff = defaultdict(list)
    # Per-rep global metrics
    rel_l2_per_rep = []
    tpr_per_rep = []
    # Per-coefficient shrinkage ratio E[hat_theta_i] / theta_i
    shrinkage_ratio_by_coeff = defaultdict(list)

    for rep in range(n_rep):
        rng = np.random.default_rng(rep)
        X = rng.standard_normal((n, p))
        beta_true = np.zeros(p)
        beta_true[:s] = coeff_values
        y = X @ beta_true + noise_sd * rng.standard_normal(n)

        res = svb_fit_linear(X, y, noise_sd=noise_sd)
        estimated = res['mu'] * res['gamma']
        selected = set(np.where(res['gamma'] > 0.5)[0])
        true_support = set(range(s))

        # Global metrics
        beta_norm = np.linalg.norm(beta_true)
        if beta_norm > 0:
            rel_l2_per_rep.append(np.linalg.norm(estimated - beta_true) / beta_norm)
        tpr_per_rep.append(len(selected & true_support) / len(true_support))

        for i, coeff in enumerate(coeff_values):
            detection_rates[coeff].append(1.0 if res['gamma'][i] > 0.5 else 0.0)
            bias_by_coeff[coeff].append(estimated[i] - coeff)
            gamma_by_coeff[coeff].append(res['gamma'][i])
            if coeff > 0:
                shrinkage_ratio_by_coeff[coeff].append(estimated[i] / coeff)

    per_coeff = {}
    for coeff in coeff_values:
        per_coeff[coeff] = {
            'detection_rate': np.mean(detection_rates[coeff]),
            'bias_mean': np.mean(bias_by_coeff[coeff]),
            'bias_std': np.std(bias_by_coeff[coeff]),
            'shrinkage': np.mean(bias_by_coeff[coeff]) / coeff if coeff > 0 else 0,
            'gamma_mean': np.mean(gamma_by_coeff[coeff]),
            'shrinkage_ratio_mean': np.mean(shrinkage_ratio_by_coeff[coeff]) if coeff > 0 else 0,
            'shrinkage_ratio_std': np.std(shrinkage_ratio_by_coeff[coeff]) if coeff > 0 else 0,
        }

    global_metrics = {
        'rel_l2_mean': np.mean(rel_l2_per_rep),
        'rel_l2_std': np.std(rel_l2_per_rep),
        'tpr_mean': np.mean(tpr_per_rep),
        'tpr_std': np.std(tpr_per_rep),
    }

    return per_coeff, global_metrics


def clip_yerr(means, stds, lo=None, hi=None):
    means, stds = np.asarray(means), np.asarray(stds)
    lower = stds if lo is None else np.minimum(stds, means - lo)
    upper = stds if hi is None else np.minimum(stds, hi - means)
    return [lower, upper]


def plot_results(coeff_values, per_coeff):
    """2x2 figure: rel l2 error, TPR, shrinkage ratio, relative shrinkage — all per coefficient."""
    coeffs = list(coeff_values)
    x_vals = np.array(coeffs)

    rel_l2 = [np.linalg.norm(per_coeff[c]['bias_mean']) / c if c > 0 else 0 for c in coeffs]
    tprs = [per_coeff[c]['detection_rate'] for c in coeffs]
    tpr_stds = [np.sqrt(per_coeff[c]['detection_rate'] * (1 - per_coeff[c]['detection_rate'])) for c in coeffs]
    shrink_means = [per_coeff[c]['shrinkage_ratio_mean'] for c in coeffs]
    shrink_stds = [per_coeff[c]['shrinkage_ratio_std'] for c in coeffs]
    rel_shrinkages = [per_coeff[c]['shrinkage'] for c in coeffs]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # [0,0] Relative l2 error per coefficient (use |bias| / theta as proxy)
    ax = axes[0, 0]
    bias_means = [abs(per_coeff[c]['bias_mean']) for c in coeffs]
    bias_stds = [per_coeff[c]['bias_std'] for c in coeffs]
    rel_err = [bias_means[i] / coeffs[i] if coeffs[i] > 0 else 0 for i in range(len(coeffs))]
    rel_err_std = [bias_stds[i] / coeffs[i] if coeffs[i] > 0 else 0 for i in range(len(coeffs))]
    ax.errorbar(x_vals, rel_err,
                yerr=clip_yerr(rel_err, rel_err_std, 0),
                marker='o', capsize=3)
    ax.set_xlabel('Signal strength'); ax.set_ylabel(r'Relative $\ell_2$ error')
    ax.set_title(r'$|\hat\theta_i - \theta_i| / \theta_i$')
    ax.set_xscale('log'); ax.grid(True, alpha=0.3)

    # [0,1] TPR (detection rate)
    ax = axes[0, 1]
    ax.errorbar(x_vals, tprs,
                yerr=clip_yerr(tprs, tpr_stds, 0, 1),
                marker='s', capsize=3, color='green')
    ax.set_xlabel('Signal strength'); ax.set_ylabel('Detection rate')
    ax.set_title('Detection rate ($\\gamma_i > 0.5$)')
    ax.set_ylim(-0.05, 1.05); ax.set_xscale('log'); ax.grid(True, alpha=0.3)

    # [1,0] Shrinkage ratio E[hat_theta] / theta
    ax = axes[1, 0]
    ax.errorbar(x_vals, shrink_means,
                yerr=clip_yerr(shrink_means, shrink_stds, 0),
                marker='v', capsize=3, color='orange')
    ax.set_xlabel('Signal strength'); ax.set_ylabel('Shrinkage ratio')
    ax.set_title(r'Shrinkage: $\mathbb{E}[\hat\theta_i] / \theta_i$')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No shrinkage')
    ax.set_ylim(-0.1, 1.1); ax.set_xscale('log'); ax.legend(); ax.grid(True, alpha=0.3)

    # [1,1] Relative shrinkage bar chart (ascending order)
    ax = axes[1, 1]
    coeffs_asc = coeffs[::-1]
    rel_shrinkages_asc = rel_shrinkages[::-1]
    x = np.arange(len(coeffs_asc))
    ax.bar(x, rel_shrinkages_asc, color='orange', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}' for c in coeffs_asc], rotation=45, ha='right')
    ax.set_xlabel(r'Coefficient value $\theta_i$')
    ax.set_ylabel('Relative shrinkage')
    ax.set_title(r'Relative shrinkage: $(\mathbb{E}[\hat\theta_i] - \theta_i) / \theta_i$')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Effect of signal strength on SparsevB\n'
                 '(variable-sized coefficients in a single model)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('4_signal_strength.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_compact(coeff_values, per_coeff):
    """Figure 2: relative l2 error + relative shrinkage bar chart, no title."""
    coeffs = list(coeff_values)
    x_vals = np.array(coeffs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Relative error per coefficient
    ax = axes[0]
    bias_means = [abs(per_coeff[c]['bias_mean']) for c in coeffs]
    bias_stds = [per_coeff[c]['bias_std'] for c in coeffs]
    rel_err = [bias_means[i] / coeffs[i] if coeffs[i] > 0 else 0 for i in range(len(coeffs))]
    rel_err_std = [bias_stds[i] / coeffs[i] if coeffs[i] > 0 else 0 for i in range(len(coeffs))]
    ax.errorbar(x_vals, rel_err,
                yerr=clip_yerr(rel_err, rel_err_std, 0),
                marker='o', capsize=3)
    ax.set_xlabel('Signal strength'); ax.set_ylabel(r'Relative $\ell_2$ error')
    ax.set_title(r'$|\hat\theta_i - \theta_i| / \theta_i$')
    ax.set_xscale('log'); ax.grid(True, alpha=0.3)

    # Relative shrinkage bar chart (ascending order)
    ax = axes[1]
    coeffs_asc = coeffs[::-1]
    rel_shrinkages = [per_coeff[c]['shrinkage'] for c in coeffs_asc]
    x = np.arange(len(coeffs_asc))
    ax.bar(x, rel_shrinkages, color='orange', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}' for c in coeffs_asc], rotation=45, ha='right')
    ax.set_xlabel(r'Coefficient value $\theta_i$')
    ax.set_ylabel('Relative shrinkage')
    ax.set_title(r'Relative shrinkage: $(\mathbb{E}[\hat\theta_i] - \theta_i) / \theta_i$')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('4_signal_strength_compact.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    n, p = 200, 400
    noise_sd = 1.0
    n_rep = 50

    print(f"n={n}, p={p}, s={len(COEFF_VALUES)}, noise_sd={noise_sd}, {n_rep} reps")
    print(f"coefficients = {list(COEFF_VALUES)}\n")

    per_coeff, global_metrics = run_experiment(n, p, COEFF_VALUES, noise_sd, n_rep)

    print(f"Global: rel_l2 = {global_metrics['rel_l2_mean']:.4f} +/- {global_metrics['rel_l2_std']:.4f}, "
          f"TPR = {global_metrics['tpr_mean']:.3f} +/- {global_metrics['tpr_std']:.3f}\n")

    print(f"{'coeff':>7} | {'detection':>10} | {'bias':>14} | {'shrinkage':>10} | {'gamma':>8}")
    print("-" * 65)
    for c in COEFF_VALUES:
        r = per_coeff[c]
        print(f"{c:7.2f} | {r['detection_rate']:10.2f} | "
              f"{r['bias_mean']:+6.4f} +/- {r['bias_std']:5.4f} | "
              f"{r['shrinkage']:+8.3f} | {r['gamma_mean']:8.3f}")

    plot_results(COEFF_VALUES, per_coeff)
    plot_compact(COEFF_VALUES, per_coeff)
