"""
Signal strength: effect of SNR on variable selection
=====================================================
We test sparsevb with decreasing signal strength (fixed noise) and
measure relative l2 error, TPR, shrinkage, and per-coefficient detection.

Experiment 1: uniform signal strength, varying from 10 to 0.05.
Experiment 2: variable-sized coefficients in the same model (5.0 to 0.01).
"""

import numpy as np
from sparsevb import svb_fit_linear
import matplotlib.pyplot as plt
from collections import defaultdict


def generate_data(n, p, s, signal_strength, noise_sd=1.0, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta_true = np.zeros(p)
    beta_true[:s] = signal_strength
    y = X @ beta_true + noise_sd * rng.standard_normal(n)
    return X, y, beta_true


def run_experiment(n, p, s, signal_values, noise_sd=1.0, n_rep=50):
    results = defaultdict(list)
    for signal in signal_values:
        snr = signal**2 * s / noise_sd**2
        rel_l2s, tprs, shrinkages = [], [], []
        gamma_actives, gamma_inactives = [], []

        for rep in range(n_rep):
            X, y, beta_true = generate_data(n, p, s, signal, noise_sd, seed=rep)
            true_support = set(np.where(beta_true != 0)[0])
            res = svb_fit_linear(X, y, noise_sd=noise_sd)
            estimated = res['mu'] * res['gamma']
            selected = set(np.where(res['gamma'] > 0.5)[0])

            beta_norm = np.linalg.norm(beta_true)
            if beta_norm > 0:
                rel_l2s.append(np.linalg.norm(estimated - beta_true) / beta_norm)
            tprs.append(len(selected & true_support) / len(true_support))
            active_mask = beta_true != 0
            if signal > 0:
                shrinkages.append(np.mean(estimated[active_mask]) / signal)
            gamma_actives.append(np.mean(res['gamma'][active_mask]))
            gamma_inactives.append(np.mean(res['gamma'][~active_mask]))

        results['signal'].append(signal)
        results['snr'].append(snr)
        results['rel_l2_mean'].append(np.mean(rel_l2s) if rel_l2s else 0)
        results['rel_l2_std'].append(np.std(rel_l2s) if rel_l2s else 0)
        results['tpr_mean'].append(np.mean(tprs))
        results['tpr_std'].append(np.std(tprs))
        results['shrinkage_mean'].append(np.mean(shrinkages) if shrinkages else 0)
        results['shrinkage_std'].append(np.std(shrinkages) if shrinkages else 0)
        results['gamma_active_mean'].append(np.mean(gamma_actives))
        results['gamma_active_std'].append(np.std(gamma_actives))
        results['gamma_inactive_mean'].append(np.mean(gamma_inactives))
        results['gamma_inactive_std'].append(np.std(gamma_inactives))
    return results


def run_variable_coefficients(n, p, s, noise_sd=1.0, n_rep=50):
    coeff_values = np.array([5.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01])
    s_actual = len(coeff_values)
    detection_rates = defaultdict(list)
    bias_by_coeff = defaultdict(list)

    for rep in range(n_rep):
        rng = np.random.default_rng(rep)
        X = rng.standard_normal((n, p))
        beta_true = np.zeros(p)
        beta_true[:s_actual] = coeff_values
        y = X @ beta_true + noise_sd * rng.standard_normal(n)

        res = svb_fit_linear(X, y, noise_sd=noise_sd)
        estimated = res['mu'] * res['gamma']
        for i, coeff in enumerate(coeff_values):
            detection_rates[coeff].append(1.0 if res['gamma'][i] > 0.5 else 0.0)
            bias_by_coeff[coeff].append(estimated[i] - coeff)

    results = {}
    for coeff in coeff_values:
        results[coeff] = {
            'detection_rate': np.mean(detection_rates[coeff]),
            'bias_mean': np.mean(bias_by_coeff[coeff]),
            'bias_std': np.std(bias_by_coeff[coeff]),
            'shrinkage': np.mean(bias_by_coeff[coeff]) / coeff if coeff > 0 else 0,
        }
    return coeff_values, results


def clip_yerr(means, stds, lo=None, hi=None):
    means, stds = np.asarray(means), np.asarray(stds)
    lower = stds if lo is None else np.minimum(stds, means - lo)
    upper = stds if hi is None else np.minimum(stds, hi - means)
    return [lower, upper]


def plot_snr_results(results, coeff_values=None, var_results=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    signal = results['signal']

    ax = axes[0, 0]
    ax.errorbar(signal, results['rel_l2_mean'],
                yerr=clip_yerr(results['rel_l2_mean'], results['rel_l2_std'], 0),
                marker='o', capsize=3)
    ax.set_xlabel('Signal strength'); ax.set_ylabel(r'Relative $\ell_2$ error')
    ax.set_title(r'$\|\hat\theta - \theta_0\|_2 / \|\theta_0\|_2$')
    ax.set_xscale('log'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.errorbar(signal, results['tpr_mean'],
                yerr=clip_yerr(results['tpr_mean'], results['tpr_std'], 0, 1),
                marker='s', capsize=3, color='green')
    ax.set_xlabel('Signal strength'); ax.set_ylabel('True positive rate')
    ax.set_title('True positive rate (Recall)')
    ax.set_ylim(-0.05, 1.05); ax.set_xscale('log'); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.errorbar(signal, results['shrinkage_mean'],
                yerr=clip_yerr(results['shrinkage_mean'], results['shrinkage_std'], 0),
                marker='v', capsize=3, color='orange')
    ax.set_xlabel('Signal strength'); ax.set_ylabel('Shrinkage ratio')
    ax.set_title(r'Shrinkage: $\mathbb{E}[\hat\theta] / \theta_0$')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No shrinkage')
    ax.set_ylim(-0.1, 1.5); ax.set_xscale('log'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if coeff_values is not None and var_results is not None:
        coeffs = list(coeff_values)
        shrinkages = [var_results[c]['shrinkage'] for c in coeffs]
        ax.bar(range(len(coeffs)), shrinkages, color='orange', alpha=0.8)
        ax.set_xticks(range(len(coeffs)))
        ax.set_xticklabels([f'{c}' for c in coeffs], rotation=45, ha='right')
        ax.set_xlabel(r'Coefficient value $\theta_i$')
        ax.set_ylabel('Relative shrinkage')
        ax.set_title(r'Relative shrinkage: bias / $\theta_i$')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.set_visible(False)

    fig.suptitle('Effect of signal-to-noise ratio on SparsevB\n'
                 '(decreasing signal, fixed noise)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('4_signal_strength.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    n, p, s = 200, 400, 10
    noise_sd = 1.0
    n_rep = 50
    signal_values = [10.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05]

    print(f"Experiment 1: n={n}, p={p}, s={s}, noise_sd={noise_sd}, {n_rep} reps")
    print(f"signal = {signal_values}\n")

    results = run_experiment(n, p, s, signal_values, noise_sd, n_rep)

    print(f"{'signal':>7} {'SNR':>6} | {'rel-l2':>12} | {'TP rate':>10} | "
          f"{'shrink':>10} | {'g_act':>8} {'g_inact':>8}")
    print("-" * 85)
    for i in range(len(results['signal'])):
        print(f"{results['signal'][i]:7.2f} {results['snr'][i]:6.1f} | "
              f"{results['rel_l2_mean'][i]:5.3f} +/- {results['rel_l2_std'][i]:4.3f} | "
              f"{results['tpr_mean'][i]:4.2f} +/- {results['tpr_std'][i]:4.2f} | "
              f"{results['shrinkage_mean'][i]:5.3f} +/- {results['shrinkage_std'][i]:4.3f} | "
              f"{results['gamma_active_mean'][i]:5.3f}  {results['gamma_inactive_mean'][i]:6.4f}")

    print(f"\nExperiment 2: variable-sized coefficients")
    coeff_values, var_results = run_variable_coefficients(n, p, s, noise_sd, n_rep)

    print(f"{'coeff':>7} | {'detection':>10} | {'bias':>14} | {'shrinkage':>10}")
    print("-" * 55)
    for c in coeff_values:
        r = var_results[c]
        print(f"{c:7.2f} | {r['detection_rate']:10.2f} | "
              f"{r['bias_mean']:+6.4f} +/- {r['bias_std']:5.4f} | "
              f"{r['shrinkage']:+8.3f}")

    plot_snr_results(results, coeff_values, var_results)
