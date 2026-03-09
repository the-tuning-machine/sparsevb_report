"""
Correlated groups: needle in a haystack
========================================
G groups of B variables with Toeplitz intra-group correlation (rho^|i-j|).
In each active group only one variable is truly causal; the B-1 others
are correlated decoys. We vary rho from 0 to 0.99 and measure whether
sparsevb recovers exactly the right variable in each group.
"""

import numpy as np
from scipy.linalg import toeplitz, cholesky
from sparsevb import svb_fit_linear
import matplotlib.pyplot as plt
from collections import defaultdict


def generate_correlated_groups(n, G, B, rho, active_per_group, signal_strength,
                               seed=None):
    rng = np.random.default_rng(seed)
    p = G * B
    X = np.zeros((n, p))

    row = rho ** np.arange(B)
    Sigma = toeplitz(row)
    L = cholesky(Sigma, lower=True)

    for g in range(G):
        Z = rng.standard_normal((n, B))
        X[:, g * B:(g + 1) * B] = Z @ L.T

    beta_true = np.zeros(p)
    active_groups = sorted(rng.choice(G, size=active_per_group, replace=False))
    true_active_indices = []
    for g in active_groups:
        idx = g * B
        beta_true[idx] = signal_strength
        true_active_indices.append(idx)

    y = X @ beta_true + rng.standard_normal(n)
    return X, y, beta_true, active_groups, true_active_indices


def run_experiment(n, G, B, rho_values, active_per_group, signal_strength, n_rep=50):
    results = defaultdict(list)

    for rho in rho_values:
        recalls, precisions = [], []
        tp_counts, fp_counts, fn_counts, l2_errors = [], [], [], []

        for rep in range(n_rep):
            X, y, beta_true, _, true_active_indices = generate_correlated_groups(
                n, G, B, rho, active_per_group, signal_strength,
                seed=rep * 1000 + int(rho * 100)
            )
            res = svb_fit_linear(X, y, noise_sd=1.0)
            estimated = res['mu'] * res['gamma']
            selected = set(np.where(res['gamma'] > 0.5)[0])
            true_support = set(true_active_indices)

            l2_errors.append(np.linalg.norm(estimated - beta_true))
            tp = len(selected & true_support)
            fp = len(selected - true_support)
            fn = len(true_support - selected)
            tp_counts.append(tp)
            fp_counts.append(fp)
            fn_counts.append(fn)
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)

        results['rho'].append(rho)
        results['l2_mean'].append(np.mean(l2_errors))
        results['l2_std'].append(np.std(l2_errors))
        results['recall_mean'].append(np.mean(recalls))
        results['recall_std'].append(np.std(recalls))
        results['precision_mean'].append(np.mean(precisions))
        results['precision_std'].append(np.std(precisions))
        results['tp_mean'].append(np.mean(tp_counts))
        results['fp_mean'].append(np.mean(fp_counts))
        results['fn_mean'].append(np.mean(fn_counts))
        results['tp_rate'].append(np.mean(tp_counts) / active_per_group)
        results['fp_rate'].append(np.mean(fp_counts) / active_per_group)
        results['fn_rate'].append(np.mean(fn_counts) / active_per_group)

    return results


def clip_yerr(means, stds, lo=None, hi=None):
    means, stds = np.asarray(means), np.asarray(stds)
    lower = stds if lo is None else np.minimum(stds, means - lo)
    upper = stds if hi is None else np.minimum(stds, hi - means)
    return [lower, upper]


def plot_results(results, active_per_group):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    rho = results['rho']

    ax = axes[0, 0]
    ax.errorbar(rho, results['recall_mean'],
                yerr=clip_yerr(results['recall_mean'], results['recall_std'], 0, 1),
                marker='o', capsize=3, color='green')
    ax.set_xlabel(r'$\rho$'); ax.set_ylabel('Recall')
    ax.set_title('Recall (TPR)'); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.errorbar(rho, results['precision_mean'],
                yerr=clip_yerr(results['precision_mean'], results['precision_std'], 0, 1),
                marker='s', capsize=3, color='steelblue')
    ax.set_xlabel(r'$\rho$'); ax.set_ylabel('Precision')
    ax.set_title('Precision'); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.errorbar(rho, results['l2_mean'],
                yerr=clip_yerr(results['l2_mean'], results['l2_std'], 0),
                marker='D', capsize=3, color='orange')
    ax.set_xlabel(r'$\rho$'); ax.set_ylabel(r'$\ell_2$ error')
    ax.set_title(r'$\ell_2$ error'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    x = np.arange(len(rho))
    width = 0.25
    ax.bar(x - width, results['tp_rate'], width, label='TP', color='seagreen', alpha=0.8)
    ax.bar(x, results['fp_rate'], width, label='FP', color='indianred', alpha=0.8)
    ax.bar(x + width, results['fn_rate'], width, label='FN', color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r:.2f}' for r in rho], rotation=45, ha='right')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(f'Rate (/ {active_per_group} active)')
    ax.set_title('TP / FP / FN rates'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Correlated groups: needle in a haystack\n'
                 f'(Toeplitz correlation, {len(rho)} values of rho)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('1_correlated_groups.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    n, G, B = 200, 200, 10
    active_per_group = 5
    signal_strength = 3.0
    rho_values = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
    n_rep = 50

    print(f"n={n}, p={G*B}, G={G} groups of {B} (1 active + {B-1} decoys)")
    print(f"{active_per_group} active groups, signal={signal_strength}")
    print(f"rho = {rho_values}, {n_rep} reps\n")

    results = run_experiment(n, G, B, rho_values, active_per_group,
                             signal_strength, n_rep=n_rep)

    print(f"{'rho':>6} | {'l2-err':>10} | {'Recall':>10} | {'Precision':>10} | "
          f"{'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 70)
    for i, rho in enumerate(results['rho']):
        print(f"{rho:6.2f} | "
              f"{results['l2_mean'][i]:4.2f}+/-{results['l2_std'][i]:4.2f} | "
              f"{results['recall_mean'][i]:5.2f}+/-{results['recall_std'][i]:4.2f} | "
              f"{results['precision_mean'][i]:5.2f}+/-{results['precision_std'][i]:4.2f} | "
              f"{results['tp_mean'][i]:5.1f} {results['fp_mean'][i]:5.1f} {results['fn_mean'][i]:5.1f}")

    plot_results(results, active_per_group)

