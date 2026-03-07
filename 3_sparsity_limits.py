"""
Sparsity limits: what happens when s -> n?
==========================================
The theory (Ray & Szabo, 2022) assumes s = o(n). We test the behavior
of sparsevb when progressively increasing s/n from very sparse toward 1.

Experiment 1: fixed n, increasing s.
Experiment 2: fixed s/n ratios, increasing n (scaling behavior).
"""

import numpy as np
from sparsevb import svb_fit_linear
import matplotlib.pyplot as plt
from collections import defaultdict


def generate_data(n, p, s, signal_strength, noise_sd=1.0, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta_true = np.zeros(p)
    active_idx = rng.choice(p, size=s, replace=False)
    beta_true[active_idx] = signal_strength * rng.choice([-1, 1], size=s)
    y = X @ beta_true + noise_sd * rng.standard_normal(n)
    return X, y, beta_true


def run_experiment(n, p, s_values, signal_strength, noise_sd=1.0, n_rep=30):
    results = defaultdict(list)
    for s in s_values:
        l2_errors, tprs, fprs = [], [], []
        for rep in range(n_rep):
            X, y, beta_true = generate_data(n, p, s, signal_strength, noise_sd,
                                            seed=rep * 1000 + s)
            true_support = set(np.where(beta_true != 0)[0])
            res = svb_fit_linear(X, y, noise_sd=noise_sd)
            estimated = res['mu'] * res['gamma']
            selected = set(np.where(res['gamma'] > 0.5)[0])

            l2_errors.append(np.linalg.norm(estimated - beta_true))
            if len(true_support) > 0:
                tprs.append(len(selected & true_support) / len(true_support))
            n_true_neg = p - len(true_support)
            if n_true_neg > 0:
                fprs.append(len(selected - true_support) / n_true_neg)

        results['s'].append(s)
        results['s_over_n'].append(s / n)
        results['l2_mean'].append(np.mean(l2_errors))
        results['l2_std'].append(np.std(l2_errors))
        results['tpr_mean'].append(np.mean(tprs) if tprs else 0)
        results['tpr_std'].append(np.std(tprs) if tprs else 0)
        results['fpr_mean'].append(np.mean(fprs) if fprs else 0)
        results['fpr_std'].append(np.std(fprs) if fprs else 0)
    return results


def run_scaling_experiment(n_values, p_factor, s_fractions, signal_strength,
                           noise_sd=1.0, n_rep=30):
    results = defaultdict(lambda: defaultdict(list))
    for s_frac in s_fractions:
        for n in n_values:
            p = int(p_factor * n)
            s = max(1, int(s_frac * n))
            if s >= p:
                continue
            l2_errors, tprs = [], []
            for rep in range(n_rep):
                X, y, beta_true = generate_data(n, p, s, signal_strength, noise_sd,
                                                seed=rep * 1000 + n + s)
                true_support = set(np.where(beta_true != 0)[0])
                res = svb_fit_linear(X, y, noise_sd=noise_sd)
                estimated = res['mu'] * res['gamma']
                selected = set(np.where(res['gamma'] > 0.5)[0])
                l2_errors.append(np.linalg.norm(estimated - beta_true))
                if len(true_support) > 0:
                    tprs.append(len(selected & true_support) / len(true_support))

            results[s_frac]['n'].append(n)
            results[s_frac]['l2_mean'].append(np.mean(l2_errors))
            results[s_frac]['l2_std'].append(np.std(l2_errors))
            results[s_frac]['tpr_mean'].append(np.mean(tprs) if tprs else 0)
    return results


def clip_yerr(means, stds, lo=None, hi=None):
    means, stds = np.asarray(means), np.asarray(stds)
    lower = stds if lo is None else np.minimum(stds, means - lo)
    upper = stds if hi is None else np.minimum(stds, hi - means)
    return [lower, upper]


def plot_sparsity_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    s_over_n = results['s_over_n']

    ax = axes[0]
    ax.errorbar(s_over_n, results['l2_mean'],
                yerr=clip_yerr(results['l2_mean'], results['l2_std'], 0),
                marker='o', capsize=3)
    ax.set_xlabel('s / n'); ax.set_ylabel(r'$\ell_2$ error')
    ax.set_title(r'$\ell_2$ error'); ax.grid(True, alpha=0.3)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='s/n = 0.5')
    ax.legend()

    ax = axes[1]
    ax.errorbar(s_over_n, results['tpr_mean'],
                yerr=clip_yerr(results['tpr_mean'], results['tpr_std'], 0, 1),
                marker='s', capsize=3, color='green')
    ax.set_xlabel('s / n'); ax.set_ylabel('True positive rate')
    ax.set_title('True positive rate'); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    ax = axes[2]
    ax.errorbar(s_over_n, results['fpr_mean'],
                yerr=clip_yerr(results['fpr_mean'], results['fpr_std'], 0, 1),
                marker='^', capsize=3, color='red')
    ax.set_xlabel('s / n'); ax.set_ylabel('False positive rate')
    ax.set_title('False positive rate: FP / (p - s)')
    ax.set_ylim(-0.05, max(0.2, max(results['fpr_mean']) * 1.2)); ax.grid(True, alpha=0.3)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    n = int(results['s'][0] / results['s_over_n'][0]) if results['s_over_n'][0] > 0 else 0
    fig.suptitle(f'Effect of sparsity s/n on SparsevB (n={n}, p={int(n*2)})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('3_sparsity_limits.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_scaling_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for s_frac, data in sorted(results.items()):
        label = f's/n = {s_frac}'
        axes[0].plot(data['n'], data['l2_mean'], marker='o', label=label)
        axes[1].plot(data['n'], data['tpr_mean'], marker='s', label=label)

    axes[0].set_xlabel('n'); axes[0].set_ylabel(r'$\ell_2$ error')
    axes[0].set_title(r'$\ell_2$ error vs. n (p = 2n)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('n'); axes[1].set_ylabel('True positive rate')
    axes[1].set_title('True positive rate vs. n (p = 2n)')
    axes[1].set_ylim(-0.05, 1.05); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle('Scaling: performance for fixed s/n as n increases',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('3_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    n, p = 100, 200
    signal_strength = 3.0
    noise_sd = 1.0
    n_rep = 30
    s_values = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]

    print(f"Experiment 1: n={n}, p={p}, signal={signal_strength}")
    print(f"s = {s_values}, {n_rep} reps\n")

    results = run_experiment(n, p, s_values, signal_strength, noise_sd, n_rep)

    print(f"{'s':>4} {'s/n':>6} | {'l2-error':>14} | {'TP rate':>12} | {'FP rate':>12}")
    print("-" * 60)
    for i in range(len(results['s'])):
        print(f"{results['s'][i]:4d} {results['s_over_n'][i]:6.2f} | "
              f"{results['l2_mean'][i]:5.2f} +/- {results['l2_std'][i]:5.2f}  | "
              f"{results['tpr_mean'][i]:4.2f} +/- {results['tpr_std'][i]:4.2f} | "
              f"{results['fpr_mean'][i]:4.3f} +/- {results['fpr_std'][i]:4.3f}")

    plot_sparsity_results(results)

    print("\n--- Experiment 2: scaling with n ---")
    n_values = [50, 100, 150, 200, 300]
    s_fractions = [0.05, 0.1, 0.2, 0.4]

    scaling_results = run_scaling_experiment(
        n_values, 2, s_fractions, signal_strength, noise_sd, 20)

    for s_frac, data in sorted(scaling_results.items()):
        print(f"\ns/n = {s_frac}:")
        for i, nv in enumerate(data['n']):
            print(f"  n={nv:4d}, p={2*nv:4d}, s={int(s_frac*nv):3d} : "
                  f"l2={data['l2_mean'][i]:.2f}, TP rate={data['tpr_mean'][i]:.2f}")

    plot_scaling_results(scaling_results)
