"""
Robustness to non-Gaussian distributions
=========================================
We test sparsevb when the Gaussian assumption is violated on three axes:
  - Noise distribution (X=Gaussian, theta=constant)
  - Design matrix distribution (noise=Gaussian, theta=constant)
  - Coefficient distribution (X=Gaussian, noise=Gaussian)

Each axis is tested with 7 distributions:
  Gaussian, Student(1), Student(2), Student(3), Laplace, Cauchy, Uniform.
"""

import numpy as np
from sparsevb import svb_fit_linear
import matplotlib.pyplot as plt


DIST_LABELS = {
    "gaussian": "Gaussian",
    "student1": "Student(1)",
    "student2": "Student(2)",
    "student3": "Student(3)",
    "laplace": "Laplace",
    "cauchy": "Cauchy",
    "uniform": "Uniform",
}

DIST_NAMES = list(DIST_LABELS.keys())


def sample_distribution(rng, dist_name, size, scale=1.0):
    if dist_name == "gaussian":
        return scale * rng.standard_normal(size)
    elif dist_name.startswith("student"):
        df = int(dist_name.replace("student", ""))
        return scale * rng.standard_t(df=df, size=size)
    elif dist_name == "laplace":
        return rng.laplace(scale=scale, size=size)
    elif dist_name == "cauchy":
        return scale * rng.standard_cauchy(size)
    elif dist_name == "uniform":
        return rng.uniform(-scale * np.sqrt(3), scale * np.sqrt(3), size=size)
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")


def noise_sd_for(dist_name, scale):
    if dist_name == "gaussian":
        return scale
    elif dist_name == "laplace":
        return scale * np.sqrt(2)
    elif dist_name == "uniform":
        return scale
    elif dist_name.startswith("student"):
        df = int(dist_name.replace("student", ""))
        return scale * np.sqrt(df / (df - 2)) if df > 2 else scale
    else:
        return scale


def generate_data(n, p, s, signal_strength, noise_dist="gaussian",
                  noise_scale=1.0, design_dist="gaussian",
                  theta_dist="constant", seed=None):
    rng = np.random.default_rng(seed)
    X = sample_distribution(rng, design_dist, (n, p), scale=1.0)

    beta_true = np.zeros(p)
    active_idx = rng.choice(p, size=s, replace=False)
    if theta_dist == "constant":
        beta_true[active_idx] = signal_strength * rng.choice([-1, 1], size=s)
    else:
        beta_true[active_idx] = sample_distribution(rng, theta_dist, s, scale=signal_strength)

    noise = sample_distribution(rng, noise_dist, n, scale=noise_scale)
    y = X @ beta_true + noise
    return X, y, beta_true


def evaluate(X, y, beta_true, noise_sd=None):
    res = svb_fit_linear(X, y, noise_sd=noise_sd)
    estimated = res['mu'] * res['gamma']
    selected = set(np.where(res['gamma'] > 0.5)[0])
    true_support = set(np.where(beta_true != 0)[0])

    l2_err = np.linalg.norm(estimated - beta_true)
    tp = len(selected & true_support)
    fp = len(selected - true_support)
    tpr = tp / max(len(true_support), 1)
    fdr = fp / max(len(selected), 1)
    return l2_err, tpr, fdr


def run_single_config(n, p, s, signal_strength, noise_dist, noise_scale,
                      design_dist, theta_dist, n_rep):
    l2s, tprs, fdrs = [], [], []
    true_sd = noise_sd_for(noise_dist, noise_scale)
    for rep in range(n_rep):
        X, y, beta_true = generate_data(
            n, p, s, signal_strength,
            noise_dist=noise_dist, noise_scale=noise_scale,
            design_dist=design_dist, theta_dist=theta_dist, seed=rep
        )
        l2, tpr, fdr = evaluate(X, y, beta_true, noise_sd=true_sd)
        l2s.append(l2); tprs.append(tpr); fdrs.append(fdr)

    return {
        'l2': (np.mean(l2s), np.std(l2s)),
        'tpr': (np.mean(tprs), np.std(tprs)),
        'fdr': (np.mean(fdrs), np.std(fdrs)),
    }


def experiment_vary_noise(n, p, s, signal_strength, noise_scale, n_rep):
    results = {}
    for dist in DIST_NAMES:
        print(f"  Noise = {DIST_LABELS[dist]}...")
        results[DIST_LABELS[dist]] = run_single_config(
            n, p, s, signal_strength, noise_dist=dist,
            noise_scale=noise_scale, design_dist="gaussian",
            theta_dist="constant", n_rep=n_rep)
    return results


def experiment_vary_design(n, p, s, signal_strength, noise_scale, n_rep):
    results = {}
    for dist in DIST_NAMES:
        print(f"  Design = {DIST_LABELS[dist]}...")
        results[DIST_LABELS[dist]] = run_single_config(
            n, p, s, signal_strength, noise_dist="gaussian",
            noise_scale=noise_scale, design_dist=dist,
            theta_dist="constant", n_rep=n_rep)
    return results


def experiment_vary_theta(n, p, s, signal_strength, noise_scale, n_rep):
    results = {}
    for dist in DIST_NAMES:
        print(f"  Theta = {DIST_LABELS[dist]}...")
        results[DIST_LABELS[dist]] = run_single_config(
            n, p, s, signal_strength, noise_dist="gaussian",
            noise_scale=noise_scale, design_dist="gaussian",
            theta_dist=dist, n_rep=n_rep)
    return results


def plot_comparison(results_noise, results_design, results_theta,
                    n, p, s, signal_strength):
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))

    experiments = [
        ("Noise distribution",
         results_noise,
         r"$X \sim \mathcal{N}(0,1)$, $\theta_i = \pm 3$"),
        ("Design distribution",
         results_design,
         r"Noise $\sim \mathcal{N}(0,1)$, $\theta_i = \pm 3$"),
        (r"$\theta$ distribution",
         results_theta,
         r"$X \sim \mathcal{N}(0,1)$, Noise $\sim \mathcal{N}(0,1)$, scale$=3$"),
    ]
    metrics = ['l2', 'tpr', 'fdr']
    metric_names = [r'$\ell_2$ error', 'True positive rate', 'False discovery rate']
    colors = ['#4c72b0', '#dd8452', '#937860', '#c49555',
              '#55a868', '#c44e52', '#8172b3']

    for row, (exp_name, results, conditions) in enumerate(experiments):
        labels = list(results.keys())
        x = np.arange(len(labels))
        width = 0.6

        for col, (metric, mname) in enumerate(zip(metrics, metric_names)):
            ax = axes[row, col]
            means = [results[l][metric][0] for l in labels]
            stds = [results[l][metric][1] for l in labels]
            ax.bar(x, means, width, yerr=stds, capsize=4,
                   color=colors[:len(labels)], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

            if row == 0:
                ax.set_title(mname, fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{exp_name}\n({conditions})',
                              fontsize=10, fontweight='bold')
            if metric in ('tpr', 'fdr'):
                ax.set_ylim(-0.05, 1.05)

            for bar, m, sd in zip(ax.patches, means * 1, stds * 1):
                pass
            for i_bar, (m, sd) in enumerate(zip(means, stds)):
                bar = ax.patches[i_bar]
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + sd + 0.02,
                        f'{m:.2f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('Robustness of SparsevB to non-Gaussian distributions\n'
                 f'($n={n}$, $p={p}$, $s_0={s}$ active coefficients)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('2_non_gaussian_robustness.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_results_table(results, title):
    print(f"\n{title}")
    print("=" * 70)
    print(f"{'Distribution':>15} | {'l2-error':>14} | {'TP rate':>12} | {'FD rate':>12}")
    print("-" * 70)
    for label, m in results.items():
        print(f"{label:>15} | {m['l2'][0]:5.2f} +/- {m['l2'][1]:5.2f}  | "
              f"{m['tpr'][0]:4.2f} +/- {m['tpr'][1]:4.2f} | "
              f"{m['fdr'][0]:4.2f} +/- {m['fdr'][1]:4.2f}")


if __name__ == "__main__":
    n, p, s = 200, 400, 10
    signal_strength = 3.0
    noise_scale = 1.0
    n_rep = 50

    print(f"n={n}, p={p}, s={s}, noise_scale={noise_scale}")
    print(f"{n_rep} repetitions per configuration")
    print(f"Distributions: {', '.join(DIST_LABELS.values())}\n")

    print("--- Vary noise distribution ---")
    results_noise = experiment_vary_noise(n, p, s, signal_strength, noise_scale, n_rep)
    print_results_table(results_noise, "Noise distribution")

    print("\n--- Vary design distribution ---")
    results_design = experiment_vary_design(n, p, s, signal_strength, noise_scale, n_rep)
    print_results_table(results_design, "Design distribution")

    print("\n--- Vary theta distribution ---")
    results_theta = experiment_vary_theta(n, p, s, signal_strength, noise_scale, n_rep)
    print_results_table(results_theta, "Theta distribution")

    plot_comparison(results_noise, results_design, results_theta,
                    n, p, s, signal_strength)
