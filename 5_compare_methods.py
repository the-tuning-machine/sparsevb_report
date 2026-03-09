"""
Compare gamma threshold vs Stability Selection
===============================================
Grid search over Stability Selection hyperparameters across multiple
problem settings. The median probability model (gamma > 0.5) is included
as a baseline in the final ranking table.
"""

import time
import numpy as np
from itertools import product
from sparsevb import svb_fit_linear
from selection_stability import select_stability

SCENARIOS = [
    {"n": 100, "p": 200, "s": 5, "signals": [3.0, -2.0, 1.5, -1.0, 0.5],
     "noise_sd": 1.0, "label": "base"},
    {"n": 200, "p": 200, "s": 5, "signals": [3.0, -2.0, 1.5, -1.0, 0.5],
     "noise_sd": 1.0, "label": "n=p"},
    {"n": 100, "p": 500, "s": 5, "signals": [3.0, -2.0, 1.5, -1.0, 0.5],
     "noise_sd": 1.0, "label": "p=500"},
    {"n": 100, "p": 200, "s": 10, "signals": np.linspace(3, 0.3, 10).tolist(),
     "noise_sd": 1.0, "label": "s=10"},
    {"n": 100, "p": 200, "s": 5, "signals": [3.0, -2.0, 1.5, -1.0, 0.5],
     "noise_sd": 2.0, "label": "noisy"},
    {"n": 100, "p": 200, "s": 5, "signals": [1.0, -0.8, 0.6, -0.4, 0.2],
     "noise_sd": 1.0, "label": "weak"},
]

N_SEEDS = 3
B_STABILITY = 20

STABILITY_GRID = {
    "subsample_frac": [0.5, 0.65, 0.8],
    "gamma_threshold": [0.3, 0.5, 0.7],
    "freq_threshold": [0.5, 0.7, 0.9],
}

def generate_data(scenario, seed):
    rng = np.random.default_rng(seed)
    n, p, s = scenario["n"], scenario["p"], scenario["s"]
    X = rng.standard_normal((n, p))
    beta_true = np.zeros(p)
    beta_true[:s] = scenario["signals"]
    y = X @ beta_true + scenario["noise_sd"] * rng.standard_normal(n)
    return X, y, set(range(s))


def compute_metrics(selected, true_support):
    sel = set(int(v) for v in selected)
    tp = len(sel & true_support)
    fp = len(sel - true_support)
    fn = len(true_support - sel)
    precision = tp / (tp + fp) if tp + fp > 0 else 1.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1}


print("Evaluating gamma > 0.5 baseline...")
baseline_results = []
for sc in SCENARIOS:
    metrics_list = []
    for seed in range(N_SEEDS):
        X, y, true_support = generate_data(sc, seed)
        res = svb_fit_linear(X, y, noise_sd=sc["noise_sd"])
        selected = np.where(res["gamma"] > 0.5)[0]
        metrics_list.append(compute_metrics(selected, true_support))
    avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    baseline_results.append({"scenario": sc["label"], **avg})

baseline_avg = {
    "f1": np.mean([r["f1"] for r in baseline_results]),
    "precision": np.mean([r["precision"] for r in baseline_results]),
    "recall": np.mean([r["recall"] for r in baseline_results]),
    "tp": np.mean([r["tp"] for r in baseline_results]),
    "fp": np.mean([r["fp"] for r in baseline_results]),
    "fn": np.mean([r["fn"] for r in baseline_results]),
}

print(f"  gamma>0.5 baseline — F1={baseline_avg['f1']:.3f}  "
      f"TP={baseline_avg['tp']:.1f}  FP={baseline_avg['fp']:.1f}\n")

stab_keys = list(STABILITY_GRID.keys())
stab_combos = list(product(*[STABILITY_GRID[k] for k in stab_keys]))
total_calls = len(SCENARIOS) * len(stab_combos) * N_SEEDS

print(f"Stability Selection grid search")
print(f"  {len(SCENARIOS)} scenarios x {len(stab_combos)} combos x {N_SEEDS} seeds "
      f"= {total_calls} calls")
print("=" * 90)

stab_results = []
done = 0
t_start = time.time()

for sc in SCENARIOS:
    for combo in stab_combos:
        params = dict(zip(stab_keys, combo))
        metrics_list = []
        t_combo = time.time()

        for seed in range(N_SEEDS):
            X, y, true_support = generate_data(sc, seed)
            sel, _ = select_stability(
                X, y, B=B_STABILITY,
                subsample_frac=params["subsample_frac"],
                gamma_threshold=params["gamma_threshold"],
                freq_threshold=params["freq_threshold"],
                rng=np.random.default_rng(100 + seed),
            )
            metrics_list.append(compute_metrics(sel, true_support))
            done += 1

        avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
        stab_results.append({"scenario": sc["label"], "params": params, **avg})

        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total_calls - done) / rate if rate > 0 else 0
        print(f"  [{done:4d}/{total_calls}] {elapsed:5.0f}s ETA {eta:4.0f}s | "
              f"{sc['label']:<6} | sub={params['subsample_frac']:.1f} "
              f"gam={params['gamma_threshold']:.1f} "
              f"freq={params['freq_threshold']:.1f} | "
              f"F1={avg['f1']:.3f} TP={avg['tp']:.1f} FP={avg['fp']:.1f} "
              f"({time.time()-t_combo:.1f}s)", flush=True)

print(f"\nDone in {time.time()-t_start:.0f}s")

param_agg = {}
for r in stab_results:
    key = tuple(sorted(r["params"].items()))
    param_agg.setdefault(key, []).append(r)

param_scores = []
for key, runs in param_agg.items():
    param_scores.append({
        "params": dict(key),
        "f1": np.mean([r["f1"] for r in runs]),
        "precision": np.mean([r["precision"] for r in runs]),
        "recall": np.mean([r["recall"] for r in runs]),
        "tp": np.mean([r["tp"] for r in runs]),
        "fp": np.mean([r["fp"] for r in runs]),
        "fn": np.mean([r["fn"] for r in runs]),
    })
param_scores.sort(key=lambda r: r["f1"], reverse=True)

print("\n" + "=" * 95)
print("RANKING — Stability Selection configs + gamma>0.5 baseline")
print("=" * 95)
header = (f"{'Rank':>4} | {'Method':<28} | "
          f"{'F1':>5} | {'Prec':>5} | {'Rec':>5} | "
          f"{'TP':>5} | {'FP':>5} | {'FN':>5}")
print(header)
print("-" * 95)

entries = []
for r in param_scores:
    p = r["params"]
    label = (f"StabSel sub={p['subsample_frac']:.2f} "
             f"g={p['gamma_threshold']:.1f} f={p['freq_threshold']:.1f}")
    entries.append({"label": label, **r})

baseline_entry = {"label": "*** gamma > 0.5 ***", **baseline_avg}

inserted = False
rank = 1
for e in entries:
    if not inserted and baseline_avg["f1"] >= e["f1"]:
        print(f"{rank:4d} | {baseline_entry['label']:<28} | "
              f"{baseline_entry['f1']:5.3f} | {baseline_entry['precision']:5.3f} | "
              f"{baseline_entry['recall']:5.3f} | {baseline_entry['tp']:5.1f} | "
              f"{baseline_entry['fp']:5.1f} | {baseline_entry['fn']:5.1f}  <---")
        inserted = True
        rank += 1
    print(f"{rank:4d} | {e['label']:<28} | "
          f"{e['f1']:5.3f} | {e['precision']:5.3f} | {e['recall']:5.3f} | "
          f"{e['tp']:5.1f} | {e['fp']:5.1f} | {e['fn']:5.1f}")
    rank += 1

if not inserted:
    print(f"{rank:4d} | {baseline_entry['label']:<28} | "
          f"{baseline_entry['f1']:5.3f} | {baseline_entry['precision']:5.3f} | "
          f"{baseline_entry['recall']:5.3f} | {baseline_entry['tp']:5.1f} | "
          f"{baseline_entry['fp']:5.1f} | {baseline_entry['fn']:5.1f}  <---")

print("=" * 95)
print(f"\nBest Stability Selection: {param_scores[0]['params']}  "
      f"(F1={param_scores[0]['f1']:.3f})")
print(f"Gamma > 0.5 baseline:    F1={baseline_avg['f1']:.3f}")
