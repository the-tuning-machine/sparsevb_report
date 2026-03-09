# SparseVB Experiments

Experiments for the Bayesian Machine Learning report on Spike-and-Slab Variational Bayes (Ray & Szabo, 2022).

## Requirements
```bash
pip install sparsevb numpy scipy scikit-learn matplotlib jupyter ipykernel
```

To register the Python kernel in Jupyter:
```bash
python -m ipykernel install --user --name sparsevb --display-name "Python (sparsevb)"
```

## Part 1: Synthetic experiments

### Scripts
| Script | Description | Output |
|--------|-------------|--------|
| `1_correlated_groups.py` | Needle-in-haystack with Toeplitz-correlated groups | `1_correlated_groups.png` |
| `2_non_gaussian_robustness.py` | Robustness to non-Gaussian noise, design and coefficients (7 distributions) | `2_non_gaussian_robustness.png` |
| `3_sparsity_limits.py` | Effect of increasing sparsity ratio s/n | `3_sparsity_limits.png`, `3_scaling.png` |
| `4_signal_strength.py` | Effect of decreasing signal-to-noise ratio | `4_signal_strength.png` |
| `5_compare_methods.py` | Compare sparsevb vs AIC/BIC/CV/Knockoffs/Stability | (console output) |
| `6_tune_hyperparams.py` | Grid search for Stability Selection hyperparameters | (console output) |

Utility modules used by `5_compare_methods.py` and `6_tune_hyperparams.py`:
- `selection_aic_bic.py` — AIC/BIC forward selection
- `selection_cv.py` — Cross-validation forward selection
- `selection_knockoffs.py` — Knockoff filter
- `selection_stability.py` — Stability selection

### Running

#### Individual scripts
```bash
python 1_correlated_groups.py
python 2_non_gaussian_robustness.py
python 3_sparsity_limits.py
python 4_signal_strength.py
python 5_compare_methods.py
python 6_tune_hyperparams.py
```

#### All at once (notebook)
```bash
jupyter notebook run_all.ipynb
```
Or open `run_all.ipynb` in Jupyter and run all cells.

#### All at once (command line)
```bash
cd report
for f in 1_*.py 2_*.py 3_*.py 4_*.py 5_*.py 6_*.py; do
    echo "=== $f ==="
    python "$f"
done
```

### Key parameters
All experiments use known `noise_sd` when the ground truth is available, as in the paper (Table 1). The main settings match the paper:
- **Design matrix**: X_ij ~ N(0, 1) (iid Gaussian)
- **Hyperparameters**: a₀ = 1, b₀ = p, λ = 1 (sparsevb defaults)
- **Selection threshold**: γ_i > 0.5

---

## Part 2: Experiments on the riboflavin dataset

Comparison of VB Laplace, VB Gaussian, and LASSO on the riboflavin dataset
(Bühlmann et al., 2014) via 10-fold cross-validation.

### Additional Requirements
```bash
pip install rpy2
```

### Running
Open `riboflavin_experiments.ipynb` in Jupyter and run all cells
in order. The dataset is downloaded automatically via OpenML.

### Notes
- LASSO alpha tuned once via `LassoCV` on the 70% train split for efficiency
- R seed fixed via `rpy2` before the CV loop for reproducibility (`set.seed(42)`)

---

## Reference
Ray, K. & Szabó, B. (2022). *Variational Bayes for high-dimensional linear regression with sparse priors*. [arXiv:1904.07150v3](https://arxiv.org/abs/1904.07150)
