# Rainbow Spread Pricing Comparison

This repository contains Jupyter notebooks that compare different pricing approaches for a **rainbow spread** option involving multiple underlying assets.

## 📄 Notebooks

- **`comparison_mvn_rank.ipynb`** — Main analysis notebook.
  - Compares four methods for a 4-asset rainbow spread:
    1. **Presentation (product-of-probabilities)** — pairwise ranking probabilities using Kirk/Margrabe pricing.
    2. **LZD-ext (pairwise conditional)** — uses Li-Zhou-Deng conditional quadrature for each pair.
    3. **Presentation-MVN (joint ranking)** — uses a bivariate normal joint probability for the 4th leg.
    4. **Monte Carlo benchmark** — correlated GBMs with antithetic variates.
  - Includes timing comparisons and an error vs correlation sweep.

- **`rainbow_spread_comparison.ipynb`** — Earlier version of the analysis (likely focused on more basic comparisons).

## 🧠 What the code does

The notebook implements several core pricing primitives:

- **Black-76 / Margrabe / Kirk** pricing functions for spread options.
- **LZD conditional quadrature** for pricing a spread option by conditioning on one leg.
- **Outranking probability** for lognormal variables (univariate normal CDF).
- **Joint MVN ranking probability**: calculates `P(S4 > S2, S4 > S3)` using a bivariate normal CDF and Monte Carlo fallback.
- **Monte Carlo benchmark** using correlated Gaussian shocks and antithetic sampling.

## 🧩 Requirements

Recommended minimal Python environment (e.g. via `conda` or `venv`):

- Python 3.8+
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `jupyter`

Install dependencies via:

```bash
pip install numpy pandas matplotlib scipy jupyter
```

## ▶️ Running the analysis

Launch Jupyter in this directory:

```bash
jupyter notebook
```

Then open `comparison_mvn_rank.ipynb` and run the cells in order.

## 📌 Notes

- The joint MVN method uses `scipy.stats.multivariate_normal.cdf()`.
- If the multivariate CDF is unavailable or fails, a Monte Carlo fallback is used.

## 🧠 Model Summary: Strengths & Weaknesses

### 1) Presentation (product-of-probabilities)
- **Strengths:** Fast, simple, and fully analytical; uses pairwise ranking probabilities which are easy to interpret.
- **Weaknesses:** Assumes independence between rank events (e.g., S4 beating S2 and S4 beating S3), so it can misprice when joint correlations are strong.

### 2) LZD-ext (pairwise conditional)
- **Strengths:** Uses conditional quadrature for each pair, typically improving accuracy over simple pairwise pricing; still reasonably fast.
- **Weaknesses:** Still relies on pairwise separability (product of terms) and does not capture full joint ranking probability.

### 3) Presentation-MVN (joint ranking)
- **Strengths:** Captures the joint probability of the 4th leg beating both competitors via a bivariate normal CDF; typically more accurate when correlation is important.
- **Weaknesses:** Slightly more complex and requires a stable implementation of multivariate normal CDF; fallback uses Monte Carlo which is slower.

### 4) Monte Carlo benchmark
- **Strengths:** Most flexible and asymptotically unbiased; captures full joint dynamics without analytic approximation assumptions.
- **Weaknesses:** Computationally expensive; results have simulation noise.

## 🔧 How to Tweak Inputs

The main notebook defines market parameters near the top of the analysis section. You can modify these to explore different market scenarios.

- **Forward prices (`F`)** — change the relative levels to adjust which asset is most likely to win.
- **Volatilities (`sigma`)** — increase to make the payoff distribution wider and change ranking probabilities.
- **Correlations (`rho`)** — the joint ranking methods (Presentation-MVN, MC) are most sensitive to this.
- **Time to maturity (`T`)** — affects how much spread option values diffuse.
- **Strike (`K`)** — currently set to 0 for Margrabe exactness; change to explore general spread option pricing.
- **Monte Carlo sims (`n_sims`)** — increase for reduced noise; decreases performance.

## 🚀 Quick Start Example

In the notebook, edit the market parameters near the top of the analysis section (the `F3`, `sig3`, `R3`, `F4`, `sig4`, `R4`, `T`, `K` definitions). Then rerun the “Test Scenarios and Experiment Runner” cell to refresh the outputs.

Example tweak (insert or replace the block in the notebook):

```python
# Example: make asset 4 much more volatile and reduce correlation
F4 = (50.0, 52.0, 51.0, 49.5)
sig4 = (0.35, 0.30, 0.30, 0.6)  # larger vol for the 4th asset
R4 = np.array([[1.0, 0.2, 0.2, 0.3],
               [0.2, 1.0, 0.2, 0.3],
               [0.2, 0.2, 1.0, 0.3],
               [0.3, 0.3, 0.3, 1.0]])
T = 0.5
K = 0.0

# Rerun the comparison cell (the one that builds the `res` DataFrame)
```

After adjusting parameters, rerun the cells that compute:
- `presentation_price(...)`
- `lzd_ext_price(...)`
- `presentation_mvn_price(...)`
- `mc_price(...)`

---

If you'd like, I can also add a ready-to-run “parameter sweep” cell (e.g., sweep correlation and plot error vs MC).