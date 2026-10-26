# Formative 3 — Probability Distributions, Bayesian Inference & Gradient Descent

**Team:** Faith Mutoni · Kenny Ishimwe Gatete · Regina

This repository contains our full submission for Formative 3, covering four parts:

| Part | Topic | Folder |
|------|-------|--------|
| 1 | Expectation-Maximization (EM) for a Gaussian Mixture Model | `part1/` |
| 2 | Bayesian sentiment analysis on IMDb reviews | `part2/` |
| 3 | Manual gradient descent (matrix calculus) | `part3/` |
| 4 | Gradient descent in code, using SciPy for the derivative | `part4/` |

**Quick links:** [Task Sheet](https://docs.google.com/spreadsheets/d/1UyCfa2_V8Yq3vcrrh_rh8Kt6UWSfntaiyWy0rkNpPl8/edit?usp=sharing) · [Google Colab Notebook](https://colab.research.google.com/drive/10OIGaow25zCMIbhboXtu6DwCwZuEGT_d?usp=sharing)

---

## Part 1 — EM Algorithm (`part1/`)

We model **Fathers vs. Children** heights from the Galton Families dataset as an unlabeled mixture of two 1-D Gaussians. We pool one father height per family (205) with every child's height (934), 1,139 heights in total, discard the labels, and recover the two groups using Expectation-Maximization implemented from scratch (no sklearn).

Component 1 is always **Children** and component 2 **Fathers**. Since fathers are taller on average (69.32 in vs. 66.75 in), the lower starting mean goes to component 1.

**Run it:**

```bash
cd part1
pip install numpy pandas matplotlib
jupyter notebook formative3_part1_fathers.ipynb   # needs GaltonFamilies.csv in the same folder
```

The notebook prints the tracking table for iterations 0–2 and classifies a test height with exact posterior probabilities (set `test_h` to the height the coach gives you).

**How EM works:**

1. **Initialise:** means at the 25th/75th percentiles, both variances at the pooled variance, `π₁ = π₂ = 0.5`.
2. **E-step:** responsibility `r_ij = π_j·N(x_i | μ_j, σ_j²) / Σ_k π_k·N(x_i | μ_k, σ_k²)`.
3. **M-step:** `N_j = Σ r_ij`, `μ_j = Σ r_ij·x_i / N_j`, `σ_j² = Σ r_ij·(x_i − μ_j)² / N_j`, `π_j = N_j / n`.
4. **Converge:** stop when the log-likelihood changes by less than `1e-6`. It never decreases between iterations.

**Should we just split at the global mean?** No — and this is the core answer for the presentation. A hard split at the mean (67.21 in):

- **Misclassifies the overlap region.** Many children are taller than the cut and many fathers shorter, so accuracy is only **60.8%**.
- **Throws away the shape of the distributions.** It ignores the spread and size of each group, and each pile's mean is biased because the pile is truncated at the cut.
- **Gives no measure of confidence.** EM instead produces a *soft assignment* and uses that uncertainty to fit better parameters, rather than forcing a binary guess.

**Tracking table:**

| Iteration | μ₁ (Children) | μ₂ (Fathers) | σ₁² | σ₂² | π₁ | π₂ | Log-Likelihood |
|-----------|---------------|--------------|-----|-----|----|----|----------------|
| 0 (Init) | 64.5000 | 70.0000 | 12.7241 | 12.7241 | 0.5000 | 0.5000 | -3119.8222 |
| 1 | 65.2205 | 69.2054 | 9.0603 | 8.4468 | 0.5011 | 0.4989 | -3057.9135 |
| 2 | 65.1609 | 69.2600 | 8.9420 | 8.1040 | 0.5005 | 0.4995 | -3056.9719 |

Converged after **319 iterations**: μ = [64.177, 69.519], σ² = [5.302, 6.041], π = [0.432, 0.568].

**Sample posterior:** a height of 68.0 in is **19.88% Child** and **80.12% Father**.

**Functions (modular, DRY):**

- `load_heights` — loads the CSV and returns pooled, parent and child heights.
- `gaussian_pdf`, `log_likelihood` — Gaussian density and mixture log-likelihood.
- `e_step`, `m_step`, `init_params`, `fit_em` — the EM loop with convergence check and history.
- `tracking_table` — builds the iteration table.
- `classify` — posterior probabilities for a test height.
- `global_mean_split` — the baseline used in the comparison.
- `plot_fit` — saves `em_fit_fathers.png`.

---

## Part 2 — Bayesian Sentiment Analysis (`part2/`)

Using the IMDb review dataset, we compute **P(Positive | keyword)** via Bayes' Theorem for positive-leaning and negative-leaning keywords, in pure Python (no ML libraries). We compute only this one direction, as the brief requires.

**Keywords:** amazing, excellent, wonderful, enjoyable (positive-leaning) · terrible, boring, awful, disappointing (negative-leaning)

**Run it:**

```bash
cd part2
python app.py
```

**Results:**

| Keyword | Prior P(Positive) | Likelihood P(kw\|Pos) | Marginal P(kw) | Posterior P(Pos\|kw) |
|---------|-------------------|-----------------------|----------------|----------------------|
| amazing | 0.50 | 0.0740 | 0.0496 | 0.7463 |
| excellent | 0.50 | 0.1174 | 0.0725 | 0.8099 |
| wonderful | 0.50 | 0.1066 | 0.0650 | 0.8203 |
| enjoyable | 0.50 | 0.0424 | 0.0309 | 0.6846 |
| terrible | 0.50 | 0.0154 | 0.0541 | 0.1419 |
| boring | 0.50 | 0.0247 | 0.0623 | 0.1983 |
| awful | 0.50 | 0.0136 | 0.0624 | 0.1093 |
| disappointing | 0.50 | 0.0074 | 0.0165 | 0.2236 |

Posterior = Likelihood × Prior / Marginal. Positive-leaning keywords push the posterior well above the 0.50 prior (0.68–0.82), and negative-leaning keywords push it well below (0.11–0.22), which is the direction Bayes' Theorem predicts.

---

## Part 3 — Manual Gradient Descent (`part3/`)

Manual, matrix-form derivation of gradient descent for `ŷ = Xm + b·1`, with **3 fully worked iterations (one per group member)**, using the chain rule to derive `∂J/∂m = (2/n)Xᵀe` and `∂J/∂b = (2/n)1ᵀe`.

- Model: `y = m₁x₁ + m₂x₂ + b`, cost `J = (1/n)eᵀe` with `e = ŷ − y`
- Start: `m = [-1, 2]`, `b = [1, 1]`, `X = [[1, 3], [4, 10]]`, `y = [5, 6]`, `α = 0.01`

| Iter | ŷ | J | ∂J/∂m | ∂J/∂b | Updated m | Updated b |
|------|---|---|-------|-------|-----------|-----------|
| 1 | [6, 17] | 61.000 | [45, 113] | 12.000 | [-1.45, 0.87] | 0.8800 |
| 2 | [2.04, 3.78] | 6.845 | [-11.84, -31.08] | -5.180 | [-1.3316, 1.1808] | 0.9318 |
| 3 | [3.1426, 7.4134] | 2.724 | [3.7962, 8.5618] | -0.444 | [-1.3696, 1.0952] | 0.9362 |

MSE across iterations: **61.00 → 6.845 → 2.724 → 2.390** (about 96% error reduction in 3 steps). `m₂` changes most and zig-zags after the first overshoot, but the swings shrink, so the updates are converging.

The handwritten calculations are in `Gradient_descent_calculation.pdf`.

---

## Part 4 — Gradient Descent in Code (`part4/`)

Converts the Part 3 derivation into working code. `scipy.differentiate.derivative` computes each partial derivative numerically (no hand-typed derivative in the update), and every update step is explicit (predictions → error → cost → gradients → update), with nothing hidden behind a `.fit()` call. The SciPy gradients are printed next to the closed-form formulas to show they agree.

**Run it:** open the [Colab notebook](https://colab.research.google.com/drive/10OIGaow25zCMIbhboXtu6DwCwZuEGT_d?usp=sharing), or locally:

```bash
cd part4
pip install scipy matplotlib numpy
jupyter notebook   # open the Part 4 notebook and run all cells
```

This produces two separate plots:

- **Parameters over iterations:** how `m₁`, `m₂` and `b` change.
- **Error over iterations:** MSE falling from 61.0 to about 2.39.

**Final result:** `m = [-1.369562, 1.095182]`, `b = 0.936240`, predictions `[2.852224, 6.409812]` (actual `[5, 6]`), `J = 2.390444`. Every printed iteration matches the Part 3 manual numbers.

---

## Technologies

- Python 3, NumPy, Pandas, SciPy, Matplotlib
- No sklearn or other ML libraries used for the from-scratch parts (EM, Bayes)

## Team Contributions

- **Part 1 (EM):** Faith Mutoni
- **Part 2 (Bayes):** Kenny Ishimwe Gatete
- **Part 3 (Manual GD):** Regina leads; one iteration per member (Iteration 1: _name_, Iteration 2: _name_, Iteration 3: _name_)
- **Part 4 (Code):** _Regina_

## Submission Checklist

- [ ] README for the presentation
- [ ] GitHub repository (git logs are reviewed)
- [ ] Jupyter notebook with all implementations and images
- [ ] Neat PDF of handwritten Part 3 calculations
- [ ] PDF of member contributions