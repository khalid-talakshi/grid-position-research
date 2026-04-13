# Cumulative Ordered-Logistic Regression — Line-by-Line

## Mathematical Specification

### Likelihood

For each observation $i = 1, \ldots, N$, the finish position $y_i \in \{1, \ldots, K\}$ is modelled as:

$$y_i \mid \mathbf{p}_i \sim \text{Categorical}(p_{i,1}, \ldots, p_{i,K})$$

The cell probabilities are derived from cumulative probabilities via:

$$P(y_i \leq k \mid \text{grid}_i) = \text{logistic}(\kappa_k - \phi_i), \quad k = 1, \ldots, K-1$$

$$p_{i,k} = P(y_i \leq k \mid \text{grid}_i) - P(y_i \leq k-1 \mid \text{grid}_i)$$

with boundary conditions $P(y_i \leq 0) = 0$ and $P(y_i \leq K) = 1$.

### Linear predictor

$$\phi_i = \beta \cdot z_i, \qquad z_i = \frac{\text{grid}_i - \bar{g}}{s_g}$$

where $\bar{g} \approx 8.31$ and $s_g \approx 5.47$ are the sample mean and SD of grid positions.

### Priors

$$\beta \sim \mathcal{N}(0,\, 1)$$

$$\kappa_1 \sim \mathcal{N}(\mu_1,\, \tau^{-1})$$

$$\kappa_k \sim \mathcal{N}(\mu_k,\, \tau^{-1})\,\mathbf{1}(\kappa_k > \kappa_{k-1}), \quad k = 2, \ldots, K-1$$

where $\boldsymbol{\mu} = \text{linspace}(-2, 2, K-1)$ and $\tau \approx 0.44$ (SD $\approx 1.5$), both supplied as data.

The ordering constraint $\kappa_1 < \kappa_2 < \cdots < \kappa_{K-1}$ is enforced via the truncated-normal indicator $\mathbf{1}(\kappa_k > \kappa_{k-1})$.

### Per-slot log-odds shift

A one grid slot improvement changes $z_i$ by $\Delta z = -1/s_g$, shifting the log-odds of finishing in any position $k$ or better by:

$$\Delta \log\text{-odds} = \frac{\beta}{s_g} \approx \frac{\beta}{5.47}$$

This quantity is constant across all cutpoints (the proportional odds assumption).

---

## Observation loop

```jags
for (i in 1:N) {
```
Loops over all N = 1,655 race–driver observations.

---

```jags
  phi[i] <- beta * grid_z[i]
```
Computes the **linear predictor** for observation i.
`beta` is the single regression coefficient; `grid_z[i]` is the z-scored
starting grid position. A larger (worse) grid position yields a larger φ,
which will shift probability mass toward worse finishing positions.

---

```jags
  for (k in 1:(K-1)) {
    cum_p[i, k] <- ilogit(cutpoint[k] - phi[i])
  }
```
Computes **K-1 = 19 cumulative probabilities**, one per threshold:

  P(finish ≤ k | grid_z[i]) = logistic(κ_k − φ_i)

Subtracting φ from the cutpoint means a higher φ shifts probabilities
*right* (toward worse finish positions).

---

```jags
  p[i, 1] <- cum_p[i, 1]
```
Cell probability for finishing **1st**:

  P(finish = 1) = P(finish ≤ 1)

---

```jags
  for (k in 2:(K-1)) {
    p[i, k] <- cum_p[i, k] - cum_p[i, k-1]
  }
```
Cell probabilities for finishing **2nd through 19th**:

  P(finish = k) = P(finish ≤ k) − P(finish ≤ k-1)

This differences adjacent cumulative probabilities to get proper
non-negative category probabilities.

---

```jags
  p[i, K] <- 1 - cum_p[i, K-1]
```
Cell probability for finishing **last (K = 20th)**:

  P(finish = K) = 1 − P(finish ≤ K-1)

Ensures all K probabilities sum to 1.

---

```jags
  classification[i] ~ dcat(p[i, 1:K])
```
**Likelihood**: observed finish position is drawn from a categorical
distribution with the K probabilities computed above.

---

## Priors

```jags
beta ~ dnorm(0, 1)
```
**Prior on the grid effect.** Normal(0, 1) in JAGS precision
parameterisation = Normal(0, σ²=1). Weakly informative on the log-odds
scale; centred at zero (no assumed effect), allowing data to push it
positive (grid position matters) or toward zero (it doesn't).

---

```jags
cutpoint[1] ~ dnorm(mu_cut[1], tau_cut)
```
**First cutpoint**, drawn from a normal with mean `mu_cut[1]` (≈ −2)
and precision `tau_cut` (≈ 0.44, i.e. SD ≈ 1.5). Anchors the bottom
of the ordinal scale.

---

```jags
for (k in 2:(K-1)) {
  cutpoint[k] ~ dnorm(mu_cut[k], tau_cut) T(cutpoint[k-1], )
}
```
**Remaining 18 cutpoints**, each drawn from a normal *truncated from
below* at the previous cutpoint. The `T(cutpoint[k-1], )` syntax
enforces the ordering constraint κ₁ < κ₂ < … < κ₁₉ required by the
cumulative logit model. `mu_cut` is a linspace of 19 evenly-spaced
values across [−2, 2], giving each cutpoint a sensible prior home.

---

## Parameter Reference

### Free parameters (estimated from data)

| Parameter | Type | Description |
|-----------|------|-------------|
| `beta` | scalar | The **grid effect**. Measures how much a one-SD increase in grid position shifts the log-odds of finishing in a worse position. A positive value (expected) means starting further back predicts finishing further back. Estimated from the data with a Normal(0,1) prior. |
| `cutpoint[k]` | vector of 19 | The **decision thresholds** that partition the log-odds scale into 20 finish-position bins. `cutpoint[k]` is the log-odds value at which a driver has a 50% chance of finishing in position k or better, holding the grid effect at zero. Must be strictly ordered: κ₁ < κ₂ < … < κ₁₉. |

### Derived quantities (computed, not sampled)

| Quantity | Shape | Description |
|----------|-------|-------------|
| `phi[i]` | N-vector | The **linear predictor** for observation i. Equal to `beta * grid_z[i]`. Represents the overall "disadvantage" of driver i's grid position on the log-odds scale. |
| `cum_p[i, k]` | N × (K-1) matrix | **Cumulative probabilities**: P(finish ≤ k \| grid_z[i]). Built by applying the logistic function to `cutpoint[k] - phi[i]`. |
| `p[i, k]` | N × K matrix | **Cell probabilities**: P(finish = k \| grid_z[i]). Derived by differencing adjacent cumulative probabilities. These are the finalised inputs to the categorical likelihood. |

### Data / hyperparameters (fixed, supplied from R)

| Name | Value | Description |
|------|-------|-------------|
| `N` | 1,655 | Number of race–driver observations. |
| `K` | 20 | Number of finish position categories (one per grid slot). |
| `grid_z` | N-vector | Z-scored starting grid positions: `(grid - 8.31) / 5.47`. |
| `classification` | N-vector (int) | Observed finish positions, 1-indexed (the response variable). |
| `mu_cut` | 19-vector | Prior means for each cutpoint, evenly spaced across [−2, 2]. Centres the truncated-normal priors in sensible regions of the log-odds scale. |
| `tau_cut` | scalar (≈ 0.44) | Prior precision (1/variance) shared by all cutpoint priors. Corresponds to SD ≈ 1.5, allowing cutpoints to shift substantially from their prior means. |
