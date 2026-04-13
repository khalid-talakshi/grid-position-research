library(tidyverse)
library(rjags)
library(coda)

# ============================================================================
# Load and prepare data
# ============================================================================

data <- read_csv("./data/grid-results.csv")

data_clean <- data |>
  filter(!is.na(GridPosition) & !is.na(ClassifiedPosition)) |>
  mutate(
    grid         = as.numeric(GridPosition),
    classification = as.integer(ClassifiedPosition)   # 1-indexed for dcat
  ) |>
  select(grid, classification)

cat("Observations :", nrow(data_clean), "\n")
cat("Grid range   :", min(data_clean$grid), "–", max(data_clean$grid), "\n")
cat("Finish range :", min(data_clean$classification), "–", max(data_clean$classification), "\n")

# ============================================================================
# Z-score grid position (mirrors PyMC preprocessing)
# ============================================================================

grid_mean <- mean(data_clean$grid)
grid_sd   <- sd(data_clean$grid)
grid_z    <- (data_clean$grid - grid_mean) / grid_sd

cat("Grid mean    :", round(grid_mean, 3), " sd:", round(grid_sd, 3), "\n")

# ============================================================================
# Prepare data for JAGS
# ============================================================================

N              <- nrow(data_clean)
classification <- data_clean$classification
K              <- max(classification)

# Cutpoint prior hyperparameters (mirror PyMC: Normal(linspace(-2,2), sd=1.5))
mu_cut  <- seq(-2, 2, length.out = K - 1)
tau_cut <- 1 / 1.5^2    # precision = 1/sigma^2

cat("K (finish categories) :", K, "\n")
cat("Cutpoints             :", K - 1, "\n")

mod_data <- list(
  N              = N,
  grid_z         = grid_z,
  classification = classification,
  K              = K,
  mu_cut         = mu_cut,
  tau_cut        = tau_cut
)

# ============================================================================
# Initialize chains
# No latent z to worry about — only beta and ordered cutpoints
# ============================================================================

init_chain <- function() {
  # Start cutpoints at evenly-spaced prior means with small jitter
  cp <- mu_cut + rnorm(K - 1, 0, 0.05)
  cp <- sort(cp)               # guarantee strict ordering at init

  list(
    beta     = rnorm(1, 0, 0.5),
    cutpoint = cp
  )
}

# ============================================================================
# Fit model
# ============================================================================

# Runtime note: ~60 iter/sec on a modern laptop.
# Defaults below (~2 min): n.chains=2, adapt=500, burn=1000, iter=2000.
# For production increase n.chains to 4 and n.iter to 5000 (~6 min).
N_CHAINS <- 3
N_ADAPT  <- 500
N_BURNIN <- 1000
N_ITER   <- 5000
N_THIN   <- 1

cat("\n--- Compiling model ---\n")

mod <- jags.model(
  file    = "./model/ordinal-logistic-grid-pos.jags",
  data    = mod_data,
  inits   = init_chain,
  n.chains = N_CHAINS,
  n.adapt  = N_ADAPT
)

cat(sprintf("\n--- Burn-in (%d iterations) ---\n", N_BURNIN))
update(mod, n.iter = N_BURNIN)

cat(sprintf("\n--- Drawing posterior samples (%d iterations x %d chains) ---\n",
            N_ITER, N_CHAINS))
samples_coda <- coda.samples(
  mod,
  variable.names = c("beta", "cutpoint"),
  n.iter = N_ITER,
  thin   = N_THIN
)

summary(samples_coda)
plot(samples_coda)

# ============================================================================
# Convergence diagnostics
# ============================================================================

cat("\n--- Gelman-Rubin Rhat ---\n")
cat("(Middle cutpoints may be mildly elevated [~1.05] due to slow JAGS mixing\n")
cat(" with truncated-normal ordering priors. Run 4 chains for production.)\n\n")
gelman_diag <- gelman.diag(samples_coda, multivariate = FALSE)
print(gelman_diag)

# Summarise
rhat_vals <- gelman_diag$psrf[, "Point est."]
n_bad     <- sum(rhat_vals > 1.05)
if (n_bad > 0) {
  cat(sprintf("\n⚠  %d parameter(s) with Rhat > 1.05 — consider N_CHAINS=4 and N_ITER=5000.\n", n_bad))
} else {
  cat("\n✓ All Rhat ≤ 1.05\n")
}

# ============================================================================
# Save trace
# ============================================================================

saveRDS(samples_coda, "./model/ordinal-logistic-grid-pos-samples.rds")
cat("\n✓ Posterior samples saved to model/ordinal-logistic-grid-pos-samples.rds\n")

# ============================================================================
# Extract posterior draws
# ============================================================================

samples_matrix <- do.call(rbind, samples_coda)

beta_samples     <- samples_matrix[, "beta"]
cutpoint_cols    <- grep("^cutpoint\\[", colnames(samples_matrix), value = TRUE)
cutpoint_samples <- samples_matrix[, cutpoint_cols]   # (n_iter, K-1)

n_iter <- nrow(samples_matrix)
cat("\nPosterior summary:\n")
cat("  beta :", round(mean(beta_samples), 3), "±", round(sd(beta_samples), 3), "\n")

# ============================================================================
# Posterior predictions: P(finish = y | grid = x) for all x in 1:20, y in 1:K
# ============================================================================

# ilogit (logistic CDF)
ilogit <- function(x) 1 / (1 + exp(-x))

grid_range  <- 1:20
prob_matrix <- matrix(NA, nrow = length(grid_range), ncol = K)
rownames(prob_matrix) <- paste0("grid_", grid_range)
colnames(prob_matrix) <- paste0("finish_", 1:K)

for (i_x in seq_along(grid_range)) {
  x     <- grid_range[i_x]
  x_z   <- (x - grid_mean) / grid_sd               # z-score

  # Linear predictor for all posterior draws: n_iter vector
  phi   <- beta_samples * x_z

  # Cumulative probabilities: P(finish <= k) = ilogit(cutpoint[k] - phi)
  # sweep(cutpoint_samples, 1, phi, `-`) computes cutpoint[k] - phi[i] row-wise
  cum_p <- ilogit(sweep(cutpoint_samples, 1, phi, `-`))     # (n_iter, K-1)

  # Cell probabilities P(finish = k)
  cell_p <- cbind(
    cum_p[, 1],                                            # k = 1
    cum_p[, -1] - cum_p[, -(K - 1)],                      # k = 2..K-1
    1 - cum_p[, K - 1]                                     # k = K
  )

  # Posterior-mean probability over all draws
  prob_matrix[i_x, ] <- colMeans(cell_p)
}

# ============================================================================
# Save probability table
# ============================================================================

prob_long <- prob_matrix |>
  as.data.frame() |>
  rownames_to_column("grid_position") |>
  mutate(grid_position = as.integer(gsub("grid_", "", grid_position))) |>
  pivot_longer(
    cols         = -grid_position,
    names_to     = "finish_position",
    names_prefix = "finish_",
    values_to    = "probability"
  ) |>
  mutate(finish_position = as.integer(finish_position))

write_csv(prob_long, "./model/ordinal-logistic-grid-pos-probs.csv")
cat("✓ Probability table saved to model/ordinal-logistic-grid-pos-probs.csv\n")

# ============================================================================
# Print sample predictions
# ============================================================================

cat("\n--- P(finish = 1 | grid = x) for all grid positions ---\n")
p_win <- prob_long |>
  filter(finish_position == 1) |>
  arrange(grid_position) |>
  mutate(probability = round(probability, 4))
print(p_win, n = 20)

cat("\n--- P(top 5 finish | grid = x) ---\n")
p_top5 <- prob_long |>
  filter(finish_position <= 5) |>
  summarise(p_top5 = sum(probability), .by = grid_position) |>
  arrange(grid_position) |>
  mutate(p_top5 = round(p_top5, 4))
print(p_top5, n = 20)

cat("\n✓ Model fitting complete!\n")
