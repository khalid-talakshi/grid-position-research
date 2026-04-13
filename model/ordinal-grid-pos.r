library(tidyverse)
library(rjags)
library(coda)

# ============================================================================
# Load and prepare data
# ============================================================================

data <- read_csv("./data/grid-results.csv")

# Remove rows with missing grid or classification positions
data_clean <- data %>%
  filter(!is.na(GridPosition) & !is.na(ClassifiedPosition)) %>%
  mutate(
    grid = as.numeric(GridPosition),
    classification = as.numeric(ClassifiedPosition)  # Keep 1-indexed for dcat
  ) %>%
  select(grid, classification)

# Summary
cat("Data shape:", nrow(data_clean), "races\n")
cat("Grid positions range:", min(data_clean$grid), "—", max(data_clean$grid), "\n")
cat("Finish positions range (1-indexed):", 
    min(data_clean$classification), "—", max(data_clean$classification), "\n")

# ============================================================================
# Prepare data for JAGS
# ============================================================================

N <- nrow(data_clean)
grid_pos <- data_clean$grid
classification <- data_clean$classification

# K is the number of finish categories (dcat expects categories 1:K)
K <- max(data_clean$classification)

cat("K (number of finish categories):", K, "\n")
cat("Number of cutpoints:", K - 1, "\n")

mod_data <- list(
  N = N,
  grid = grid_pos,
  classification = classification,
  K = K
)

# ============================================================================
# Initialize latent variable z and cutoffs per chain
# ============================================================================

init_chain <- function(chain_id) {
  # Initialize cutoffs: wider spacing to avoid overlap issues
  cutoff_init <- seq(-10, 10, length.out = K - 1)
  
  # Ensure cutoffs are strictly ordered
  cutoff_init <- sort(cutoff_init)
  
  # Initialize latent z constrained to the interval for each observed category
  z_init <- numeric(N)
  
  for (i in 1:N) {
    y <- classification[i]
    
    if (y == 1) {
      # Category 1: z <= cutoff[1]
      z_init[i] <- cutoff_init[1] - abs(rnorm(1, 0.5, 0.1))
    } else if (y == K) {
      # Category K: z > cutoff[K-1]
      z_init[i] <- cutoff_init[K - 1] + abs(rnorm(1, 0.5, 0.1))
    } else {
      # Category y: cutoff[y-1] < z <= cutoff[y]
      lower <- cutoff_init[y - 1]
      upper <- cutoff_init[y]
      
      # Add safety margin
      z_init[i] <- lower + (upper - lower) * runif(1, 0.1, 0.9)
    }
  }
  
  list(
    alpha = runif(1, -0.5, 0.5),
    beta = runif(1, 0.001, 0.05),
    cutoff = cutoff_init,
    z = z_init
  )
}

# ============================================================================
# Fit model
# ============================================================================

cat("\n--- Initializing model ---\n")

mod <- jags.model(
  file = "./model/ordinal-grid-pos.jags",
  data = mod_data,
  inits = function() init_chain(sample(1:3, 1)),  # per-chain inits
  n.chains = 3,
  n.adapt = 1000
)

cat("\n--- Burn-in ---\n")
update(mod, n.iter = 2000)

cat("\n--- Drawing posterior samples ---\n")
samples_coda <- coda.samples(
  mod,
  variable.names = c("alpha", "beta", "cutoff"),
  n.iter = 5000,
  thin = 2
)

# ============================================================================
# Diagnostic: Gelman-Rubin (Rhat)
# ============================================================================

cat("\n--- Convergence diagnostics (Rhat, should be < 1.01) ---\n")
gelman_diag <- gelman.diag(samples_coda, confidence = 0.95, transform = FALSE)
print(gelman_diag)

# ============================================================================
# Save posterior samples
# ============================================================================

saveRDS(samples_coda, "./model/ordinal-grid-pos-samples.rds")
cat("\n✓ Posterior samples saved to model/ordinal-grid-pos-samples.rds\n")

# ============================================================================
# Extract posterior draws and compute probability table
# ============================================================================

# Convert coda object to matrix: rows = iterations, cols = variables
samples_matrix <- do.call(rbind, samples_coda)

# Extract alpha, beta, and cutoffs from all iterations
alpha_samples <- samples_matrix[, "alpha"]
beta_samples <- samples_matrix[, "beta"]

# Cutoff columns are "cutoff[1]", "cutoff[2]", ..., "cutoff[K-1]"
cutoff_cols <- grep("^cutoff\\[", colnames(samples_matrix), value = TRUE)
cutoff_samples <- samples_matrix[, cutoff_cols]

n_iter <- nrow(samples_matrix)

cat("\nPosterior summary (all chains combined):\n")
cat("  alpha:", mean(alpha_samples), "±", sd(alpha_samples), "\n")
cat("  beta:", mean(beta_samples), "±", sd(beta_samples), "\n")

# ============================================================================
# Compute P(finish = y | grid = x) for all x in 1:20, y in 1:K
# ============================================================================

# Grid positions to evaluate (1 to 20)
grid_range <- 1:20

# Build prediction matrix: P[x, y] = P(finish = y | grid = x)
prob_matrix <- matrix(NA, nrow = length(grid_range), ncol = K)
rownames(prob_matrix) <- paste0("grid_", grid_range)
colnames(prob_matrix) <- paste0("finish_", 1:K)

# For each grid position
for (i_x in seq_along(grid_range)) {
  x <- grid_range[i_x]
  
  # Compute mu = alpha + beta * x for all posterior samples
  mu <- alpha_samples + beta_samples * x
  
  # For each finish position y (1 to K)
  for (y in 1:K) {
    # P(finish = y | grid = x) = Phi(cutoff[y] - mu) - Phi(cutoff[y-1] - mu)
    # where Phi is the standard normal CDF
    
    if (y == 1) {
      # P(y = 1) = Phi(cutoff[1] - mu)
      upper <- pnorm(cutoff_samples[, 1] - mu)
      lower <- 0
    } else if (y == K) {
      # P(y = K) = 1 - Phi(cutoff[K-1] - mu)
      upper <- 1
      lower <- pnorm(cutoff_samples[, K - 1] - mu)
    } else {
      # P(y in middle) = Phi(cutoff[y] - mu) - Phi(cutoff[y-1] - mu)
      upper <- pnorm(cutoff_samples[, y] - mu)
      lower <- pnorm(cutoff_samples[, y - 1] - mu)
    }
    
    # Average probability across all posterior samples
    prob_y <- mean(upper - lower)
    prob_matrix[i_x, y] <- prob_y
  }
}

# Convert to long format for CSV
prob_long <- prob_matrix %>%
  as.data.frame() %>%
  rownames_to_column("grid_position") %>%
  mutate(grid_position = as.numeric(gsub("grid_", "", grid_position))) %>%
  pivot_longer(
    cols = -grid_position,
    names_to = "finish_position",
    names_prefix = "finish_",
    values_to = "probability"
  ) %>%
  mutate(finish_position = as.numeric(finish_position))

# Save to CSV
write_csv(prob_long, "./model/ordinal-grid-pos-summary.csv")
cat("✓ Probability summary saved to model/ordinal-grid-pos-summary.csv\n")

# ============================================================================
# Display sample probabilities
# ============================================================================

cat("\n--- Sample probabilities: P(finish = y | grid = x) ---\n")
cat("(Showing grid positions 1, 5, 10, 15, 20)\n")

sample_grids <- c(1, 5, 10, 15, 20)
for (x in sample_grids) {
  idx <- which(grid_range == x)
  cat("\nGrid position", x, ":\n")
  probs <- prob_matrix[idx, ]
  for (y in 1:min(5, K)) {  # show first 5 finish positions
    cat(sprintf("  P(finish = %2d) = %.4f\n", y, probs[y]))
  }
  if (K > 5) cat("  ...\n")
}

cat("\n✓ Model fitting complete!\n")
