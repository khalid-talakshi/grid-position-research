library(tidyverse)
library(rjags)

data = read_csv("./data/grid-results.csv")
circuit_info = read_csv("./data/circuit-type.csv")

merged_data = merge(data, circuit_info, by.x = "circuitId", by.y = "circuitId")

classified_position = as.numeric(merged_data$ClassifiedPosition) 
max_classified_position = max(classified_position)

grid_position = as.numeric(merged_data$GridPosition) - 1

mod1_string = "
model {
  for (i in 1:N) {
    z[i] ~ dnorm(mu[i], tau) T(lower[i], upper[i])
    mu[i] <- alpha + beta * grid[i]
  }

  # Cutpoints
  cutoff[1] ~ dnorm(0, 1)
  for (k in 2:(max_finish - 1)) {
    cutoff[k] ~ dnorm(cutoff[k-1] + 1, 1) T(cutoff[k-1] + 0.01, 1.0E6)
  }

  # Interval bounds derived from observed category
  for (i in 1:N) {
    lower[i] <- ifelse(classification[i] == 1,
                       -1.0E6,
                       cutoff[classification[i] - 1])

    upper[i] <- ifelse(classification[i] == max_finish,
                       1.0E6,
                       cutoff[classification[i]])
  }

  # priors for regression / scale (add if needed)
  alpha ~ dnorm(0, 0.001)
  beta  ~ dnorm(0, 0.001)
  sigma ~ dunif(0, 10)
  tau <- 1 / (sigma * sigma)
}
"


mod1_data = list(
    classification = classified_position,
    grid = grid_position,
    N = nrow(merged_data),
    max_finish = max_classified_position
)

z_init <- classified_position + rnorm(length(classified_position), 0, 0.01)

cutoff_init <- seq(
  min(z_init) - 1,
  max(z_init) - 1,
  length.out = max_classified_position - 1
)

inits <- function() {
  list(
    alpha = 0,
    beta = 0,
    sigma = 1,
    z = z_init,
    cutoff = cutoff_init
  )
}


mod1_jags = jags.model(
    textConnection(mod1_string),
    data = mod1_data,
    inits = inits,
    n.chains = 3
)

