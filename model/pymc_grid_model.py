"""
Simple Bayesian ordinal regression: grid position -> finish position.

Model
-----
finish_i ~ OrderedLogistic(eta=beta * grid_z_i, cutpoints=alpha)

where:
  grid_z_i  = (grid_i - mean(grid)) / std(grid)   # z-scored predictor
  beta      ~ Normal(0, 1)                          # grid effect
  alpha_k   ~ ordered cutpoints, K-1 of them        # finish thresholds

Usage
-----
  uv run python model/pymc_grid_model.py
  uv run python model/pymc_grid_model.py --draws 2000 --tune 1000 --chains 4
"""

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (grid_z, finish_0idx, grid_mean, grid_std)."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["GridPosition", "ClassifiedPosition"])
    df = df.astype({"GridPosition": int, "ClassifiedPosition": int})

    grid = df["GridPosition"].to_numpy(dtype=float)
    finish = df["ClassifiedPosition"].to_numpy(dtype=int)

    grid_mean = grid.mean()
    grid_std = grid.std()
    grid_z = (grid - grid_mean) / grid_std

    finish_0idx = finish - 1  # OrderedLogistic expects 0-indexed categories

    print(f"Observations : {len(grid)}")
    print(f"Grid range   : {grid.min():.0f} – {grid.max():.0f}")
    print(f"Finish range : {finish.min()} – {finish.max()}")
    print(f"K categories : {finish_0idx.max() + 1}")

    return grid_z, finish_0idx, grid_mean, grid_std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(grid_z: np.ndarray, finish_0idx: np.ndarray) -> pm.Model:
    K = int(finish_0idx.max()) + 1  # number of finish categories

    with pm.Model() as model:
        # --- predictor data containers (swappable at predict time) ----------
        grid_z_data = pm.Data("grid_z", grid_z, dims="obs")

        # --- priors ----------------------------------------------------------
        beta = pm.Normal("beta", mu=0, sigma=1)

        # Initialise K-1 cutpoints spread across (-2, 2); enforce ordering
        # via the 'ordered' transform built into pm.Normal with initval trick.
        cutpoints = pm.Normal(
            "cutpoints",
            mu=np.linspace(-2, 2, K - 1),
            sigma=1.5,
            transform=pm.distributions.transforms.ordered,
            initval=np.linspace(-2, 2, K - 1),
            shape=K - 1,
        )

        # --- linear predictor ------------------------------------------------
        eta = pm.Deterministic("eta", beta * grid_z_data, dims="obs")

        # --- likelihood ------------------------------------------------------
        pm.OrderedLogistic(
            "finish",
            eta=eta,
            cutpoints=cutpoints,
            observed=finish_0idx,
            dims="obs",
        )

    return model


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_model(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    seed: int = 42,
) -> az.InferenceData:
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=True,
        )
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)
    return idata


# ---------------------------------------------------------------------------
# Posterior predictions
# ---------------------------------------------------------------------------

def posterior_finish_probs(
    idata: az.InferenceData,
    grid_mean: float,
    grid_std: float,
    grid_positions: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Compute P(finish=k | grid=x) for each grid position, averaged over
    all posterior draws.

    Returns a DataFrame with columns: grid_position, finish_position, probability.
    """
    if grid_positions is None:
        grid_positions = np.arange(1, 21)

    posterior = idata.posterior
    beta_samples = posterior["beta"].values.flatten()          # (draws,)
    cutpoints_samples = posterior["cutpoints"].values.reshape(-1, posterior["cutpoints"].shape[-1])  # (draws, K-1)

    K = cutpoints_samples.shape[1] + 1
    records = []

    for gp in grid_positions:
        gp_z = (gp - grid_mean) / grid_std
        eta = beta_samples * gp_z    # (draws,)

        # Cumulative probabilities via logistic CDF
        # P(finish <= k) = logistic(cutpoints[k] - eta)
        cum_probs = 1.0 / (1.0 + np.exp(-(cutpoints_samples - eta[:, None])))  # (draws, K-1)

        # Cell probabilities: P(finish = k) = P(finish<=k) - P(finish<=k-1)
        # Prepend 0 and append 1 for boundary conditions
        zeros = np.zeros((cum_probs.shape[0], 1))
        ones  = np.ones((cum_probs.shape[0], 1))
        cum_full = np.hstack([zeros, cum_probs, ones])           # (draws, K+1)
        cell_probs = np.diff(cum_full, axis=1)                    # (draws, K)

        mean_probs = cell_probs.mean(axis=0)                      # (K,)

        for k, prob in enumerate(mean_probs):
            records.append({
                "grid_position": int(gp),
                "finish_position": k + 1,     # back to 1-indexed
                "probability": float(prob),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_diagnostics(idata: az.InferenceData) -> None:
    summary = az.summary(idata, var_names=["beta", "cutpoints"], round_to=3)
    print("\n--- Posterior summary ---")
    print(summary.to_string())

    rhats = summary["r_hat"]
    if (rhats > 1.05).any():
        print("\n⚠  Some Rhat values > 1.05 — consider more tuning steps.")
    else:
        print("\n✓ All Rhat ≤ 1.05 — chains have converged.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_heatmap(prob_df: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of P(finish=y | grid=x) for all 20×20 combinations."""
    pivot = prob_df.pivot(
        index="grid_position", columns="finish_position", values="probability"
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", origin="upper")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Finish position")
    ax.set_ylabel("Grid position")
    ax.set_title("P(finish = y | grid = x)")

    plt.colorbar(im, ax=ax, label="Probability")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"✓ Heatmap saved to {out_path}")


def plot_selected_grids(
    prob_df: pd.DataFrame,
    grid_positions: list[int],
    out_path: Path,
) -> None:
    """Bar chart of finish distribution for selected grid positions."""
    fig, axes = plt.subplots(1, len(grid_positions), figsize=(4 * len(grid_positions), 4), sharey=True)

    for ax, gp in zip(axes, grid_positions):
        sub = prob_df[prob_df["grid_position"] == gp].sort_values("finish_position")
        ax.bar(sub["finish_position"], sub["probability"], color="steelblue", alpha=0.8)
        ax.set_title(f"Grid P{gp}")
        ax.set_xlabel("Finish position")
        if ax is axes[0]:
            ax.set_ylabel("Probability")

    plt.suptitle("P(finish = y | grid = x)", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ Bar charts saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bayesian ordinal model: grid → finish")
    p.add_argument("--data",          default="./data/grid-results.csv", type=Path)
    p.add_argument("--draws",         default=2000, type=int)
    p.add_argument("--tune",          default=1000, type=int)
    p.add_argument("--chains",        default=4,    type=int)
    p.add_argument("--target-accept", default=0.9,  type=float)
    p.add_argument("--seed",          default=42,   type=int)
    p.add_argument("--out-dir",       default="./model", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data
    print("=== Loading data ===")
    grid_z, finish_0idx, grid_mean, grid_std = load_data(args.data)

    # 2. Model
    print("\n=== Building model ===")
    model = build_model(grid_z, finish_0idx)
    print(model.debug())

    # 3. Sample
    print("\n=== Sampling ===")
    idata = sample_model(
        model,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        seed=args.seed,
    )

    # 4. Diagnostics
    print_diagnostics(idata)

    # 5. Save trace
    trace_path = out_dir / "pymc_grid_trace.nc"
    idata.to_netcdf(str(trace_path))
    print(f"✓ Trace saved to {trace_path}")

    # 6. Posterior predictions
    print("\n=== Computing posterior predictions ===")
    prob_df = posterior_finish_probs(idata, grid_mean, grid_std)

    csv_path = out_dir / "pymc_grid_probs.csv"
    prob_df.to_csv(csv_path, index=False)
    print(f"✓ Probabilities saved to {csv_path}")

    # 7. Print sample predictions
    print("\n--- P(finish=1 | grid=x) for all grid positions ---")
    p1 = (
        prob_df[prob_df["finish_position"] == 1]
        .sort_values("grid_position")[["grid_position", "probability"]]
    )
    print(p1.to_string(index=False))

    print("\n--- P(finish in top 5 | grid=x) ---")
    top5 = (
        prob_df[prob_df["finish_position"] <= 5]
        .groupby("grid_position")["probability"]
        .sum()
        .reset_index()
        .rename(columns={"probability": "p_top5"})
    )
    print(top5.to_string(index=False))

    # 8. Plots
    print("\n=== Saving plots ===")
    plot_heatmap(prob_df, out_dir / "pymc_grid_heatmap.png")
    plot_selected_grids(prob_df, [1, 5, 10, 15, 20], out_dir / "pymc_grid_bars.png")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
